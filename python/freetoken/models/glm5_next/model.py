"""GLM-5.3-Flash (glm5_next) model: hybrid KDA/DSA decoder with mHC residual streams.

Layer layout comes from the checkpoint's ``layer_types`` (34 KDA linear-attention
layers, 11 NoPE-MLA/DSA layers at 3:1) and ``mlp_layer_types`` (3 dense + 42 MoE).
The residual stream is mHC-widened to ``hc_mult`` (4) parallel streams:

    layer 0:  residual = hc_expand(x);  (post, comb, x) = mhc_pre(residual, hc_attn_*)
    each sublayer boundary fuses the previous hc_post with the next hc_pre
    (mhc_fused_post_pre), and the sublayer input is RMS-normed AFTER the mix
    (the reference fuses the norm into its hc kernels; decomposed here, same math).
    last layer: x = mhc_post(...); x = hc_contract(x)  -> final norm -> lm_head.

The deferred (post, comb) pair threads through the layer loop exactly like
glm_moe_dsa's (x, residual) pair. lm_head quant mirrors glm_moe_dsa (optional
W8A16 fp8 at load; the ~1.2 GiB bf16 head is read every decode step).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, NamedTuple, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.layers.mhc import hc_contract, hc_expand, mhc_fused_post_pre, mhc_post, mhc_pre
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.utils import init_logger, nvtx_annotate

from .attention import Glm5NextAttention
from .kda import Glm5NextKDA
from .mlp import Glm5NextGatedMLP
from .moe import Glm5NextSparseBlock
from .vision import Glm5NextVisionModel

if TYPE_CHECKING:
    from freetoken.message import MMItem
    from freetoken.models.config import ModelConfig

logger = init_logger(__name__)

# FREETOKEN_GLM_MTP_PROBE=1: on single-request prefills, log how often the MTP layer's
# next-next-token argmax agrees with the target's (an offline acceptance estimate).
_MTP_PROBE = os.environ.get("FREETOKEN_GLM_MTP_PROBE", "0") == "1"
# FREETOKEN_GLM_MTP_PROBE_EXACT=1 (with the probe): also run the MTP layer with its routed experts
# dequantized from the checkpoint's block FP8, to price the NVFP4 re-quantization.
_MTP_PROBE_EXACT = os.environ.get("FREETOKEN_GLM_MTP_PROBE_EXACT", "0") == "1"
# FREETOKEN_GLM_MTP_GRAPH_CHECK=N: compare the first N MTP graph replays against the eager layer.
_MTP_GRAPH_CHECK = int(os.environ.get("FREETOKEN_GLM_MTP_GRAPH_CHECK", "0") or 0)
# FREETOKEN_GLM_MTP_GRAPHS=0: run the MTP layer eagerly (no draft graphs).
_MTP_GRAPHS = os.environ.get("FREETOKEN_GLM_MTP_GRAPHS", "1") != "0"


class Glm5NextDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.glm5_args
        self._layer_id = layer_id
        self._is_last = layer_id == config.num_layers - 1
        self.mhc = args.mhc
        self._n = args.mhc_num_residual_streams
        self._hc_eps = args.hc_eps
        self._rms_eps = args.norm_eps
        self._post_mult = args.mhc_post_mult_value
        self._sinkhorn = args.mhc_sinkhorn_iterations

        if args.is_kda_layer(layer_id):
            self.self_attn: BaseOP = Glm5NextKDA(config, layer_id, prefix=f"{prefix}.self_attn")
        else:
            self.self_attn = Glm5NextAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = Glm5NextSparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
        else:
            self.mlp = Glm5NextGatedMLP(
                config.hidden_size, config.intermediate_size, swiglu_limit=config.swiglu_limit,
                tensor_parallel=True, quant_config=config.quant, prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(size=config.hidden_size, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(size=config.hidden_size, eps=args.norm_eps)

        if self.mhc:
            n, hidden = self._n, config.hidden_size
            mix = 2 * n + n * n
            # fp32 mHC weights (models/weight.py exempts hc_* from the dtype downcast).
            self.hc_attn_fn = torch.empty(mix, n * hidden, dtype=torch.float32)
            self.hc_attn_base = torch.empty(mix, dtype=torch.float32)
            self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
            self.hc_ffn_fn = torch.empty(mix, n * hidden, dtype=torch.float32)
            self.hc_ffn_base = torch.empty(mix, dtype=torch.float32)
            self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def _pre(self, residual, fn, scale, base):
        # Layer 0's standalone pre rides the fused kernel too (HAS_POST=False
        # path; x/post/comb are the no-post sentinels) -- same dispatch, same
        # numerics, and the kernel wins at every batch size (see layers/mhc.py).
        if residual.is_cuda:
            _, post, comb, x = mhc_fused_post_pre(
                residual.new_empty(residual.shape[0], residual.shape[-1]),
                residual, None, None, fn, scale, base,
                self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
            )
            return post, comb, x
        return mhc_pre(
            residual, fn, scale, base,
            self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
        )

    def _fused(self, x, residual, post, comb, fn, scale, base):
        return mhc_fused_post_pre(
            x, residual, post, comb, fn, scale, base,
            self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if post is None:
            if residual is None:
                residual = hc_expand(x, self._n)
            post, comb, x = self._pre(
                residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        else:
            residual, post, comb, x = self._fused(
                x, residual, post, comb,
                self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            )
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x)

        residual, post, comb, x = self._fused(
            x, residual, post, comb,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
        )
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)

        if self._is_last:
            x = mhc_post(x, residual, post, comb)
            return hc_contract(x), None, None, None
        return x, residual, post, comb


class _SharedHead(BaseOP):
    """``shared_head``: the MTP layer's final norm; its projection is the model's lm_head."""

    def __init__(self, hidden_size: int, eps: float):
        self.norm = RMSNorm(size=hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm.forward(x)


class Glm5NextMTPLayer(BaseOP):
    """One GLM-5.3-Flash next-token-prediction layer (checkpoint ``layers.{num_layers + k}``).

    Ported from vLLM's ``Glm5NextMultiTokenPredictorLayer``: the token embedding (zeroed at
    position 0) and the previous hidden state are RMS-normed separately, concatenated and
    projected by ``eh_proj``; one MLA/DSA + MoE block runs WITHOUT mHC (plain pre-norm
    residuals); ``shared_head.norm`` gives the hidden state that feeds both the draft logits
    and the next draft step. The block keeps its own KV, indexer slot and expert bank under
    ``layer_id``.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args = config.glm5_args
        hidden, eps = config.hidden_size, args.norm_eps
        self._layer_id = layer_id
        self.enorm = RMSNorm(size=hidden, eps=eps)
        self.hnorm = RMSNorm(size=hidden, eps=eps)
        self.eh_proj = LinearReplicated(2 * hidden, hidden, has_bias=False, prefix=f"{prefix}.eh_proj")
        self.input_layernorm = RMSNorm(size=hidden, eps=eps)
        self.self_attn = Glm5NextAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = RMSNorm(size=hidden, eps=eps)
        # The checkpoint's quant config names this layer's experts block-FP8; they are served
        # re-quantized to NVFP4 (iter_mtp_expert_pieces), so the MoE takes the last decoder
        # layer's scheme -- one offload cache holds one expert format.
        self.mlp = Glm5NextSparseBlock(config, layer_id, prefix=f"model.layers.{config.num_layers - 1}.mlp")
        self.shared_head = _SharedHead(hidden, eps)

    @nvtx_annotate("MTP")
    def forward(
        self, embeds: torch.Tensor, positions: torch.Tensor, prev_hidden: torch.Tensor, moe=None,
    ) -> torch.Tensor:
        embeds = embeds.masked_fill((positions == 0).view(-1, 1), 0.0)
        x = self.eh_proj.forward(
            torch.cat([self.enorm.forward(embeds), self.hnorm.forward(prev_hidden)], dim=-1)
        )
        residual = x
        x = residual + self.self_attn.forward(self.input_layernorm.forward(x))
        residual = x
        x = residual + (moe or self.mlp.forward)(self.post_attention_layernorm.forward(x))
        return self.shared_head.forward(x)


class _ExactMTPExperts:
    """The MTP layer's MoE with routed experts dequantized from the checkpoint's block FP8
    (this rank's shard, bf16, one expert at a time) -- a reference for the served NVFP4 banks."""

    def __init__(self, model_path: str, layer_id: int):
        import json

        from freetoken.models.glm5_next.weight import _CKPT, _ShardReader
        from freetoken.utils import download_hf_weight

        folder = download_hf_weight(model_path)
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        self._reader = _ShardReader(folder, weight_map, torch.device("cpu"))
        self._prefix = f"{_CKPT}.layers.{layer_id}.mlp.experts"

    def _weight(self, e: int, proj: str, device) -> torch.Tensor:
        from freetoken.distributed import get_tp_info
        from freetoken.models.glm5_next.weight import _dequant_fp8_block

        src = f"{self._prefix}.{e}.{proj}"
        w = _dequant_fp8_block(self._reader.get(f"{src}.weight").to(device), self._reader.get(f"{src}.weight_scale").to(device))
        tp = get_tp_info()
        axis = 1 if proj == "down_proj" else 0
        step = w.shape[axis] // tp.size
        return w.narrow(axis, tp.rank * step, step).to(torch.bfloat16)

    @torch.no_grad()
    def __call__(self, block: Glm5NextSparseBlock, x: torch.Tensor) -> torch.Tensor:
        from freetoken.layers import swiglu_clamp_and_mul

        weights, ids = block._route(x)
        shared = block.shared_experts.forward(x)
        experts = block.experts
        out = torch.zeros_like(x)
        for e in ids.unique().tolist():
            rows, slot = (ids == e).nonzero(as_tuple=True)
            xe = x[rows]
            gate = xe @ self._weight(e, "gate_proj", x.device).T
            up = xe @ self._weight(e, "up_proj", x.device).T
            h = swiglu_clamp_and_mul(torch.cat([gate, up], dim=-1), alpha=float(experts.alpha), limit=float(experts.limit))
            y = h @ self._weight(e, "down_proj", x.device).T
            out.index_add_(0, rows, y * weights[rows, slot].unsqueeze(-1).to(y.dtype))
        return experts._maybe_all_reduce(out) + shared


class MTPTargetFeatures(NamedTuple):
    """The target forward's final hidden rows (a CUDA-graph output under replay)."""

    hidden: torch.Tensor


@dataclass
class _DraftSeed:
    """The MTP step already taken for a request's next block: its first proposal."""

    uid: int
    position: int          # the verify anchor position this seed belongs to
    token: torch.Tensor    # [1] proposal for position + 1
    probs: torch.Tensor    # [vocab] the distribution it was drawn from
    hidden: torch.Tensor   # [1, hidden] MTP output that produced it (recycled by step 2)


@dataclass
class _SpecJournal:
    """Per-verify state the commit needs once acceptance is known."""

    start: int                          # anchor position
    ring_k: torch.Tensor | None         # [index_layers, kpool, Di] tail rings before drafting
    ring_gate: torch.Tensor | None
    kda: list = field(default_factory=list)    # KDAVerifyRows, one per KDA layer
    index: list = field(default_factory=list)  # (slot, k [rows, Di], gate [rows, Di])


class Glm5NextModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Glm5NextDecoderLayer(config, i, prefix=f"{prefix}.layers.{i}") for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        residual = post = comb = None
        for layer in self.layers.op_list:
            x, residual, post, comb = layer.forward(x, residual, post, comb)
        return self.norm.forward(x)


class Glm5NextForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self._config = config
        self.model = Glm5NextModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        # Built after lm_head so the offload walk yields the MTP experts after the target's.
        mtp_ids = config.glm5_args.mtp_layer_ids if config.num_layers == len(config.glm5_args.layer_types) else ()
        self.mtp_layers = OPList(
            [Glm5NextMTPLayer(config, lid, prefix=f"mtp_layers.{k}") for k, lid in enumerate(mtp_ids)]
        )
        self._last_hidden: torch.Tensor | None = None
        self._target_features: MTPTargetFeatures | None = None
        self._draft_seed: _DraftSeed | None = None
        self._think_end_id: int | None = None
        self._thinking_width: int | None = None
        self._think_scan: tuple = (None, 0, False)
        # Verify-graph journals by span: a replay runs no Python, so the rows the capture
        # recorded (graph-owned tensors, rewritten by every replay) are what commit reads.
        self._graph_journals: dict[int, _SpecJournal] = {}
        # MTP layer graphs by row count (draft steps, catch-ups), sharing the verify staging.
        self._mtp_graphs: dict[int, tuple] = {}
        self._mtp_buf: dict[str, torch.Tensor] | None = None
        self._mtp_check_left = _MTP_GRAPH_CHECK

    def full_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits for EVERY row of ``h`` (ParallelLMHead.forward keeps one row per
        request); rank-major all_gather concatenates the vocabulary shards in order."""
        head = self.lm_head
        local = head.quant_method.apply(head, h) if head.tied_embedding is None else torch.nn.functional.linear(h, head.tied_embedding.weight)
        if head.tp_size == 1:
            return local[:, : head.num_embeddings]
        rows = local.shape[0]
        gathered = head._comm.all_gather(local).view(head.tp_size, rows, -1)
        return gathered.permute(1, 0, 2).reshape(rows, -1)[:, : head.num_embeddings]

    @torch.no_grad()
    def _probe_mtp(self, input_ids: torch.Tensor, hidden: torch.Tensor) -> None:
        """Teacher-forced MTP agreement over one prompt: the MTP output at position t (fed token
        t+1 and the target hidden at t) should predict what the target predicts at t+1."""
        batch = get_global_ctx().batch
        n = hidden.shape[0]
        if not self.mtp_layers.op_list or not batch.is_prefill or batch.size != 1 or n < 4:
            return
        if getattr(batch, "speculative", False) or torch.cuda.is_current_stream_capturing():
            return  # a verify block or a graph capture, not a prompt
        ids = input_ids.view(-1)
        next_ids = torch.cat([ids[1:], ids[-1:]])
        embeds = self.model.embed_tokens.forward(next_ids)
        mtp = self.mtp_layers.op_list[0]
        mtp_h = mtp.forward(embeds, batch.positions.view(-1), hidden)
        exact_h = None
        if _MTP_PROBE_EXACT:
            if getattr(self, "_exact_experts", None) is None:
                self._exact_experts = _ExactMTPExperts(self._model_path, mtp._layer_id)
            exact_h = mtp.forward(
                embeds, batch.positions.view(-1), hidden, moe=lambda x: self._exact_experts(mtp.mlp, x)
            )
        # MTP row t sees token t+1 and the target hidden at t, so it should predict token t+2: the
        # target's own argmax at t+1 and the prompt's token t+2. The t / t+2 columns catch an
        # off-by-one; the target-vs-truth column says how predictable the text itself is.
        from freetoken.models.deepseek_v4.dspark import sampling_probs

        counts = {"target@t+1": 0, "target@t": 0, "target@t+2": 0, "truth@t+2": 0, "target_vs_truth": 0}
        if exact_h is not None:
            counts.update({"exact@t+1": 0, "exact_vs_nvfp4": 0})
        # Expected per-token acceptance of standard speculative sampling, sum_x min(p, q), at the
        # recommended sampling (temperature 1.0, top_p 0.95).
        expected = {"E[accept] nvfp4": 0.0}
        if exact_h is not None:
            expected["E[accept] exact"] = 0.0
        total = 0
        for lo in range(1, n - 3, 512):
            hi = min(lo + 512, n - 3)
            draft_logits = self.full_logits(mtp_h[lo:hi])
            draft = draft_logits.argmax(-1)
            target_logits = self.full_logits(hidden[lo - 1:hi + 2])  # rows lo-1 .. hi+1
            target = target_logits.argmax(-1)
            p = sampling_probs(target_logits[2:hi - lo + 2], 1.0, 0.95)
            expected["E[accept] nvfp4"] += float(torch.minimum(p, sampling_probs(draft_logits, 1.0, 0.95)).sum())
            if exact_h is not None:
                exact_logits = self.full_logits(exact_h[lo:hi])
                exact = exact_logits.argmax(-1)
                counts["exact@t+1"] += int((exact == target[2:hi - lo + 2]).sum())
                counts["exact_vs_nvfp4"] += int((exact == draft).sum())
                expected["E[accept] exact"] += float(torch.minimum(p, sampling_probs(exact_logits, 1.0, 0.95)).sum())
            t0 = target[1:hi - lo + 1]
            counts["target@t"] += int((draft == t0).sum())
            counts["target@t+1"] += int((draft == target[2:hi - lo + 2]).sum())
            counts["target@t+2"] += int((draft == target[3:hi - lo + 3]).sum())
            counts["truth@t+2"] += int((draft == ids[lo + 2:hi + 2]).sum())
            counts["target_vs_truth"] += int((target[2:hi - lo + 2] == ids[lo + 2:hi + 2]).sum())
            total += hi - lo
        summary = ", ".join(f"{k} {100.0 * v / max(total, 1):.1f}%" for k, v in {**counts, **expected}.items())
        logger.info_rank0(f"MTP probe over {total} positions: {summary}")

    # ----- MTP speculative decoding --------------------------------------------------------
    # Serving shape: one request (max_running_req == 1), like DSpark's adaptive path.
    #
    # Alignment (validated by _probe_mtp): the MTP row at position t is fed token t+1 and the
    # target hidden at t, and predicts token t+2. So after the target has scored positions
    # ..p and sampled token p+1, one MTP row at p yields the proposal for p+2 -- the first
    # draft of a verify anchored at p+1. Later steps recycle the MTP hidden in place of the
    # target's. The MTP layer's KV at a position is always rewritten from the TARGET hidden
    # once that position is committed (catch-up), so draft-step KV never outlives a block.

    def dspark_target_features(self) -> MTPTargetFeatures | None:
        return self._target_features

    def speculative_width(self, req, max_width: int) -> int:
        """Narrow blocks while the request is still inside ``<think>`` (lower acceptance)."""
        if self._thinking_width is None or self._think_end_id is None:
            return max_width
        uid, scanned, closed = self._think_scan
        if uid != req.uid:
            scanned, closed = req.max_device_len - req.output_len, False  # generated tokens only
        if not closed:
            ids = req.input_ids
            closed = bool((ids[scanned:] == self._think_end_id).any())
            scanned = ids.numel()
        self._think_scan = (req.uid, scanned, closed)
        return max_width if closed else min(max_width, self._thinking_width)

    def configure_speculative_phases(self, think_end_id: int | None, thinking_width: int | None) -> None:
        self._think_end_id = think_end_id
        self._thinking_width = thinking_width
        self._think_scan = (None, 0, False)

    @contextmanager
    def _rows(self, batch, start: int, lo: int, hi: int):
        """Run the MTP layer over rows ``[lo, hi)`` of a single-request verify batch.

        Those rows' positions (``start + lo``..) already own paged slots (allocated for the
        whole block), so only the view changes: positions, out_loc and the request's
        cached/device length, then attention metadata rebuilt for that range.
        """
        ctx = get_global_ctx()
        req = batch.reqs[0]
        saved = (batch.positions, batch.out_loc, batch.attn_metadata, req.cached_len, req.device_len)
        batch.positions = saved[0][lo:hi]
        batch.out_loc = saved[1][lo:hi]
        req.cached_len, req.device_len = start + lo, start + hi
        ctx.attn_backend.prepare_metadata(batch)
        try:
            yield
        finally:
            batch.positions, batch.out_loc, batch.attn_metadata = saved[:3]
            req.cached_len, req.device_len = saved[3:]

    def _propose(self, logits: torch.Tensor, sp) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw one proposal from an MTP logits row, shaped like the target's sampler."""
        from freetoken.models.deepseek_v4.dspark import sampling_probs

        probs = sampling_probs(logits, sp.temperature, sp.top_p, sp.top_k)
        token = probs.argmax(dim=-1) if sp.is_greedy else torch.multinomial(probs, 1).squeeze(-1)
        return token, probs[0]

    def _mtp_rows(self, tokens: torch.Tensor, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        embeds = self.model.embed_tokens.forward(tokens)
        return self.mtp_layers.op_list[0].forward(embeds, positions, hidden)

    @torch.no_grad()
    def capture_draft_graphs(self, stream, pool, dummy_req, max_span: int, reset_moe) -> None:
        """Capture the MTP layer (+ last-row logits) for 1 .. ``max_span`` rows of one request."""
        from freetoken.core import Batch

        ctx = get_global_ctx()
        stage = getattr(ctx.attn_backend, "_stage_spec_verify", None)
        if not self.mtp_layers.op_list or stage is None or max_span < 1 or not _MTP_GRAPHS:
            return
        dev = ctx.page_table.device
        hidden_size = self._config.hidden_size
        buf = self._mtp_buf = {
            "tokens": torch.zeros(max_span, dtype=torch.int64, device=dev),
            "positions": torch.zeros(max_span, dtype=torch.int32, device=dev),
            "out_loc": torch.zeros(max_span, dtype=torch.int32, device=dev),
            "hidden": torch.zeros(max_span, hidden_size, dtype=torch.bfloat16, device=dev),
        }
        slot = dummy_req.linear_slot_idx if dummy_req.linear_slot_idx is not None else dummy_req.table_idx
        for span in range(max_span, 0, -1):
            batch = Batch(reqs=[dummy_req], phase="decode")
            batch.padded_reqs = batch.reqs
            batch.input_ids = buf["tokens"][:span]
            batch.positions = buf["positions"][:span]
            batch.out_loc = buf["out_loc"][:span]
            buf["positions"][:span].copy_(torch.arange(span, dtype=torch.int32, device=dev))
            stage(batch, dummy_req.table_idx, slot, 0)
            graph = torch.cuda.CUDAGraph()
            with ctx.forward_batch(batch):
                self._mtp_rows(buf["tokens"][:span], buf["positions"][:span], buf["hidden"][:span])
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    out = self._mtp_rows(buf["tokens"][:span], buf["positions"][:span], buf["hidden"][:span])
                    logits = self.full_logits(out[span - 1 : span])
                reset_moe()
            self._mtp_graphs[span] = (graph, out, logits)
        logger.info_rank0(f"Captured MTP draft graphs for 1..{max_span} rows")

    def _mtp_run(self, batch, start: int, lo: int, hi: int, tokens: torch.Tensor, hidden: torch.Tensor):
        """MTP output rows and last-row logits for positions ``start+lo .. start+hi-1`` of the
        batch's one request: a captured graph when one fits, else the eager layer."""
        span = hi - lo
        entry = self._mtp_graphs.get(span)
        if entry is None:
            if batch.is_decode and lo == 0 and hi == batch.positions.numel():
                out = self._mtp_rows(tokens, batch.positions[lo:hi], hidden)
            else:
                with self._rows(batch, start, lo, hi):
                    out = self._mtp_rows(tokens, batch.positions, hidden)
            return out, self.full_logits(out[span - 1 : span])
        graph, out, logits = entry
        buf = self._mtp_buf
        req = batch.reqs[0]
        buf["tokens"][:span].copy_(tokens)
        buf["positions"][:span].copy_(batch.positions[lo:hi])
        buf["out_loc"][:span].copy_(batch.out_loc[lo:hi])
        buf["hidden"][:span].copy_(hidden)
        backend = get_global_ctx().attn_backend
        saved = (batch.input_ids, batch.attn_metadata, batch.fla_metadata)
        batch.input_ids = buf["tokens"][:span]
        slot = req.linear_slot_idx if getattr(req, "linear_slot_idx", None) is not None else req.table_idx
        try:
            backend._stage_spec_verify(batch, req.table_idx, slot, start + lo)
            graph.replay()
        finally:
            batch.input_ids, batch.attn_metadata, batch.fla_metadata = saved
        if self._mtp_check_left > 0:
            self._mtp_check_left -= 1
            with self._rows(batch, start, lo, hi):
                ref = self._mtp_rows(tokens, batch.positions, hidden)
            err = (ref.float() - out[:span].float()).abs().max().item() / (ref.float().abs().max().item() + 1e-8)
            logger.info_rank0(f"MTP graph check: span={span} start={start + lo} rel_err={err:.2e}")
        return out[:span], logits

    def _seed(self, req, position: int, row_hidden: torch.Tensor, logits: torch.Tensor) -> None:
        token, probs = self._propose(logits, req.sampling_params)
        self._draft_seed = _DraftSeed(req.uid, position, token, probs, row_hidden.clone())

    @torch.no_grad()
    def commit_target_forward(self, batch, features: MTPTargetFeatures | None, next_tokens: torch.Tensor) -> None:
        """After an ordinary prefill/decode: MTP KV over its rows, and the next block's seed."""
        self._draft_seed = None
        if not self.mtp_layers.op_list or features is None or len(batch.reqs) != 1:
            return
        if getattr(batch, "padded_size", 1) != 1 or batch.mm_embeds is not None:
            return
        req = batch.reqs[0]
        n = batch.input_ids.numel() if batch.is_prefill else 1
        hidden = features.hidden[:n]
        # complete_one has run: cached_len is the position of the token after the last row.
        nxt = req.cached_len
        if req.input_ids.numel() > nxt:  # a chunked prompt continues
            follow = req.input_ids[nxt : nxt + 1].to(batch.input_ids.device, non_blocking=True)
        else:
            follow = next_tokens[:1]
        tokens = torch.cat([batch.input_ids[1:n].long(), follow.long().view(1)])
        if batch.is_prefill:
            out = self._mtp_rows(tokens, batch.positions[:n], hidden)
            logits = self.full_logits(out[n - 1 : n]) if req.input_ids.numel() <= nxt else None
        else:
            out, logits = self._mtp_run(batch, nxt - 1, 0, 1, tokens, hidden)
        if req.input_ids.numel() <= nxt:
            self._seed(req, nxt, out[n - 1 : n], logits)

    @torch.no_grad()
    def draft(self, sampling_params) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Propose ``spec_block`` tokens for the prepared verify batch (one request)."""
        if not self.mtp_layers.op_list:
            return None
        ctx = get_global_ctx()
        batch = ctx.batch
        if len(batch.reqs) != 1:
            raise RuntimeError("GLM MTP speculation serves one request per batch")
        req = batch.reqs[0]
        k = int(batch.spec_block)
        start = req.cached_len
        seed = self._draft_seed
        if seed is None or seed.uid != req.uid or seed.position != start:
            raise RuntimeError(
                f"MTP has no seed for request {req.uid} at position {start} "
                f"(have {None if seed is None else (seed.uid, seed.position)})"
            )
        kv = ctx.kv_cache
        if not hasattr(kv, "_tail_k"):
            raise RuntimeError("GLM MTP speculation needs the kpool DSA cache (FREETOKEN_GLM_DSA=1)")
        journal = _SpecJournal(
            start=start,
            ring_k=kv._tail_k[:, req.table_idx].clone(),
            ring_gate=kv._tail_gate[:, req.table_idx].clone(),
        )
        tokens, probs, hidden = [seed.token.view(1)], [seed.probs], seed.hidden
        sp = sampling_params[0]
        for j in range(1, k):
            # Step j+1 sits at position start + j - 1 (row j - 1 of the verify block).
            hidden, logits = self._mtp_run(batch, start, j - 1, j, tokens[-1].long(), hidden)
            token, q = self._propose(logits, sp)
            tokens.append(token.view(1))
            probs.append(q)
        proposed = torch.cat(tokens).long()
        q = torch.stack(probs)
        confidence = q.gather(1, proposed.view(-1, 1)).squeeze(1)
        # Journal only the verify that follows; the draft steps above wrote nothing to keep.
        batch.spec_journal = journal
        return proposed, q, confidence

    @torch.no_grad()
    def commit_speculative(
        self, batch, accepted: List[int], emitted: List[torch.Tensor], features: MTPTargetFeatures | None = None,
    ) -> None:
        """Roll per-token state back to the accepted prefix, then seed the next block."""
        journal: _SpecJournal | None = getattr(batch, "spec_journal", None)
        batch.spec_journal = None
        self._draft_seed = None
        if journal is None:
            raise RuntimeError("an MTP verify finished without its journal")
        if getattr(batch, "spec_verify_decode", False) and not journal.kda:
            captured = self._graph_journals.get(int(batch.spec_block) + 1)
            if captured is None:
                raise RuntimeError(f"no captured verify journal for span {int(batch.spec_block) + 1}")
            journal.kda, journal.index = captured.kda, captured.index
        ctx = get_global_ctx()
        req = batch.reqs[0]
        span = int(batch.spec_block) + 1
        n = accepted[0]
        for rows in journal.kda:
            rows.layer.commit_verify(rows, accepted, span)

        kv = ctx.kv_cache
        r = req.table_idx
        kv._tail_k[:, r].copy_(journal.ring_k)
        kv._tail_gate[:, r].copy_(journal.ring_gate)
        kp = kv._tail_k.shape[2]
        lo = max(0, n + 1 - kp)
        dev = batch.positions.device
        keep_rows = torch.arange(lo, n + 1, device=dev)
        residues = torch.tensor([(journal.start + i) % kp for i in range(lo, n + 1)], device=dev)
        for slot, k_rows, gate_rows in journal.index:
            kv.tail_k(slot)[r].index_copy_(0, residues, k_rows.index_select(0, keep_rows).to(kv._tail_k.dtype))
            kv.tail_gate(slot)[r].index_copy_(0, residues, gate_rows.index_select(0, keep_rows).to(kv._tail_gate.dtype))

        hidden = features.hidden if features is not None else self._last_hidden
        if hidden is None or not self.mtp_layers.op_list:
            return
        tokens = emitted[0].to(dev, non_blocking=True).long()  # accepted drafts + bonus
        out, logits = self._mtp_run(batch, journal.start, 0, n + 1, tokens, hidden[: n + 1])
        self._seed(req, journal.start + n + 1, out[n : n + 1], logits)

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing hook: materialize the DSA layers' bmm-ready
        kv_b splits and free the checkpoint-layout originals (glm_moe_dsa
        precedent)."""
        for layer in self.model.layers.op_list:
            if isinstance(layer.self_attn, Glm5NextAttention):
                layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        input_ids = batch.input_ids
        if getattr(batch, "spec_verify_decode", False) and torch.cuda.is_current_stream_capturing():
            # Graph capture of a verify: record this span's rows for every later replay.
            journal = _SpecJournal(start=0, ring_k=None, ring_gate=None)
            batch.spec_journal = journal
            self._graph_journals[input_ids.numel()] = journal
        elif getattr(batch, "spec_verify_decode", False) and getattr(batch, "spec_journal", None) is None:
            batch.spec_journal = _SpecJournal(start=0, ring_k=None, ring_gate=None)  # capture warmup
        output = self.model.forward(input_ids)
        if self.mtp_layers.op_list:
            self._last_hidden = output
            self._target_features = MTPTargetFeatures(output)
            if _MTP_PROBE:
                self._probe_mtp(input_ids, output)
        if getattr(batch, "speculative", False):
            return self.full_logits(output)  # acceptance reads every verify row
        return self.lm_head.forward(output)


class Glm5NextForConditionalGeneration(Glm5NextForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            self.visual = Glm5NextVisionModel(config.vision_config)

    def place_encoder_weights(self, mode: str) -> None:
        self.visual.place_weights(mode)

    def encode(self, item: MMItem) -> torch.Tensor:
        return self.visual.forward(item.feature, [item.grid_thw])


__all__ = ["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"]
