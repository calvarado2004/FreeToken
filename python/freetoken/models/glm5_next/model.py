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
from typing import TYPE_CHECKING, Tuple

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
        self.mlp = Glm5NextSparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
        self.shared_head = _SharedHead(hidden, eps)

    @nvtx_annotate("MTP")
    def forward(self, embeds: torch.Tensor, positions: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        embeds = embeds.masked_fill((positions == 0).view(-1, 1), 0.0)
        x = self.eh_proj.forward(
            torch.cat([self.enorm.forward(embeds), self.hnorm.forward(prev_hidden)], dim=-1)
        )
        residual = x
        x = residual + self.self_attn.forward(self.input_layernorm.forward(x))
        residual = x
        x = residual + self.mlp.forward(self.post_attention_layernorm.forward(x))
        return self.shared_head.forward(x)


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
        ids = input_ids.view(-1)
        next_ids = torch.cat([ids[1:], ids[-1:]])
        embeds = self.model.embed_tokens.forward(next_ids)
        mtp_h = self.mtp_layers.op_list[0].forward(embeds, batch.positions.view(-1), hidden)
        agree = total = 0
        for lo in range(0, n - 2, 512):
            hi = min(lo + 512, n - 2)
            draft = self.full_logits(mtp_h[lo:hi]).argmax(-1)
            target = self.full_logits(hidden[lo + 1:hi + 1]).argmax(-1)
            agree += int((draft == target).sum())
            total += hi - lo
        logger.info_rank0(f"MTP probe: {agree}/{total} teacher-forced top-1 agreement ({100.0 * agree / max(total, 1):.1f}%)")

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing hook: materialize the DSA layers' bmm-ready
        kv_b splits and free the checkpoint-layout originals (glm_moe_dsa
        precedent)."""
        for layer in self.model.layers.op_list:
            if isinstance(layer.self_attn, Glm5NextAttention):
                layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        input_ids = get_global_ctx().batch.input_ids
        output = self.model.forward(input_ids)
        if self.mtp_layers.op_list:
            self._last_hidden = output
            if _MTP_PROBE:
                self._probe_mtp(input_ids, output)
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
