"""GLM-5.3-Flash MTP draft graphs: speculative cycles through the captured graphs must match the
same cycles run eagerly.

A tiny hybrid model with its MTP layer (routed experts replaced by the shared experts: no offload
cache here) captures the verify graph and the MTP draft graphs, then runs draft -> verify -> commit
cycles with partial and full acceptance. The verify logits and the next block's seed distribution
must equal an eager run from the same cache state. The draft graphs replay between the verify
graph's replays, which is the order that exposed stale capture-time tensors before.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("glm5_next_model_rig_mtp", os.path.join(_HERE, "test_glm5_next_model.py"))
_rig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rig)

K = 4
DEV = "cuda"


class _Req:
    def __init__(self, table_idx: int, cached_len: int, device_len: int, slot: int = 1):
        self.table_idx, self.cached_len, self.device_len = table_idx, cached_len, device_len
        self.linear_slot_idx, self.mamba_ping_pong = slot, None
        self.uid = 7
        self.sampling_params = SimpleNamespace(temperature=0.0, top_p=1.0, top_k=-1, is_greedy=True)

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


class _Ctx(SimpleNamespace):
    @contextmanager
    def forward_batch(self, batch):
        old, self.batch = self.batch, batch
        try:
            yield
        finally:
            self.batch = old


@pytest.fixture()
def mtp_rig(monkeypatch):
    from freetoken.attention.dsa_indexer_kpool import Glm5NextDSABackend
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.dsa_pool import KpoolDSAKVCache
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.glm5_next.args import set_mtp_enabled
    from freetoken.models.glm5_next.config import parse_config
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM
    from freetoken.utils.hf import RawConfigShim

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    set_mtp_enabled(True)
    raw = _rig._hf_config()._data
    raw["text_config"]["num_nextn_predict_layers"] = 1
    config = parse_config(RawConfigShim(raw))

    prev = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(DEV)
    try:
        model = Glm5NextForCausalLM(config)
    finally:
        torch.set_default_dtype(prev[0])
        torch.set_default_device(prev[1])
    gen = torch.Generator(device=DEV).manual_seed(0)
    model.load_state_dict({
        k: ((torch.randn(v.shape, device=DEV, generator=gen) * 0.02).abs() + 0.5 if "norm" in k
            else torch.randn(v.shape, device=DEV, generator=gen) * 0.02).to(v.dtype)
        for k, v in model.state_dict().items()
    })
    mtp = model.mtp_layers.op_list[0]
    mtp.mlp.forward = lambda x: mtp.mlp.shared_experts.forward(x)

    kv = KpoolDSAKVCache(
        latent_dim=_rig.LATENT, num_layers=3, num_pages=4, page_size=64, dtype=torch.bfloat16,
        device=torch.device(DEV), index_head_dim=_rig.IDX_D, num_index_layers=2, index_ratio=4, num_req_slots=4,
    )
    page_table = torch.full((3, 256), -1, dtype=torch.int32, device=DEV)
    page_table[0, :192] = torch.arange(192, dtype=torch.int32, device=DEV)
    page_table[2, :64] = torch.arange(192, 256, dtype=torch.int32, device=DEV)  # capture dummy row
    pool = LinearStatePool(config.linear_attention_group(), num_slots=4, dtype=torch.bfloat16, device=torch.device(DEV), tp_size=1)
    ctx = _Ctx(kv_cache=kv, page_table=page_table, linear_state_pool=pool, attn_backend=None, batch=None)
    for mod in (
        "freetoken.attention.dsa.get_global_ctx",
        "freetoken.models.glm5_next.kda.get_global_ctx",
        "freetoken.models.glm5_next.attention.get_global_ctx",
        "freetoken.models.glm5_next.model.get_global_ctx",
        "freetoken.layers.embedding.get_global_ctx",
    ):
        monkeypatch.setattr(mod, lambda: ctx)
    ctx.attn_backend = Glm5NextDSABackend(config)
    yield model, ctx
    set_mtp_enabled(False)


def test_draft_graph_cycles_match_eager(mtp_rig):
    from freetoken.models.glm5_next.model import _SpecJournal

    model, ctx = mtp_rig
    kv, lpool, backend = ctx.kv_cache, ctx.linear_state_pool, ctx.attn_backend

    def batch_for(ids, start, phase, req):
        b = _rig._batch(ctx, ids, start, phase)
        b.reqs = b.padded_reqs = [req]
        b.speculative, b.spec_block, b.spec_verify_decode, b.spec_journal = False, 0, False, None
        backend.prepare_metadata(b)
        return b

    def cache_tensors():
        return [lpool.recurrent_states, lpool.conv_states, kv._kv_buffer, kv._index_k_buffer, kv._tail_k, kv._tail_gate]

    torch.manual_seed(1)
    prompt = 40
    ids = torch.randint(0, _rig.VOCAB, (prompt + 30,)).tolist()
    b = batch_for(ids[:prompt], 0, "prefill", _Req(0, 0, prompt))
    with ctx.forward_batch(b):
        model.forward()
        hidden = model._target_features.hidden
        model._mtp_rows(torch.tensor(ids[1:prompt + 1], device=DEV), b.positions, hidden)
    seed_hidden = hidden[-1:].clone()

    stream = torch.cuda.Stream()
    backend._spec_buffers(K + 1)
    static = {name: torch.zeros(K + 1, dtype=torch.int32, device=DEV) for name in ("ids", "pos", "loc")}
    dummy = _Req(2, 0, K + 1, slot=3)
    before_capture = [t.clone() for t in cache_tensors()]
    with torch.cuda.stream(stream):
        cb = SimpleNamespace(
            reqs=[dummy], padded_reqs=[dummy], phase="prefill", is_prefill=True, is_decode=False, size=1,
            input_ids=static["ids"], positions=static["pos"], out_loc=static["loc"], speculative=True,
            spec_block=K, spec_verify_decode=True, spec_journal=None, fla_metadata=None, mm_embeds=None,
            active_table_idx=None,
        )
        static["pos"].copy_(torch.arange(K + 1, device=DEV))
        backend.prepare_for_spec_capture(cb)
        verify_graph = torch.cuda.CUDAGraph()
        with ctx.forward_batch(cb):
            model.forward()
            with torch.cuda.graph(verify_graph, stream=stream):
                verify_logits = model.forward()
        verify_features = model._target_features
        model.capture_draft_graphs(stream, verify_graph.pool(), dummy, K + 1, lambda: None)
    torch.cuda.current_stream().wait_stream(stream)
    for live, saved in zip(cache_tensors(), before_capture):
        live.copy_(saved)
    draft_graphs = dict(model._mtp_graphs)
    assert sorted(draft_graphs) == list(range(1, K + 2))
    start_state = [t.clone() for t in cache_tensors()]

    def cycles(use_graphs: bool):
        for live, saved in zip(cache_tensors(), start_state):
            live.copy_(saved)
        model._mtp_graphs = draft_graphs if use_graphs else {}
        pos, h = prompt, seed_hidden
        logits_out, seeds = [], []
        for cycle in range(4):
            vb = batch_for([ids[pos]] + [0] * K, pos, "prefill", _Req(0, pos, pos + K + 1))
            vb.speculative, vb.spec_block = True, K
            with ctx.forward_batch(vb):
                tok = torch.tensor([ids[pos + 1]], device=DEV)
                for j in range(1, K):
                    h, _ = model._mtp_run(vb, pos, j - 1, j, tok, h)
                    tok = torch.tensor([ids[pos + 1 + j]], device=DEV)
                vb.input_ids = torch.tensor(ids[pos:pos + K + 1], dtype=torch.int32, device=DEV)
                vb.spec_journal = _SpecJournal(start=pos, ring_k=kv._tail_k[:, 0].clone(), ring_gate=kv._tail_gate[:, 0].clone())
                if use_graphs:
                    static["ids"].copy_(vb.input_ids)
                    static["pos"].copy_(vb.positions)
                    static["loc"].copy_(vb.out_loc)
                    vb.input_ids, vb.positions, vb.out_loc = static["ids"], static["pos"], static["loc"]
                    vb.spec_verify_decode = True
                    backend.prepare_for_spec_replay(vb)
                    verify_graph.replay()
                    logits, features = verify_logits.float().clone(), verify_features
                else:
                    logits, features = model.forward().float(), model._target_features
                n = 2 if cycle % 2 == 0 else K
                emitted = torch.tensor(ids[pos + 1:pos + n + 2], dtype=torch.int32)
                model.commit_speculative(vb, [n], [emitted], features)
            h = model._draft_seed.hidden
            logits_out.append(logits[: n + 1])
            seeds.append(model._draft_seed.probs.float().clone())
            pos += n + 1
        torch.cuda.synchronize()
        return torch.cat(logits_out), torch.stack(seeds)

    graph_logits, graph_seeds = cycles(True)
    eager_logits, eager_seeds = cycles(False)
    assert torch.isfinite(graph_logits).all()
    err = (graph_logits - eager_logits).abs().max() / (eager_logits.abs().max() + 1e-8)
    assert err < 3e-2, f"verify logits through graphs diverge from eager: {err}"
    assert (graph_seeds - eager_seeds).abs().max() < 1e-3, "next-block seed distribution diverges"
