"""GLM-5.3-Flash speculative verify: rolling state back to the accepted prefix.

A verify runs the anchor plus ``k`` drafted tokens as one extend. KDA keeps a recurrent
state and a conv window, and the kpool indexer keeps per-request tail rings; all three
would absorb the rejected rows. After ``commit_speculative`` the model must be exactly
where token-by-token decoding of the accepted prefix leaves it: same verify logits on the
kept rows, same KDA state, same conv window, same rings, and the same logits for every
token decoded afterwards.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("glm5_next_model_rig", os.path.join(_HERE, "test_glm5_next_model.py"))
_rig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rig)
rig = _rig.rig  # the fixture

K = 4


def _state(ctx) -> dict[str, torch.Tensor]:
    pool, kv = ctx.linear_state_pool, ctx.kv_cache
    return {
        "recurrent": pool.recurrent_states[:, 1].float().clone(),
        "conv": pool.conv_states[:, 1].float().clone(),
        "ring_k": kv._tail_k[:, 0].float().clone(),
        "ring_gate": kv._tail_gate[:, 0].float().clone(),
    }


def _close(got: torch.Tensor, want: torch.Tensor, what: str, tol: float = 3e-2) -> None:
    err = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item() + 1e-8
    assert err / scale < tol, f"{what}: {err} (scale {scale})"


def _cache_tensors(ctx) -> list[torch.Tensor]:
    pool, kv = ctx.linear_state_pool, ctx.kv_cache
    return [pool.recurrent_states, pool.conv_states, kv._kv_buffer, kv._index_k_buffer, kv._tail_k, kv._tail_gate]


def _verify(model, ctx, batch, mode: str) -> torch.Tensor:
    """Run the prepared verify block eagerly on the prefill path, eagerly on the
    decode-shaped graph path, or captured into a CUDA graph and replayed."""
    if mode == "prefill":
        return model.forward().float()
    batch.spec_verify_decode = True
    ctx.attn_backend.prepare_for_spec_replay(batch)
    if mode == "decode":
        return model.forward().float()
    saved = [t.clone() for t in _cache_tensors(ctx)]
    journal = batch.spec_journal
    batch.spec_journal = None
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        model.forward()  # warmup (autotune) outside the graph
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = model.forward()
    for live, before in zip(_cache_tensors(ctx), saved):
        live.copy_(before)  # capture ran the block twice; replay must start clean
    batch.spec_journal = journal
    ctx.attn_backend.prepare_for_spec_replay(batch)
    graph.replay()
    return out.float()


@pytest.mark.parametrize("mode", ["prefill", "decode", "graph"])
@pytest.mark.parametrize("start", [40, 42])
@pytest.mark.parametrize("accepted", [0, 2, K])
def test_verify_commit_matches_token_by_token_decode(rig, start, accepted, mode):
    from freetoken.models.glm5_next.model import _SpecJournal

    model, ctx = rig
    torch.manual_seed(3)
    total = start + K + 12  # decode past index_topk so the pooled indexer selects
    ids = torch.randint(0, _rig.VOCAB, (total,)).tolist()

    _rig._reset(ctx)
    _rig._batch(ctx, ids[:start], 0, "prefill")
    model.forward()
    ref_logits, ref_state = [], None
    for t in range(start, total):
        _rig._batch(ctx, ids[t : t + 1], t, "decode")
        ref_logits.append(model.forward().float()[0])
        if t == start + accepted:
            ref_state = _state(ctx)

    _rig._reset(ctx)
    _rig._batch(ctx, ids[:start], 0, "prefill")
    model.forward()
    # anchor + the accepted drafts, then drafts the target disagrees with
    block = ids[start : start + 1 + accepted] + [
        (ids[start + 1 + j] + 1) % _rig.VOCAB for j in range(accepted, K)
    ]
    batch = _rig._batch(ctx, block, start, "prefill")
    batch.speculative, batch.spec_block = True, K
    kv = ctx.kv_cache
    batch.spec_journal = _SpecJournal(
        start=start, ring_k=kv._tail_k[:, 0].clone(), ring_gate=kv._tail_gate[:, 0].clone()
    )
    rows = _verify(model, ctx, batch, mode)
    assert rows.shape == (K + 1, _rig.VOCAB)
    for j in range(accepted + 1):
        _close(rows[j], ref_logits[j], f"verify row {j}")

    model.commit_speculative(batch, [accepted], [torch.tensor(block[1 : accepted + 2])])
    got = _state(ctx)
    for name in ("recurrent", "conv"):
        _close(got[name], ref_state[name], f"{name} after commit", tol=1e-2)
    for name in ("ring_k", "ring_gate"):
        # only the residues of committed positions are live; the rest are never read
        _close(got[name], ref_state[name], f"{name} after commit", tol=1e-2)

    for t in range(start + accepted + 1, total):
        _rig._batch(ctx, ids[t : t + 1], t, "decode")
        _close(model.forward().float()[0], ref_logits[t - start], f"decode at {t} after commit")


def test_kpool_plan_is_rebuilt_inside_a_capture(rig):
    """A cached write plan built eagerly (a capture's warmup) must not be reused by the captured
    forward: its tensors live outside the graph pool and are freed after capture. The MTP draft
    graphs hold only the MTP indexer slot, so no slot 0 rebuilds it for them."""
    model, ctx = rig
    _rig._reset(ctx)
    batch = _rig._batch(ctx, [1, 2, 3], 8, "prefill")
    ctx.attn_backend.prepare_for_spec_replay(batch)
    md = batch.attn_metadata
    backend = ctx.attn_backend

    eager = backend._plan_kpool_writes(md, batch, slot=1)
    assert backend._plan_kpool_writes(md, batch, slot=1) is eager  # same forward reuses it
    assert md.kpool_plan_in_graph is False

    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            captured = backend._plan_kpool_writes(md, batch, slot=1)
    torch.cuda.current_stream().wait_stream(stream)
    assert captured is not eager
    assert md.kpool_plan_in_graph is True
