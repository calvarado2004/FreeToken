"""DSpark verification must never enter prompt-style whole-layer MoE streaming."""

from types import SimpleNamespace

import torch

from freetoken.layers.moe import OffloadMoELayer
from freetoken.models.deepseek_v4 import moe as dsv4_moe


class _RoutingProbe(dsv4_moe.DSV4OffloadMoELayer):
    """BaseOP construction without allocating the multi-GB expert banks."""

    def __init__(self, cache):
        self.layer_id = 0
        self.top_k = 12
        self.num_experts = 64
        self.offload_cache = cache


def _layer(cache):
    return _RoutingProbe(cache)


def _routes(rows=6):
    return (
        torch.zeros((rows, 8), dtype=torch.bfloat16),
        torch.zeros((rows, 12), dtype=torch.float32),
        torch.zeros((rows, 12), dtype=torch.int32),
    )


def test_wide_hybrid_verify_uses_decode_before_prompt_crossover(monkeypatch):
    cache = SimpleNamespace(decode_target="hybrid")
    layer = _layer(cache)
    expected = torch.tensor([17.0])
    layer._decode_routed = lambda *_: expected
    monkeypatch.setattr(
        dsv4_moe,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(speculative=True)),
    )
    monkeypatch.setattr(
        OffloadMoELayer,
        "_prefill_routed",
        lambda *_: (_ for _ in ()).throw(AssertionError("entered prefill overlap")),
    )

    assert layer._prefill_routed(*_routes()) is expected


def test_wide_offload_verify_uses_on_demand_slots(monkeypatch):
    calls = []

    class Cache:
        decode_target = "offload"
        collect_stats = False

        def ensure_experts(self, layer_id, ids):
            calls.append(("ensure", layer_id))

        def copy_missing(self):
            calls.append(("copy",))

        def bank_views(self):
            return ()

        def alphas_for_slots(self, layer_id):
            return None

    layer = _layer(Cache())
    expected = torch.tensor([23.0])
    layer._expert_gemm = lambda *args, **kwargs: expected
    monkeypatch.setattr(
        dsv4_moe,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(speculative=True)),
    )
    monkeypatch.setattr(
        OffloadMoELayer,
        "_prefill_routed",
        lambda *_: (_ for _ in ()).throw(AssertionError("entered prefill overlap")),
    )

    assert layer._prefill_routed(*_routes()) is expected
    assert calls == [("ensure", 0), ("copy",)]


def test_wide_prompt_still_uses_prefill_overlap_crossover(monkeypatch):
    layer = _layer(SimpleNamespace(decode_target="offload"))
    expected = torch.tensor([31.0])
    monkeypatch.setattr(
        dsv4_moe,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=SimpleNamespace(speculative=False)),
    )
    monkeypatch.setattr(OffloadMoELayer, "_prefill_routed", lambda *_: expected)

    assert layer._prefill_routed(*_routes()) is expected
