"""Fixed-shape DSpark backbone graph contracts.

The graph is intentionally narrower than vLLM's full-step graph: FreeToken's live
request objects still own sampling controls, so the RNG-sensitive sequential Markov
stage remains eager. These tests pin the compact graph inputs and the device-addressed
non-causal window that make the expensive parallel backbone safe to replay.
"""

from types import SimpleNamespace

import torch

import freetoken.attention.dsv4_sparse as sparse_mod
from freetoken.attention.dsv4_sparse import DSV4AttnMetadata
from freetoken.engine.graph import DraftGraphCaptureBuffer


def test_draft_graph_buffer_compacts_anchor_and_noise_from_target_spans():
    buf = DraftGraphCaptureBuffer.init(
        max_reqs=2,
        gamma=3,
        noise_token_id=99,
        device=torch.device("cpu"),
    )
    batch = SimpleNamespace(
        padded_size=2,
        spec_block=3,
        # Target verify is anchor + gamma, request-major.
        input_ids=torch.tensor([11, 70, 71, 72, 22, 80, 81, 82], dtype=torch.int32),
        positions=torch.tensor([4, 5, 6, 7, 20, 21, 22, 23], dtype=torch.int32),
        active_table_idx=torch.tensor([7, 9], dtype=torch.int64),
    )

    buf.copy_from(batch)

    assert buf.input_ids.view(2, 3).tolist() == [[11, 99, 99], [22, 99, 99]]
    assert buf.positions.view(2, 3).tolist() == [[4, 5, 6], [20, 21, 22]]
    assert buf.request_table_idx.tolist() == [7, 9]


def test_draft_window_is_eager_order_then_end_padding(monkeypatch):
    # Identity full-slot -> window-slot mapping plus 100 makes the selected absolute
    # positions directly visible in the assertion.
    monkeypatch.setattr(
        sparse_mod,
        "get_global_ctx",
        lambda: SimpleNamespace(
            kv_cache=SimpleNamespace(translate_full_to_window=lambda x: x + 100)
        ),
    )
    snap = torch.arange(20, dtype=torch.int64).view(2, 10)
    md = DSV4AttnMetadata(
        last_indices=torch.tensor([0, 0], dtype=torch.int32),
        full_snap=snap,
        window_ar=torch.arange(4),
    )
    pos = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int64)
    rows = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)

    write, topk = md.draft_window_ctx(pos, rows, num_reqs=2, span=3)

    assert write.tolist() == [102, 103, 104, 115, 116, 117]
    # Request 0 has only two context tokens: [0,1] + block [2,3,4] + end padding.
    assert topk[0, 0].tolist() == [100, 101, 102, 103, 104, -1, -1]
    # Request 1 has a full four-token context: [1,2,3,4] + block [5,6,7].
    assert topk[3, 0].tolist() == [111, 112, 113, 114, 115, 116, 117]
    assert torch.equal(topk[0], topk[1]) and torch.equal(topk[1], topk[2])


def test_markov_sampling_stays_outside_the_backbone_graph():
    import inspect

    from freetoken.models.deepseek_v4.dspark import DSparkDrafter

    backbone = inspect.getsource(DSparkDrafter.graph_backbone)
    assert "sample_block" not in backbone
    assert "draft_block" in backbone
