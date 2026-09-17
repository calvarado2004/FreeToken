from __future__ import annotations

from types import SimpleNamespace

from freetoken.server.stats import StatsTracker


def _reply(**kw):
    base = dict(uid=1, finished=False, completion_tokens_delta=1)
    base.update(kw)
    return SimpleNamespace(**base)


def test_tracker_keeps_the_newest_speculative_totals():
    tr = StatsTracker()
    assert tr.spec_blocks_total == 0
    tr.observe(_reply(spec_accepted_total=6, spec_drafted_total=9, spec_blocks_total=3, spec_accepted_per_pos=[2, 2, 1, 1]))
    # a reply without counters (e.g. a prompt reply) leaves them untouched
    tr.observe(_reply(completion_tokens_delta=0))
    assert (tr.spec_accepted_total, tr.spec_drafted_total, tr.spec_blocks_total) == (6, 9, 3)
    assert tr.spec_accepted_per_pos == [2, 2, 1, 1]
    tr.observe(_reply(spec_accepted_total=8, spec_drafted_total=13, spec_blocks_total=4, spec_accepted_per_pos=[3, 2, 2, 1]))
    assert (tr.spec_accepted_total, tr.spec_blocks_total) == (8, 4)
