"""
CPU unit tests for dynamic_sampling — no torch, no GPU. Run: python3 test_dynamic_sampling.py

Verifies the MECHANICS only: rows in all-same-binary-outcome groups get their response_mask
zeroed (dropped from the masked-mean loss = un-dilution), while contrastful groups are kept.

These tests say nothing about whether the gate HELPS. It does not: a 2-seed x 2-arm controlled
A/B (2026-07-16) measured ~no effect on veRL + Adam + binary reward, because the rows it drops
already have advantage exactly 0 and the un-dilution is just a loss rescale that Adam cancels.
The "+0.14 lever" framing this file used to carry has been withdrawn — that +0.14 was
SFT 0.405 -> GRPO 0.545 (the effect of GRPO itself), never attributed to the gate by any control.
See ../results_20260712/RESULTS_ab_gating.md.
"""

import numpy as np

from dynamic_sampling import dead_group_rows, apply_dynamic_sampling

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if cond: _passed += 1
    else: _failed += 1


class _Data:
    """Minimal stand-in for veRL DataProto: .batch and .non_tensor_batch dicts."""
    def __init__(self, response_mask, uid, outcome):
        self.batch = {"response_mask": response_mask}
        self.non_tensor_batch = {"uid": np.array(uid), "outcome_binary": np.array(outcome, dtype=float)}


def test_dead_group_rows():
    print("test_dead_group_rows")
    # groups 0,2 all-fail (dead); group 1 contrastful
    uid =     [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    outcome = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    drop = dead_group_rows(uid, outcome)
    check("drops exactly the two all-fail groups' rows", drop == [0, 1, 2, 3, 8, 9, 10, 11],
          f"drop={drop}")
    check("keeps the contrastful group (rows 4-7 not dropped)",
          all(i not in drop for i in [4, 5, 6, 7]))


def test_all_success_group_also_dropped():
    """An all-SUCCESS group is zero-variance too (no contrast) -> dropped."""
    print("test_all_success_group_also_dropped")
    uid =     [0, 0, 0, 0, 1, 1, 1, 1]
    outcome = [1, 1, 1, 1, 1, 0, 1, 0]     # group 0 all-success (dead), group 1 contrastful
    drop = dead_group_rows(uid, outcome)
    check("all-success group is dropped", set(drop) == {0, 1, 2, 3}, f"drop={drop}")


def test_response_mask_zeroed():
    print("test_response_mask_zeroed")
    uid =     [0, 0, 1, 1]
    outcome = [0, 0, 1, 0]                  # group 0 dead, group 1 contrastful
    mask = np.ones((4, 5), dtype=float)
    data = _Data(mask, uid, outcome)
    data, stats = apply_dynamic_sampling(data)
    m = data.batch["response_mask"]
    check("dead group's rows zeroed in the mask", m[0].sum() == 0 and m[1].sum() == 0)
    check("contrastful group's rows untouched", m[2].sum() == 5 and m[3].sum() == 5)
    check("stats report the drop", stats["dropped_rows"] == 2 and stats["dropped_groups"] == 1,
          f"stats={stats}")


def test_noop_when_outcome_absent():
    """Safe to always call: with no outcome_binary field it is a no-op."""
    print("test_noop_when_outcome_absent")
    data = _Data(np.ones((4, 3)), [0, 0, 1, 1], [0, 0, 1, 0])
    del data.non_tensor_batch["outcome_binary"]
    data, stats = apply_dynamic_sampling(data)
    check("no-op when outcome_binary missing", data.batch["response_mask"].sum() == 12 and
          stats["dropped_rows"] == 0, f"stats={stats}")


def test_tau2_like_batch():
    """~20% success, n=8, 16 groups: quantify how much dead weight dynamic sampling removes."""
    print("test_tau2_like_batch")
    rng = np.random.default_rng(0)
    n, groups, p = 8, 16, 0.20
    uid = np.repeat(np.arange(groups), n)
    outcome = (rng.random(groups * n) < p).astype(float)
    data = _Data(np.ones((groups * n, 4)), uid, outcome)
    data, stats = apply_dynamic_sampling(data)
    print(f"    dropped {stats['dropped_rows']}/{stats['total_rows']} rows "
          f"({stats['dropped_groups']}/{stats['total_groups']} groups), "
          f"live_frac={stats['live_frac']}")
    check("removes a non-trivial dead fraction (>=15%)", (1 - stats["live_frac"]) >= 0.15,
          f"{100*(1-stats['live_frac']):.0f}% of the batch was dead weight, now dropped")


if __name__ == "__main__":
    print("=" * 78)
    print("dynamic_sampling unit tests  (drop all-same-outcome groups from the loss)")
    print("=" * 78)
    for t in (test_dead_group_rows, test_all_success_group_also_dropped, test_response_mask_zeroed,
              test_noop_when_outcome_absent, test_tau2_like_batch):
        t(); print()
    print("=" * 78)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 78)
    raise SystemExit(1 if _failed else 0)
