"""
CPU unit tests for group_filter — no torch, no GPU. Run: python3 test_group_filter.py

These pin down the two claims the GSM8K bake-off is built on:

  A. gate="outcome" and gate="std" AGREE under a binary reward, and DISAGREE under a shaped one.
     That is why the tau2 A/B (binary reward) measured nothing, and why the shaped/PRM regime is
     the one where binary-outcome gating could actually earn its keep.

  B. drop="mask" changes the loss denominator (i.e. is confounded with LR) while drop="physical"
     keeps it constant. This is the confound the whole experiment exists to remove.
"""

import numpy as np

from group_filter import (
    GroupAccumulator,
    is_dead_group,
    partition,
    stats_for,
    zero_rows,
)

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if cond:
        _passed += 1
    else:
        _failed += 1


# ---------------------------------------------------------------- gate semantics


def test_dead_group_basics():
    print("\n=== gate: what counts as a dead group ===")
    # one group of 4: all fail
    idx = ["g0"] * 4
    out = [0.0, 0.0, 0.0, 0.0]
    sc = [0.0, 0.0, 0.0, 0.0]
    keep, drop, dead, live = partition(idx, out, sc, gate="outcome")
    check("all-fail group is dropped", len(drop) == 4 and len(keep) == 0, f"dead={dead}")

    # all pass -> also dead (no contrast either)
    out = [1.0] * 4
    sc = [1.0] * 4
    keep, drop, dead, live = partition(idx, out, sc, gate="outcome")
    check("all-PASS group is dropped too", len(drop) == 4, "zero variance cuts both ways")

    # mixed -> kept
    out = [1.0, 0.0, 0.0, 1.0]
    sc = [1.0, 0.0, 0.0, 1.0]
    keep, drop, dead, live = partition(idx, out, sc, gate="outcome")
    check("mixed group is kept", len(keep) == 4 and len(drop) == 0)

    # singleton -> dead (nothing to be relative to)
    keep, drop, dead, live = partition(["s"], [1.0], [1.0], gate="outcome")
    check("singleton is dropped", len(drop) == 1, "group-relative needs >= 2")


def test_binary_gates_agree():
    print("\n=== claim A1: under BINARY reward, outcome-gate == std-gate ===")
    rng = np.random.default_rng(0)
    agree = True
    for _ in range(200):
        n = int(rng.integers(2, 9))
        out = rng.integers(0, 2, size=n).astype(float)
        sc = out.copy()  # binary reward: score IS the outcome
        rows = list(range(n))
        a = is_dead_group(rows, out, sc, gate="outcome")
        b = is_dead_group(rows, out, sc, gate="std")
        if a != b:
            agree = False
            break
    check("the two gates agree on every binary case", agree,
          "so the tau2 A/B could not have distinguished them")


def test_shaped_gates_disagree():
    print("\n=== claim A2: under SHAPED reward, the gates DISAGREE (the phantom advantage) ===")
    # an all-FAIL group whose shaped rewards differ only by length
    idx = ["g"] * 4
    out = [0.0, 0.0, 0.0, 0.0]          # nobody solved the task
    sc = [0.9, 0.6, 0.4, 0.1]           # but the shaped reward spreads them out

    dead_outcome = is_dead_group([0, 1, 2, 3], out, sc, gate="outcome")
    dead_std = is_dead_group([0, 1, 2, 3], out, sc, gate="std")
    check("outcome-gate kills the all-fail group", dead_outcome is True)
    check("std-gate KEEPS it (DAPO would train on it)", dead_std is False,
          "std>0 so DAPO's filter lets it through")

    # and what GRPO would do with it: a nonzero advantage on a group where nothing succeeded
    mean = sum(sc) / len(sc)
    adv = [(s - mean) / (float(np.std(sc)) + 1e-6) for s in sc]
    check("the surviving group carries a nonzero 'phantom' advantage",
          max(abs(a) for a in adv) > 0.5,
          f"adv={[round(a, 2) for a in adv]} — reinforces the highest-shaped FAILURE")


# ---------------------------------------------------------------- the confound


def test_mask_changes_denominator():
    print("\n=== claim B1: drop='mask' shrinks the token-mean denominator (== LR change) ===")
    # 4 groups x 3 rows, 2 groups dead
    idx = sum(([f"g{g}"] * 3 for g in range(4)), [])
    out = [0, 0, 0] + [1, 0, 1] + [1, 1, 1] + [0, 1, 0]
    sc = [float(x) for x in out]
    keep, drop, dead, live = partition(idx, out, sc, gate="outcome")
    st = stats_for(keep, drop, dead, live)

    mask = np.ones((12, 5), dtype=float)
    denom_before = mask.sum()
    masked = zero_rows(mask, drop)
    denom_after = masked.sum()

    check("2 of 4 groups are dead", st["dropped_groups"] == 2, f"dead={dead}")
    check("denominator actually shrinks", denom_after < denom_before,
          f"{denom_before:.0f} -> {denom_after:.0f}")
    scale = denom_before / denom_after
    check("loss scale == 1/live_frac (the confound, quantified)",
          abs(scale - st["loss_scale_if_masked"]) < 1e-6,
          f"scale={scale:.3f}, live_frac={st['live_frac']}")


def test_physical_keeps_denominator_constant():
    print("\n=== claim B2: drop='physical' keeps the batch (and denominator) constant ===")
    # Every generated batch has 4 groups of 3; roughly half are dead.
    def gen(seed):
        rng = np.random.default_rng(seed)
        idx, out = [], []
        for g in range(4):
            n = 3
            idx += [f"b{seed}g{g}"] * n
            # force a mix: some groups all-same, some contrastful
            out += list(rng.integers(0, 2, size=n).astype(float)) if g % 2 == 0 else [0.0] * n
        return idx, out, list(out)

    acc = GroupAccumulator(target_groups=4, max_gen_batches=10, gate="outcome")
    n_batches = 0
    while not acc.full:
        n_batches += 1
        acc.add(*gen(n_batches))
    rows = acc.take()
    groups = acc.take_groups()

    check("accumulator reaches exactly the target", len(groups) == 4, f"got {len(groups)}")
    check("no over-fill (would GROW the denominator)", len(groups) == acc.target_groups)
    check("batch size is constant regardless of how many were dropped", len(rows) == 4 * 3,
          f"{len(rows)} rows = 4 groups x 3")
    eff = acc.efficiency()
    check("the real cost is extra generation, not a smaller gradient",
          eff["generation_overhead"] >= 1.0,
          f"generated {eff['groups_generated']} groups to keep {eff['groups_kept']}")


def test_refuses_short_batch():
    print("\n=== safety valve: never silently train on a short batch ===")
    # a task so hard that every group is all-fail
    def gen_all_dead(_):
        idx = sum(([f"g{g}"] * 4 for g in range(4)), [])
        out = [0.0] * 16
        return idx, out, out

    acc = GroupAccumulator(target_groups=8, max_gen_batches=3, gate="outcome")
    raised = False
    try:
        for i in range(10):
            acc.add(*gen_all_dead(i))
    except RuntimeError as e:
        raised = "Refusing to train on a short batch" in str(e)
    check("raises instead of shrinking the batch", raised,
          "a short batch would reintroduce the very confound we removed")

    raised2 = False
    try:
        GroupAccumulator(target_groups=4).take()
    except RuntimeError:
        raised2 = True
    check("take() refuses before the batch is full", raised2)


def test_determinism():
    print("\n=== determinism: same input -> same selection ===")
    idx = sum(([f"g{g}"] * 3 for g in range(6)), [])
    out = [1, 0, 1] * 6
    sc = [float(x) for x in out]
    a = partition(idx, out, sc, gate="outcome")
    b = partition(idx, out, sc, gate="outcome")
    check("partition is deterministic", a == b)
    check("kept rows come back sorted", a[0] == sorted(a[0]))


if __name__ == "__main__":
    for t in (
        test_dead_group_basics,
        test_binary_gates_agree,
        test_shaped_gates_disagree,
        test_mask_changes_denominator,
        test_physical_keeps_denominator_constant,
        test_refuses_short_batch,
        test_determinism,
    ):
        t()
    print("\n" + "=" * 78)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 78)
    raise SystemExit(1 if _failed else 0)
