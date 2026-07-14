"""
CPU unit tests for grpo_gated — no torch, no GPU. Run:  python3 test_grpo_gated.py

Each test contrasts veRL's STANDARD GRPO advantage (standard_grpo_advantage, a faithful
reimpl of core_algos.py::compute_grpo_outcome_advantage) against the binary-outcome-gated
version, on synthetic groups that reproduce the tau2-airline failure mode.
"""

import numpy as np

from grpo_gated import (
    standard_grpo_advantage,
    gated_grpo_advantage,
    dead_weight_fraction,
)

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    mark = "PASS" if cond else "FAIL"
    if cond:
        _passed += 1
    else:
        _failed += 1
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


# ----------------------------------------------------------------------------------------
def test_phantom_advantage_killed():
    """THE money test. An all-FAIL group (binary outcome all 0) whose SHAPED rewards differ
    by trajectory length. Standard GRPO fabricates a non-zero 'phantom' advantage that
    rewards the shortest failure (== train the model to give up faster). The binary-outcome
    gate zeroes it. DAPO's std-gate would NOT catch this (shaped std > 0)."""
    print("test_phantom_advantage_killed")
    # one group, 4 rollouts, ALL failed the task (outcome 0), but shaped reward decreases
    # with trajectory length -> shortest fail looks 'best'.
    index = np.array([0, 0, 0, 0])
    shaped = np.array([0.30, 0.20, 0.10, 0.05])   # what GRPO sees as 'reward'
    outcome = np.array([0, 0, 0, 0])              # raw task success: all failed

    std_adv = standard_grpo_advantage(shaped, index)
    gated_adv, keep = gated_grpo_advantage(shaped, outcome, index)

    check("standard GRPO fabricates a phantom advantage on the all-fail group",
          np.abs(std_adv).max() > 0.1,
          f"max|adv|={np.abs(std_adv).max():.3f} (rewards shortest fail: adv={std_adv.round(2).tolist()})")
    check("gated advantage is exactly zero on the all-fail group",
          np.allclose(gated_adv, 0.0),
          f"gated adv={gated_adv.round(3).tolist()}")
    check("gated keep-mask drops all rows of the no-contrast group",
          (~keep).all(),
          f"keep={keep.tolist()}")


# ----------------------------------------------------------------------------------------
def test_zero_variance_binary_group_flagged():
    """Pure-binary all-fail group. Standard GRPO gives ~0 advantage but KEEPS the rows
    (dilution). The gate flags them for removal via keep-mask (= dynamic sampling)."""
    print("test_zero_variance_binary_group_flagged")
    index = np.array([0, 0, 0, 0])
    binary = np.array([0, 0, 0, 0])

    std_adv = standard_grpo_advantage(binary, index)
    gated_adv, keep = gated_grpo_advantage(binary, binary, index)

    check("standard GRPO leaves the group in the batch (advantage ~0 but present)",
          np.allclose(std_adv, 0.0),
          "these rows still count in the loss average = dilution")
    check("gate flags every row for removal (keep all False)",
          (~keep).all(),
          f"keep={keep.tolist()}")


# ----------------------------------------------------------------------------------------
def test_contrastful_group_preserved():
    """A group WITH contrast (2 success, 2 fail). The gate must NOT change a good group:
    advantages match standard GRPO and every row is kept."""
    print("test_contrastful_group_preserved")
    index = np.array([0, 0, 0, 0])
    binary = np.array([1, 1, 0, 0])   # reward == binary outcome here

    std_adv = standard_grpo_advantage(binary, index)
    gated_adv, keep = gated_grpo_advantage(binary, binary, index)

    check("gated advantage equals standard on a contrastful group",
          np.allclose(std_adv, gated_adv),
          f"std={std_adv.round(3).tolist()}  gated={gated_adv.round(3).tolist()}")
    check("successes get positive advantage, failures negative",
          gated_adv[0] > 0 and gated_adv[2] < 0)
    check("all rows kept", keep.all(), f"keep={keep.tolist()}")


# ----------------------------------------------------------------------------------------
def test_mixed_batch_dilution_quantified():
    """A realistic batch: some contrastful groups, some all-fail groups. Show how much of
    the batch is dead weight under standard GRPO, and that the gate keeps exactly the
    contrastful rows."""
    print("test_mixed_batch_dilution_quantified")
    # 4 groups x 4 rollouts. groups 0,1 contrastful; groups 2,3 all-fail.
    index = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    binary = np.array([1, 0, 0, 0,  1, 1, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0])

    _, keep = gated_grpo_advantage(binary, binary, index)
    dead = dead_weight_fraction(binary, index)

    check("half the batch is dead weight under standard GRPO",
          abs(dead - 0.5) < 1e-9,
          f"dead-weight fraction = {dead:.0%} (groups 2 & 3 all-fail)")
    check("gate keeps only the 8 contrastful rows",
          keep.sum() == 8 and keep[:8].all() and (~keep[8:]).all(),
          f"kept {keep.sum()}/16 rows")


# ----------------------------------------------------------------------------------------
def test_tau2_like_success_rate():
    """Simulate a batch at tau2-airline's observed ~20% success, n=8 per group, 16 groups.
    Quantify the dead weight that standard GRPO silently averages over — the concrete
    mechanism behind the flat 2026-07-13 curve."""
    print("test_tau2_like_success_rate")
    rng = np.random.default_rng(0)   # deterministic
    n, groups, p = 8, 16, 0.20
    index = np.repeat(np.arange(groups), n)
    binary = (rng.random(groups * n) < p).astype(float)

    dead = dead_weight_fraction(binary, index)
    _, keep = gated_grpo_advantage(binary, binary, index)
    empirical_all_fail = 1 - keep.reshape(groups, n).any(axis=1).mean()  # groups fully dead

    print(f"    success rate p={p:.0%}, n={n}, groups={groups}")
    print(f"    dead-weight fraction (rows in no-contrast groups) = {dead:.0%}")
    print(f"    fully-dead groups                                 = {empirical_all_fail:.0%}")
    print(f"    theoretical P(all {n} fail) at p={p:.0%}           = {(1-p)**n:.0%}")
    check("a non-trivial slice of the batch is dead weight (>=15%)",
          dead >= 0.15,
          f"{dead:.0%} of trajectories carry zero gradient yet dilute the loss")


if __name__ == "__main__":
    print("=" * 78)
    print("grpo_gated unit tests  (veRL standard GRPO  vs  binary-outcome-gated GRPO)")
    print("=" * 78)
    for t in (
        test_phantom_advantage_killed,
        test_zero_variance_binary_group_flagged,
        test_contrastful_group_preserved,
        test_mixed_batch_dilution_quantified,
        test_tau2_like_success_rate,
    ):
        t()
        print()
    print("=" * 78)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 78)
    raise SystemExit(1 if _failed else 0)
