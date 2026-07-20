"""
group_filter — separate the two things "dynamic sampling" actually does.

A 2026-07-16 controlled A/B (2 seeds x 2 arms) on tau2-airline found that binary-outcome gating
implemented as *response_mask zeroing* is approximately a no-op under veRL + Adam. The mechanism
turned out to be a confound rather than an effect:

  1. veRL's GRPO advantage is (score - mean)/(std + eps). An all-same-outcome group has std == 0,
     so its advantage is EXACTLY 0. Those rows already contribute no gradient.
  2. Zeroing their response_mask therefore leaves the numerator alone but SHRINKS the denominator
     of veRL's token-mean loss (loss = masked_sum / loss_mask.sum()), scaling the loss up by
     1/live_frac. Measured on that run: 2.4x to 24x, varying step to step.
  3. Adam is approximately invariant to a global gradient rescale, so most of (2) is absorbed.

So "mask zeroing" is not a filter, it is a time-varying learning-rate multiplier. Any A/B built on
it is confounded with LR and cannot answer whether *dropping dead groups* helps.

DAPO's filter_groups does the same job without that confound: it physically drops the rows and
RESAMPLES more prompts until the batch is full again, so the denominator stays constant.

This module exposes both axes so the confound can be measured instead of argued about:

    gate  = "outcome" | "std"        which signal decides a group is dead
    drop  = "mask"    | "physical"   how the dead rows leave the loss

    (outcome, mask)     <- what was actually run on tau2; confounded with LR
    (outcome, physical) <- the un-confounded version of that same idea
    (std,     physical) <- DAPO's filter_groups
    (std,     mask)     <- included for completeness of the 2x2

Why the gate signal matters too (this is the part DAPO does not do):
under BINARY reward an all-fail group has std == 0, so "std" and "outcome" agree and there is
nothing to find -- which is exactly why the tau2 A/B measured nothing. Under SHAPED/PRM reward an
all-fail group has std > 0, so DAPO's std gate KEEPS it and GRPO fabricates a length-concordant
"phantom advantage" that rewards the shortest failure. Gating on the raw binary outcome kills that.
That regime is untested and is the reason both axes are kept separate here.

Pure-numpy core so it unit-tests with no torch and no GPU.
"""

from __future__ import annotations

from collections import defaultdict

# A group is "dead" when it carries no learning signal: every rollout in it got the same
# result, so the group-relative advantage is zero (or, for a singleton, undefined).
GATES = ("outcome", "std")
DROPS = ("mask", "physical")


def group_rows(index) -> dict:
    """Map group id -> list of row indices, preserving row order within a group."""
    groups = defaultdict(list)
    for i, u in enumerate(index):
        groups[u].append(i)
    return dict(groups)


def is_dead_group(rows, outcome_binary, scores, gate: str, std_eps: float = 1e-8) -> bool:
    """Does this group carry contrast?

    gate="outcome": dead when every rollout shares the same binary task outcome. This is the
        signal that survives reward shaping -- it asks "did the task succeed", not "did the
        scalar reward differ".
    gate="std":     dead when the reward-metric spread is ~0 (DAPO's rule). Under a shaped
        reward an all-fail group can still have std > 0 and thus survive this gate.
    """
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if len(rows) < 2:
        return True  # a singleton has no one to be relative to
    if gate == "outcome":
        return len({round(float(outcome_binary[i]), 6) for i in rows}) < 2
    vals = [float(scores[i]) for i in rows]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return var**0.5 <= std_eps


def partition(index, outcome_binary, scores, gate: str = "outcome", std_eps: float = 1e-8) -> tuple:
    """Split rows into (keep, drop) and report which groups died.

    Returns (keep_rows, drop_rows, dead_groups, live_groups), all sorted / deterministic.
    """
    groups = group_rows(index)
    keep, drop, dead, live = [], [], [], []
    for gid, rows in groups.items():
        if is_dead_group(rows, outcome_binary, scores, gate, std_eps):
            drop.extend(rows)
            dead.append(gid)
        else:
            keep.extend(rows)
            live.append(gid)
    return sorted(keep), sorted(drop), dead, live


def stats_for(keep, drop, dead, live) -> dict:
    total_rows = len(keep) + len(drop)
    total_groups = len(dead) + len(live)
    return {
        "kept_rows": len(keep),
        "dropped_rows": len(drop),
        "total_rows": total_rows,
        "kept_groups": len(live),
        "dropped_groups": len(dead),
        "total_groups": total_groups,
        # live_frac is the fraction of rows that survive the gate. Under drop="mask" it is also
        # exactly the factor by which the token-mean loss gets scaled up (1/live_frac), which is
        # the confound this module exists to expose.
        "live_frac": round(len(keep) / total_rows, 4) if total_rows else 1.0,
        "loss_scale_if_masked": round(total_rows / len(keep), 4) if keep else float("inf"),
    }


# --------------------------------------------------------------------------------------------
# drop = "physical": DAPO-style accumulate-and-resample
# --------------------------------------------------------------------------------------------


class GroupAccumulator:
    """Accumulate contrastful groups across generation batches until the batch is full.

    This is the un-confounded drop mechanism. Instead of masking dead rows out of a fixed-size
    batch (which shrinks the loss denominator), it keeps generating fresh prompts and collecting
    only the live groups, so the batch handed to the optimizer is always the same size -- the
    denominator is constant and the comparison is clean.

    Mirrors DAPO's filter_groups loop, including its safety valve: if the target is never reached
    within max_gen_batches, raise rather than silently train on a short batch (a short batch would
    reintroduce exactly the denominator change we are trying to avoid).

    Usage is trainer-driven so this stays testable without torch:

        acc = GroupAccumulator(target_groups=16, max_gen_batches=8)
        while not acc.full:
            batch = generate()                       # trainer's job
            acc.add(batch.uid, batch.outcome, batch.score)
        rows = acc.take()                            # row indices, per add() call
    """

    def __init__(self, target_groups: int, max_gen_batches: int = 8, gate: str = "outcome",
                 std_eps: float = 1e-8):
        if target_groups < 1:
            raise ValueError("target_groups must be >= 1")
        self.target_groups = target_groups
        self.max_gen_batches = max_gen_batches
        self.gate = gate
        self.std_eps = std_eps
        # kept groups in arrival order: (batch_idx, group_id, [row indices])
        self.groups: list = []
        self.n_gen_batches = 0
        self.seen_rows = 0
        self.seen_groups = 0

    @property
    def kept_groups(self) -> int:
        return len(self.groups)

    @property
    def full(self) -> bool:
        return self.kept_groups >= self.target_groups

    def add(self, index, outcome_binary, scores) -> dict:
        """Feed one freshly generated batch; keep its live groups."""
        bidx = self.n_gen_batches
        self.n_gen_batches += 1
        keep, drop, dead, live = partition(index, outcome_binary, scores, self.gate, self.std_eps)
        rows_by_gid = group_rows(index)
        for gid in live:
            self.groups.append((bidx, gid, sorted(rows_by_gid[gid])))
        self.seen_rows += len(keep) + len(drop)
        self.seen_groups += len(dead) + len(live)
        s = stats_for(keep, drop, dead, live)
        s["gen_batch"] = self.n_gen_batches
        s["accumulated_groups"] = self.kept_groups
        s["target_groups"] = self.target_groups
        if not self.full and self.n_gen_batches >= self.max_gen_batches:
            raise RuntimeError(
                f"group_filter: only {self.kept_groups}/{self.target_groups} contrastful groups "
                f"after {self.n_gen_batches} generation batches. Refusing to train on a short "
                f"batch -- that would change the loss denominator, which is the exact confound "
                f"this module avoids. Raise max_gen_batches, raise rollout.n, or accept that the "
                f"task's success rate is too extreme for group-relative advantage."
            )
        return s

    def take(self) -> list:
        """Exactly target_groups worth of kept rows, as (batch_idx, row_idx) pairs.

        Trimming to exactly the target matters in both directions: a short batch shrinks the loss
        denominator and an over-full one grows it. Either way the comparison is no longer clean,
        which is the whole point of this mechanism.
        """
        if not self.full:
            raise RuntimeError(
                f"take() called with only {self.kept_groups}/{self.target_groups} groups. "
                f"Keep calling add() until .full, or let add() raise at max_gen_batches."
            )
        out = []
        for bidx, _gid, rows in self.groups[: self.target_groups]:
            out.extend((bidx, r) for r in rows)
        return out

    def take_groups(self) -> list:
        """Same selection as take(), but grouped: [(batch_idx, group_id, [rows]), ...]."""
        if not self.full:
            raise RuntimeError(f"take_groups() called with {self.kept_groups}/{self.target_groups}")
        return list(self.groups[: self.target_groups])

    def efficiency(self) -> dict:
        """How much generation did the gate cost? This is the real price of dropping groups --
        not a smaller gradient, but more rollouts to fill the same batch."""
        return {
            "gen_batches": self.n_gen_batches,
            "rows_generated": self.seen_rows,
            "groups_generated": self.seen_groups,
            "groups_kept": self.kept_groups,
            "group_yield": round(self.kept_groups / self.seen_groups, 4) if self.seen_groups else 0.0,
            # >1 means the gate made you generate more than a plain run would have
            "generation_overhead": round(self.seen_groups / self.target_groups, 4)
            if self.target_groups else 0.0,
        }


# --------------------------------------------------------------------------------------------
# drop = "mask": the confounded mechanism, kept so the A/B can include it
# --------------------------------------------------------------------------------------------


def zero_rows(response_mask, rows):
    """Zero the given rows of a 2D (bs, L) mask. Works for numpy arrays and torch tensors."""
    if not rows:
        return response_mask
    m = response_mask.clone() if hasattr(response_mask, "clone") else response_mask.copy()
    for i in rows:
        m[i] = 0
    return m
