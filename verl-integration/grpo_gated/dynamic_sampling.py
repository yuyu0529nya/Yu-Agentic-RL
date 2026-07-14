"""
dynamic_sampling — the +0.14 lever: drop all-same-outcome groups from the GRPO loss so the
batch isn't diluted by zero-gradient dead weight. This is the hand-written pipeline's
"variance gating" (gate=True in grpo_update.py: all-same-outcome groups get advantage 0 and
are then DROPPED), ported onto veRL's standard trainer, which lacks it (filter_groups lives
only in the DAPO recipe).

Diagnosed on tau2-airline (2026-07-13): at ~20% binary success with n=8, 17-31% of groups are
all-fail -> zero variance -> zero gradient, yet veRL keeps them in the batch where they dilute
the masked-mean loss. The hand-written +0.14 (SFT 0.405 -> GRPO 0.545, mega_vanilla) came from
dropping them, on the BINARY reward. PRM-Lite added only +0.005 on top.

KEY TRICK (no batch resize needed): zero the `response_mask` for rows in dead groups. veRL's
policy loss is a masked-mean over response_mask, so those tokens leave BOTH the numerator and
the denominator -> the group is truly removed from the loss average = un-dilution. A physical
row-drop + resample (DAPO-style) is possible too, but mask-zeroing is a one-liner on the batch
and needs no trainer-loop surgery.

Pure-numpy core so it unit-tests with no torch. `apply_dynamic_sampling(data)` is the veRL
DataProto entry point (call it in RayPPOTrainer.fit right before compute_advantage).
"""

from __future__ import annotations

from collections import defaultdict


def dead_group_rows(index, outcome_binary) -> list:
    """Row indices belonging to a group whose binary outcomes are all identical (no contrast).
    index: (bs,) group id per row. outcome_binary: (bs,) task success in {0,1}."""
    groups = defaultdict(list)
    for i, u in enumerate(index):
        groups[u].append(i)
    drop = []
    for _, rows in groups.items():
        outs = {round(float(outcome_binary[i]), 6) for i in rows}
        if len(rows) < 2 or len(outs) < 2:
            drop.extend(rows)
    return sorted(drop)


def zero_rows(response_mask, rows):
    """Zero the given rows of a 2D (bs, L) mask. Works for numpy arrays and torch tensors."""
    if not rows:
        return response_mask
    m = response_mask.clone() if hasattr(response_mask, "clone") else response_mask.copy()
    for i in rows:
        m[i] = 0
    return m


def apply_dynamic_sampling(data, outcome_key: str = "outcome_binary",
                           uid_key: str = "uid", mask_key: str = "response_mask") -> tuple:
    """veRL entry point. Zero response_mask for rows in all-same-outcome groups.

    Reads data.non_tensor_batch[uid_key] and [outcome_key], edits data.batch[mask_key] in place.
    Returns (data, stats) where stats = {dropped_rows, dropped_groups, total_rows, total_groups,
    live_frac}. No-op (with stats) if outcome_key is absent, so it's safe to always call.
    """
    ntb = data.non_tensor_batch
    if outcome_key not in ntb or uid_key not in ntb:
        return data, {"dropped_rows": 0, "note": f"{outcome_key}/{uid_key} absent — dynamic sampling skipped"}

    index = list(ntb[uid_key])
    outcome = list(ntb[outcome_key])
    total_rows = len(index)
    total_groups = len(set(index))

    drop = dead_group_rows(index, outcome)
    data.batch[mask_key] = zero_rows(data.batch[mask_key], drop)

    dropped_groups = total_groups - len({index[i] for i in range(total_rows) if i not in set(drop)})
    return data, {
        "dropped_rows": len(drop),
        "dropped_groups": dropped_groups,
        "total_rows": total_rows,
        "total_groups": total_groups,
        "live_frac": round(1 - len(drop) / total_rows, 4) if total_rows else 1.0,
    }
