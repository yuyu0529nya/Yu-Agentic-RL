"""
dynamic_sampling — zero the response_mask of all-same-outcome groups (binary-outcome gating).
This is the hand-written pipeline's "variance gating" (gate=True in grpo_update.py), ported onto
veRL's standard trainer, which lacks it (filter_groups lives only in the DAPO recipe).

==============================================================================================
STATUS (2026-07-16, after a 2-seed x 2-arm controlled A/B): ON veRL + Adam + BINARY reward,
THIS IS APPROXIMATELY A NO-OP. Do not claim it lifts the curve.
See ../results_20260712/RESULTS_ab_gating.md for the full write-up.

WHY it is a no-op here (code-level, not speculation):
  1. veRL's GRPO advantage is (score - mean)/(std + eps) with norm_adv_by_std_in_grpo=true.
     An all-same-outcome group has std == 0, so its advantage is EXACTLY 0. Those rows already
     contribute nothing to the gradient. Gating cannot remove a gradient that is already zero.
  2. veRL's default loss_agg_mode is "token-mean": loss = masked_sum(...) / loss_mask.sum().
     Zeroing dead rows leaves the numerator unchanged but SHRINKS THE DENOMINATOR, so the loss
     is scaled up by 1/live_frac. Measured here: 2.4x to 24x, varying step to step.
     => in veRL, this gate is not a filter; it is a time-varying loss AMPLIFIER.
  3. Adam is (approximately) invariant to a global rescaling of the gradient (update ~ m/sqrt(v)),
     so that amplification is largely absorbed and the policy update is unchanged.

MEASURED (seed=123, gating ON vs OFF, everything else identical):
  pg_loss    : arm(ON) is 4-17x arm(OFF)   -> amplification is real          (confirms 2)
  grad_norm  : arm(ON) is ~2-3x arm(OFF)   -> amplification is real          (confirms 2)
  ppo_kl     : the two arms are EQUAL      -> the update did NOT get bigger  (confirms 3)
  train (paired over 20 steps, same seed => same task batches):
               mean diff = -0.0063, 95% CI [-0.0183, +0.0058], t(19) = -1.013  -> NET EFFECT 0
  val@20     : 0.5375 (ON) vs 0.55 (OFF)   -> no difference (baseline noise is +/-0.05)
  Even at live_frac = 0.0417 (276/288 rows dropped, 24x amplification, ONE live group), val was
  unaffected. Pushing the mechanism to its extreme changed nothing.

HISTORY — two refuted claims of mine, kept here on purpose:
  * "This is the sample-efficiency fix for the flat curve."   REFUTED. The flat curve was an LR
    problem (grad_norm ~0.05, 20x under the clip threshold; pg_clipfrac ~0.001).
  * "Gating HURTS on low-success tasks; it starves the gradient." (written 2026-07-15 from a
    SINGLE seed: 0.4125 vs 0.5625) REFUTED by seed=123: the gap collapsed to 0.5375 vs 0.55.
    Arm(ON)'s own cross-seed spread (0.125) exceeds the between-arm mean gap (0.081) -> at n=2
    the effect is not there. "Starvation" was also mechanistically wrong: the dropped rows had
    advantage exactly 0, so nothing could be starved.
  METHODOLOGICAL LESSON: an n=1 RL A/B fabricated a 0.15-sized effect that does not exist.

The "+0.14" attribution this file used to carry has been WITHDRAWN: the hand-written +0.14 is
SFT 0.405 -> GRPO 0.545, i.e. the effect of running GRPO at all. The hand-written 2x2 ablation
varied process-reward x length-aware-advantage, never gating (gating was ON in all four cells),
so that pipeline never ran a gating on/off control. The A/B above is the only controlled test of
this gate that has ever been run.

STILL TRUE: the implementation is correct (11 CPU unit tests). Under SHAPED/PRM rewards an
all-fail group can have std > 0 and thus a nonzero "phantom" advantage that rewards the shortest
failure; binary-outcome gating kills that, and DAPO's std-based filter_groups would not. That is
the regime where this gate could earn its keep -- it was never tested there. Under BINARY reward
there is no phantom to kill, which is exactly why this A/B found nothing.

NOT YET DONE (the right sequel): DAPO-style physical row-drop + RESAMPLE to refill the batch.
That keeps the denominator constant and cleanly separates "dropping dead groups" from
"shrinking the denominator". Cheap to run on GSM8K (deterministic reward, no API cost).

KEY TRICK (no batch resize needed): zero the `response_mask` for rows in dead groups. veRL's
policy loss is a masked-mean over response_mask, so those tokens leave BOTH the numerator and
the denominator. NOTE: this is exactly the un-dilution described in (2) above -- which is why
this implementation is confounded with a learning-rate change and a physical drop+resample is
the correct design.
==============================================================================================

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
