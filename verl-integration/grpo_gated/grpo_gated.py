"""
grpo_gated — a binary-outcome-gated GRPO advantage estimator for veRL.

WHY THIS EXISTS (diagnosed on tau2-airline, 2026-07-13)
-------------------------------------------------------
Running veRL's *standard* GRPO (main_ppo / RayPPOTrainer, adv_estimator=grpo) on
tau2-airline from an SFT checkpoint produced a FLAT learning curve across two
learning rates (lr=4e-6 for 10 steps, lr=2e-5 for 4 steps): train task-score stuck
at ~0.19-0.29, ppo_kl bigger at the higher lr but entropy RISING (0.70 -> 0.88) —
the policy moved, but into a more diffuse region, not toward higher reward.

Root cause is structural, not a tuning problem. veRL's GRPO advantage
(verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage) does:

    adv[i] = (score[i] - group_mean) / (group_std + 1e-6)

For a group whose n rollouts all share the same outcome (e.g. all-fail — common
when success rate is ~20% and n=8: P(all 8 fail) ~ 0.81^8 ~ 0.19), std == 0 so the
advantage is ~0. BUT veRL keeps those zero-advantage rows in the batch. They
contribute no gradient yet still count in the loss average, DILUTING the few groups
that carry signal.

Two gaps in veRL vs. the hand-written pipeline this ports from:

  Gap 1 (dilution / dynamic sampling):
     veRL DOES have dynamic sampling that drops no-contrast groups
     (recipe/dapo/dapo_ray_trainer.py::filter_groups, keep iff std>0), but ONLY in
     the DAPO recipe — the standard GRPO trainer everyone uses has nothing.

  Gap 2 (phantom advantage — subtler, DAPO misses it too):
     DAPO's filter_groups gates on the *reward-metric std*. The moment you use a
     SHAPED / process reward (PRM-Lite), an all-FAIL group has non-zero shaped-reward
     variance, so DAPO KEEPS it and GRPO fabricates a "phantom advantage" on it. That
     phantom is ~length-concordant: it rewards the *shortest failure* = it trains the
     model to GIVE UP FASTER. (In the hand-written run this caused an iter1 0.35->0.20
     dip.) The fix is to gate on the *binary task outcome*, not on the shaped-reward
     variance — so an all-fail group teaches nothing even when its shaped rewards differ.

This module implements the gate. `gated_grpo_advantage` computes the GRPO advantage on
the (possibly shaped) score but ZEROES any group with no BINARY-OUTCOME contrast, and
returns a keep-mask so a trainer can additionally DROP those rows (dynamic sampling).

Pure-numpy core so it runs / unit-tests with no torch. The veRL-registerable torch
wrapper (`compute_grpo_gated_outcome_advantage`) is at the bottom and imports torch lazily.
"""

from __future__ import annotations

import numpy as np


def _group_rows(index) -> dict:
    """Map each group id -> list of row indices that belong to it (preserves order)."""
    groups: dict = {}
    for i, idx in enumerate(index):
        groups.setdefault(idx, []).append(i)
    return groups


def standard_grpo_advantage(scores, index, epsilon: float = 1e-6, norm_by_std: bool = True):
    """Reference reimplementation of veRL's compute_grpo_outcome_advantage
    (verl/trainer/ppo/core_algos.py:266), for side-by-side comparison in tests.

    scores : (bs,) scalar reward per trajectory  ( == token_level_rewards.sum(-1) )
    index  : (bs,) group id per trajectory
    Returns per-trajectory advantage (bs,). No gate: no-contrast groups get ~0 but
    are NOT flagged for removal (that is exactly the dilution problem).
    """
    scores = np.asarray(scores, dtype=float)
    adv = np.zeros_like(scores)
    for _, rows in _group_rows(index).items():
        g = scores[rows]
        if len(g) == 1:
            mean, std = 0.0, 1.0                      # veRL's single-sample convention
        else:
            mean, std = g.mean(), g.std(ddof=1)       # torch.std default is ddof=1
        for r in rows:
            adv[r] = (scores[r] - mean) / (std + epsilon) if norm_by_std else (scores[r] - mean)
    return adv


def gated_grpo_advantage(
    scores_shaped,
    outcome_binary,
    index,
    epsilon: float = 1e-6,
    norm_by_std: bool = True,
    adv_clip: float | None = None,
    denom_eps: float = 0.0,
):
    """Binary-outcome-gated GRPO advantage (the hand-written trick veRL's standard
    GRPO lacks).

    A group teaches only if its BINARY outcomes contain BOTH a success and a failure;
    otherwise every rollout in it gets advantage 0 AND keep=False. This simultaneously:
      * removes zero-variance dilution (all-same binary groups carry no gradient), and
      * kills "phantom advantage": an all-fail group whose SHAPED rewards differ
        (scores_shaped varies) would otherwise get a spurious length-concordant
        gradient. We gate on `outcome_binary`, not on scores_shaped's variance, so it
        is zeroed regardless.

    Args:
      scores_shaped  : (bs,) the reward the advantage is computed FROM (may be shaped).
      outcome_binary : (bs,) raw task success in {0,1} — what the GATE looks at.
      index          : (bs,) group id per trajectory.
      adv_clip       : optional symmetric clip on the advantage (hand-written uses 3.0).
      denom_eps      : optional extra denominator floor (hand-written uses 0.25) so tiny
                       shaped-reward gaps can't blow up to unit scale. 0.0 == veRL-like.

    Returns:
      advantages : (bs,) float
      keep_mask  : (bs,) bool — False for rows in a no-contrast group; a trainer can
                   drop these (= dynamic sampling on the standard GRPO path).
    """
    scores_shaped = np.asarray(scores_shaped, dtype=float)
    outcome_binary = np.asarray(outcome_binary, dtype=float)
    adv = np.zeros_like(scores_shaped)
    keep = np.zeros(len(scores_shaped), dtype=bool)

    for _, rows in _group_rows(index).items():
        outs = outcome_binary[rows]
        # GATE: contrast requires both a success and a failure in the BINARY outcome.
        no_contrast = len(rows) < 2 or outs.std(ddof=0) == 0.0
        if no_contrast:
            continue  # adv stays 0, keep stays False  -> row is dead weight, drop it

        g = scores_shaped[rows]
        mean = g.mean()
        std = g.std(ddof=1) if len(g) > 1 else 0.0
        for r in rows:
            a = (scores_shaped[r] - mean) / (std + denom_eps + epsilon) if norm_by_std \
                else (scores_shaped[r] - mean)
            if adv_clip is not None:
                a = max(-adv_clip, min(adv_clip, a))
            adv[r] = a
            keep[r] = True
    return adv, keep


def dead_weight_fraction(outcome_binary, index) -> float:
    """Fraction of trajectories that live in a no-contrast group (== the batch fraction
    that is pure dilution under standard GRPO). Handy diagnostic to quantify the problem
    at a given success rate."""
    outcome_binary = np.asarray(outcome_binary, dtype=float)
    dead = 0
    for _, rows in _group_rows(index).items():
        outs = outcome_binary[rows]
        if len(rows) < 2 or outs.std(ddof=0) == 0.0:
            dead += len(rows)
    return dead / len(outcome_binary) if len(outcome_binary) else 0.0


# --------------------------------------------------------------------------------------
# veRL-registerable estimator (torch). Import guarded so this module still loads on a
# machine without torch (the numpy core above + the unit tests do not need torch).
#
# To activate on the GPU box, from a module imported at trainer startup:
#     from grpo_gated import register_grpo_gated
#     register_grpo_gated()
# then launch with  algorithm.adv_estimator=grpo_gated
#
# NOTE on the binary outcome source: the faithful gate reads the RAW task success. In
# veRL that is `token_level_scores` (pre-shaping), which DAPO already surfaces as
# "seq_reward". The registerable wrapper below derives the binary outcome by thresholding
# the trajectory score at `gate_success_threshold` — EXACT when the reward is already
# binary (our tau2-airline case). When you add PRM-Lite shaping, wire the raw
# token_level_scores through instead of thresholding the shaped reward (see integration
# note in README).
# --------------------------------------------------------------------------------------

def compute_grpo_gated_outcome_advantage(
    token_level_rewards,
    response_mask,
    index,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
):
    import torch

    with torch.no_grad():
        scores_shaped = token_level_rewards.sum(dim=-1)
        sn = scores_shaped.detach().cpu().numpy()

        threshold = 1.0 - 1e-6
        adv_clip = None
        denom_eps = 0.0
        if config is not None:
            threshold = getattr(config, "gate_success_threshold", threshold)
            adv_clip = getattr(config, "gate_adv_clip", adv_clip)
            denom_eps = getattr(config, "gate_denom_eps", denom_eps)
        outcome = (sn >= threshold).astype(float)

        adv, _keep = gated_grpo_advantage(
            sn, outcome, np.asarray(index),
            epsilon=epsilon, norm_by_std=norm_adv_by_std_in_grpo,
            adv_clip=adv_clip, denom_eps=denom_eps,
        )
        adv_t = torch.tensor(adv, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
        scores = adv_t.unsqueeze(-1) * response_mask
    return scores, scores


def register_grpo_gated():
    """Register `grpo_gated` with veRL's advantage-estimator registry (call at startup)."""
    from verl.trainer.ppo.core_algos import register_adv_est
    register_adv_est("grpo_gated")(compute_grpo_gated_outcome_advantage)
    return "grpo_gated"
