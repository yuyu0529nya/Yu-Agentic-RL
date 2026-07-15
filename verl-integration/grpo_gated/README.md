# grpo_gated — binary-outcome-gated GRPO advantage for veRL

A drop-in advantage estimator that ports two things veRL's **standard** GRPO trainer
(`main_ppo` / `RayPPOTrainer`) lacks, distilled from a hand-written RL pipeline:

1. **Dynamic sampling on the standard path.** veRL only filters no-contrast groups in the
   *DAPO recipe* (`recipe/dapo/dapo_ray_trainer.py::filter_groups`). The standard GRPO
   trainer keeps zero-variance groups in the batch, where they carry no gradient but still
   dilute the loss average.
2. **Phantom-advantage gating** — the subtle bit DAPO also misses. DAPO filters on the
   *reward-metric std*; with a shaped/process reward (PRM-Lite) an all-fail group has
   non-zero shaped variance, so it survives and GRPO fabricates a length-concordant
   "phantom advantage" that **trains the model to give up faster**. Gating on the *binary
   task outcome* kills it.

## Why (diagnosed on tau2-airline, 2026-07-13)

veRL standard GRPO from an SFT checkpoint gave a **flat** curve across two LRs
(lr=4e-6 ×10 steps, lr=2e-5 ×4 steps): train score stuck ~0.19–0.29, `ppo_kl` larger at
the higher LR but `entropy` **rising** 0.70→0.88 — the policy moved into a more diffuse
region, not toward reward.

> ⚠️ **UPDATE (2026-07-15) — a controlled A/B refuted this section's hypothesis.**
> The flat curve was an **LR problem**, not a structural sample-efficiency one: raising
> lr to 1e-4 with **gating OFF** reaches val **0.5625**, while **gating ON** stalls at
> **0.4125**. Gating *hurts* here — at ~20% success it drops **54–75%** of the batch
> (`live_frac` 0.25–0.46) and starves the gradient. Full result:
> [`../results_20260712/RESULTS_ab_gating.md`](../results_20260712/RESULTS_ab_gating.md).
> The phantom-advantage mechanism below is still real and unit-tested; it simply did not
> translate into a val gain on this low-success-rate task.

My original hypothesis was that the root cause is structural, not tuning:
`compute_grpo_outcome_advantage` gives `(score-mean)/(std+eps)`, so an all-same-outcome
group gets ~0 advantage **but stays in the batch**. At ~20% success with n=8, **17–31%**
of the batch is such dead weight (see `test_tau2_like_success_rate`). *(This part is true;
what the A/B showed is that removing that dead weight does not help — and hurts — because
at low success rate the "dead" groups are the majority.)*

## The gate

```python
adv, keep = gated_grpo_advantage(scores_shaped, outcome_binary, index)
```
A group teaches only if `outcome_binary` contains **both** a success and a failure;
otherwise every rollout in it gets advantage 0 and `keep=False`. `keep` lets a trainer
drop those rows (dynamic sampling). Advantage is still computed from `scores_shaped`, so
shaping/PRM-Lite is supported — but the *gate* looks at the binary outcome, so an all-fail
group is zeroed even when its shaped rewards differ.

## Run the tests (no torch, no GPU)

```bash
python3 test_grpo_gated.py     # 11 passing checks
```
Highlight — standard GRPO on an all-FAIL group with length-varying shaped reward:
`adv = [1.24, 0.34, -0.56, -1.01]` (rewards the shortest failure); gated → `[0,0,0,0]`.

## Wire into veRL (on the GPU box, needs torch)

```python
# in a module imported at trainer startup
from grpo_gated import register_grpo_gated
register_grpo_gated()
# launch with:  algorithm.adv_estimator=grpo_gated
```

The registerable wrapper derives the binary outcome by thresholding the trajectory score
(exact when the reward is already binary — the tau2-airline case). With PRM-Lite shaping,
pass the raw `token_level_scores` through as the gate signal instead of thresholding the
shaped reward.

## Status / next

- [x] gate + keep-mask core, veRL-registerable wrapper, 11 CPU unit tests passing
- [ ] **phase 2 (trainer-level):** use `keep` to actually drop rows + resample generations
      until the batch is full of contrastful groups (port DAPO's `filter_groups` loop onto
      the standard trainer). The estimator gate alone already fixes phantom advantage; the
      resampling additionally removes dilution.
- [x] **GPU A/B done (2026-07-15) — the payoff run, and it refuted the hypothesis.**
      `grpo` (gating off) reaches val **0.5625**; `grpo_gated` (gating on) stalls at
      **0.4125**. The flat curve was an LR problem, not sample efficiency; gating *hurts*
      on tau2 (drops 54–75% of samples). Honest negative result — the estimator itself is
      correct (11 unit tests). See
      [`../results_20260712/RESULTS_ab_gating.md`](../results_20260712/RESULTS_ab_gating.md).
- [ ] Multi-seed (2–3 seeds, mean±std) to promote "gating hurts" from strong to airtight.
