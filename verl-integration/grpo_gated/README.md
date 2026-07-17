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

> ⚠️ **UPDATE (2026-07-16) — a 2-seed × 2-arm controlled A/B refuted this section's hypothesis,
> and then refuted my first refutation too.** Read this before believing anything below.
>
> 1. The flat curve was an **LR problem**, not structural sample efficiency
>    (`grad_norm≈0.05`, 20× under the clip threshold; `pg_clipfrac≈0.001`). Raising lr to 1e-4
>    with **gating OFF** reaches val **0.5625 / 0.55** (2 seeds).
> 2. I then claimed from a **single seed** that gating *hurts* (0.4125 vs 0.5625). **That was
>    wrong.** On seed=123 the gap collapsed to **0.5375 vs 0.55**. Arm(ON)'s own cross-seed
>    spread (0.125) exceeds the between-arm mean gap (0.081) → at n=2 there is no effect.
> 3. **On veRL + Adam + BINARY reward this gate is ≈ a no-op**, and provably so: the rows it
>    drops have advantage **exactly 0** (std=0 → `(0−0)/(0+eps)`), so nothing can be starved.
>    Its only mechanical effect is shrinking the `token-mean` denominator (`loss_mask.sum()`),
>    which scales the loss by `1/live_frac` (measured 2.4×–24×) — and **Adam's scale-invariance
>    absorbs it**. Measured: `pg_loss` ↑4–17×, `grad_norm` ↑2–3×, but **`ppo_kl` identical** and
>    **paired train diff = −0.006, 95% CI [−0.018, +0.006]**.
>
> Full write-up: [`../results_20260712/RESULTS_ab_gating.md`](../results_20260712/RESULTS_ab_gating.md).
> The phantom-advantage mechanism below is still real and unit-tested — it just has nothing to do
> under a **binary** reward (see *Status / next*).

My original hypothesis was that the root cause is structural, not tuning:
`compute_grpo_outcome_advantage` gives `(score-mean)/(std+eps)`, so an all-same-outcome
group gets ~0 advantage **but stays in the batch**. At ~20% success with n=8, **17–31%**
of the batch is such dead weight (see `test_tau2_like_success_rate`). *(The dilution itself is
real. What the A/B showed is that **removing it changes nothing**, because those rows' advantage
is already exactly 0 — so un-diluting them only rescales the loss, and Adam cancels the rescale.)*

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

**Use the ready-made runtime wiring — do not hand-patch veRL.**

```bash
../run_tau2_grpo_gated_seed123.sh          # the gated arm, seed pinned
../run_tau2_grpo_gated_seed123_managed.sh  # + artifact snapshot & verified shutdown
```

They put [`../gated_runtime/`](../gated_runtime/) on `PYTHONPATH`, so Python auto-imports
`sitecustomize.py`, which wraps `ray_trainer.compute_advantage` **inside the Ray worker**
(registering the estimator in the shell's driver process is not enough — `fit` runs in Ray).
It activates only when `USE_DYNAMIC_SAMPLING=1`, so ordinary GRPO launches are untouched.

It is deliberately **fail-closed** — it raises instead of silently degrading into plain GRPO if:
- `Tau2AgentLoop` did not export `outcome_binary` (the raw, unshaped success label);
- the batch has no `uid` grouping key;
- the installed veRL lacks the expected `RayPPOTrainer.fit` → `compute_advantage(` seam;
- dynamic sampling masked the **entire** batch (a 0/0 masked loss).

That first check matters: `apply_dynamic_sampling` no-ops when `outcome_binary` is absent, so
without a hard failure a "gated" run would quietly be an ordinary GRPO run — and the A/B would
compare two identical arms. `outcome_binary` is emitted by
[`../tau2_agent_loop.py`](../tau2_agent_loop.py) from the **unshaped** evaluator reward on purpose:
under PRM/shaping, deriving the gate label from the shaped reward would reintroduce the very
phantom advantage this gate exists to kill.

Manual alternative (driver process only — insufficient for Ray, kept for reference):

```python
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
- [x] **GPU A/B done (2026-07-15) — refuted my "sample efficiency" hypothesis.** The flat curve
      was an LR problem. Single-seed result said gating *hurts* (0.4125 vs 0.5625).
- [x] **Multi-seed done (2026-07-16) — and it refuted the "gating hurts" claim too.**
      Filling the 2×2 (`{42,123} × {gating on,off}`) collapsed the effect from **−0.15** (seed=42)
      to **−0.0125** (seed=123). Same-seed paired train over 20 steps: **−0.006, 95% CI
      [−0.018, +0.006]** — no effect. **Conclusion: on veRL + Adam + BINARY reward this gate is
      ≈ a no-op**, mechanistically explained (dropped rows already have advantage 0; the gate only
      rescales the loss; Adam cancels the rescale). See
      [`../results_20260712/RESULTS_ab_gating.md`](../results_20260712/RESULTS_ab_gating.md).
      **Methodological lesson: an n=1 RL A/B fabricated a 0.15-sized effect that does not exist.**

### Where this gate could still earn its keep (untested)

Under a **binary** reward an all-fail group has `std = 0` → advantage exactly 0 → **there is no
phantom advantage to kill**, which is precisely why the A/B above found nothing. The phantom only
appears under **shaped / PRM rewards**, where an all-fail group has `std > 0` and GRPO fabricates a
length-concordant advantage (and where DAPO's *std-based* `filter_groups` would **not** catch it —
that is this gate's genuine differentiator). **That regime was never tested.**

- [ ] **The right test:** run this gate under PRM-Lite / shaped reward, where the phantom exists.
- [ ] **phase 2 (trainer-level) — now the priority, and it is a confound fix, not an add-on:**
      use `keep` to physically **drop rows + resample** until the batch is refilled (port DAPO's
      `filter_groups` loop onto the standard trainer). Mask-zeroing shrinks the `token-mean`
      denominator, so it is **confounded with a learning-rate change**; drop+resample keeps the
      denominator constant and cleanly separates "dropping dead groups" from "shrinking the
      denominator". Cheap to run on GSM8K (deterministic reward → no API cost, low noise).
