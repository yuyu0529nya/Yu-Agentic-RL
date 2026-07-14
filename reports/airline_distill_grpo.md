# Airline: Teacher-Distillation → GRPO — turning three nulls into a significant, mechanistic win

**TL;DR.** On τ-bench airline (a multi-turn, multi-tool, self-hosted-user-simulator dialogue
benchmark), GRPO from a raw base policy had repeatedly produced null results. Diagnosis traced the
cause to **gradient starvation**: a weak base (~20% success) makes most on-policy rollout groups
all-fail, so intra-group advantage variance vanishes and there is no learning signal. Fix:
**warm-start the policy with teacher-trajectory distillation, then run GRPO.** Result — held-out
pass^1 climbs **base 0.20 → distilled SFT 0.41 → GRPO 0.55** (10-trial nail-down, n=200 per
checkpoint, every stage's paired CI excluding 0; single-session peak 0.56), with all four RL
reward configurations significantly beating both base and their own SFT start. The decisive
evidence is a controlled contrast: the *identical* GRPO ablation from base is all-null; the only
change that unlocks it is the distillation floor. Extended training saturates at ~0.55–0.56 by
iteration ~5 and stays stable through 10 iterations (no over-optimization collapse).

All numbers below are recomputed from raw per-trajectory JSONL; recompute commands are at the end.

---

## 1. The problem, from prior work

- **GRPO-from-base 2×2 ablation was all-null** (vanilla / process-reward / length-aware-advantage /
  combined): nothing beat base 0.23 at n=20 held-out tasks. Root cause verified from the rollouts:
  with base success ~20%, most of the 30 training tasks return all-fail groups → zero intra-group
  variance → the group-relative advantage is ~0 everywhere → no gradient.
- **Self-RFT is skill-bounded**: the base policy's own successful rollouts *never* invoke the
  write-tools (`update_reservation_flights/passengers/baggages`) that held-out tasks require. You
  cannot reinforce a skill the policy never exhibits.
- **Naive tool-fragment SFT hurt** (earlier phase): single-turn tool-biased SFT collapsed held-out
  0.28 → 0.10 by over-calling tools and under-communicating.

So the missing ingredient was a way to inject the *complete multi-turn behavior* — including the
write-skills — that RL could then optimize.

## 2. Method

**Teacher distillation.** A strong API model (deepseek-class) plays the *agent* through the exact
same τ-bench harness used for training and evaluation (same system prompt, tool schemas, dialogue
loop). Successful trajectories (reward = 1) are filtered, de-duplicated by action signature, and
capped per task, then used as the target for behavior-cloning SFT (render-twice-diff assistant-token
loss mask, QLoRA on Qwen2.5-7B).

**GRPO from the SFT checkpoint.** The same self-built GRPO loop (group-relative advantage
`(r − mean)/(std + ε)`, outcome-variance gating, optional length-aware advantage and rule-based
dense process reward), warm-started from the distilled adapter instead of raw base.

**Evaluation.** 30 train / 20 held-out tasks (zero overlap), pass^1 over 5 trials at agent temp
0.5, self-hosted base-7B user-simulator pinned to temp 0 with a constant eval seed, so every
checkpoint faces the same user-sim conversations (fair paired comparison). Same-session base is the
comparator (the temp-0.5 agent drifts ~1 SE across nights via user-sim stochasticity).

## 3. The distillation floor (SFT alone)

114 teacher trajectories (76% teacher success vs base ~20%; 51% contain write-actions — the
self-RFT blind spot, now covered).

| checkpoint | train loss | held-out pass^1 | Δ vs base | 95% CI |
|---|---|---|---|---|
| base | — | 0.180 | — | — |
| SFT 3 epochs | 1.061 | 0.220 | +0.040 | [−0.03, +0.13] ns |
| **SFT 9 epochs** | **0.743** | **0.380** | **+0.200** | **[+0.13, +0.27] ★** |

Behavior cloning has a **training-depth threshold**: at 3 epochs the loss "looked trained" (1.25 →
1.06) but behavior barely moved; the capability jump only appeared once loss was pushed into the
~0.7 regime. Continuing to 15 epochs (loss 0.58) gave no further held-out gain → the 114-example
set is saturated at ~9 epochs.

Mechanism (behavioral diff over 100 held-out sims, base → SFT): reads state 2.3× more before
acting, halves reckless write-calls, transfers policy-refusal requests ~3× more, communicates ~35%
more per turn — the exact inverse of the earlier over-calling SFT failure.

## 4. GRPO from the SFT floor — the thesis proven

Warm-started from the 9-epoch distilled adapter; same 2×2 that was null from base. pass^1 per iter
(n = 20×5 = 100 per checkpoint):

| arm | it1 | it2 | it3 | endpoint Δ vs base | peak Δ vs SFT-init |
|---|---|---|---|---|---|
| vanilla (binary) | 0.540 | 0.460 | 0.480 | +0.25 [+0.12,+0.38] ★ | +0.22 [+0.10,+0.34] ★ |
| prm_lite + LATA | 0.450 | 0.460 | 0.540 | +0.31 [+0.17,+0.46] ★ | +0.22 [+0.09,+0.36] ★ |
| prm_lite only | 0.500 | 0.490 | 0.480 | +0.25 [+0.12,+0.39] ★ | +0.18 [+0.08,+0.29] ★ |
| LATA only | 0.440 | 0.520 | 0.540 | +0.31 [+0.15,+0.47] ★ | +0.22 [+0.07,+0.38] ★ |

(base 0.23; SFT-init this session 0.32. init-alone vs base +0.09 ns — SFT-alone not individually
significant this session due to user-sim drift, but every RL arm is.)

**Every arm significantly beats base AND its own SFT start.** The reward configuration barely
matters (all tie ~0.48–0.54); the *warm start* is what unlocked learning. This is the controlled
proof: identical algorithm from base → null; from the distillation floor → uniformly significant.
The earlier airline nulls were a **starting-policy problem (gradient starvation), not an algorithm
problem**. (Suggestive: the LATA-bearing arms are still climbing at it3 while vanilla/prm-only peak
at it1 — LATA may sustain the climb; a multi-seed check is the natural follow-up.)

## 5. A second, subtler finding: the RL ceiling is set by skill coverage, not floor height

A larger/stronger teacher set (mega-212: three collection runs merged, 30/30 tasks covered,
including v4-pro attempts on the hardest tasks) was distilled and then GRPO'd:

| | SFT floor (held-out) | GRPO peak (held-out) |
|---|---|---|
| 114-example set (9 epochs, loss 0.74) | 0.32–0.38 | 0.54 |
| mega-212 set (8 epochs, loss 0.81)   | **0.14** | **0.56** |

The mega set's SFT floor was *lower* (0.14 — under-trained at loss 0.81, and not above base), yet
GRPO reached the **same or slightly higher** ceiling (0.56, +0.39 over that session's base 0.17,
CI [+0.23,+0.55]). Interpretation: what a warm start contributes to GRPO is **skill coverage** (the
write-tools and multi-turn structure needed to occasionally succeed on every task → outcome-mixed
rollout groups → live gradient), not a high greedy-eval score. An under-trained checkpoint with full
skill coverage un-starves the gradient just as well as a well-trained one; RL then elicits the latent
skills either way. Corollary: bigger teacher data did **not** raise the RL ceiling here (~0.54–0.56
either way) — the ceiling is bounded by skill coverage + measurement, not floor height.

Also documented (honest negative): even the stronger v4-pro teacher could not solve tasks 7/29/39
(≈2/35) → these are structurally hard (ambiguous / refusal / uncooperative-user tasks), not a
teacher-quality gap.

## 6. Engineering notes

- **20K-token trajectory training on one 32GB card** required three stacked fixes, each caught by a
  5-min smoke on the longest trajectories before the full run: eager attention → SDPA; drop the
  explicit all-ones attention mask (it forces SDPA off its fused causal path); compute the loss as
  chunked cross-entropy over hidden states rather than materializing full-sequence fp32 vocab logits.
- **CPU-only render pre-flight** (tokenizer-level) caught that the trainer's default 4096-token
  right-truncation would silently drop the late write-action turns from 67/114 examples — fixed to
  20480 before spending any GPU time.
- Ops hardening after live incidents: flock replay-guards (an SSH sleep/wake replayed a launch),
  append-only logs, exporting box knobs to child scripts (stale `/home` + `uv run tau2` defaults),
  and a free-memory (not just process-count) check before chaining runs (a dying vLLM holds VRAM
  after it leaves the process list).

## 6a. Method parity: Turn-Discounted Advantage (the one lever we were missing)

For completeness we added Turn-Discounted Advantage (weight each assistant token by
α^((N-1)−rank), α=1.05, mean-normalized — up-weighting early-turn reasoning) and ran it at the
*identical* budget (ITERS=3, N=6, from e9) alongside a re-run LATA anchor, same session:

| method (from e9 floor 0.37) | end pass^1 | Δ vs SFT init | vs LATA (same session) |
|---|---|---|---|
| Turn-Discount | 0.360 | −0.01 (peak +0.02) ns | — |
| LATA | 0.550 | +0.18 [+0.08,+0.29] ★ | LATA +0.19 [+0.09,+0.30] ★ |

Turn-Discount is flat off the SFT floor and significantly worse than LATA head-to-head. This
independently matches the ranking reported in published agentic-RL ablations (turn-discounting
stagnates while length-aware normalization keeps climbing) — here on tau2 v2, with our self-built
trainer, at matched budget. Mechanism-targeted length normalization (LATA) remains the winning
lever, consistent with the search-agent leg.

## 6c. Training-budget saturation (the "just train longer" lever, answered)

The best arm (LATA) was continued from its iter-3 checkpoint for 7 more on-policy iterations
(10 total), with the checkpoint re-anchored in the new session (paired, same usersim/seed):

```
base 0.23 · anchor(it3) 0.53 · it4 0.54 · it5 0.56 (peak) · it6–it10 0.55 ×5
```

- Peak (it5) vs anchor: +0.03, CI [−0.01, +0.08] ns; endpoint (it10) vs anchor: +0.02 [+0.00, +0.05].
- **The curve saturates at ~0.55–0.56 by iteration ~5** — more budget does not raise the ceiling,
  consistent with §5 (the ceiling is skill-coverage/measurement-bound, not budget-bound).
- Equally notable: **no over-optimization collapse through 10 iterations.** The binary-reward
  search-agent run collapsed at iter 4; here the length-aware advantage holds a stable plateau at
  3× that depth — the anti-collapse property persists under extended training.

## 6b. Nail-down (10-trial confirmation)

A same-session eval-only pass at **10 trials/task (n=200 per checkpoint)** confirms the arc at
higher power (paired bootstrap CI over the 20 held-out tasks):

| checkpoint | pass^1 | Δ vs base | 95% CI |
|---|---|---|---|
| base | 0.200 | — | — |
| distillation floor (e9) | 0.405 | +0.205 | [+0.105, +0.315] ★ |
| GRPO from e9 (prmlata) | **0.550** | +0.350 | [+0.210, +0.495] ★ |
| GRPO from mega (vanilla) | 0.545 | +0.345 | [+0.200, +0.495] ★ |

Every delta's CI excludes 0. The distillation floor is now significant *on its own* (the 10-trial
0.405 is the reliable estimate; the earlier 0.32/0.38 single-session reads were user-sim drift).
The two best RL checkpoints tie (0.550 vs 0.545) — the bigger mega teacher set did not beat the
114-set at the RL ceiling, consistent with §5. Headline arc: **base 0.20 → distill 0.41 → RL 0.55**.

## 6d. Training-seed robustness (3 independent replicates)

Three GRPO replicates from the same e9 floor with fully isolated training randomness
(rollout-sampling and update RNG both re-seeded; eval seed held constant for fair pairing):

| replicate | curve (it1/2/3) | endpoint | Δ vs base | Δ vs SFT-init |
|---|---|---|---|---|
| seed 101 | 0.51 / 0.53 / 0.52 | 0.520 | +0.34 ★ | +0.17 ★ |
| seed 202 | 0.43 / 0.52 / 0.55 | 0.550 | +0.37 ★ | +0.20 ★ |
| seed 303 | 0.52 / 0.43 / 0.49 | 0.490 | +0.31 ★ | +0.14 ★ |

**Every seed independently replicates both stages with individual significance** (all six paired
CIs exclude 0). Trajectories differ (early or mid-run dips), but endpoints converge to
0.49–0.55 (mean 0.52 ± 0.03). The effect is robust to training randomness; only the exact
endpoint carries ~±0.03 seed variance.

## 7. Honest caveats

- n = 20 held-out tasks (airline's full held-out set); CIs are wide even where significant. The
  10-trial nail-down (§6b) tightens the per-task rate estimates but the task count is fixed at 20.
- Endpoint seed-variance is ±0.03 (§6d): quote "0.49–0.55 across seeds, best checkpoint 0.55",
  not "always 0.55".
- The self-hosted user-simulator drifts run-to-run (base ranged 0.17–0.23 across sessions), which is
  why same-session base/init are the only valid comparators and cross-night absolute numbers are not.
- Teacher dialogues used an API user-simulator; evals use the self-hosted 7B — a mild train/eval
  style mismatch that did not prevent the gain but may cap it.

## 8. Recompute

```
# distillation floor (114-set)
python3 scripts/grpo/tau2_eval_analyze.py autodl_artifacts/distill_sft_0702/base_eval.jsonl \
  e3=autodl_artifacts/distill_sft_0702/distill_eval.jsonl \
  e9=autodl_artifacts/distill_sft_0702_ext/distill_ext_eval.jsonl
# GRPO-from-SFT 2x2 (114-set warm start)
python3 scripts/grpo/tau2_eval_analyze.py autodl_artifacts/grpo_from_sft_0703c/base_eval.jsonl \
  init=.../init_eval.jsonl vanilla=.../vanilla_eval_it3.jsonl prmlata=.../prmlata_eval_it3.jsonl \
  prmonly=.../prmonly_eval_it3.jsonl lataonly=.../lataonly_eval_it3.jsonl
# GRPO-from-mega
python3 scripts/grpo/tau2_eval_analyze.py autodl_artifacts/grpo_from_mega_0704/base_eval.jsonl \
  init_mega=.../init_eval.jsonl vanilla=.../vanilla_eval_it3.jsonl prmlata=.../prmlata_eval_it2.jsonl
```
