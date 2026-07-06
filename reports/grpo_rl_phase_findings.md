# GRPO RL Post-Training on tau2-bench Airline — Findings & Honest Negative Result

Qwen2.5-7B-Instruct, single-GPU QLoRA, custom GRPO loop. This report covers the RL phase
(retail pipeline validation → airline R4 → airline R5) and the evaluation-methodology work.
SFT-phase findings are in `m1_m2_sft_endtoend_findings.md`; the RL design is in `m3_grpo_design.md`.

## TL;DR
I built a complete, single-GPU agent RL post-training system from scratch (rollout collection
→ reward → group-relative advantage → QLoRA update → serve → eval) and ran it end-to-end on
tau2-bench airline. **No method (rejection-sampling fine-tuning or gated GRPO) produced a
statistically significant pass^1 gain on the 20 held-out tasks (all McNemar p = 1.0).** The
value of this work is not a leaderboard number — it is a rigorous, mechanistic diagnosis of
*why* small-data post-training does not move the needle here, and an evaluation methodology
that can tell a real effect from noise. The two binding constraints I identified and proved
from the data are (1) **evaluation noise dominated by the LLM user-simulator** exceeds the
plausible effect size, and (2) a **train/held-out skill-coverage gap** (the training successes
never exercise the write-tools the held-out tasks require).

## 1. Setup
- **Task**: tau2-bench airline (single-control, 50 tasks). Long-horizon multi-tool dialogue;
  a task is "solved" only when the agent issues the correct database-modifying (write) action.
- **Policy**: Qwen2.5-7B-Instruct, 4-bit QLoRA (nf4), LoRA on all attn+MLP projections.
- **Harness**: policy served by vLLM (OpenAI-compatible); user simulated by `deepseek/deepseek-chat`
  via API (fast, no rate limit). Reward = tau2 binary task outcome.
- **Trainer (my own)**: REINFORCE-with-group-baseline (the core of GRPO without PPO clipping);
  per task a group of N rollouts, advantage_i = (reward_i − group_mean)/(group_std + eps);
  advantage-weighted, assistant-only token NLL with a length-aware normalization option.
  Reward modes: binary outcome, or outcome + a rule-based dense process score.
- **Split**: 30 train / 20 held-out, deterministic and disjoint (no leakage).

## 2. Evaluation methodology (a contribution in itself)
- Headline metric: pass^1 on the 20 held-out tasks; also reported on a **LIVE-10 pool** (the
  tasks any checkpoint ever solves) because the other 10 are unsolvable by every checkpoint and
  only add variance.
- Significance: paired **McNemar exact test** on per-task base-vs-adapter outcomes; for
  multi-trial runs, a paired per-task rate test (bootstrap 95% CI + sign test).
- **Key methodological finding — the user-simulator is the dominant noise source.** Because the
  user is itself a sampling LLM, the same base policy (greedy, temperature 0) scored 0.35 /
  LIVE-10 0.70 in one run and 0.20 / LIVE-10 0.30 in another — purely from different user
  utterances. So single-run pass^1 is not reproducible, and any cross-run comparison must be
  paired and variance-controlled. I added a user-simulator temperature pin (cut the noise at
  the source) plus the paired rate test as the correct evaluation protocol.

## 3. Results
All pass^1 on the 20 held-out, temperature-0 (greedy) agent, deepseek user-sim.

| Stage | Setup | pass^1 (overall) | pass^1 (LIVE-10) | vs base | significance |
|---|---|---|---|---|---|
| Pipeline check | retail, GRPO 1 update | 0.20 → 0.20 | — | flat | (validation only) |
| R4 base | airline | 0.35 (7/20) | 0.70 | — | — |
| R4 GRPO (prm-shaped, chained) | iter1/2/3 | 0.20 / 0.25 / 0.35 | — | dip→recover | within noise |
| R5 base | airline (re-run) | 0.20 | 0.30 | — | — |
| R5 RFT (from base) | rejection-sampling FT | 0.25 | 0.40 | net +1 | McNemar p=1.0 |
| R5 gated GRPO | binary+gate, curriculum, N=12, LR 5e-6 | 0.20–0.25 | 0.30–0.40 | net 0/+1 | McNemar p=1.0 |

**Honest read:** every adapter lands within the ±0.15 binomial noise band of the base over 20
tasks; none of the +1-task swings survives a paired significance test.

## 4. Diagnosis (the core contribution)
1. **SFT collapse, mechanistically.** (prior phase) Behavior-cloning on a small hard-task
   trajectory set *hurt* end-to-end pass^1 (0.28 → 0.10) even though offline single-turn tool
   accuracy was 0.75. Cross-trajectory analysis: the SFT policy over-calls tools (≈4× more
   tool calls/task, ~90% of turns become tool turns) and stops communicating — a multi-turn
   failure that offline metrics completely miss.
2. **Gradient starvation + a phantom gradient.** With binary reward, ~11/15 train groups had
   zero intra-group outcome variance, so GRPO had almost no real contrast to learn from. Worse,
   the dense process reward *manufactured* tiny non-zero advantages on all-fail groups
   (so "usable rollouts" read 90/90 — a bug masquerading as a feature), and those phantom
   advantages were ~63% concordant with simply rewarding *shorter* failures, i.e. the dominant
   gradient trained the model to give up faster (the mechanical cause of the iter-1 dip).
3. **A reward-code bug under the symptom.** The process-reward module parsed tau2's *flat*
   tool-call format as the OpenAI *nested* format, so every tool name resolved to empty → all
   tool-category rules silently no-op'd → only length/verbosity penalties ever fired. Fixing the
   parser revived tool classification (and a previously dead data-chaining reward that had been
   hard-coded off). Lesson: a reward function can be "running" yet measuring nothing.
4. **Train/eval regime mismatch.** Training collected rollouts at temperature 1.0 / 256 max
   tokens (84–112 truncated tool calls per iteration), while eval ran greedy / 512 tokens
   (0 truncated). The policy was optimized in a token-budget-crippled regime it was never scored
   in. Fixed by matching both to 768 tokens + a broken-tool-call sanity gate.
5. **Train/held-out skill-coverage gap (proved from data).** The 42 successful training
   trajectories use write-tools `cancel_reservation` (5×) and `book_reservation` (2×) and
   **never** `update_reservation_flights/passengers/baggages`. The successful held-out
   trajectories require exactly those: `update_reservation_flights` 13×, `update_reservation_
   passengers` 9×, `update_reservation_baggages` 1×. So imitation/RL on the available successes
   can teach read-and-route format but cannot transfer the write-skills the held-out tasks need —
   a hard ceiling independent of the algorithm.

## 5. What worked (engineering / methodology)
- A working single-GPU GRPO loop (alternating vLLM serving and QLoRA training to fit a 7B in 32GB).
- A version-pinning fix + fail-fast self-test for a silent assistant-token masking bug
  (a transformers v5 change zeroed the loss mask without raising — pinned 4.57.x + a 0.1s probe).
- The advantage gate: dropping zero-outcome-variance groups removed the phantom gradient
  (verified: usable rollouts 90/90 → 24/90, all from genuine success/fail contrasts).
- Variance-controlled paired evaluation (user-sim temperature pin + paired per-task rate test).
- A reproducible single-command pipeline and an honest results-vs-noise reporting discipline.

## 6. Conclusion
On tau2-bench airline at this scale (7B, single GPU, ~15–30 train tasks, 20 held-out, an LLM
user-simulator), **the evaluation noise and the train/held-out skill gap exceed what small-data
post-training can overcome**, so neither RFT nor gated GRPO yields a credible pass^1 gain. This
is a clean, well-characterized negative result: the bottleneck is the *signal* (reward
sparsity/quality and eval noise) and *data coverage*, not the optimization algorithm — which I
demonstrated works mechanically once the signal is gated and the bugs removed.

## 7. What would move it (not done here)
- **Statistical power**: expand held-out from 20 → 50 tasks; pin user-sim temperature and use
  the paired rate test (both implemented, `run_paired_eval.sh`).
- **Data coverage**: collect/seed successful trajectories that exercise the write-tools the
  held-out tasks require (the proven gap), or train on tasks overlapping the held-out skills.
- **Verifiable reward**: move the RL showcase to a task with a deterministic, exact-match reward
  (the noise we fought here disappears when the reward is verifiable), reusing this trainer.

## Appendix — pointers
- Code: `scripts/grpo/{collect_rollouts,grpo_update,prm_lite_reward,summarize_eval,build_rft_dataset}.py`,
  `{run_grpo_min,run_rft,run_paired_eval,base_eval_retail,adapter_eval_retail}.sh`, `NEXT_RUN.md`.
- Artifacts: `autodl_artifacts/{r2_retail_106_75_*,grpo_airline_r4_westc_*,r5_westc_*}/`.
- Split: `scripts/grpo/airline_rl_split.json` (train 30 / held-out 20).
