# Agent Post-Training with GRPO — Portfolio Overview

A from-scratch, single-GPU reinforcement-learning post-training system for LLM agents, plus a
rigorous evaluation methodology, applied to three settings that together tell one coherent story:
**when agent RL fails, diagnose the exact mechanism, fix it, and prove the fix with a controlled
contrast — from three rigorous nulls to pass^1 0.20 → 0.41 → 0.55 on a noisy multi-turn tool agent,
with the RL science underneath validated in a clean verifiable-reward environment.**

## The system (built from scratch, one GPU)
collect rollouts → reward → group-relative advantage → QLoRA update → serve (vLLM) → evaluate.
- Custom GRPO loop: group-relative advantage, outcome-variance advantage gating, length-aware
  normalization, advantage-weighted assistant-only token NLL; QLoRA (4-bit) so a 7B fits in 32 GB
  by alternating vLLM serving and training.
- Evaluation harness with a discipline most projects skip: held-out splits (no leakage),
  multi-trial pass@1, paired McNemar + bootstrap significance testing, and explicit
  results-vs-noise reporting.

## Result 1 — the headline: three nulls diagnosed, then flipped into the biggest win (tau2 airline)
tau2-bench airline (multi-turn tool agent, Qwen2.5-7B). End-to-end RL from the raw base first
produced three rigorous nulls — which I diagnosed precisely from the data:
- **Gradient starvation**: base ~20% success → most rollout groups all-fail → zero intra-group
  variance → no learning signal; plus a phantom gradient from a dense reward fabricating
  advantage on all-fail groups, and a reward-code bug (flat-vs-nested tool-call parsing).
- A self-sampling **skill-coverage gap** (the policy's own successes never call the
  reservation-change write-tools that held-out tasks need).
- The dominant eval-noise source identified as the LLM user-simulator, motivating a
  variance-controlled paired evaluation (pinned-greedy user-sim, fixed seed, paired-by-task).

Then the fix, proven by controlled contrast: **teacher-trajectory distillation as a warm start,
then GRPO** — held-out **pass^1 0.20 → 0.41 (distilled SFT) → 0.55 (GRPO)**, 10-trial nail-down
(n=200/checkpoint), every stage's paired bootstrap CI excluding 0. The *identical* GRPO 2×2
ablation is all-null from base and uniformly significant from the distillation floor — the
blocker was the starting policy, not the algorithm. Mechanistic extras: the RL ceiling is set by
skill coverage, not SFT-floor height; training saturates by ~iter 5 and stays collapse-free
through 10 iters (length-aware advantage), while turn-discounted advantage flatlines at matched
budget. → `airline_distill_grpo.md`, `airline_distill_sft.md`, `grpo_rl_phase_findings.md`,
`m1_m2_sft_endtoend_findings.md`.

## Result 2 — the RL-science leg (multi-turn search agent, controlled verifiable reward)
A multi-turn retrieval agent (`<search>` → BM25 over the HotpotQA distractor corpus → `<answer>`),
Qwen2.5-7B, on-policy GRPO. The deterministic reward isolates measurement noise, which is what
made the mechanism work possible: the binary-EM run climbed to 0.46 then *collapsed* at iter 4 —
a textbook reward over-optimization diagnosed mechanistically (answers shrank 24→7 chars) — and
the fix loop (token-F1 partial credit + a controlled 3-way of anti-collapse levers, where
length-aware advantage wins) pushed the final result to **held-out EM 38.7% → 49.3%
(+10.7, McNemar p<0.001, n=300; reconfirmed at n=2400, p<1e-30)** with the late-stage collapse
removed. → companion repo `search-agent-rlvr` and `search_agent_rlvr_findings.md`.

## Result 3 — trainer cross-validation (GSM8K, verifiable reward)
GSM8K, Qwen2.5-1.5B, 4-iteration on-policy GRPO with exact-match reward:
**pass@1 61.4% → 67.4% (+6.0, McNemar p<0.001, n=1319).** Training success rose monotonically
across iterations. → `gsm8k_rlvr_findings.md`.

## Why the three results are one story
Same trainer, three reward regimes. The airline nulls were diagnosed as a *starting-policy /
signal* problem, not an algorithm problem. The GSM8K win confirmed the trainer on single-turn
reasoning; the search agent confirmed it on a multi-turn tool-use agent with a verifiable reward;
and the airline line closed the loop — inject the missing skills by distillation and the very
same trainer that was null from base delivers its largest gain (0.20 → 0.55). Diagnosis →
prediction → confirmation → fix, plus the discipline to spot when more training stops helping.

## Map
- Code: `scripts/grpo/` (collect/update/reward/eval, gsm8k_*, search_agent/qa_reward/search_retriever,
  run_*.sh, NEXT_RUN.md).
- Reports: `reports/airline_distill_grpo.md`, `reports/airline_distill_sft.md`,
  `reports/grpo_rl_phase_findings.md`, `reports/m1_m2_sft_endtoend_findings.md`,
  `reports/search_agent_rlvr_findings.md`, `reports/gsm8k_rlvr_findings.md`.
- Artifacts: `autodl_artifacts/` (eval jsonls, rollouts, metrics, adapters, logs).
