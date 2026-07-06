# Agent Post-Training with GRPO — Portfolio Overview

A from-scratch, single-GPU reinforcement-learning post-training system for LLM agents, plus a
rigorous evaluation methodology, applied to three settings that together tell one coherent story:
**RL works when the reward is verifiable — including for a multi-turn tool-use agent — and when
it isn't, here is exactly why it fails and how I proved it.**

## The system (built from scratch, one GPU)
collect rollouts → reward → group-relative advantage → QLoRA update → serve (vLLM) → evaluate.
- Custom GRPO loop: group-relative advantage, outcome-variance advantage gating, length-aware
  normalization, advantage-weighted assistant-only token NLL; QLoRA (4-bit) so a 7B fits in 32 GB
  by alternating vLLM serving and training.
- Evaluation harness with a discipline most projects skip: held-out splits (no leakage),
  multi-trial pass@1, paired McNemar + bootstrap significance testing, and explicit
  results-vs-noise reporting.

## Result 1 — a clean, significant RL win (verifiable reward)
GSM8K, Qwen2.5-1.5B, 4-iteration on-policy GRPO with exact-match reward:
**pass@1 61.4% → 67.4% (+6.0, McNemar p<0.001, n=1319).** Training success rose monotonically
across iterations. → `gsm8k_rlvr_findings.md`.

## Result 2 — a rigorous negative result + deep diagnosis (noisy reward)
tau2-bench airline (multi-turn tool agent, Qwen2.5-7B). End-to-end RL produced no significant
pass@1 gain, and I diagnosed precisely why, from the data:
- SFT collapse mechanism (tool over-calling; offline metrics hide a multi-turn failure).
- A phantom gradient from a dense reward fabricating advantage on all-fail groups; a reward-code
  bug (flat-vs-nested tool-call parsing) that silenced all tool-category rules.
- A train/eval token-budget mismatch; a proven train/held-out skill-coverage gap.
- The dominant eval-noise source identified as the LLM user-simulator (the same base scored 0.35
  and 0.20 across runs), motivating a variance-controlled paired evaluation.
→ `grpo_rl_phase_findings.md`, `m1_m2_sft_endtoend_findings.md`.

## Result 3 — the Agent + RL flagship (multi-turn search agent, verifiable reward)
A multi-turn retrieval agent (`<search>` → BM25 over the HotpotQA distractor corpus → `<answer>`),
Qwen2.5-7B, 4-iteration on-policy GRPO with an exact-match reward:
**held-out EM 39.0% → 46.0% (+7.0, McNemar p=0.0099, n=300)** at the best checkpoint (iter 3).
Iteration 4 then *regressed* to 24.7% — a textbook reward over-optimization I diagnosed
mechanistically (answers shrank 24→7 chars as the policy over-paid for brevity past the EM
sweet spot), and handled with checkpoint selection / early stopping. → `search_agent_rlvr_findings.md`.

## Why the three results are one story
Same trainer, three reward regimes. The airline analysis concluded the blocker was the
reward/eval *signal*, not the algorithm. The GSM8K win confirmed it on single-turn reasoning, and
the search agent extends it to a **multi-turn tool-use agent**: swap in a verifiable reward and the
identical trainer produces a significant gain — and shows the next failure mode (over-optimization)
once the gain is real. Diagnosis → prediction → confirmation, plus the discipline to spot when more
training makes things worse.

## Map
- Code: `scripts/grpo/` (collect/update/reward/eval, gsm8k_*, search_agent/qa_reward/search_retriever,
  run_*.sh, NEXT_RUN.md).
- Reports: `reports/search_agent_rlvr_findings.md`, `reports/gsm8k_rlvr_findings.md`,
  `reports/grpo_rl_phase_findings.md`, `reports/m1_m2_sft_endtoend_findings.md`,
  `reports/m3_grpo_design.md`.
- Artifacts: `autodl_artifacts/` (eval jsonls, rollouts, metrics, adapters, logs).
