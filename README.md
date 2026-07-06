# Agentic RL — Reinforcement Learning for LLM Agents (self-built GRPO)

A **from-scratch GRPO reinforcement-learning stack** for post-training LLM agents —
trainer, on-policy rollout + evaluation harness, reward design, and diagnostics, all
built end to end on top of vLLM + QLoRA. No veRL, no TRL — the entire loop is my own.

## Highlights
- 🚩 **Multi-turn retrieval agent (RLVR): +10.7 pts held-out Exact-Match** — 38.7% → 49.3%
  (McNemar **p < 0.001**, n = 300; reconfirmed at **n = 2400, p < 1e-30**).
- **GSM8K (RLVR): 61.4% → 67.4% pass@1** (+6.0 pts, McNemar p < 0.001, n = 1319).
- Diagnosed and **fixed reward over-optimization** (Goodhart), then ran a controlled
  **three-way comparison of anti-over-optimization techniques** — the mechanism-targeted
  fix won and eliminated the late-training collapse.
- **Built the whole GRPO pipeline solo**, and extended it to multi-turn tool-calling RL on
  tau2-bench with rigorous paired evaluation (bootstrap CI + McNemar).

## 🚩 Flagship — a search agent trained with verifiable-reward RL (RLVR)
A multi-turn retrieval agent (`<search>` → BM25 over a HotpotQA corpus → `<answer>`),
trained with on-policy GRPO and a **verifiable** exact-match / token-F1 reward.

**Result: held-out Exact-Match 38.7% → 49.3% (+10.7 points)** — McNemar p < 0.001 (n = 300),
reconfirmed with multi-trial evaluation at **n = 2400, p < 1e-30**. This is a large,
statistically decisive gain, independently re-verifiable from the raw rollouts.

The engineering story is a complete **diagnose → fix → improve** loop:
1. A binary exact-match reward induces **reward over-optimization** — the policy games the
   metric by collapsing answers from ~24 to ~7 characters (textbook Goodhart).
2. Redesigning the reward as **token-F1 partial credit** raises the ceiling *and* stabilizes
   training.
3. A head-to-head of three anti-over-optimization levers — **KL-to-base anchor vs. dense
   process reward vs. length-aware advantage (LATA)** — shows the mechanism-targeted
   **length-aware advantage wins**, removing the late-training collapse entirely.

## GSM8K — a clean RLVR win
Qwen2.5-1.5B, the same self-built GRPO loop, exact-match reward:
**61.4% → 67.4% pass@1** (+6.0 pts, McNemar p < 0.001, n = 1319) — proving the trainer is
correct and the gains are real on a fully verifiable reward.

## The stack I built (`scripts/grpo/`)
- **`grpo_update.py`** — GRPO from scratch: group-relative advantage `(r − mean)/(std + ε)`,
  **outcome-variance advantage gating**, **length-aware advantage (LATA)**, QLoRA, batched
  loss, optional KL-to-base anchor.
- **`collect_rollouts.py`** — on-policy rollout collection driving the tau2 machinery.
- **`prm_lite_reward.py`** — a rule-based **dense process reward** (outcome + β·process).
- **`search_agent.py` / `search_retriever.py` / `qa_reward.py`** — the retrieval-agent
  episode loop, BM25 retriever, and verifiable QA reward.
- **`gsm8k_collect.py` / `gsm8k_eval.py`** — the GSM8K RLVR pipeline.
- **`tau2_eval_analyze.py` / `summarize_eval.py`** — paired evaluation with bootstrap
  confidence intervals and McNemar significance testing.

## Beyond the wins — multi-turn tool-calling RL (tau2-bench airline)
Extended the same pipeline to multi-turn, tool-calling customer-service dialogue: base-model
characterization, SFT-vs-RL comparison, dense process-reward shaping, and a careful study of
user-simulator **evaluation methodology** (paired testing, bootstrap CIs, controlled
ablations) — demonstrating end-to-end RL engineering on a hard, realistic agent benchmark.

## Tech
Qwen2.5 (0.5B / 1.5B / 7B) · vLLM · PEFT / QLoRA · bitsandbytes · transformers ·
tau2-bench · on-policy GRPO with verifiable rewards (RLVR).

*Trained adapters, model weights, and run artifacts are kept out of the repo for size —
this repository is the code and write-ups.*
