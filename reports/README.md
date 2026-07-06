# Reports

Findings for a from-scratch, single-GPU GRPO post-training system for LLM agents.
**Start here:** [PORTFOLIO_OVERVIEW.md](PORTFOLIO_OVERVIEW.md) · résumé bullets in [RESUME_BULLETS.md](RESUME_BULLETS.md).

Flagships are listed below, strongest first. The full experiment trail (SFT / tool-protocol
phases, PRM-rerank studies, per-run analyses and gates) lives in [`archive/`](archive/) for
reproducibility — it documents the process, but is not the first read.

## Flagship results

### 1. τ-bench airline — teacher-distillation → GRPO (headline)
Turned three rigorous nulls into a significant multi-turn tool-agent win. Diagnosed the
GRPO-from-base null as **gradient starvation** (weak base → all-fail rollout groups → zero
intra-group variance → no gradient) plus a skill-coverage gap, then fixed it with a
teacher-trajectory distillation warm start before GRPO: held-out **pass^1 0.20 → 0.41 → 0.55**
(10-trial nail-down, n=200; every stage's paired bootstrap CI excludes 0). The *identical* 2×2
ablation is all-null from base and uniformly significant from the distillation floor — the
blocker was the starting policy, not the algorithm. Reward-design ablation: length-aware
advantage is effective; turn-discounted advantage is null at matched budget.
→ [airline_distill_grpo.md](airline_distill_grpo.md) ·
[airline_distill_sft.md](airline_distill_sft.md) ·
[grpo_rl_phase_findings.md](grpo_rl_phase_findings.md) ·
[m1_m2_sft_endtoend_findings.md](m1_m2_sft_endtoend_findings.md)

### 2. Multi-turn search agent RLVR + over-optimization study
`<search>` → BM25 retrieval → `<answer>`, on-policy GRPO with a token-F1 verifiable reward.
Held-out **EM 38.7% → 49.3%** (+10.7, McNemar p<0.001, n=300; re-confirmed at n=2400). The
deterministic reward isolates measurement noise, enabling a clean over-optimization mechanism
study: the collapse was answer-length collapse (24→7 chars, a Goodhart effect), and length-aware
advantage beat KL / process-reward anti-collapse levers.
→ [search_agent_rlvr_findings.md](search_agent_rlvr_findings.md) ·
[search_agent_overopt_curve.svg](search_agent_overopt_curve.svg)

### 3. GSM8K single-turn RLVR — trainer cross-validation
Qwen2.5-1.5B, 4-iteration GRPO with exact-match reward: **pass@1 61.4% → 67.4%**
(+6.0, McNemar p<0.001, n=1319). Confirms the trainer itself is correct.
→ [gsm8k_rlvr_findings.md](gsm8k_rlvr_findings.md)

---
*A unifying thread: mechanism-targeted, length-aware reward design beats generic regularizers —
independently validated on the search agent (length normalization cures over-optimization) and
on airline (length-aware advantage effective, turn-discount null).*
