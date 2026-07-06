# When Offline SFT Gains Don't Transfer: A Diagnosed Failure of Behavior-Cloned Tool Policies on tau2-bench Airline

**Date:** 2026-06-23
**Scope:** End-to-end (full-rollout) evaluation of base vs SFT Qwen2.5-7B on tau2-bench airline, with a credible 50-task, leakage-aware harness and a mechanistic failure diagnosis.

## TL;DR

On a credible 50-task evaluation, **behavior-cloning SFT did not improve and substantially HURT** end-to-end task success:

| Model (agent) | overall pass^1 | seen-10 | unseen-40 |
| --- | ---: | ---: | ---: |
| **base Qwen2.5-7B** | **0.280** (14/50) | 0.300 (3/10) | 0.275 (11/40) |
| + Phase2H SFT (mixed dialogue+tool) | 0.120 (6/50) | 0.300 (3/10) | 0.075 (3/40) |
| + Phase2I SFT (decision-balanced) | 0.100 (5/50) | 0.100 (1/10) | 0.100 (4/40) |

**Diagnosed mechanism (not a serving artifact):** the SFT models collapse into **tool-call spamming**. They issue **18.5 tool calls/task vs base's 4.7**, with **89.6% of assistant turns containing a tool call vs base's 30.8%**, and only **~88 chars of dialogue text/turn vs base's ~268**. They stop communicating with the user, so both the DB-state and COMMUNICATE reward components fail. Offline single-turn metrics (0.75 tool-name/arg accuracy) completely missed this multi-turn degeneration.

## 1. Problem & motivation

Prior offline "behavior proxy" evals were encouraging (Phase2F/2H reached ~0.75 tool-name/argument accuracy and ~0.875 protocol-wrapper rate). But Phase2G hinted these gains did not transfer to real tau2 rollout on a 5-task slice (base = v3 = v4 = 0.20). That slice was too small (1 task = 20%) to trust. **M1's goal: build a statistically credible, leakage-aware end-to-end baseline. M2's goal: measure whether the SFT stack actually moves end-to-end pass^1.**

## 2. Evaluation setup (controlled comparison)

- **Agent under test:** Qwen2.5-7B-Instruct, served via vLLM (`--enforce-eager`), OpenAI-compatible endpoint. SFT variants = same base weights + a LoRA adapter (`--enable-lora`). The ONLY variable across conditions is the adapter.
- **User simulator (environment counterpart):** `anthropic/glm-5.1` (API), identical across all conditions.
- **Tasks:** all 50 airline tasks (ids 0–49). `NUM_TRIALS=1`, `max_steps=80`, `temperature=0`, `agent_max_tokens=128`, `max_concurrency=4`.
- **Metric:** pass^1 via `scripts/summarize_tau2_results.py`; leakage-aware split via `scripts/summarize_tau2_seen_unseen.py` + `scripts/m1_eval_task_split.json`.
- **seen/unseen split:** "seen" = the 10 tasks used in any SFT train/valid split (`1,12,15,20,23,27,33,34,38,42`); "unseen" = the other 40. Reusable, documented.

### A note on the split (control for difficulty)
A reference GLM-5.1 baseline (agent = GLM-5.1) scored overall 0.58 but **seen 0/10, unseen 0.725** — i.e., the SFT was trained on the *hardest* tasks. For our actual model (Qwen2.5-7B base) seen (0.30) ≈ unseen (0.275), so the difficulty confound is mild for the base, making base-vs-SFT comparisons clean.

## 3. Results

See the TL;DR table. Both SFT variants reduce overall pass^1 by ~55–65% relative to base. Phase2I (built to reduce over-calling) did **not** recover performance; its seen-10 is actually worse than Phase2H (it traded away tool accuracy without fixing rollout behavior).

## 4. Mechanism: behavioral collapse into tool-call loops

Cross-model trajectory analysis (all 50 rollouts each):

| metric | base | Phase2I (SFT) |
| --- | ---: | ---: |
| avg assistant turns / task | 15.3 | 20.6 |
| **avg tool calls / task** | **4.7** | **18.5** |
| **% assistant turns with a tool call** | **30.8%** | **89.6%** |
| **avg dialogue text chars / turn** | **267.7** | **87.8** |

The SFT agent calls tools ~4× as often, on ~90% of turns, and barely talks. Sampled failed trajectories show consecutive empty-content tool calls (the model hammers tools without communicating). This is a genuine behavioral pathology, **not** a parsing/truncation artifact: of 1032 SFT assistant turns, 925 tool calls parsed cleanly and only 16 showed truncation.

**Interpretation:** behavior-cloning on a small set (136 train rows) of hard-task trajectories, with single-executable-tool-call targets, taught the model tool-call *formatting* but destroyed its multi-turn *policy* (when to talk, confirm, and stop). Offline single-turn probes cannot see this; only full rollout reveals the collapse. Even Phase2I's offline over-call fix ("text_no_tool" 0.81) did not translate to rollout behavior.

## 5. Caveats (honest)

- `NUM_TRIALS=1` → single-sample noise per task. But the drop (0.28→0.10) and the 4× over-calling far exceed single-task noise.
- `agent_max_tokens=128` was identical across conditions; truncation was checked and largely ruled out (16/1032).
- GLM-5.1 cost-tracking warnings in logs are non-fatal (completions returned 200 OK).

## 6. What this means / next steps

The current behavior-cloning SFT line is **net-negative end-to-end**, and we now know *why*. A one-shot decision label is unlikely to fix rollout over-calling (Phase2I already tried via data balancing). Candidate directions:

- **(A) Two-stage decision-gated policy** (Phase2J gate + Phase2H tool generator) with a *hard* constraint on when tools may be called — directly targets the diagnosed over-calling. Requires custom serving/routing.
- **(B) Rebuild SFT data** — larger, full-trajectory, less single-tool-target bias; preserve dialogue turns so policy isn't destroyed.
- **(C) Reward-driven policy (RL / GRPO + PRM-Lite/LATA-style)** — learn *when* to act from task reward rather than imitating; matches the long-term project thesis.

**Recommended framing:** base Qwen2.5-7B at pass^1 0.28 is the real starting point; the value-add must *preserve* base's healthy dialogue while improving tool use — which BC-SFT, as constructed here, does the opposite of.

## 7. Reproduce

```bash
# base
RUN_BASE=1 RUN_SFT=0 TASK_IDS=0..49 MAX_CONCURRENCY=4 VLLM_ENFORCE_EAGER=1 \
  bash scripts/run_tau2_base_vs_sft_vllm_autodl.sh
# SFT (adapter)
RUN_BASE=0 RUN_SFT=1 ADAPTER_PATH=outputs/<adapter>/checkpoint ... bash scripts/run_tau2_base_vs_sft_vllm_autodl.sh
# leakage-aware summary
python scripts/summarize_tau2_seen_unseen.py <sim_dir> --split scripts/m1_eval_task_split.json
```

Artifacts: `autodl_artifacts/m1_base50_5090d_20260623/`, `autodl_artifacts/m2_phase2h_5090_20260623/`, `autodl_artifacts/m2_phase2i_5090d_20260623/`.
