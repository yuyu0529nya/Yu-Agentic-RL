# Conversation Handoff

This is the current handoff for continuing the `yuyu` long-horizon multi-tool
agent post-training project after migrating from the old Windows workspace.

## Project Goal

Build a strong big-model algorithm internship project around long-horizon
tool-using agents:

- tau2-bench airline baseline and pass-rate evaluation
- failure taxonomy and process-reward analysis
- PRM-Lite reranking as an intermediate diagnostic
- Qwen2.5-7B QLoRA SFT for tool-call behavior
- later transition to stronger full-rollout SFT / constrained decoding / GRPO

The user is learning while building. Explanations should be concrete,
teacher-like, and preferably in Chinese.

## Current Workspace

Current migrated workspace:

```text
/Users/yuyu/Desktop/yuyu_from_E_100.81.157.87
```

Original workspace:

```text
E:\yuyu
```

The root workspace is not currently a git repository. The official tau2-bench
checkout is under:

```text
third_party/tau2-bench
```

Important private archive:

```text
conversation_archive/current_yuyu_codex_thread.jsonl
```

Latest full Codex rollout archive used for this migration:

```text
/Users/yuyu/.codex/sessions/2026/05/16/rollout-2026-05-16T18-25-58-019e3052-44ab-7e83-ad57-251314e67603.jsonl
```

Do not publish `.env`, API keys, raw conversation archives, model checkpoints,
or large AutoDL tarballs.

## Current Status

The project has moved past early PRM-rerank work. The latest completed
end-to-end stage is:

```text
Phase2G: Base vs v3 vs v4 full tau2 rollout comparison
```

Phase2G showed that Phase2F's offline protocol gain did not transfer to full
tau2 pass^1 on the continuity 5-task slice. Current prepared next stage:

```text
Phase2H: Mixed Dialogue + Tool Policy SFT
```

Phase2H has now completed one AutoDL RTX 4090D training run. It trained the
model on assistant text turns, grounded single-tool turns, protocol-only
single-tool turns, and sequentialized one-call targets converted from gold
parallel tool turns.

Phase2G full tau2 result:

| Run | pass^1 | success | tasks |
| --- | ---: | ---: | ---: |
| Base Qwen2.5-7B | 0.2000 | 1/5 | 5 |
| Slot-Grounded v3 | 0.2000 | 1/5 | 5 |
| Single-Tool Protocol v4 | 0.2000 | 1/5 | 5 |

Phase2F offline behavior result, still useful as a local protocol capability:

| Metric | Base Qwen2.5-7B | Phase2F v4 SFT |
| --- | ---: | ---: |
| protocol wrapper rate | 0.000 | 0.875 |
| single-call rate | 0.000 | 0.875 |
| multi-call rate | 0.000 | 0.000 |
| tool-name accuracy | 0.000 | 0.750 |
| argument exact accuracy | 0.000 | 0.750 |
| clean single exact | 0.000 | 0.750 |

Training was healthy:

| Item | Value |
| --- | ---: |
| steps | 160 |
| trainable LoRA params | 40,370,176 |
| valid loss before | 1.9999 |
| valid loss after | 0.2489 |
| peak CUDA memory | 12,024.8 MB |

Important latest files:

- `reports/phase2h_mixed_dialogue_tool_policy_4090_results.md`
- `reports/behavior_sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048.md`
- `reports/sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048.md`
- `reports/phase2g_full_tau2_compare_phase2g_basev3_eager64_v4_eager32_20260616_115520.md`
- `reports/airline_qwen25_7b_base_vs_v3_vs_v4_phase2g_basev3_eager64_v4_eager32_20260616_115520.json`
- `reports/mixed_dialogue_tool_policy_dataset_v1_2048.md`
- `reports/phase2h_mixed_dialogue_tool_policy_runbook.md`
- `reports/phase2f_single_tool_protocol_v4_4090_results.md`
- `reports/behavior_sft_single_tool_protocol_v4_qwen25_7b_qlora_1536.md`
- `reports/sft_single_tool_protocol_v4_qwen25_7b_qlora_1536.md`
- `reports/single_tool_protocol_dataset_v4_1536.md`
- `autodl_artifacts/phase2f_single_tool_protocol_v4_4090_20260615_224720`

## How We Got Here

Older but still useful milestones:

- 50-task GLM-5.1 tau2 airline baseline: pass^1 / avg reward `0.5800`
  with 29 successes out of 50.
- PRM-Lite hard5 N=4 rerank showed the PRM rules could select successful
  trajectories on some hard tasks; hard5 oracle pass@4 and PRM-rerank pass@4
  reached `0.8000`.
- Phase2E Slot-Grounded Action-Prefix v3 fixed many grounded argument issues:
  tool-name accuracy and arg exact accuracy reached `0.708`, but single-call
  rate was only `0.083` and multi-call rate was `0.625`.
- Phase2F v4 changed the target to exactly one executable
  `<tool_call>...</tool_call>` span, which fixed the protocol boundary on the
  offline behavior proxy.

Useful earlier reports:

- `reports/prm_rerank_experiment_hard5_n4.md`
- `reports/prm_ablation_experiment_failed21_n4_timeout300_v2.md`
- `reports/phase2e_slot_grounded_v3_4090_results.md`
- `reports/phase2_base_vs_sft_tau2_eval_plan.md`

## Important Scripts

Core tau2 evaluation:

- `scripts/run_tau2_airline_baseline.sh`
- `scripts/run_tau2_airline_baseline.ps1`
- `scripts/run_tau2_base_vs_sft_vllm_autodl.sh`
- `scripts/run_phase2g_full_tau2_compare_autodl.sh`
- `scripts/summarize_tau2_results.py`
- `scripts/compare_tau2_runs.py`

PRM and diagnostics:

- `scripts/process_reward_scorer.py`
- `scripts/prm_rerank_tau2.py`
- `scripts/prm_ablation_tau2.py`
- `scripts/inspect_tau2_trajectory.py`

SFT data and training:

- `scripts/build_slot_grounded_action_prefix_dataset.py`
- `scripts/build_single_tool_protocol_dataset.py`
- `scripts/build_mixed_dialogue_tool_policy_dataset.py`
- `scripts/evaluate_sft_behavior.py`
- `scripts/evaluate_mixed_policy_behavior.py`
- `scripts/run_slot_grounded_action_prefix_sft_4090_autodl.sh`
- `scripts/run_single_tool_protocol_sft_4090_autodl.sh`
- `scripts/run_mixed_dialogue_tool_policy_sft_4090.sh`
- `scripts/run_mixed_dialogue_tool_policy_sft_4090_autodl.sh`

Tests:

- `tests/test_slot_grounded_action_prefix_dataset.py`
- `tests/test_single_tool_protocol_dataset.py`
- `tests/test_mixed_dialogue_tool_policy_dataset.py`

## AutoDL Policy

Do not proactively shut down AutoDL instances unless the user explicitly asks
for shutdown for that specific run. The user has changed this preference during
the project depending on GPU availability, so ask or infer only from the newest
instruction in the active conversation.

For Phase2F, the previous run left the 4090D instance running as requested in
that moment. That remote state may now be stale.

## Next Best Step

Phase2I is now the next step and its local preflight is prepared:

```text
Phase2I: keep Phase2H's strong tool behavior while fixing text/tool decision,
so assistant text turns stop over-calling tools before full tau2.
```

Phase2H behavior summary:

```text
base:   action_type=0.531, text_no_tool=1.000, tool_call=0.062, arg_exact=0.062
phase2h: action_type=0.734, text_no_tool=0.719, tool_call=0.750, arg_exact=0.750
```

The next iteration should reduce text-turn over-calling, likely by increasing
assistant-text supervision weight/ratio or by adding a decision gate before
full tau2.

Current result report:

```text
reports/phase2h_mixed_dialogue_tool_policy_4090_results.md
```

Decision rule:

- Do not call Phase2H fully successful yet.
- Do not spend a full tau2 run until text no-tool behavior is recovered.
- Keep the strong 0.750 tool-call/tool-name/arg-exact behavior as a useful
  positive signal.

Phase2I local implementation:

- `scripts/build_decision_balanced_mixed_policy_dataset.py`
- `scripts/run_decision_balanced_mixed_policy_sft_4090.sh`
- `scripts/run_decision_balanced_mixed_policy_sft_4090_autodl.sh`
- `tests/test_decision_balanced_mixed_policy_dataset.py`
- `reports/mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048.md`
- `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_manifest_v1_2048.json`

Phase2I local preflight result:

```text
train raw:      80 text / 139 tool, 219 rows
train balanced: 160 text / 113 tool, 273 rows
valid:          unchanged, 34 text / 60 tool, 94 rows
heldout:        unchanged, 108 text / 161 tool, 269 rows
validation_errors: 0
```

Validation already run locally:

- `python3 -m py_compile scripts/build_decision_balanced_mixed_policy_dataset.py ...`
- `bash -n scripts/run_decision_balanced_mixed_policy_sft_4090.sh scripts/run_decision_balanced_mixed_policy_sft_4090_autodl.sh`
- Direct execution of the three Phase2I balance tests.
- `python3 scripts/build_decision_balanced_mixed_policy_dataset.py --skip-tokenizer --max-sample-tokens 2048 --text-repeat 2 --protocol-keep-ratio 0.5 --balance-seed 11`

Remote status:

```text
4090D training has not started yet.
The old saved SSH gateway endpoint no longer connects, and direct
100.81.157.87:22 timed out. Need the current AutoDL/SeetaCloud SSH command
or host/port before uploading and running Phase2I remotely.
```

## Suggested New-Session Prompt

```text
请先读取 docs/conversation_handoff.md。
我们继续 yuyu 的 tau2-bench airline 长链路工具智能体后训练项目。
当前 Phase2G full tau2 对比已经完成：base/v3/v4 都是 1/5，pass^1=0.2。
Phase2H Mixed Dialogue + Tool Policy SFT 已经跑完：工具指标到 0.750，但 text_no_tool 降到 0.719。
Phase2I Decision-Balanced Mixed Policy 代码和本地 preflight 已完成；下一步需要新的 4090D SSH 地址，然后跑远端 tokenizer-aware 构建、QLoRA 和 behavior eval。
```
