# Phase2H Mixed Dialogue Tool Policy 4090 Results

## Status

Run completed successfully on AutoDL RTX 4090D.

- Remote run dir: `/root/autodl-tmp/yuyu`
- Local artifact dir: `autodl_artifacts/phase2h_mixed_dialogue_tool_policy_4090_20260616_134216`
- Artifact tarball: `autodl_artifacts/phase2h_mixed_dialogue_tool_policy_4090_20260616_134216/yuyu_mixed_dialogue_tool_policy_artifacts_20260616_134216.tar.gz`
- Remote artifact tarball: `/root/autodl-tmp/yuyu_mixed_dialogue_tool_policy_artifacts_20260616_134216.tar.gz`
- Remote instance policy: `AUTO_SHUTDOWN=0`; the instance was left running.

## Dataset

Tokenizer-aware dataset build succeeded with validation errors at 0.

| Split | Rows kept | Rejected | Text targets | Tool targets | Mean tokens | Max tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 192 | 27 | 80 | 112 | 1030.9 | 2048 |
| valid | 82 | 16 | 34 | 48 | 1331.6 | 2041 |
| heldout | 233 | 36 | 108 | 125 | 1165.4 | 2048 |

The extra rejects compared with the local no-tokenizer build are from strict
grounding after context trimming. They are excluded instead of training on
tool calls whose evidence was trimmed away.

## Training

| Item | Value |
| --- | ---: |
| base model | Qwen2.5-7B-Instruct |
| max seq len | 2048 |
| steps | 260 |
| LoRA rank | 16 |
| trainable parameters | 40,370,176 |
| valid loss before | 2.3769 |
| valid loss after | 0.4015 |
| final rolling train loss | 0.2326 |
| elapsed seconds | 275.6 |
| max CUDA memory MB | 13,435.5 |

Training was mechanically healthy.

## Mixed-Policy Behavior

| Model | Action type acc | Text no-tool | Text nonempty | Tool-call rate | Wrapper | Single-call | Multi-call | Tool-name acc | Arg exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.531 | 1.000 | 1.000 | 0.062 | 0.062 | 0.062 | 0.000 | 0.062 | 0.062 |
| phase2h | 0.734 | 0.719 | 1.000 | 0.750 | 0.750 | 0.750 | 0.000 | 0.750 | 0.750 |

## Interpretation

Phase2H fixed the main under-calling/tool-protocol failure mode much better than
base: tool-call rate, wrapper rate, single-call rate, tool-name accuracy, and
argument exact accuracy all reached 0.750 on the 64 heldout probes.

The remaining problem is over-calling. Text no-tool rate dropped from 1.000 to
0.719, meaning the adapter calls tools on about 9 of 32 text probes where the
gold target is a natural assistant response. That is a real full-tau2 risk.

Decision: do not call this phase fully successful yet. It is a useful positive
step for tool execution, but the next code/data iteration should reduce
assistant-text over-calling before spending a full tau2 comparison run.

## Files

- `reports/mixed_dialogue_tool_policy_dataset_v1_2048_remote.md`
- `reports/sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048.md`
- `reports/behavior_sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048.md`
- `autodl_artifacts/phase2h_mixed_dialogue_tool_policy_4090_20260616_134216/metrics.json`
- `autodl_artifacts/phase2h_mixed_dialogue_tool_policy_4090_20260616_134216/behavior_summary.json`
