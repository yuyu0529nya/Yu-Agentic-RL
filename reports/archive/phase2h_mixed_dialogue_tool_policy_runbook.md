# Phase2H Mixed Dialogue Tool Policy Runbook

## Goal

Phase2G proved that the Phase2F single-tool protocol improvement was not enough
to improve full tau2 pass^1. Phase2H trains the next missing layer: when to talk,
when to call a tool, and how to execute one valid tool call at a time under
`parallel_tool_calls=false`.

## Local Preparation Status

Prepared files:

- `scripts/build_mixed_dialogue_tool_policy_dataset.py`
- `scripts/evaluate_mixed_policy_behavior.py`
- `scripts/run_mixed_dialogue_tool_policy_sft_4090.sh`
- `scripts/run_mixed_dialogue_tool_policy_sft_4090_autodl.sh`
- `tests/test_mixed_dialogue_tool_policy_dataset.py`
- `reports/mixed_dialogue_tool_policy_dataset_v1_2048.md`

Local checks already passed:

```bash
python3 -m py_compile scripts/build_mixed_dialogue_tool_policy_dataset.py scripts/evaluate_mixed_policy_behavior.py
bash -n scripts/run_mixed_dialogue_tool_policy_sft_4090.sh
bash -n scripts/run_mixed_dialogue_tool_policy_sft_4090_autodl.sh
python3 scripts/build_mixed_dialogue_tool_policy_dataset.py --skip-tokenizer --max-sample-tokens 2048
```

Dataset build result without tokenizer stats:

| Split | Rows | Rejected | Text targets | Tool targets |
| --- | ---: | ---: | ---: | ---: |
| train | 219 | 0 | 80 | 139 |
| valid | 94 | 4 | 34 | 60 |
| heldout | 269 | 0 | 108 | 161 |

The 4 valid rejects are strict-grounding rejects for an unobserved
`reservation_id`; they are intentionally excluded.

## 4090D Command

On AutoDL, from the project root:

```bash
cd /root/autodl-tmp/yuyu

AUTO_SHUTDOWN=0 \
SAFETY_SHUTDOWN_MINUTES=0 \
bash scripts/run_mixed_dialogue_tool_policy_sft_4090_autodl.sh
```

Useful explicit defaults:

```bash
cd /root/autodl-tmp/yuyu

MAX_SAMPLE_TOKENS=2048 \
STEPS=260 \
LR=1.5e-4 \
LORA_R=16 \
LORA_ALPHA=32 \
MAX_NEW_TOKENS=128 \
EVAL_PROBES=64 \
AUTO_SHUTDOWN=0 \
bash scripts/run_mixed_dialogue_tool_policy_sft_4090_autodl.sh
```

The wrapper installs 4090 dependencies, downloads or reuses Qwen2.5-7B-Instruct,
builds the dataset with tokenizer stats, trains QLoRA, runs mixed-policy
behavior eval, and packages artifacts.

## Outputs

Expected run name:

```text
sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048
```

Important outputs:

```text
reports/mixed_dialogue_tool_policy_dataset_v1_2048.md
reports/sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048.md
reports/behavior_sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048.md
outputs/sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048/checkpoint
outputs/behavior_sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_2048/summary.json
```

The AutoDL wrapper also writes:

```text
/root/autodl-tmp/yuyu_mixed_dialogue_tool_policy_artifacts_<timestamp>.tar.gz
```

## Success Criteria

First gate: mixed-policy behavior eval should improve over base on:

- action type accuracy
- text no-tool rate
- tool-call rate
- protocol wrapper rate
- single-call rate
- tool-name accuracy
- argument exact accuracy

Second gate: if the behavior eval is healthy, run a full tau2 pass^1 comparison.

Do not call Phase2H successful from training loss alone.
