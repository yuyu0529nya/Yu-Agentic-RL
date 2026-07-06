# Phase2I Decision-Balanced Preflight

## Status

- Local code: `READY`
- Local validation: `PASS`
- Remote 4090D training: `BLOCKED_ON_CURRENT_SSH_ENDPOINT`

## Goal

Phase2I keeps Phase2H's strong executable single-tool behavior while reducing tool over-calling on assistant text turns. The change is data-level: repeat assistant text targets in the train split and downsample protocol-only tool variants. Valid and heldout remain unchanged so behavior eval is still comparable.

## Local Files

- `scripts/build_decision_balanced_mixed_policy_dataset.py`
- `scripts/run_decision_balanced_mixed_policy_sft_4090.sh`
- `scripts/run_decision_balanced_mixed_policy_sft_4090_autodl.sh`
- `tests/test_decision_balanced_mixed_policy_dataset.py`
- `reports/mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048.md`
- `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_manifest_v1_2048.json`
- `tmp/phase2i_decision_balanced_code_update_20260616.tar.gz`

## Local Data Mix

| Split | Stage | Text | Tool | Rows |
| --- | --- | ---: | ---: | ---: |
| train | Phase2H-style raw clean | 80 | 139 | 219 |
| train | Phase2I balanced | 160 | 113 | 273 |
| valid | unchanged | 34 | 60 | 94 |
| heldout | unchanged | 108 | 161 | 269 |

Train balance details:

- `text_repeat=2`
- `protocol_keep_ratio=0.5`
- `balance_seed=11`
- Protocol-only rows downsampled: `26`
- Balance roles: `assistant_text_original=80`, `assistant_text_repeat=80`, `protocol_tool_kept=27`, `tool_original=86`

## Validation

Commands completed locally:

```text
python3 -m py_compile scripts/build_decision_balanced_mixed_policy_dataset.py scripts/build_mixed_dialogue_tool_policy_dataset.py scripts/evaluate_mixed_policy_behavior.py scripts/train_sft_smoke.py
bash -n scripts/run_decision_balanced_mixed_policy_sft_4090.sh scripts/run_decision_balanced_mixed_policy_sft_4090_autodl.sh
python3 scripts/build_decision_balanced_mixed_policy_dataset.py --skip-tokenizer --max-sample-tokens 2048 --text-repeat 2 --protocol-keep-ratio 0.5 --balance-seed 11
```

The three Phase2I unit-style balance tests were executed directly and passed.

## Remote Blocker

The saved old SSH gateway endpoint no longer accepts connections. Direct `100.81.157.87:22` also timed out. Phase2I remote training has not started.

Needed to continue:

```text
current AutoDL/SeetaCloud SSH command, or current host + port
```
