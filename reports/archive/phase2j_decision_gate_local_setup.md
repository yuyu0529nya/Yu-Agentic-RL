# Phase2J Decision Gate Local Setup

## Goal

Separate long-horizon agent control into two decisions:

1. gate: decide whether the next assistant action is `assistant_text` or `tool_call`;
2. tool policy: if the gate chooses `tool_call`, let the stronger Phase2H-style tool generator produce the exact tool name and arguments.

This addresses the Phase2H/2I tradeoff: Phase2H generated tools well but over-called tools, while Phase2I talked more safely but lost tool accuracy.

## Local Artifacts

- Dataset builder: `scripts/build_decision_gate_dataset.py`
- Gate behavior evaluator: `scripts/evaluate_decision_gate_behavior.py`
- 4090 training script: `scripts/run_decision_gate_sft_4090.sh`
- AutoDL wrapper: `scripts/run_decision_gate_sft_4090_autodl.sh`
- Dataset report: `reports/decision_gate_dataset_v1_2048.md`
- Manifest: `data/decision_gate/tau2_airline_decision_gate_manifest_v1_2048.json`

## Dataset

| Split | Rows | assistant_text | tool_call |
| --- | ---: | ---: | ---: |
| train | 152 | 66 | 86 |
| valid | 65 | 30 | 35 |
| heldout | 195 | 94 | 101 |

Initial greeting turns are skipped by default. Duplicate decision points are removed, and when a protocol-only and non-protocol row describe the same decision, the non-protocol row is kept.

## Validation

Passed locally:

```bash
python scripts/build_decision_gate_dataset.py
python -m pytest tests/test_decision_gate_dataset.py tests/test_mixed_dialogue_tool_policy_dataset.py tests/test_decision_balanced_mixed_policy_dataset.py
python scripts/evaluate_decision_gate_behavior.py --help
bash -n scripts/run_decision_gate_sft_4090.sh scripts/run_decision_gate_sft_4090_autodl.sh
```

Result: 10 tests passed, dataset structural validation errors = 0.

## GPU Command

On AutoDL, from `/root/autodl-tmp/yuyu`:

```bash
AUTO_SHUTDOWN=0 bash scripts/run_decision_gate_sft_4090_autodl.sh
```

The wrapper packages artifacts and leaves the instance running by default. It does not schedule an early safety shutdown.

## Decision Criteria

Proceed to gated policy integration only if the decision gate improves over Phase2I on heldout action selection:

- high `assistant_text` recall: avoids over-calling tools;
- high `tool_call` recall: does not block needed tool calls;
- overall accuracy clearly above the one-stage Phase2I gate behavior.

If the gate works, combine it with Phase2H tool generation and rerun the gated upper-bound analysis with real gate predictions.
