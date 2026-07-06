# Decision Gate Dataset v1

## Goal

Train a decision-only gate that predicts exactly one label for the next assistant action: `assistant_text` or `tool_call`.

This is Phase2J's first artifact. It separates action decision from tool-call generation.

## Outputs

- train: `data/decision_gate/tau2_airline_decision_gate_v1_2048_train.jsonl`
- valid: `data/decision_gate/tau2_airline_decision_gate_v1_2048_valid.jsonl`
- heldout: `data/decision_gate/tau2_airline_decision_gate_v1_2048_heldout.jsonl`

## Configuration

- Input source: mixed-policy v1 rows
- Include initial greeting: `False`
- Duplicate decision points are removed by `(source_id, turn_index, call_index, action, prefix_count)`.
- If a protocol-only and non-protocol row describe the same decision, the non-protocol row is kept.

## Summary

| Split | Rows | Actions | Tasks | Mean prefix msgs | Max prefix msgs | Dropped duplicates | Skipped |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| train | 152 | `assistant_text:66, tool_call:86` | `1:28, 12:30, 20:15, 27:15, 34:21, 38:14, 42:29` | 12.1 | 30 | 53 | `initial_greeting:14` |
| valid | 65 | `assistant_text:30, tool_call:35` | `15:13, 23:18, 33:34` | 18.0 | 40 | 25 | `initial_greeting:4` |
| heldout | 195 | `assistant_text:94, tool_call:101` | `2:30, 16:10, 18:17, 25:41, 32:55, 37:42` | 14.6 | 43 | 60 | `initial_greeting:14` |

## Validation

No structural validation errors.

## Training Use

- Train with normal assistant-content SFT; the target is only the short label.
- At inference, classify generated text into `assistant_text` or `tool_call`.
- Then route `tool_call` cases to a stronger tool generator such as Phase2H.
