# PRM-Lite Process Reward Report: fixtures

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 2 |
| Avg reward | 0.5000 |
| Success count | 1 |
| Avg process score | -4.7500 |
| Avg process score, success | -0.5000 |
| Avg process score, failure | -9.0000 |

## Risk Tags

| Tag | Count |
| --- | ---: |
| `argument_hallucination` | 1 |
| `ask_identifier_and_call_tool_same_turn` | 1 |
| `ask_then_tool_same_turn` | 1 |
| `communication_target_miss` | 2 |
| `incomplete_evidence` | 2 |
| `premature_deferral` | 1 |
| `slot_grounding_error` | 1 |
| `transfer_after_ungrounded_slot_error` | 1 |
| `ungrounded_user_id` | 1 |

## Per-task Scores

| Task | Reward | DB | Process | Tags | Components |
| --- | ---: | --- | ---: | --- | --- |
| 18 | 0.0000 | False | -9.0 | `argument_hallucination`, `ask_identifier_and_call_tool_same_turn`, `ask_then_tool_same_turn`, `communication_target_miss`, `incomplete_evidence`, `premature_deferral`, `slot_grounding_error`, `transfer_after_ungrounded_slot_error`, `ungrounded_user_id` | `missing_required_communication` -1.5<br>`ungrounded_user_id` -3.0<br>`ask_identifier_and_call_tool_same_turn` -2.0<br>`transfer_after_ungrounded_slot_error` -2.5 |
| 18 | 1.0000 | True | -0.5 | `communication_target_miss`, `incomplete_evidence` | `missing_required_communication` -1.5<br>`slot_grounding_clean` +1.0 |
