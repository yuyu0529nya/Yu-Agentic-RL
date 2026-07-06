# PRM-Lite Process Reward Report: phase2c_recovery_v5_4090_20260615_1113

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 3 |
| Avg reward | 0.0000 |
| Success count | 0 |
| Avg process score | -13.8333 |
| Avg process score, success | - |
| Avg process score, failure | -13.8333 |

## Risk Tags

| Tag | Count |
| --- | ---: |
| `action_mismatch` | 3 |
| `argument_hallucination` | 2 |
| `ask_identifier_and_call_tool_same_turn` | 2 |
| `ask_then_tool_same_turn` | 2 |
| `communication_target_miss` | 1 |
| `incomplete_evidence` | 3 |
| `premature_deferral` | 2 |
| `slot_grounding_error` | 2 |
| `transfer_after_ungrounded_slot_error` | 2 |
| `ungrounded_user_id` | 2 |

## Per-task Scores

| Task | Reward | DB | Process | Tags | Components |
| --- | ---: | --- | ---: | --- | --- |
| 18 | 0.0000 | False | -14.0 | `action_mismatch`, `argument_hallucination`, `ask_identifier_and_call_tool_same_turn`, `ask_then_tool_same_turn`, `communication_target_miss`, `incomplete_evidence`, `premature_deferral`, `slot_grounding_error`, `transfer_after_ungrounded_slot_error`, `ungrounded_user_id` | `reference_action_mismatch` -5.0<br>`missing_required_communication` -1.5<br>`ungrounded_user_id` -3.0<br>`ask_identifier_and_call_tool_same_turn` -2.0<br>`transfer_after_ungrounded_slot_error` -2.5 |
| 25 | 0.0000 | False | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 44 | 0.0000 | False | -26.5 | `action_mismatch`, `argument_hallucination`, `ask_identifier_and_call_tool_same_turn`, `ask_then_tool_same_turn`, `incomplete_evidence`, `premature_deferral`, `slot_grounding_error`, `transfer_after_ungrounded_slot_error`, `ungrounded_user_id` | `reference_action_mismatch` -19.0<br>`ungrounded_user_id` -3.0<br>`ask_identifier_and_call_tool_same_turn` -2.0<br>`transfer_after_ungrounded_slot_error` -2.5 |
