# PRM-Lite Process Reward Report: airline_prm_rerank_glm51_failed21_n4_max80

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 32 |
| Avg reward | 0.3750 |
| Success count | 12 |
| Avg process score | -2.2031 |
| Avg process score, success | 0.7083 |
| Avg process score, failure | -3.9500 |

## Risk Tags

| Tag | Count |
| --- | ---: |
| `action_mismatch` | 17 |
| `calculation_error` | 3 |
| `communication_db_gap` | 15 |
| `compensation_policy_error` | 3 |
| `incomplete_evidence` | 17 |
| `object_selection_error` | 1 |
| `payment_planning_error` | 4 |
| `premature_write` | 9 |
| `temporal_policy_error` | 3 |
| `user_pressure_susceptibility` | 2 |

## Per-task Scores

| Task | Reward | DB | Process | Tags | Components |
| --- | ---: | --- | ---: | --- | --- |
| 1 | 0.0000 | False | -4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0<br>`user_pressure_susceptibility` -1.0 |
| 1 | 0.0000 | False | -4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0<br>`user_pressure_susceptibility` -1.0 |
| 2 | 0.0000 | False | -11.0 | `action_mismatch`, `communication_db_gap`, `compensation_policy_error`, `incomplete_evidence`, `object_selection_error`, `premature_write` | `reference_action_mismatch` -1.0<br>`recent_reservation_candidates_missing` -2.0<br>`unexpected_write_tool` -1.5<br>`certificate_forbidden_by_task` -3.0<br>`delayed_certificate_without_change_or_cancel` -1.5<br>`compensation_requires_user_goal` -2.0 |
| 2 | 1.0000 | True | 2.5 | - | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5 |
| 7 | 0.0000 | False | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -2.0<br>`cancel_precondition_supported` +1.0 |
| 7 | 0.0000 | False | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -2.0<br>`cancel_precondition_supported` +1.0 |
| 12 | 1.0000 | True | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 12 | 1.0000 | True | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 14 | 0.0000 | False | -3.0 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error` | `reference_action_mismatch` -2.0<br>`calculation_trace_inconsistent` -1.0 |
| 14 | 0.0000 | False | -15.0 | `action_mismatch`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -2.0<br>`basic_economy_update_attempt` -3.0<br>`basic_economy_update_attempt` -3.0<br>`write_tool_payment_error` -2.0<br>`expected_write_plan_missing` -2.0<br>`unexpected_write_tool` -1.5<br>`unexpected_write_tool` -1.5 |
| 15 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 15 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 16 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 16 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 18 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 18 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 20 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 20 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 21 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 21 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 23 | 0.0000 | False | -5.0 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -3.0<br>`write_tool_payment_error` -2.0<br>`calculation_trace_inconsistent` -1.0<br>`cancel_precondition_supported` +1.0 |
| 23 | 0.0000 | False | -2.0 | `action_mismatch`, `calculation_error`, `communication_db_gap`, `incomplete_evidence`, `payment_planning_error` | `reference_action_mismatch` -3.0<br>`split_booking_policy_supported` +1.0<br>`calculation_trace_inconsistent` -1.0<br>`cancel_precondition_supported` +1.0 |
| 25 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 27 | 0.0000 | False | -2.0 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 29 | 0.0000 | False | 0.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0<br>`cancel_precondition_supported` +1.0 |
| 32 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 33 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 34 | 1.0000 | True | 0.0 | - | - |
| 37 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 38 | 0.0000 | False | -2.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5<br>`compensation_requires_user_goal` -2.0 |
| 42 | 0.0000 | False | -4.0 | `communication_db_gap`, `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_precondition_supported` +1.0 |
| 44 | 0.0000 | False | -10.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_write` | `reference_action_mismatch` -10.0<br>`unexpected_write_tool` -1.5<br>`cancel_precondition_supported` +1.0 |
