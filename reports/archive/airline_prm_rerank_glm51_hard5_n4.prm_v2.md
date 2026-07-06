# PRM-Lite Process Reward Report: airline_prm_rerank_glm51_hard5_n4

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 20 |
| Avg reward | 0.3500 |
| Success count | 7 |
| Avg process score | -4.9500 |
| Avg process score, success | 1.7857 |
| Avg process score, failure | -8.5769 |

## Risk Tags

| Tag | Count |
| --- | ---: |
| `action_mismatch` | 8 |
| `calculation_error` | 4 |
| `communication_db_gap` | 12 |
| `compensation_policy_error` | 3 |
| `incomplete_evidence` | 8 |
| `object_selection_error` | 1 |
| `payment_planning_error` | 4 |
| `premature_deferral` | 2 |
| `premature_write` | 8 |
| `temporal_policy_error` | 3 |
| `tool_affordance_error` | 3 |
| `user_pressure_susceptibility` | 3 |

## Per-task Scores

| Task | Reward | DB | Process | Tags | Components |
| --- | ---: | --- | ---: | --- | --- |
| 1 | 0.0000 | False | -4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0<br>`user_pressure_susceptibility` -1.0 |
| 1 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 1 | 0.0000 | False | -4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0<br>`user_pressure_susceptibility` -1.0 |
| 1 | 0.0000 | False | -4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0<br>`user_pressure_susceptibility` -1.0 |
| 2 | 1.0000 | True | 2.5 | - | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5 |
| 2 | 0.0000 | False | -3.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5<br>`unexpected_write_tool` -1.5<br>`certificate_forbidden_by_task` -3.0<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 2 | 0.0000 | False | -9.0 | `action_mismatch`, `communication_db_gap`, `compensation_policy_error`, `incomplete_evidence`, `object_selection_error`, `premature_write` | `reference_action_mismatch` -1.0<br>`recent_reservation_candidates_missing` -2.0<br>`unexpected_write_tool` -1.5<br>`certificate_forbidden_by_task` -3.0<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 2 | 0.0000 | False | -5.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5<br>`unexpected_write_tool` -1.5<br>`certificate_forbidden_by_task` -3.0<br>`delayed_certificate_without_change_or_cancel` -1.5<br>`compensation_requires_user_goal` -2.0 |
| 23 | 0.0000 | False | -5.5 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error` | `reference_action_mismatch` -3.0<br>`split_booking_policy_supported` +1.0<br>`optimal_mastercard_charge_mismatch` -3.5<br>`calculation_trace_inconsistent` -1.0<br>`cancel_precondition_supported` +1.0 |
| 23 | 0.0000 | False | -11.5 | `action_mismatch`, `calculation_error`, `communication_db_gap`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -4.0<br>`basic_economy_update_attempt` -3.0<br>`expected_write_plan_missing` -2.0<br>`calculation_trace_inconsistent` -1.0<br>`unexpected_write_tool` -1.5 |
| 23 | 0.0000 | False | -9.0 | `action_mismatch`, `calculation_error`, `communication_db_gap`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -4.0<br>`write_tool_payment_error` -2.0<br>`book_before_cancel_rebook_plan` -2.0<br>`calculation_trace_inconsistent` -1.0 |
| 23 | 1.0000 | True | 5.0 | `calculation_error`, `payment_planning_error` | `reference_actions_matched` +1.0<br>`split_booking_policy_supported` +1.0<br>`optimal_mastercard_charge_matched` +3.0<br>`calculation_trace_inconsistent` -1.0<br>`cancel_precondition_supported` +1.0 |
| 37 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 37 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 37 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 37 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 44 | 0.0000 | False | -8.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -8.0 |
| 44 | 0.0000 | False | -15.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `tool_affordance_error` | `reference_action_mismatch` -13.0<br>`wrong_tool_for_schedule_lookup` -2.0 |
| 44 | 0.0000 | False | -16.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` | `reference_action_mismatch` -13.0<br>`wrong_tool_for_schedule_lookup` -2.0<br>`premature_deferral` -1.5 |
| 44 | 0.0000 | False | -14.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` | `reference_action_mismatch` -13.0<br>`premature_deferral` -1.5 |
