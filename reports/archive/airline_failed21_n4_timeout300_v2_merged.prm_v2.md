# PRM-Lite Process Reward Report: airline_failed21_n4_timeout300_v2_merged

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 84 |
| Avg reward | 0.3810 |
| Success count | 32 |
| Avg process score | -2.1250 |
| Avg process score, success | 0.5156 |
| Avg process score, failure | -3.7500 |

## Risk Tags

| Tag | Count |
| --- | ---: |
| `action_mismatch` | 42 |
| `calculation_error` | 8 |
| `communication_db_gap` | 43 |
| `compensation_policy_error` | 8 |
| `incomplete_evidence` | 43 |
| `object_selection_error` | 1 |
| `payment_planning_error` | 8 |
| `premature_deferral` | 5 |
| `premature_write` | 25 |
| `temporal_policy_error` | 9 |
| `tool_affordance_error` | 5 |
| `user_pressure_susceptibility` | 1 |

## Per-task Scores

| Task | Reward | DB | Process | Tags | Components |
| --- | ---: | --- | ---: | --- | --- |
| 1 | 0.0000 | False | -4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0<br>`user_pressure_susceptibility` -1.0 |
| 1 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 1 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 1 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 2 | 0.0000 | False | -9.0 | `action_mismatch`, `communication_db_gap`, `compensation_policy_error`, `incomplete_evidence`, `object_selection_error`, `premature_write` | `reference_action_mismatch` -1.0<br>`recent_reservation_candidates_missing` -2.0<br>`unexpected_write_tool` -1.5<br>`certificate_forbidden_by_task` -3.0<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 2 | 0.0000 | False | -4.5 | `communication_db_gap`, `compensation_policy_error`, `incomplete_evidence`, `premature_write` | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5<br>`unexpected_write_tool` -1.5<br>`certificate_forbidden_by_task` -3.0<br>`certificate_without_status_confirmation` -1.0<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 2 | 1.0000 | True | 2.5 | - | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5 |
| 2 | 1.0000 | True | 2.5 | - | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5 |
| 7 | 0.0000 | True | -1.0 | `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_precondition_supported` +1.0 |
| 7 | 0.0000 | True | -1.0 | `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_precondition_supported` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0 |
| 7 | 0.0000 | False | -3.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 7 | 0.0000 | True | -1.0 | `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_precondition_supported` +1.0 |
| 12 | 0.0000 | False | -2.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_write` | `reference_action_mismatch` -1.0<br>`unexpected_write_tool` -1.5 |
| 12 | 1.0000 | True | -4.5 | `action_mismatch`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` | `reference_action_mismatch` -3.0<br>`premature_deferral` -1.5 |
| 12 | 1.0000 | True | -4.5 | `action_mismatch`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` | `reference_action_mismatch` -3.0<br>`premature_deferral` -1.5 |
| 12 | 1.0000 | True | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 14 | 0.0000 | False | -1.0 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error` | `reference_action_mismatch` -1.0<br>`calculation_trace_inconsistent` -1.0<br>`cancel_precondition_supported` +1.0 |
| 14 | 0.0000 | False | -9.5 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -2.0<br>`basic_economy_update_attempt` -3.0<br>`expected_write_plan_missing` -2.0<br>`calculation_trace_inconsistent` -1.0<br>`unexpected_write_tool` -1.5 |
| 14 | 0.0000 | False | -9.5 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -2.0<br>`basic_economy_update_attempt` -3.0<br>`expected_write_plan_missing` -2.0<br>`calculation_trace_inconsistent` -1.0<br>`unexpected_write_tool` -1.5 |
| 14 | 0.0000 | False | -3.0 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -1.0<br>`write_tool_payment_error` -2.0<br>`calculation_trace_inconsistent` -1.0<br>`cancel_precondition_supported` +1.0 |
| 15 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 15 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 15 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 15 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 16 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 16 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 16 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 16 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 18 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 18 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 18 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 18 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 20 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 20 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 20 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 20 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 21 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 21 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 21 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 21 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 23 | 0.0000 | False | -5.0 | `action_mismatch`, `calculation_error`, `communication_db_gap`, `incomplete_evidence`, `payment_planning_error` | `reference_action_mismatch` -4.0<br>`calculation_trace_inconsistent` -1.0 |
| 23 | 1.0000 | True | 5.0 | `calculation_error`, `payment_planning_error` | `reference_actions_matched` +1.0<br>`split_booking_policy_supported` +1.0<br>`optimal_mastercard_charge_matched` +3.0<br>`calculation_trace_inconsistent` -1.0<br>`cancel_precondition_supported` +1.0 |
| 23 | 0.0000 | False | -3.0 | `action_mismatch`, `calculation_error`, `communication_db_gap`, `incomplete_evidence`, `payment_planning_error` | `reference_action_mismatch` -3.0<br>`calculation_trace_inconsistent` -1.0<br>`cancel_precondition_supported` +1.0 |
| 23 | 0.0000 | False | -11.5 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -4.0<br>`basic_economy_update_attempt` -3.0<br>`expected_write_plan_missing` -2.0<br>`calculation_trace_inconsistent` -1.0<br>`unexpected_write_tool` -1.5 |
| 25 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 25 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 25 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 25 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 27 | 0.0000 | False | -2.0 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 27 | 0.0000 | False | -2.0 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 27 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 27 | 0.0000 | False | -2.0 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 29 | 0.0000 | False | -5.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_write` | `reference_action_mismatch` -2.0<br>`unexpected_write_tool` -1.5<br>`unexpected_write_tool` -1.5 |
| 29 | 0.0000 | False | -5.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_write` | `reference_action_mismatch` -2.0<br>`unexpected_write_tool` -1.5<br>`unexpected_write_tool` -1.5 |
| 29 | 0.0000 | False | 0.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0<br>`cancel_precondition_supported` +1.0 |
| 29 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 32 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 32 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 32 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 32 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 33 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 33 | 1.0000 | True | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 33 | 1.0000 | True | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 33 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 34 | 0.0000 | False | 0.0 | `communication_db_gap` | - |
| 34 | 1.0000 | True | 0.0 | - | - |
| 34 | 1.0000 | True | 0.0 | - | - |
| 34 | 1.0000 | True | 0.0 | - | - |
| 37 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 37 | 0.0000 | False | -3.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0 |
| 37 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 37 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 38 | 1.0000 | True | 2.5 | - | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5 |
| 38 | 0.0000 | False | -0.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 38 | 0.0000 | False | -2.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5<br>`compensation_requires_user_goal` -2.0 |
| 38 | 0.0000 | False | -0.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 42 | 1.0000 | True | -1.0 | `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_precondition_supported` +1.0 |
| 42 | 0.0000 | False | -3.0 | `communication_db_gap`, `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_precondition_supported` +1.0<br>`cancel_precondition_supported` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_after_24h_without_policy_basis` -3.0 |
| 42 | 1.0000 | True | -1.0 | `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_precondition_supported` +1.0 |
| 42 | 0.0000 | False | -7.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_precondition_supported` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_precondition_supported` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`unexpected_write_tool` -1.5 |
| 44 | 0.0000 | False | -16.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` | `reference_action_mismatch` -13.0<br>`wrong_tool_for_schedule_lookup` -2.0<br>`premature_deferral` -1.5 |
| 44 | 0.0000 | False | -16.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` | `reference_action_mismatch` -13.0<br>`wrong_tool_for_schedule_lookup` -2.0<br>`premature_deferral` -1.5 |
| 44 | 0.0000 | False | -14.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` | `reference_action_mismatch` -13.0<br>`premature_deferral` -1.5 |
| 44 | 0.0000 | False | -9.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_write` | `reference_action_mismatch` -8.0<br>`unexpected_write_tool` -1.5<br>`cancel_precondition_supported` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_precondition_supported` +1.0 |
