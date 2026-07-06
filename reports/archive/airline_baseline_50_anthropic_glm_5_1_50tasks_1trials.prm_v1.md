# PRM-Lite Process Reward Report: airline_baseline_50_anthropic_glm_5_1_50tasks_1trials

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 50 |
| Avg reward | 0.5800 |
| Success count | 29 |
| Avg process score | -1.1300 |
| Avg process score, success | 0.7931 |
| Avg process score, failure | -3.7857 |

## Risk Tags

| Tag | Count |
| --- | ---: |
| `action_mismatch` | 17 |
| `calculation_error` | 2 |
| `communication_db_gap` | 18 |
| `compensation_policy_error` | 3 |
| `incomplete_evidence` | 17 |
| `object_selection_error` | 1 |
| `payment_planning_error` | 2 |
| `policy_precedence_error` | 1 |
| `premature_deferral` | 1 |
| `premature_write` | 9 |
| `temporal_policy_error` | 3 |
| `tool_affordance_error` | 1 |
| `user_pressure_susceptibility` | 1 |

## Per-task Scores

| Task | Reward | DB | Process | Tags | Components |
| --- | ---: | --- | ---: | --- | --- |
| 0 | 1.0000 | True | 0.0 | - | - |
| 1 | 0.0000 | False | -4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0<br>`user_pressure_susceptibility` -1.0 |
| 2 | 0.0000 | False | -11.0 | `action_mismatch`, `communication_db_gap`, `compensation_policy_error`, `incomplete_evidence`, `object_selection_error`, `premature_write` | `reference_action_mismatch` -1.0<br>`recent_reservation_candidates_missing` -2.0<br>`unexpected_write_tool` -1.5<br>`certificate_forbidden_by_task` -3.0<br>`delayed_certificate_without_change_or_cancel` -1.5<br>`compensation_requires_user_goal` -2.0 |
| 3 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 4 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 5 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 6 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 7 | 0.0000 | False | -1.0 | `action_mismatch`, `incomplete_evidence` | `reference_action_mismatch` -2.0<br>`cancel_precondition_supported` +1.0 |
| 8 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 9 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 10 | 1.0000 | True | 0.0 | - | - |
| 11 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 12 | 0.0000 | False | -2.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_write` | `reference_action_mismatch` -1.0<br>`unexpected_write_tool` -1.5 |
| 13 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 14 | 0.0000 | False | -6.5 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -2.0<br>`expected_write_plan_missing` -2.0<br>`calculation_trace_inconsistent` -1.0<br>`unexpected_write_tool` -1.5 |
| 15 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 16 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 17 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 18 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 19 | 1.0000 | True | 2.0 | - | `reference_actions_matched` +1.0<br>`cancel_precondition_supported` +1.0 |
| 20 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 21 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 22 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 23 | 0.0000 | False | -8.5 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` | `reference_action_mismatch` -4.0<br>`expected_write_plan_missing` -2.0<br>`calculation_trace_inconsistent` -1.0<br>`unexpected_write_tool` -1.5 |
| 24 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 25 | 0.0000 | False | -1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0 |
| 26 | 1.0000 | True | 0.0 | - | - |
| 27 | 0.0000 | False | -2.0 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 28 | 1.0000 | True | 0.0 | - | - |
| 29 | 0.0000 | False | 0.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -1.0<br>`cancel_precondition_supported` +1.0 |
| 30 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 31 | 1.0000 | True | 0.0 | - | - |
| 32 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 33 | 0.0000 | False | -3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -3.0 |
| 34 | 0.0000 | False | 0.0 | `communication_db_gap` | - |
| 35 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 36 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 37 | 0.0000 | False | -10.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `policy_precedence_error`, `premature_write`, `temporal_policy_error` | `reference_action_mismatch` -2.0<br>`unexpected_write_tool` -1.5<br>`past_flight_cancel_attempt` -3.0<br>`policy_precedence_error` -1.0<br>`cancel_after_24h_without_policy_basis` -3.0 |
| 38 | 0.0000 | False | -0.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` | `reference_actions_matched` +1.0<br>`recent_reservation_candidates_checked` +1.5<br>`unexpected_write_tool` -1.5<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 39 | 1.0000 | True | 0.0 | `premature_write`, `temporal_policy_error` | `reference_actions_matched` +1.0<br>`cancel_precondition_supported` +1.0<br>`cancel_after_24h_without_policy_basis` -3.0<br>`cancel_precondition_supported` +1.0 |
| 40 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 41 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 42 | 0.0000 | False | -2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` | `reference_action_mismatch` -2.0 |
| 43 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 44 | 0.0000 | False | -16.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` | `reference_action_mismatch` -13.0<br>`wrong_tool_for_schedule_lookup` -2.0<br>`premature_deferral` -1.5 |
| 45 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 46 | 1.0000 | True | 0.0 | - | - |
| 47 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 48 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 49 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
