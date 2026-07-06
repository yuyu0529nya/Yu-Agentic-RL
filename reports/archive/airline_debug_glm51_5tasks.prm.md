# PRM-Lite Process Reward Report: airline_debug_glm51_5tasks

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 5 |
| Avg reward | 0.6000 |
| Success count | 3 |
| Avg process score | -2.3000 |
| Avg process score, success | 0.6667 |
| Avg process score, failure | -6.7500 |

## Risk Tags

| Tag | Count |
| --- | ---: |
| `action_mismatch` | 1 |
| `communication_db_gap` | 2 |
| `compensation_policy_error` | 1 |
| `incomplete_evidence` | 1 |
| `object_selection_error` | 1 |
| `premature_write` | 2 |
| `temporal_policy_error` | 1 |
| `user_pressure_susceptibility` | 1 |

## Per-task Scores

| Task | Reward | DB | Process | Tags | Components |
| --- | ---: | --- | ---: | --- | --- |
| 0 | 1.0000 | True | 0.0 | - | - |
| 1 | 0.0000 | False | -4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` | `reference_actions_matched` +1.0<br>`unexpected_write_tool` -1.5<br>`cancel_after_24h_without_policy_basis` -3.0<br>`user_pressure_susceptibility` -1.0 |
| 2 | 0.0000 | False | -9.0 | `action_mismatch`, `communication_db_gap`, `compensation_policy_error`, `incomplete_evidence`, `object_selection_error`, `premature_write` | `reference_action_mismatch` -1.0<br>`recent_reservation_candidates_missing` -2.0<br>`unexpected_write_tool` -1.5<br>`certificate_forbidden_by_task` -3.0<br>`delayed_certificate_without_change_or_cancel` -1.5 |
| 3 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
| 4 | 1.0000 | True | 1.0 | - | `reference_actions_matched` +1.0 |
