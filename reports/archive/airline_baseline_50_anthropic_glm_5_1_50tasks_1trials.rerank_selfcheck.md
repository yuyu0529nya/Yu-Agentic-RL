# PRM-rerank Report: airline_baseline_50_anthropic_glm_5_1_50tasks_1trials

## Summary

| Metric | Value |
| --- | ---: |
| Tasks | 50 |
| Simulations | 50 |
| Samples per task | 1 |
| First-trial pass | 0.5800 |
| Raw sample success rate | 0.5800 |
| Oracle pass@N | 0.5800 |
| PRM-rerank pass@N | 0.5800 |
| PRM gain vs first trial | +0.0000 |
| PRM gap to oracle | 0.0000 |
| Selection accuracy on solvable tasks | 1.0000 |

## Per-task Rerank

| Task | Samples | Successes | First | Oracle | PRM Selected | Best Success | Scores | Tags |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 1 | 1 | 0 | t0: 0.0000 / -4.5 | False | t0: 0.0000 / -4.5 | - | r=0.0000, p=-4.5 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` |
| 2 | 1 | 0 | t0: 0.0000 / -11.0 | False | t0: 0.0000 / -11.0 | - | r=0.0000, p=-11.0 | `action_mismatch`, `communication_db_gap`, `compensation_policy_error`, `incomplete_evidence`, `object_selection_error`, `premature_write` |
| 3 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 4 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 5 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 6 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 7 | 1 | 0 | t0: 0.0000 / -1.0 | False | t0: 0.0000 / -1.0 | - | r=0.0000, p=-1.0 | `action_mismatch`, `incomplete_evidence` |
| 8 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 9 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 10 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 11 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 12 | 1 | 0 | t0: 0.0000 / -2.5 | False | t0: 0.0000 / -2.5 | - | r=0.0000, p=-2.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_write` |
| 13 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 14 | 1 | 0 | t0: 0.0000 / -6.5 | False | t0: 0.0000 / -6.5 | - | r=0.0000, p=-6.5 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` |
| 15 | 1 | 0 | t0: 0.0000 / -1.0 | False | t0: 0.0000 / -1.0 | - | r=0.0000, p=-1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 16 | 1 | 0 | t0: 0.0000 / -1.0 | False | t0: 0.0000 / -1.0 | - | r=0.0000, p=-1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 17 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 18 | 1 | 0 | t0: 0.0000 / -3.0 | False | t0: 0.0000 / -3.0 | - | r=0.0000, p=-3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 19 | 1 | 1 | t0: 1.0000 / 2.0 | True | t0: 1.0000 / 2.0 | t0: 2.0 | r=1.0000, p=2.0 | - |
| 20 | 1 | 0 | t0: 0.0000 / -1.0 | False | t0: 0.0000 / -1.0 | - | r=0.0000, p=-1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 21 | 1 | 0 | t0: 0.0000 / -2.0 | False | t0: 0.0000 / -2.0 | - | r=0.0000, p=-2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 22 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 23 | 1 | 0 | t0: 0.0000 / -8.5 | False | t0: 0.0000 / -8.5 | - | r=0.0000, p=-8.5 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` |
| 24 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 25 | 1 | 0 | t0: 0.0000 / -1.0 | False | t0: 0.0000 / -1.0 | - | r=0.0000, p=-1.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 26 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 27 | 1 | 0 | t0: 0.0000 / -2.0 | False | t0: 0.0000 / -2.0 | - | r=0.0000, p=-2.0 | `communication_db_gap`, `compensation_policy_error`, `premature_write` |
| 28 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 29 | 1 | 0 | t0: 0.0000 / 0.0 | False | t0: 0.0000 / 0.0 | - | r=0.0000, p=0.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 30 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 31 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 32 | 1 | 0 | t0: 0.0000 / -2.0 | False | t0: 0.0000 / -2.0 | - | r=0.0000, p=-2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 33 | 1 | 0 | t0: 0.0000 / -3.0 | False | t0: 0.0000 / -3.0 | - | r=0.0000, p=-3.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 34 | 1 | 0 | t0: 0.0000 / 0.0 | False | t0: 0.0000 / 0.0 | - | r=0.0000, p=0.0 | `communication_db_gap` |
| 35 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 36 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 37 | 1 | 0 | t0: 0.0000 / -10.5 | False | t0: 0.0000 / -10.5 | - | r=0.0000, p=-10.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `policy_precedence_error`, `premature_write`, `temporal_policy_error` |
| 38 | 1 | 0 | t0: 0.0000 / -0.5 | False | t0: 0.0000 / -0.5 | - | r=0.0000, p=-0.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` |
| 39 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | `premature_write`, `temporal_policy_error` |
| 40 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 41 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 42 | 1 | 0 | t0: 0.0000 / -2.0 | False | t0: 0.0000 / -2.0 | - | r=0.0000, p=-2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 43 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 44 | 1 | 0 | t0: 0.0000 / -16.5 | False | t0: 0.0000 / -16.5 | - | r=0.0000, p=-16.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_deferral`, `tool_affordance_error` |
| 45 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 46 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 47 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 48 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
| 49 | 1 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0 | - |
