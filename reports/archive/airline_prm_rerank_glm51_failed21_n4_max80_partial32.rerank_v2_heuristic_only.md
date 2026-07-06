# PRM-rerank Report: airline_prm_rerank_glm51_failed21_n4_max80

## Summary

| Metric | Value |
| --- | ---: |
| Score mode | `heuristic_only` |
| Tasks | 21 |
| Simulations | 32 |
| Samples per task | 1-2 |
| First-trial pass | 0.3333 |
| Raw sample success rate | 0.3750 |
| Oracle pass@N | 0.3810 |
| PRM-rerank pass@N | 0.3810 |
| PRM gain vs first trial | +0.0476 |
| PRM gap to oracle | 0.0000 |
| Selection accuracy on solvable tasks | 1.0000 |

## Per-task Rerank

| Task | Samples | Successes | First | Oracle | PRM Selected | Best Success | Scores | Tags |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | t0: 0.0000 / -4.0 | False | t0: 0.0000 / -4.0 | - | r=0.0000, p=-4.0<br>r=0.0000, p=-4.0 | `communication_db_gap`, `premature_write`, `temporal_policy_error`, `user_pressure_susceptibility` |
| 2 | 2 | 1 | t0: 0.0000 / -8.5 | True | t1: 1.0000 / 1.5 | t1: 1.5 | r=0.0000, p=-8.5<br>r=1.0000, p=1.5 | - |
| 7 | 2 | 0 | t0: 0.0000 / 1.0 | False | t0: 0.0000 / 1.0 | - | r=0.0000, p=1.0<br>r=0.0000, p=1.0 | `unclassified_db_failure` |
| 12 | 2 | 2 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0<br>r=1.0000, p=0.0 | - |
| 14 | 2 | 0 | t0: 0.0000 / -1.0 | False | t0: 0.0000 / -1.0 | - | r=0.0000, p=-1.0<br>r=0.0000, p=-8.0 | `calculation_error`, `payment_planning_error` |
| 15 | 2 | 2 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0<br>r=1.0000, p=0.0 | - |
| 16 | 2 | 2 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0<br>r=1.0000, p=0.0 | - |
| 18 | 2 | 0 | t0: 0.0000 / 0.0 | False | t0: 0.0000 / 0.0 | - | r=0.0000, p=0.0<br>r=0.0000, p=0.0 | `communication_db_gap` |
| 20 | 2 | 2 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0<br>r=1.0000, p=0.0 | - |
| 21 | 2 | 0 | t0: 0.0000 / 0.0 | False | t0: 0.0000 / 0.0 | - | r=0.0000, p=0.0<br>r=0.0000, p=0.0 | `communication_db_gap` |
| 23 | 2 | 0 | t0: 0.0000 / -2.0 | False | t1: 0.0000 / 1.0 | - | r=0.0000, p=-2.0<br>r=0.0000, p=1.0 | `calculation_error`, `communication_db_gap`, `payment_planning_error` |
| 25 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 27 | 1 | 0 | t0: 0.0000 / -1.5 | False | t0: 0.0000 / -1.5 | - | r=0.0000, p=-1.5 | `communication_db_gap`, `compensation_policy_error`, `premature_write` |
| 29 | 1 | 0 | t0: 0.0000 / 1.0 | False | t0: 0.0000 / 1.0 | - | r=0.0000, p=1.0 | `communication_db_gap` |
| 32 | 1 | 0 | t0: 0.0000 / 0.0 | False | t0: 0.0000 / 0.0 | - | r=0.0000, p=0.0 | `communication_db_gap` |
| 33 | 1 | 0 | t0: 0.0000 / 0.0 | False | t0: 0.0000 / 0.0 | - | r=0.0000, p=0.0 | `communication_db_gap` |
| 34 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 37 | 1 | 1 | t0: 1.0000 / 0.0 | True | t0: 1.0000 / 0.0 | t0: 0.0 | r=1.0000, p=0.0 | - |
| 38 | 1 | 0 | t0: 0.0000 / -2.0 | False | t0: 0.0000 / -2.0 | - | r=0.0000, p=-2.0 | `communication_db_gap`, `compensation_policy_error`, `premature_write` |
| 42 | 1 | 0 | t0: 0.0000 / -5.0 | False | t0: 0.0000 / -5.0 | - | r=0.0000, p=-5.0 | `communication_db_gap`, `premature_write`, `temporal_policy_error` |
| 44 | 1 | 0 | t0: 0.0000 / 1.0 | False | t0: 0.0000 / 1.0 | - | r=0.0000, p=1.0 | `communication_db_gap` |
