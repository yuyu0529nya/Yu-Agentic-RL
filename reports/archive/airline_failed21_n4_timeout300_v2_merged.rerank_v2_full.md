# PRM-rerank Report: airline_failed21_n4_timeout300_v2_merged

## Summary

| Metric | Value |
| --- | ---: |
| Score mode | `full` |
| Tasks | 21 |
| Simulations | 84 |
| Samples per task | 4 |
| First-trial pass | 0.3333 |
| Raw sample success rate | 0.3810 |
| Oracle pass@N | 0.7619 |
| PRM-rerank pass@N | 0.7619 |
| PRM gain vs first trial | +0.4286 |
| PRM gap to oracle | 0.0000 |
| Selection accuracy on solvable tasks | 1.0000 |

## Per-task Rerank

| Task | Samples | Successes | First | Oracle | PRM Selected | Best Success | Scores | Tags |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 3 | t0: 0.0000 / -4.5 | True | t1: 1.0000 / 1.0 | t1: 1.0 | r=0.0000, p=-4.5<br>r=1.0000, p=1.0<br>r=1.0000, p=1.0<br>r=1.0000, p=1.0 | - |
| 2 | 4 | 2 | t0: 0.0000 / -9.0 | True | t2: 1.0000 / 2.5 | t2: 2.5 | r=0.0000, p=-9.0<br>r=0.0000, p=-4.5<br>r=1.0000, p=2.5<br>r=1.0000, p=2.5 | - |
| 7 | 4 | 0 | t0: 0.0000 / -1.0 | False | t0: 0.0000 / -1.0 | - | r=0.0000, p=-1.0<br>r=0.0000, p=-1.0<br>r=0.0000, p=-3.0<br>r=0.0000, p=-1.0 | `premature_write`, `temporal_policy_error` |
| 12 | 4 | 3 | t0: 0.0000 / -2.5 | True | t3: 1.0000 / -1.0 | t3: -1.0 | r=0.0000, p=-2.5<br>r=1.0000, p=-4.5<br>r=1.0000, p=-4.5<br>r=1.0000, p=-1.0 | `action_mismatch`, `incomplete_evidence` |
| 14 | 4 | 0 | t0: 0.0000 / -1.0 | False | t0: 0.0000 / -1.0 | - | r=0.0000, p=-1.0<br>r=0.0000, p=-9.5<br>r=0.0000, p=-9.5<br>r=0.0000, p=-3.0 | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error` |
| 15 | 4 | 1 | t0: 0.0000 / -1.0 | True | t2: 1.0000 / 1.0 | t2: 1.0 | r=0.0000, p=-1.0<br>r=0.0000, p=-1.0<br>r=1.0000, p=1.0<br>r=0.0000, p=-1.0 | - |
| 16 | 4 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0<br>r=0.0000, p=-1.0<br>r=0.0000, p=-1.0<br>r=0.0000, p=-1.0 | - |
| 18 | 4 | 1 | t0: 0.0000 / -3.0 | True | t2: 1.0000 / 1.0 | t2: 1.0 | r=0.0000, p=-3.0<br>r=0.0000, p=-3.0<br>r=1.0000, p=1.0<br>r=0.0000, p=-3.0 | - |
| 20 | 4 | 1 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0<br>r=0.0000, p=-1.0<br>r=0.0000, p=-1.0<br>r=0.0000, p=-1.0 | - |
| 21 | 4 | 0 | t0: 0.0000 / -2.0 | False | t0: 0.0000 / -2.0 | - | r=0.0000, p=-2.0<br>r=0.0000, p=-2.0<br>r=0.0000, p=-2.0<br>r=0.0000, p=-2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 23 | 4 | 1 | t0: 0.0000 / -5.0 | True | t1: 1.0000 / 5.0 | t1: 5.0 | r=0.0000, p=-5.0<br>r=1.0000, p=5.0<br>r=0.0000, p=-3.0<br>r=0.0000, p=-11.5 | `calculation_error`, `payment_planning_error` |
| 25 | 4 | 4 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0<br>r=1.0000, p=1.0<br>r=1.0000, p=1.0<br>r=1.0000, p=1.0 | - |
| 27 | 4 | 1 | t0: 0.0000 / -2.0 | True | t2: 1.0000 / 1.0 | t2: 1.0 | r=0.0000, p=-2.0<br>r=0.0000, p=-2.0<br>r=1.0000, p=1.0<br>r=0.0000, p=-2.0 | - |
| 29 | 4 | 0 | t0: 0.0000 / -5.0 | False | t2: 0.0000 / 0.0 | - | r=0.0000, p=-5.0<br>r=0.0000, p=-5.0<br>r=0.0000, p=0.0<br>r=0.0000, p=-2.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 32 | 4 | 3 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0<br>r=1.0000, p=1.0<br>r=1.0000, p=1.0<br>r=0.0000, p=-2.0 | - |
| 33 | 4 | 2 | t0: 0.0000 / -3.0 | True | t1: 1.0000 / -1.0 | t1: -1.0 | r=0.0000, p=-3.0<br>r=1.0000, p=-1.0<br>r=1.0000, p=-1.0<br>r=0.0000, p=-3.0 | `action_mismatch`, `incomplete_evidence` |
| 34 | 4 | 3 | t0: 0.0000 / 0.0 | True | t1: 1.0000 / 0.0 | t1: 0.0 | r=0.0000, p=0.0<br>r=1.0000, p=0.0<br>r=1.0000, p=0.0<br>r=1.0000, p=0.0 | - |
| 37 | 4 | 3 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0<br>r=0.0000, p=-3.5<br>r=1.0000, p=1.0<br>r=1.0000, p=1.0 | - |
| 38 | 4 | 1 | t0: 1.0000 / 2.5 | True | t0: 1.0000 / 2.5 | t0: 2.5 | r=1.0000, p=2.5<br>r=0.0000, p=-0.5<br>r=0.0000, p=-2.5<br>r=0.0000, p=-0.5 | - |
| 42 | 4 | 2 | t0: 1.0000 / -1.0 | True | t0: 1.0000 / -1.0 | t0: -1.0 | r=1.0000, p=-1.0<br>r=0.0000, p=-3.0<br>r=1.0000, p=-1.0<br>r=0.0000, p=-7.5 | `premature_write`, `temporal_policy_error` |
| 44 | 4 | 0 | t0: 0.0000 / -16.5 | False | t3: 0.0000 / -9.0 | - | r=0.0000, p=-16.5<br>r=0.0000, p=-16.5<br>r=0.0000, p=-14.5<br>r=0.0000, p=-9.0 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence`, `premature_write` |
