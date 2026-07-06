# PRM-rerank Report: airline_prm_rerank_glm51_hard5_n4

## Summary

| Metric | Value |
| --- | ---: |
| Score mode | `full` |
| Tasks | 5 |
| Simulations | 20 |
| Samples per task | 4 |
| First-trial pass | 0.4000 |
| Raw sample success rate | 0.3500 |
| Oracle pass@N | 0.8000 |
| PRM-rerank pass@N | 0.8000 |
| PRM gain vs first trial | +0.4000 |
| PRM gap to oracle | 0.0000 |
| Selection accuracy on solvable tasks | 1.0000 |

## Per-task Rerank

| Task | Samples | Successes | First | Oracle | PRM Selected | Best Success | Scores | Tags |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 1 | t0: 0.0000 / -4.5 | True | t1: 1.0000 / 1.0 | t1: 1.0 | r=0.0000, p=-4.5<br>r=1.0000, p=1.0<br>r=0.0000, p=-4.5<br>r=0.0000, p=-4.5 | - |
| 2 | 4 | 1 | t0: 1.0000 / 2.5 | True | t0: 1.0000 / 2.5 | t0: 2.5 | r=1.0000, p=2.5<br>r=0.0000, p=-3.5<br>r=0.0000, p=-9.0<br>r=0.0000, p=-5.5 | - |
| 23 | 4 | 1 | t0: 0.0000 / -5.5 | True | t3: 1.0000 / 5.0 | t3: 5.0 | r=0.0000, p=-5.5<br>r=0.0000, p=-11.5<br>r=0.0000, p=-9.0<br>r=1.0000, p=5.0 | `calculation_error`, `payment_planning_error` |
| 37 | 4 | 4 | t0: 1.0000 / 1.0 | True | t0: 1.0000 / 1.0 | t0: 1.0 | r=1.0000, p=1.0<br>r=1.0000, p=1.0<br>r=1.0000, p=1.0<br>r=1.0000, p=1.0 | - |
| 44 | 4 | 0 | t0: 0.0000 / -8.0 | False | t0: 0.0000 / -8.0 | - | r=0.0000, p=-8.0<br>r=0.0000, p=-15.0<br>r=0.0000, p=-16.5<br>r=0.0000, p=-14.5 | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
