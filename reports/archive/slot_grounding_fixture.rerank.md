# PRM-rerank Report: fixtures

## Summary

| Metric | Value |
| --- | ---: |
| Score mode | `heuristic_only` |
| Tie-break mode | `online_safe` |
| Tasks | 1 |
| Simulations | 2 |
| Samples per task | 2 |
| First-trial pass | 0.0000 |
| Raw sample success rate | 0.5000 |
| Oracle pass@N | 1.0000 |
| PRM-rerank pass@N | 1.0000 |
| PRM gain vs first trial | +1.0000 |
| PRM gap to oracle | 0.0000 |
| Selection accuracy on solvable tasks | 1.0000 |

## Per-task Rerank

| Task | Samples | Successes | First | Oracle | PRM Selected | Best Success | Scores | Tags |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 18 | 2 | 1 | t0: 0.0000 / -9.0 | True | t1: 1.0000 / -0.5 | t1: -0.5 | r=0.0000, p=-9.0<br>r=1.0000, p=-0.5 | `communication_target_miss`, `incomplete_evidence` |
