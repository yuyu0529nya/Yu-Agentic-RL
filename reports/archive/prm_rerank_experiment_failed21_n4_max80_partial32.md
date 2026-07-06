# PRM-Rerank Experiment: GLM-5.1 Failed-21 N=4 Max80 Partial32

## Status

This run is a partial diagnostic run, not the final N=4 result.

- Run name: `airline_prm_rerank_glm51_failed21_n4_max80`
- Domain: `airline`
- Agent model: `anthropic/glm-5.1`
- User simulator model: `anthropic/glm-5.1`
- Target tasks: 21 failed tasks from the 50-task GLM-5.1 baseline
- Target samples: 84 simulations, 21 tasks x 4 trials
- Completed samples: 32 simulations
- Reason stopped: one trajectory, task 18 trial 1, reached 711.2s wall time. This exposed a missing wall-clock timeout in the text-mode batch runner.

## Partial Baseline Metrics

From `reports/airline_prm_rerank_glm51_failed21_n4_max80.summary.json`:

| Metric | Value |
| --- | ---: |
| Simulations | 32 |
| Completed task ids | 21 |
| Samples per task | 1-2 |
| Avg reward | 0.3750 |
| Success count | 12 |
| Pass^1 | 0.3571 |
| DB match | 12 |
| DB mismatch | 20 |

Important caveat: this pass^1 is over an uneven partial sample set. It is useful for diagnosis, but not for final comparison.

## PRM-Lite v2 Summary

From `reports/airline_prm_rerank_glm51_failed21_n4_max80_partial32.prm_v2.json`:

| Metric | Value |
| --- | ---: |
| Avg process score | -2.2031 |
| Avg process score, success | 0.7083 |
| Avg process score, failure | -3.9500 |

Top risk tags:

| Tag | Count |
| --- | ---: |
| `action_mismatch` | 17 |
| `incomplete_evidence` | 17 |
| `communication_db_gap` | 15 |
| `premature_write` | 9 |
| `payment_planning_error` | 4 |
| `calculation_error` | 3 |
| `compensation_policy_error` | 3 |
| `temporal_policy_error` | 3 |
| `user_pressure_susceptibility` | 2 |
| `object_selection_error` | 1 |

The score separation is healthy: successful trajectories score much higher than failed trajectories on average.

## Rerank Result

All three scoring modes reached the partial oracle on this uneven 1-2 sample set:

| Score mode | First-trial pass | Raw sample success | Oracle pass@N | PRM pass@N | Gain vs first | Gap to oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.3333 | 0.3750 | 0.3810 | 0.3810 | +0.0476 | 0.0000 |
| `no_reference_actions` | 0.3333 | 0.3750 | 0.3810 | 0.3810 | +0.0476 | 0.0000 |
| `heuristic_only` | 0.3333 | 0.3750 | 0.3810 | 0.3810 | +0.0476 | 0.0000 |

Interpretation:

- The partial run has only one true sample-split task so far: task 2.
- PRM-Lite v2 correctly selected the successful task 2 trial.
- This is encouraging, but still not enough as a final claim because many tasks only have one sample.

## Task Classes

| Class | Tasks | Count |
| --- | --- | ---: |
| Stable success, 2/2 | 12, 15, 16, 20 | 4 |
| Stable fail, 0/2 | 1, 7, 14, 18, 21, 23 | 6 |
| Sample split | 2 | 1 |
| Single success | 25, 34, 37 | 3 |
| Single fail | 27, 29, 32, 33, 38, 42, 44 | 7 |

The most useful next data is not more easy successes. It is more samples for the split and stable-fail tasks, especially tasks 2, 18, 23, 32, 33, 42, and 44.

## Slow Trajectories

| Task | Trial | Reward | Process score | Duration | Main tags |
| ---: | ---: | ---: | ---: | ---: | --- |
| 18 | 1 | 0 | -3.0 | 711.2s | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 23 | 0 | 0 | -5.0 | 338.3s | `action_mismatch`, `calculation_error`, `incomplete_evidence`, `payment_planning_error`, `premature_write` |
| 2 | 1 | 1 | 2.5 | 246.9s | - |
| 23 | 1 | 0 | -2.0 | 227.1s | `action_mismatch`, `calculation_error`, `communication_db_gap`, `incomplete_evidence`, `payment_planning_error` |
| 32 | 0 | 0 | -1.0 | 206.2s | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 18 | 0 | 0 | -3.0 | 200.6s | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 33 | 0 | 0 | -2.0 | 200.5s | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |
| 20 | 0 | 1 | 1.0 | 190.2s | - |
| 21 | 1 | 0 | -2.0 | 180.0s | `action_mismatch`, `communication_db_gap`, `incomplete_evidence` |

Task 18 is the clear timeout outlier. This motivates a timeout-safe sharded evaluator before the full failed-21 N=4 run is retried.

## Failure Taxonomy v1.5

Observed recurring failure families:

| Failure family | Typical tasks | Pattern |
| --- | --- | --- |
| Rebook/cancel plan not closed | 14, 23, 29 | Cancel may be correct, but booking replacement is wrong or unsupported. |
| Read/search evidence incomplete | 32, 33, 44 | User and reservation lookup may be correct, but flight search or object selection drifts. |
| Write after weak evidence | 1, 38, 42 | Communication looks acceptable, but DB ends mismatched after premature or unsupported writes. |
| Repeated write drift | 18, 21 | Multiple update calls are made; some are correct and some are wrong, producing final DB mismatch. |
| Evaluation/process mismatch candidates | 12, 42 | Final or action checks can disagree with process details; these need manual inspection before new rules. |

## Timeout-Safe Sharded Evaluator

The monolithic 84-simulation run exposed a real long-horizon evaluation risk: one pathological trajectory can stall the whole experiment. This is now handled by a timeout-safe sharded evaluator.

Implemented:

1. `scripts/run_tau2_airline_sharded.ps1`: Windows runner, one task-trial shard per process, per-shard timeout, resumable skips, stdout/stderr logs, manifest CSV.
2. `scripts/run_tau2_airline_sharded.py`: cross-platform runner with the same shard/timeout/merge behavior.
3. `scripts/merge_tau2_shards.py`: merges completed shard `results.json` files into one tau2-style result directory, with missing-shard accounting.

Validated:

- Python scripts compile successfully.
- PowerShell and Python runners both pass dry-run command generation.
- Merge script was tested offline on existing task-2 shard data.
- PRM-rerank can run on the merged output.

Recommended failed-21 rerun:

```powershell
.\scripts\run_tau2_airline_sharded.ps1 `
  -AgentLlm "anthropic/glm-5.1" `
  -UserLlm "anthropic/glm-5.1" `
  -TaskIds "1,2,7,12,14,15,16,18,20,21,23,25,27,29,32,33,34,37,38,42,44" `
  -NumTrials 4 `
  -MaxSteps 80 `
  -TimeoutSeconds 300 `
  -ShardPrefix "airline_failed21_n4_timeout300" `
  -MergedSaveTo "airline_failed21_n4_timeout300_merged"
```

After the rerun, regenerate:

- PRM-Lite v2 scoring
- Full/no-reference/heuristic rerank
- Timeout and trajectory-length diagnostics

This turns the interruption into a project feature: robust long-horizon evaluation under pathological long trajectories.

## Artifacts

- `reports/airline_prm_rerank_glm51_failed21_n4_max80.summary.json`
- `reports/airline_prm_rerank_glm51_failed21_n4_max80_partial32.prm_v2.md`
- `reports/airline_prm_rerank_glm51_failed21_n4_max80_partial32.rerank_v2_full.md`
- `reports/airline_prm_rerank_glm51_failed21_n4_max80_partial32.rerank_v2_no_reference_actions.md`
- `reports/airline_prm_rerank_glm51_failed21_n4_max80_partial32.rerank_v2_heuristic_only.md`
- `scripts/run_tau2_airline_sharded.ps1`
- `scripts/run_tau2_airline_sharded.py`
- `scripts/merge_tau2_shards.py`
