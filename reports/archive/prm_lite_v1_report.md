# PRM-Lite v1 Report

Date: 2026-05-26

Run: `airline_baseline_50_anthropic_glm_5_1_50tasks_1trials`

## Scope

PRM-Lite v1 extends the original process reward scorer with rules derived from GLM 5.1 failure cases `44`, `2`, `37`, and `23`.

The scorer remains rule-based. It is not a learned reward model yet. The purpose is to make failure attribution explicit and measurable before running PRM-rerank, SFT data filtering, or GRPO.

## New Rules

| Rule | Component | Target Failure |
| --- | --- | --- |
| Schedule/duration tasks must use schedule-returning search tools. | `wrong_tool_for_schedule_lookup` | Task 44 tool affordance error. |
| Do not defer when reference actions show available search tools. | `premature_deferral` | Task 44 over-deferral. |
| Do not cancel already-flown reservations. | `past_flight_cancel_attempt` | Task 37 past flight cancellation. |
| Blocking policy rules override permissive rules. | `policy_precedence_error` | Task 37 "business can cancel" overgeneralization. |
| Delayed-flight compensation requires compatible user intent. | `compensation_requires_user_goal` | Task 2 certificate despite no change/cancel intent. |
| Payment optimization tasks must execute expected write strategy. | `expected_write_plan_missing` | Task 23 update-only instead of cancel-and-rebook. |
| Payment reasoning should not communicate inconsistent Mastercard charges. | `calculation_trace_inconsistent` | Task 23 arithmetic inconsistency. |

## Aggregate Change

| Metric | v0 | v1 |
| --- | ---: | ---: |
| Avg reward | 0.5800 | 0.5800 |
| Avg process score | -0.8200 | -1.1300 |
| Success process score | 0.7931 | 0.7931 |
| Failure process score | -3.0476 | -3.7857 |

The outcome reward is unchanged because no trajectories were rerun. The process scorer became more discriminative on failed trajectories while preserving the successful trajectory average.

## Target Case Changes

| Task | v0 Process | v1 Process | New Signals |
| --- | ---: | ---: | --- |
| 44 | -13.0 | -16.5 | `wrong_tool_for_schedule_lookup`, `premature_deferral` |
| 2 | -9.0 | -11.0 | `compensation_requires_user_goal` |
| 37 | -6.5 | -10.5 | `past_flight_cancel_attempt`, `policy_precedence_error` |
| 23 | -5.5 | -8.5 | `expected_write_plan_missing`, `calculation_trace_inconsistent` |

## v1 Risk Tag Counts

| Tag | Count |
| --- | ---: |
| `action_mismatch` | 17 |
| `communication_db_gap` | 18 |
| `incomplete_evidence` | 17 |
| `premature_write` | 9 |
| `compensation_policy_error` | 3 |
| `temporal_policy_error` | 3 |
| `calculation_error` | 2 |
| `payment_planning_error` | 2 |
| `object_selection_error` | 1 |
| `policy_precedence_error` | 1 |
| `premature_deferral` | 1 |
| `tool_affordance_error` | 1 |
| `user_pressure_susceptibility` | 1 |

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `reports/airline_baseline_50_anthropic_glm_5_1_50tasks_1trials.prm_v1.md` | Per-task process reward report. |
| `reports/airline_baseline_50_anthropic_glm_5_1_50tasks_1trials.prm_v1.csv` | Spreadsheet-friendly PRM rows. |
| `reports/airline_baseline_50_anthropic_glm_5_1_50tasks_1trials.prm_v1.json` | Machine-readable PRM details. |

## Current Limitations

1. `reference_action_mismatch` is still broad. It tells us an expected action is missing, but not always why.
2. Payment reasoning is only checked through communicated Mastercard amounts and expected write strategy. It does not yet recompute the globally optimal payment plan.
3. Tool affordance scoring is currently tuned for schedule/duration failures. Other tool-affordance classes still need rules.
4. Some successful trajectories can still have negative process components if they used a risky path but ended with the right DB state. This is acceptable for diagnosis, but rerank experiments must check false negatives.

## Next Experiment

Run PRM-rerank on a small task set:

1. Sample `N=4` trajectories per task for 10-20 tasks.
2. Score all trajectories with PRM-Lite v1.
3. Select the highest-process-score trajectory per task.
4. Compare:
   - raw `pass^1`,
   - oracle `pass^4`,
   - PRM-rerank `pass@4`.

This tests whether the process reward can improve selection before any model training.

