# PRM Ablation Experiment: failed-21 N=4

## Purpose

This experiment tests which PRM-Lite v2 rule families actually contribute to reranking on the failed-21 N=4 GLM-5.1 run.

Source artifacts:

- PRM scores: `reports/airline_failed21_n4_timeout300_v2_merged.prm_v2.json`
- Ablation output: `reports/airline_failed21_n4_timeout300_v2_merged.prm_ablation_v1.json`
- CSV: `reports/airline_failed21_n4_timeout300_v2_merged.prm_ablation_v1.csv`
- Auto report: `reports/airline_failed21_n4_timeout300_v2_merged.prm_ablation_v1.md`

## Important Leakage Check

The first rerank report showed `PRM pass@4 = 0.7619`. During ablation, a sanity check found that a scoreless selector could also reach `0.7619` if it kept reward-info-derived DB/communication tags for tie-breaking.

That means the original `0.7619` result is an optimistic diagnostic number, not the clean online-safe metric. The stricter metric must not use `db_match` or evaluator communication success as tie-break signals.

Correct headline:

| Metric | Value |
| --- | ---: |
| First-trial pass | 0.3333 |
| Online-safe PRM pass@4 | 0.7143 |
| Eval-tiebreak PRM pass@4 | 0.7619 |
| Oracle pass@4 | 0.7619 |
| Online-safe gain vs first | +0.3810 |
| Online-safe gap to oracle | 0.0476 |

Interpretation:

- The clean PRM still improves from `7/21` to `15/21`.
- It misses one solvable task: task `34`.
- The eval-derived tie-break recovers task `34`, but that should be treated as leakage for training/deployment claims.

## Ablation Summary

| Ablation | Pass@4 | Selected | Drop vs online-safe full | Meaning |
| --- | ---: | ---: | ---: | --- |
| `full` | 0.7143 | 15/21 | 0.0000 | Clean PRM-Lite v2 |
| `zero_score` | 0.3333 | 7/21 | -0.3810 | First-trial sanity baseline |
| `full_eval_tiebreak` | 0.7619 | 16/21 | +0.0476 | Optimistic diagnostic with evaluator tie-break |
| `scoreless_eval_tiebreak` | 0.7619 | 16/21 | +0.0476 | Shows why eval tie-break is not allowed |
| `no_reference_actions` | 0.5714 | 12/21 | -0.1429 | Reference action signals matter |
| `no_privileged_action_rules` | 0.5238 | 11/21 | -0.1905 | Privileged action/write rules matter more |
| `no_premature_write` | 0.5238 | 11/21 | -0.1905 | Premature-write penalties are a major contributor |
| `no_temporal_policy` | 0.6667 | 14/21 | -0.0476 | Temporal policy helps one task |
| `no_evidence_object` | 0.6667 | 14/21 | -0.0476 | Evidence/object tags help one task |
| `positive_only` | 0.4286 | 9/21 | -0.2857 | Positive support alone is weak |
| `negative_only` | 0.7143 | 15/21 | 0.0000 | Negative penalties carry most useful signal |

## Rule-Family Findings

### Strong Contributors

`premature_write` is the strongest single family:

- Removing it drops pass@4 from `0.7143` to `0.5238`.
- It newly misses tasks `2, 12, 27, 42`.
- Keeping only write-safety rules reaches `0.5238`, which is far above first-trial.

Reference-action and privileged-action rules are also important:

- `no_reference_actions`: `0.5714`
- `no_privileged_action_rules`: `0.5238`
- `reference_only`: `0.5238`

This means a large part of the current PRM gain comes from penalizing wrong or unsupported writes and direct action mismatches.

### Moderate Contributors

Evidence/object rules help:

- `no_evidence_object`: `0.6667`
- `evidence_object_only`: `0.5714`

Temporal policy helps one task:

- `no_temporal_policy`: `0.6667`
- `temporal_policy_only`: `0.3810`

### Weak on This Dataset, Still Useful for Stable-Fail Analysis

Payment-planning and tool-affordance rules do not change task-level pass@4 on this N=4 set:

- `no_payment_planning`: `0.7143`
- `no_tool_affordance`: `0.7143`

This does not mean they are useless. It means they mostly fire on tasks that are either already stable-fail (`14`, `44`) or have enough other signals to select the same sample. They remain important for explaining and repairing stable-fail tasks.

## Task-Level Notes

The clean PRM selects successful samples for 15 of 16 solvable tasks.

The missed solvable task is:

| Task | Successes | Clean PRM selected | Why missed |
| ---: | ---: | --- | --- |
| 34 | 3/4 | trial 0 failure | All four process scores are `0`, so online-safe tie-break selects the first trial |

This points to a concrete PRM improvement target: add non-leaky signals that distinguish task 34's successful and failed trajectories.

Leave-one-family-out misses:

| Ablation | New misses |
| --- | --- |
| `no_reference_actions` | 15, 18, 33 |
| `no_privileged_action_rules` | 12, 15, 18, 33 |
| `no_premature_write` | 2, 12, 27, 42 |
| `no_temporal_policy` | 42 |
| `no_evidence_object` | 33 |

## Project Implication

This ablation changes the story in a good way:

Before:

- "PRM-Lite reaches oracle pass@4."

After leakage audit:

- "A strict online-safe PRM improves first-trial pass from `0.3333` to `0.7143`, with one remaining solvable miss; an evaluator-derived tie-break reaches the oracle but is explicitly identified as leakage."

That is a stronger and more credible algorithmic claim.

## Next Actions

1. Add an online-safe rerank mode to the main rerank script.
2. Add a PRM rule for task 34-style failures.
3. Add `missing_required_communication` for task 7-style failures.
4. Add route-duration and payment-method argument diagnostics for stable-fail tasks.
5. Move to SFT seed dataset construction once the PRM report uses only online-safe metrics.
