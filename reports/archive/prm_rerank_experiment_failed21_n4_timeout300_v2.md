# PRM-Rerank Experiment: failed-21 N=4 timeout-safe GLM-5.1 run

## Run Setup

- Domain: `tau2-bench airline`
- Agent model: `anthropic/glm-5.1`
- User simulator model: `anthropic/glm-5.1`
- Task set: 21 previously failed or unstable airline tasks
- Sampling: 4 trials per task
- Total simulations: 84
- Runner: timeout-safe sharded evaluator
- Per-shard timeout: 300s
- Result directory: `third_party/tau2-bench/data/simulations/airline_failed21_n4_timeout300_v2_merged`

## Engineering Result

The monolithic failed-21 run previously stalled on a pathological long trajectory. The sharded evaluator fixed this failure mode.

| Metric | Value |
| --- | ---: |
| Shards launched | 84 |
| Shards merged | 84 |
| Missing shards | 0 |
| Timeout shards | 0 |
| Failed result writes | 0 |
| Termination mode | 84 user_stop |

Slowest shards:

| Task | Trial | Duration |
| ---: | ---: | ---: |
| 14 | 3 | 234.9s |
| 32 | 0 | 212.8s |
| 33 | 2 | 208.8s |
| 23 | 2 | 202.9s |
| 23 | 1 | 190.8s |
| 42 | 3 | 170.8s |
| 20 | 0 | 160.6s |
| 23 | 3 | 154.7s |

This validates the evaluator design: long trajectories are observable and bounded, not batch-breaking.

## Baseline Result

| Metric | Value |
| --- | ---: |
| Tasks | 21 |
| Simulations | 84 |
| Raw sample success | 32/84 |
| Avg reward | 0.3810 |
| First-trial pass | 7/21 = 0.3333 |
| Solvable by any of 4 samples | 16/21 = 0.7619 |

Task success counts:

| Class | Tasks |
| --- | --- |
| Stable success, 4/4 | 25 |
| Strong success, 3/4 | 1, 12, 32, 34, 37 |
| Medium split, 2/4 | 2, 33, 42 |
| Rare success, 1/4 | 15, 16, 18, 20, 23, 27, 38 |
| Stable fail, 0/4 | 7, 14, 21, 29, 44 |

Interpretation:

- The task set is not merely too hard. 16/21 tasks have at least one successful sample.
- This is exactly the setting where PRM-based reranking can matter.
- The 5 stable-fail tasks are the next candidates for SFT data construction or policy improvement.

## PRM-Lite v2 Result

| Metric | Value |
| --- | ---: |
| Avg process score | -2.1250 |
| Success avg process score | 0.5156 |
| Failure avg process score | -3.7500 |

Top risk tags:

| Tag | Count |
| --- | ---: |
| `communication_db_gap` | 43 |
| `incomplete_evidence` | 43 |
| `action_mismatch` | 42 |
| `premature_write` | 25 |
| `temporal_policy_error` | 9 |
| `calculation_error` | 8 |
| `payment_planning_error` | 8 |
| `compensation_policy_error` | 8 |
| `premature_deferral` | 5 |
| `tool_affordance_error` | 5 |

The score separation is useful: successful trajectories score much higher than failed trajectories on average. The most common failure pattern is still "insufficient evidence / action mismatch / DB mismatch", which aligns with the intended process reward of "check fully, compute correctly, then act".

## PRM-Rerank Result

The original evaluator-tie-break scoring modes reached the oracle on this run. A later ablation audit revised the clean online-safe headline to `0.7143` pass@4.

Important later audit note:

The original rerank script used evaluator-derived `db_match` / communication tags as tie-break signals through `risk_tags`. A follow-up ablation identified this as a leakage risk. The clean online-safe PRM metric is `0.7143` pass@4, while the evaluator-tie-break diagnostic remains `0.7619`. Use the online-safe number for final project claims.

| Score mode | First-trial pass | Raw sample success | Oracle pass@4 | PRM pass@4 | Gain vs first | Gap to oracle | Selection acc. on solvable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.3333 | 0.3810 | 0.7619 | 0.7619 | +0.4286 | 0.0000 | 1.0000 |
| `no_reference_actions` | 0.3333 | 0.3810 | 0.7619 | 0.7619 | +0.4286 | 0.0000 | 1.0000 |
| `heuristic_only` | 0.3333 | 0.3810 | 0.7619 | 0.7619 | +0.4286 | 0.0000 | 1.0000 |

Key examples:

| Task | Successes | First trial | PRM selected | Pattern |
| ---: | ---: | --- | --- | --- |
| 1 | 3/4 | fail | success | PRM avoids premature write / temporal error |
| 2 | 2/4 | fail | success | PRM separates successful evidence-complete paths |
| 15 | 1/4 | fail | success | PRM finds rare success |
| 18 | 1/4 | fail | success | PRM finds rare success in repeated-write task |
| 23 | 1/4 | fail | success | PRM finds rare payment-planning success |
| 27 | 1/4 | fail | success | PRM finds rare success |
| 33 | 2/4 | fail | success | PRM handles object/evidence split |
| 34 | 3/4 | fail | success | PRM upgrades first-trial failure |

This is the strongest project milestone so far: the system now demonstrates a concrete algorithmic improvement on a long-horizon agent benchmark, not only a baseline run.

## Failure Taxonomy v2

| Failure family | Tasks | Evidence |
| --- | --- | --- |
| Stable hard fail | 7, 14, 21, 29, 44 | 0/4 samples succeeded. These need data or policy changes, not rerank alone. |
| Payment / calculation planning | 14, 23 | `calculation_error` and `payment_planning_error` repeatedly appear. |
| Evidence/object selection drift | 2, 32, 33, 44 | `incomplete_evidence`, `action_mismatch`, and `communication_db_gap` dominate. |
| Premature write / DB mismatch | 1, 7, 29, 38, 42 | `premature_write` appears 25 times overall. |
| Tool affordance / premature deferral | 12, 44 | Some trajectories defer or fail to use available tools properly. |

## What This Means for the Project

This run upgrades the project from "we can run tau2" to a small but real experimental system:

1. A robust long-horizon evaluator handles slow pathological trajectories.
2. A process reward scorer produces interpretable failure tags.
3. A clean online-safe PRM-reranker improves first-trial pass from 0.3333 to 0.7143 on failed-21 N=4.
4. The evaluator-tie-break diagnostic reaches oracle at 0.7619, but this is not used as the final deployable metric.
5. The remaining 0/4 tasks provide a clean target set for SFT or RL.

## Next Steps

Recommended next technical step:

1. Manually inspect stable-fail tasks `7,14,21,29,44`.
2. Build a small SFT dataset from successful trajectories plus repaired stable-fail trajectories.
3. Add a PRM ablation report that proves which rule families drive selection.
4. Convert PRM-Lite into a training-time reward signal for GRPO-style experiments.
5. Add an "unseen task split" so rerank and future RL claims are not contaminated by task-specific tuning.

Artifacts:

- `reports/airline_failed21_n4_timeout300_v2_merged.summary.json`
- `reports/airline_failed21_n4_timeout300_v2_merged.prm_v2.md`
- `reports/airline_failed21_n4_timeout300_v2_merged.rerank_v2_full.md`
- `reports/airline_failed21_n4_timeout300_v2_merged.rerank_v2_no_reference_actions.md`
- `reports/airline_failed21_n4_timeout300_v2_merged.rerank_v2_heuristic_only.md`
- `reports/sharded_logs/airline_failed21_n4_timeout300_v2/manifest.csv`
- `third_party/tau2-bench/data/simulations/airline_failed21_n4_timeout300_v2_merged/results.json`
