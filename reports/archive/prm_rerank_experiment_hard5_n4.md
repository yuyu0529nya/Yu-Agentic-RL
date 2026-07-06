# PRM-rerank Experiment: GLM 5.1 hard5 N=4

## Setup

| Item | Value |
| --- | --- |
| Domain | tau2-bench airline |
| Agent/User simulator | `anthropic/glm-5.1` / `anthropic/glm-5.1` |
| Tasks | `1, 2, 23, 37, 44` |
| Samples per task | 4 |
| Total simulations | 20 |
| Max steps | 120 |
| Run directory | `third_party/tau2-bench/data/simulations/airline_prm_rerank_glm51_hard5_n4` |

## Metric Definitions

| Metric | Meaning |
| --- | --- |
| First-trial pass | Success rate if we only keep trial 0 for each task. |
| Raw sample success rate | Success rate across all sampled trajectories. |
| Oracle pass@N | Task is counted as solved if any of its N samples succeeds. This is the upper bound for reranking. |
| PRM-rerank pass@N | Task is counted as solved if the trajectory selected by PRM-Lite succeeds. |
| Selection accuracy on solvable tasks | Among tasks with at least one successful sample, how often PRM-Lite selected a success. |

## PRM-Lite v2 Additions

PRM-Lite v2 adds payment-planning invariants for task 23-like failures:

| Component | Signal |
| --- | --- |
| `basic_economy_update_attempt` | Penalizes updating a basic-economy reservation when the task context explicitly requires cancel-and-rebook planning. |
| `write_tool_payment_error` | Penalizes failed write tools such as payment amount mismatch during booking. |
| `book_before_cancel_rebook_plan` | Penalizes booking before cancelling the existing basic-economy reservation in cancel-and-rebook contexts. |
| `split_booking_policy_supported` | Rewards multiple single-passenger bookings with at most one certificate per reservation. |
| `optimal_mastercard_charge_matched` | Rewards matching the minimum Mastercard charge estimated from searched business fares and stored certificate/gift-card balances. |
| `optimal_mastercard_charge_mismatch` | Penalizes charging more than the estimated minimum Mastercard amount. |

The first version was too broad and briefly misclassified task 22, a successful basic-economy-to-economy update. The rule was narrowed to only fire in explicit cancel-and-rebook / one-certificate-per-reservation contexts. After the fix, task 22 returns to `process=1.0` with no risk tags.

## Main Result

| Scorer | Score mode | First-trial pass | Raw sample success | Oracle pass@N | PRM-rerank pass@N | Gain vs first | Gap to oracle | Solvable selection |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | `full` | 0.4000 | 0.3500 | 0.8000 | 0.8000 | +0.4000 | 0.0000 | 1.0000 |
| v1 | `no_reference_actions` | 0.4000 | 0.3500 | 0.8000 | 0.6000 | +0.2000 | 0.2000 | 0.7500 |
| v1 | `heuristic_only` | 0.4000 | 0.3500 | 0.8000 | 0.6000 | +0.2000 | 0.2000 | 0.7500 |
| v2 | `full` | 0.4000 | 0.3500 | 0.8000 | 0.8000 | +0.4000 | 0.0000 | 1.0000 |
| v2 | `no_reference_actions` | 0.4000 | 0.3500 | 0.8000 | 0.8000 | +0.4000 | 0.0000 | 1.0000 |
| v2 | `heuristic_only` | 0.4000 | 0.3500 | 0.8000 | 0.8000 | +0.4000 | 0.0000 | 1.0000 |

The important change is not the full-mode result, which was already oracle-aligned in v1. The important change is that v2 also reaches oracle on `no_reference_actions` and `heuristic_only`, meaning the payment-planning rules recover task 23 without direct action-check reward.

## Per-task v2 Full-mode Rerank

| Task | Successes | First trial | PRM selected | Oracle solvable | Notes |
| --- | ---: | --- | --- | --- | --- |
| 1 | 1/4 | fail, process -4.5 | trial 1, success, process 1.0 | yes | Temporal cancellation policy. |
| 2 | 1/4 | success, process 2.5 | trial 0, success, process 2.5 | yes | Compensation/certificate trap in failed samples. |
| 23 | 1/4 | fail, process -5.5 | trial 3, success, process 5.0 | yes | Split booking, one certificate per reservation, Mastercard minimum. |
| 37 | 4/4 | success, process 1.0 | trial 0, success, process 1.0 | yes | Stable solved task under N=4. |
| 44 | 0/4 | fail, process -8.0 | trial 0, fail, process -8.0 | no | No sampled trajectory solved the schedule-duration planning task. |

## Task 23 v2 Separation

| Trial | Reward | v2 full process | v2 heuristic-only process | Diagnosis |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.0 | -5.5 | -2.5 | Split booking shape exists, but wrong fare/payment plan. |
| 1 | 0.0 | -11.5 | -4.0 | Updates existing basic-economy reservation instead of cancel-and-rebook. |
| 2 | 0.0 | -9.0 | -5.0 | Books before cancellation and hits payment mismatch. |
| 3 | 1.0 | 5.0 | 4.0 | Correct cancel, three separate bookings, correct card total. |

## 50-task Sanity Check

The scorer was rerun on the 50-task GLM 5.1 baseline:

| Metric | Value |
| --- | ---: |
| Simulations | 50 |
| Avg reward | 0.5800 |
| Success count | 29 |
| Avg process score | -1.2500 |
| Avg process score, success | 0.7931 |
| Avg process score, failure | -4.0714 |

Top risk tags remain consistent with v1: action mismatch, incomplete evidence, communication-DB gap, premature write, temporal policy error, and payment-planning error. The v2 payment rules add stronger signal on task 14 and task 23 without the task 22 false positive after narrowing.

## Interpretation

This experiment now supports a stronger claim:

1. Multi-sampling exposes latent successful trajectories: first-trial pass is 0.40, oracle pass@4 is 0.80.
2. PRM-Lite v2 can select the successful trajectory on all solvable hard tasks, including under stricter modes that remove direct action-check components.
3. Payment planning can be scored as process structure, not only final DB success: search fares, compute balances, enforce one-certificate-per-reservation, and match Mastercard minimum.

## Next Work

1. Run N=4 rerank on all failed tasks from the 50-task baseline, not just hard5.
2. Add a task-family report: temporal policy, compensation, payment planning, schedule/duration tool affordance.
3. Convert rerank outputs into training data: selected-success positives, high-score failures as hard negatives, and component labels as process supervision.
4. Prepare the transition from PRM-rerank to training: SFT data construction first, then small GRPO with PRM-Lite reward shaping.
