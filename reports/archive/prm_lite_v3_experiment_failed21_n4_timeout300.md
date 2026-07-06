# PRM-Lite v3 Experiment: Failed-21 N=4

## Goal

This experiment turns PRM-Lite from a promising rerank heuristic into a cleaner
process-reward module with an online-safe evaluation path.

The key correction is separating two signals:

- **Main metric:** online-safe PRM tie-break, using only process components and
  their negative tags.
- **Diagnostic metric:** `--eval-tiebreak`, which can use post-hoc evaluator
  DB/communication tags only to audit leakage and upper-bound behavior.

## Dataset

| Item | Value |
| --- | ---: |
| Run | `airline_failed21_n4_timeout300_v2_merged` |
| Tasks | 21 |
| Samples per task | 4 |
| Simulations | 84 |
| Raw sample success | 32/84 = 0.3810 |
| First-trial pass | 7/21 = 0.3333 |
| Oracle pass@4 | 16/21 = 0.7619 |

## PRM-Lite v3 Changes

### 1. Online-safe rerank default

`scripts/prm_rerank_tau2.py` now uses online-safe tie-break by default:

1. Sort by `process_score`.
2. Sort by normalized process score.
3. Break ties only with process-component risk tags.
4. Finally prefer earlier trial index.

The old post-hoc behavior is preserved behind `--eval-tiebreak` and should be
reported only as a diagnostic.

### 2. Required communication reward

`scripts/process_reward_scorer.py` now checks task `communicate_info` against
assistant messages:

- `required_communication_met`: all required information was communicated.
- `missing_required_communication`: required information was absent.

Numeric matching is normalized, so examples like `1628` and `$1,628` are treated
as the same value.

This mainly improves failure taxonomy quality for stable-fail tasks such as
task 7 and task 14, where all samples fail because the agent never communicates
the required target information.

### 3. No-change write safety

The new `forbidden_db_write` component detects trajectories where the task says
the agent should not change the database, but the agent still calls write tools.

This fixed task 34:

| Trial | Reward | v3 Process | Component |
| ---: | ---: | ---: | --- |
| 0 | 0.0 | -4.0 | `forbidden_db_write` |
| 1 | 1.0 | 0.0 | - |
| 2 | 1.0 | 0.0 | - |
| 3 | 1.0 | 0.0 | - |

Task 34 moved from an eval-tag-only tie-break case to a true process-reward win.

## Rerank Results

| Method | Tie-break | Selected | Pass@4 |
| --- | --- | ---: | ---: |
| First trial | deterministic | 7/21 | 0.3333 |
| PRM-Lite v3 full | online-safe | 16/21 | 0.7619 |
| PRM-Lite v3 full | eval diagnostic | 16/21 | 0.7619 |
| No reference actions | online-safe | 13/21 | 0.6190 |
| Heuristic only | online-safe | 12/21 | 0.5714 |
| Oracle | evaluator | 16/21 | 0.7619 |

Main result: **online-safe PRM-Lite v3 reaches the oracle upper bound on this
N=4 set**, selecting all 16 solvable tasks.

## Ablation Highlights

| Ablation | Selected | Pass@4 | Gap to oracle |
| --- | ---: | ---: | ---: |
| Full | 16/21 | 0.7619 | 0.0000 |
| Zero score | 7/21 | 0.3333 | 0.4286 |
| Scoreless eval tie-break | 16/21 | 0.7619 | 0.0000 |
| No reference actions | 13/21 | 0.6190 | 0.1429 |
| No privileged action rules | 12/21 | 0.5714 | 0.1905 |
| No premature-write rules | 11/21 | 0.5238 | 0.2381 |
| No temporal-policy rules | 14/21 | 0.6667 | 0.0952 |
| No evidence/object rules | 15/21 | 0.7143 | 0.0476 |
| Negative-only | 16/21 | 0.7619 | 0.0000 |

Interpretation:

- Premature-write safety is the strongest non-reference family.
- Reference/action-match signals still matter a lot.
- Negative penalty components are currently doing most of the selection work.
- `scoreless_eval_tiebreak` reaching 16/21 proves why evaluator tags must stay
  out of the main metric.

## Current Conclusion

Phase 0 rerank is now in a clean state:

- Baseline and N=4 sampling are complete.
- PRM-Lite v3 has a clean online-safe main result.
- The leakage-prone tie-break is isolated as a diagnostic.
- Stable-fail tasks now have clearer process labels for SFT data construction.

The next research step should be moving from rerank to training data:

1. Convert stable-fail analyses into SFT examples.
2. Build preference pairs from successful vs failed trajectories.
3. Train an open model such as Qwen2.5/Qwen3-7B on the airline tool-use format.
4. Re-run the same PRM and rerank suite to test whether learned behavior improves
   task success, not just selection.

## Output Files

- `reports/airline_failed21_n4_timeout300_v2_merged.prm_v3.json`
- `reports/airline_failed21_n4_timeout300_v2_merged.prm_v3.md`
- `reports/airline_failed21_n4_timeout300_v2_merged.rerank_v3_full_online_safe.json`
- `reports/airline_failed21_n4_timeout300_v2_merged.rerank_v3_full_online_safe.md`
- `reports/airline_failed21_n4_timeout300_v2_merged.rerank_v3_full_eval_tiebreak.md`
- `reports/airline_failed21_n4_timeout300_v2_merged.prm_ablation_v3.md`
