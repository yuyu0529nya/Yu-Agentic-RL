# SFT Dataset v1 Card

## Source

- Tau2 run: `third_party/tau2-bench/data/simulations/airline_failed21_n4_timeout300_v2_merged`
- PRM scores: `reports/airline_failed21_n4_timeout300_v2_merged.prm_v3.json`
- Domain: `airline`
- Official split source: `third_party/tau2-bench/data/tau2/domains/airline/split_tasks.json`

## Outputs

- SFT imitation: `data/sft/tau2_airline_sft_v1.jsonl`
- Preference pairs: `data/preferences/tau2_airline_pairs_v1.jsonl`
- Correction blueprints: `data/sft/tau2_airline_correction_blueprints_v1.jsonl`

## Summary

| Dataset | Rows | Trainable | Split counts |
| --- | ---: | ---: | --- |
| Success imitation | 32 | 18 | `{"test": 14, "train": 18}` |
| Preference pairs | 32 | 22 | `{"test": 10, "train": 22}` |
| Correction blueprints | 5 | 3 | `{"test": 2, "train": 3}` |

## Length Statistics

Token counts are approximate character-based estimates for Phase 1A. Phase 1B should replace this with Qwen tokenizer rendering.

| Dataset | Mean | P50 | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| SFT imitation | 4195.6 | 3978 | 6216 | 9440 |
| Preference chosen+rejected | 9713.4 | 8984 | 14959 | 17588 |

## Task Coverage

- Trainable SFT tasks: `1, 12, 15, 20, 23, 27, 33, 34, 38, 42`
- Heldout SFT tasks: `2, 16, 18, 25, 32, 37`
- Preference-pair tasks: `1, 2, 12, 15, 16, 18, 20, 23, 27, 32, 33, 34, 37, 38, 42`
- Trainable correction blueprint tasks: `7, 14, 21`
- Heldout correction blueprint tasks: `29, 44`

### Success Imitation Rows Per Task

| Task | Rows |
| ---: | ---: |
| 1 | 3 |
| 2 | 2 |
| 12 | 3 |
| 15 | 1 |
| 16 | 1 |
| 18 | 1 |
| 20 | 1 |
| 23 | 1 |
| 25 | 4 |
| 27 | 1 |
| 32 | 3 |
| 33 | 2 |
| 34 | 3 |
| 37 | 3 |
| 38 | 1 |
| 42 | 2 |

## Failure-Signal Coverage From Rejected Preference Sides

| Risk tag | Count |
| --- | ---: |
| `action_mismatch` | 20 |
| `calculation_error` | 3 |
| `communication_target_miss` | 1 |
| `compensation_policy_error` | 8 |
| `incomplete_evidence` | 21 |
| `object_selection_error` | 1 |
| `payment_planning_error` | 3 |
| `policy_precedence_error` | 1 |
| `premature_write` | 15 |
| `temporal_policy_error` | 4 |
| `user_pressure_susceptibility` | 1 |

## Data Policy

- `success_imitation` rows are real reward=1 trajectories from the N=4 run.
- `preference_pair` rows compare the best successful trajectory against failed trajectories for the same task.
- `targeted_correction_blueprint` rows are not real demonstrations; they are planning specs for future synthetic or human-audited correction data.
- Official test tasks are marked `trainable=false` to keep heldout evaluation clean.
- Tool responses are retained as context. The intended SFT loss is only on assistant messages, including assistant tool calls and final answers.

## Next Checks

1. Render these rows with the target Qwen tokenizer/chat template.
2. Verify assistant-only loss masks at token level.
3. Filter `trainable=true` for actual SFT training.
4. Keep official-test rows for heldout diagnostics, not training.
