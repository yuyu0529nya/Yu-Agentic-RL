# Phase 1G SFT Adapter Offline Evaluation

## Goal

Compare the base Qwen model against SFT LoRA adapters using assistant-only masked NLL on train, valid, and heldout splits.

## Key Findings

- `action_prefix_v1`: heldout -0.5805 -> **improves selected splits**.
- `action_prefix_v2`: heldout -0.4630 -> **improves selected splits**.
- Selected adapter for follow-up: `action_prefix_v1` based on `heldout` masked NLL.

## Setup

- Base model: `models/Qwen2.5-0.5B-Instruct`
- Max sequence length: `1536`
- Device: `cuda`
- FP16: `True`
- Eval splits: `heldout`
- Max rows per split: `12`

## Split Metrics

| Model | Split | Rows | Weighted loss | Mean loss | PPL | Target tokens | Truncated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | heldout | 12 | 1.2214 | 1.2585 | 3.39 | 857 | 0 |
| `action_prefix_v1` | heldout | 12 | 0.6409 | 0.5234 | 1.90 | 857 | 0 |
| `action_prefix_v2` | heldout | 12 | 0.7585 | 0.6953 | 2.13 | 857 | 0 |

## Delta Vs Base

| Adapter | Split | Loss delta | Relative change |
| --- | --- | ---: | ---: |
| `action_prefix_v1` | heldout | -0.5805 | -47.5% |
| `action_prefix_v2` | heldout | -0.4630 | -37.9% |

## Row Deltas: action_prefix_v1

### heldout

| Sample | Task | Base | Adapter | Delta | Target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `action_prefix_v2_heldout_sft_success_task16_trial0_turn4` | 16 | 2.5991 | 0.0629 | -2.5363 | 38 |
| `action_prefix_v2_heldout_sft_success_task2_trial2_turn8` | 2 | 2.3111 | 0.6440 | -1.6671 | 46 |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn8` | 2 | 2.0868 | 0.4942 | -1.5926 | 51 |
| `action_prefix_v2_heldout_sft_success_task16_trial0_turn6` | 16 | 1.1135 | 0.4548 | -0.6587 | 77 |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn10` | 2 | 1.2102 | 0.6456 | -0.5646 | 58 |
| ... worst regressions below ... |  |  |  |  |  |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn18` | 2 | 1.4379 | 1.0761 | -0.3618 | 78 |
| `action_prefix_v2_heldout_sft_success_task16_trial0_turn9` | 16 | 1.4049 | 1.1784 | -0.2265 | 144 |
| `action_prefix_v2_heldout_sft_success_task2_trial2_turn17` | 2 | 0.5462 | 0.3690 | -0.1773 | 167 |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn20` | 2 | 0.1247 | 0.0142 | -0.1105 | 31 |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn12` | 2 | 0.0473 | 0.0004 | -0.0469 | 18 |


## Row Deltas: action_prefix_v2

### heldout

| Sample | Task | Base | Adapter | Delta | Target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `action_prefix_v2_heldout_sft_success_task16_trial0_turn4` | 16 | 2.5991 | 0.7637 | -1.8355 | 38 |
| `action_prefix_v2_heldout_sft_success_task2_trial2_turn8` | 2 | 2.3111 | 0.9479 | -1.3632 | 46 |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn8` | 2 | 2.0868 | 0.8785 | -1.2083 | 51 |
| `action_prefix_v2_heldout_sft_success_task16_trial0_turn16` | 16 | 1.2507 | 0.7341 | -0.5166 | 87 |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn10` | 2 | 1.2102 | 0.8223 | -0.3879 | 58 |
| ... worst regressions below ... |  |  |  |  |  |
| `action_prefix_v2_heldout_sft_success_task2_trial2_turn10` | 2 | 0.9694 | 0.6578 | -0.3116 | 62 |
| `action_prefix_v2_heldout_sft_success_task2_trial2_turn17` | 2 | 0.5462 | 0.3333 | -0.2130 | 167 |
| `action_prefix_v2_heldout_sft_success_task16_trial0_turn9` | 16 | 1.4049 | 1.2128 | -0.1922 | 144 |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn12` | 2 | 0.0473 | 0.0090 | -0.0383 | 18 |
| `action_prefix_v2_heldout_sft_success_task2_trial3_turn20` | 2 | 0.1247 | 0.0892 | -0.0355 | 31 |

## Interpretation

- Lower masked NLL means the model assigns higher probability to the reference assistant/tool-call tokens.
- Train improvements without valid/heldout improvements indicate memorization or distribution mismatch.
- This is an offline teacher-forced metric, not an end-to-end tau2 pass rate.
