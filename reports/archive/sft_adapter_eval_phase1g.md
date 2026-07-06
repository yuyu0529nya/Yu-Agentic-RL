# Phase 1G SFT Adapter Offline Evaluation

## Goal

Compare the base Qwen model against SFT LoRA adapters using assistant-only masked NLL on train, valid, and heldout splits.

## Key Findings

- `early20`: train -0.5404, valid -0.2708, heldout -0.3681 -> **improves generalization**.
- `final120`: train -1.3204, valid +0.2506, heldout +0.1627 -> **overfits or regresses**.
- Selected adapter for follow-up: `early20` based on heldout masked NLL.

## Setup

- Base model: `models/Qwen2.5-0.5B-Instruct`
- Max sequence length: `2048`
- Device: `cuda`
- FP16: `True`

## Split Metrics

| Model | Split | Rows | Weighted loss | Mean loss | PPL | Target tokens | Truncated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | train | 14 | 1.3835 | 1.3014 | 3.99 | 6550 | 14 |
| `base` | valid | 4 | 1.2633 | 1.2835 | 3.54 | 2758 | 4 |
| `base` | heldout | 14 | 1.3177 | 1.3154 | 3.73 | 6388 | 14 |
| `early20` | train | 14 | 0.8431 | 0.7171 | 2.32 | 6550 | 14 |
| `early20` | valid | 4 | 0.9926 | 0.9649 | 2.70 | 2758 | 4 |
| `early20` | heldout | 14 | 0.9496 | 0.8733 | 2.58 | 6388 | 14 |
| `final120` | train | 14 | 0.0632 | 0.0380 | 1.07 | 6550 | 14 |
| `final120` | valid | 4 | 1.5139 | 1.4752 | 4.54 | 2758 | 4 |
| `final120` | heldout | 14 | 1.4804 | 1.3474 | 4.39 | 6388 | 14 |

## Delta Vs Base

| Adapter | Split | Loss delta | Relative change |
| --- | --- | ---: | ---: |
| `early20` | train | -0.5404 | -39.1% |
| `early20` | valid | -0.2708 | -21.4% |
| `early20` | heldout | -0.3681 | -27.9% |
| `final120` | train | -1.3204 | -95.4% |
| `final120` | valid | +0.2506 | +19.8% |
| `final120` | heldout | +0.1627 | +12.3% |

## Row Deltas: early20

### train

| Sample | Task | Base | Adapter | Delta | Target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sft_success_task1_trial2` | 1 | 1.2349 | 0.4189 | -0.8160 | 254 |
| `sft_success_task1_trial1` | 1 | 1.1699 | 0.3746 | -0.7952 | 247 |
| `sft_success_task1_trial3` | 1 | 1.1151 | 0.4081 | -0.7070 | 257 |
| `sft_success_task34_trial1` | 34 | 1.3037 | 0.6306 | -0.6731 | 550 |
| `sft_success_task34_trial2` | 34 | 1.2669 | 0.6163 | -0.6506 | 565 |
| ... worst regressions below ... |  |  |  |  |  |
| `sft_success_task12_trial3` | 12 | 1.6518 | 1.1007 | -0.5511 | 517 |
| `sft_success_task27_trial2` | 27 | 1.7332 | 1.2618 | -0.4713 | 435 |
| `sft_success_task20_trial0` | 20 | 1.0834 | 0.6374 | -0.4459 | 341 |
| `sft_success_task12_trial2` | 12 | 1.6072 | 1.2094 | -0.3978 | 897 |
| `sft_success_task12_trial1` | 12 | 1.5937 | 1.2712 | -0.3225 | 932 |

### valid

| Sample | Task | Base | Adapter | Delta | Target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sft_success_task15_trial2` | 15 | 1.3910 | 0.8049 | -0.5862 | 314 |
| `sft_success_task33_trial2` | 33 | 1.3289 | 1.0626 | -0.2663 | 810 |
| `sft_success_task33_trial1` | 33 | 1.1837 | 0.9206 | -0.2631 | 840 |
| `sft_success_task23_trial1` | 23 | 1.2302 | 1.0715 | -0.1587 | 794 |

### heldout

| Sample | Task | Base | Adapter | Delta | Target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sft_success_task16_trial0` | 16 | 1.4641 | 0.8379 | -0.6261 | 343 |
| `sft_success_task32_trial0` | 32 | 1.2509 | 0.6298 | -0.6211 | 247 |
| `sft_success_task37_trial0` | 37 | 1.2019 | 0.5857 | -0.6162 | 243 |
| `sft_success_task32_trial1` | 32 | 1.2128 | 0.6277 | -0.5851 | 246 |
| `sft_success_task32_trial2` | 32 | 1.2070 | 0.6273 | -0.5797 | 245 |
| ... worst regressions below ... |  |  |  |  |  |
| `sft_success_task25_trial3` | 25 | 1.5038 | 1.1417 | -0.3622 | 502 |
| `sft_success_task37_trial3` | 37 | 1.3342 | 1.0041 | -0.3301 | 511 |
| `sft_success_task37_trial2` | 37 | 1.3785 | 1.0538 | -0.3247 | 531 |
| `sft_success_task25_trial2` | 25 | 1.0989 | 0.9117 | -0.1872 | 979 |
| `sft_success_task25_trial0` | 25 | 1.1958 | 1.0259 | -0.1699 | 990 |


## Row Deltas: final120

### train

| Sample | Task | Base | Adapter | Delta | Target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sft_success_task27_trial2` | 27 | 1.7332 | 0.0431 | -1.6901 | 435 |
| `sft_success_task12_trial3` | 12 | 1.6518 | 0.0549 | -1.5970 | 517 |
| `sft_success_task38_trial0` | 38 | 1.4752 | 0.0144 | -1.4608 | 401 |
| `sft_success_task12_trial2` | 12 | 1.6072 | 0.1773 | -1.4299 | 897 |
| `sft_success_task12_trial1` | 12 | 1.5937 | 0.1912 | -1.4026 | 932 |
| ... worst regressions below ... |  |  |  |  |  |
| `sft_success_task1_trial1` | 1 | 1.1699 | 0.0038 | -1.1660 | 247 |
| `sft_success_task1_trial3` | 1 | 1.1151 | 0.0024 | -1.1128 | 257 |
| `sft_success_task20_trial0` | 20 | 1.0834 | 0.0083 | -1.0750 | 341 |
| `sft_success_task42_trial0` | 42 | 0.8792 | 0.0015 | -0.8778 | 243 |
| `sft_success_task42_trial2` | 42 | 0.8338 | 0.0024 | -0.8315 | 292 |

### valid

| Sample | Task | Base | Adapter | Delta | Target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sft_success_task15_trial2` | 15 | 1.3910 | 1.2492 | -0.1418 | 314 |
| `sft_success_task33_trial2` | 33 | 1.3289 | 1.5291 | +0.2002 | 810 |
| `sft_success_task33_trial1` | 33 | 1.1837 | 1.4204 | +0.2368 | 840 |
| `sft_success_task23_trial1` | 23 | 1.2302 | 1.7020 | +0.4718 | 794 |

### heldout

| Sample | Task | Base | Adapter | Delta | Target tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sft_success_task32_trial0` | 32 | 1.2509 | 0.8124 | -0.4385 | 247 |
| `sft_success_task32_trial1` | 32 | 1.2128 | 0.8296 | -0.3832 | 246 |
| `sft_success_task32_trial2` | 32 | 1.2070 | 0.8388 | -0.3682 | 245 |
| `sft_success_task18_trial2` | 18 | 0.8498 | 0.5264 | -0.3234 | 203 |
| `sft_success_task37_trial0` | 37 | 1.2019 | 0.9426 | -0.2593 | 243 |
| ... worst regressions below ... |  |  |  |  |  |
| `sft_success_task2_trial3` | 2 | 1.6389 | 1.9933 | +0.3544 | 441 |
| `sft_success_task37_trial3` | 37 | 1.3342 | 1.6981 | +0.3639 | 511 |
| `sft_success_task25_trial0` | 25 | 1.1958 | 1.6001 | +0.4043 | 990 |
| `sft_success_task37_trial2` | 37 | 1.3785 | 1.8599 | +0.4814 | 531 |
| `sft_success_task2_trial2` | 2 | 1.5331 | 2.0432 | +0.5101 | 436 |

## Interpretation

- Lower masked NLL means the model assigns higher probability to the reference assistant/tool-call tokens.
- Train improvements without valid/heldout improvements indicate memorization or distribution mismatch.
- This is an offline teacher-forced metric, not an end-to-end tau2 pass rate.
