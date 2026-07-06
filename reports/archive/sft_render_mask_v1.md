# SFT Render/Mask Check v1

## Summary

| Metric | Value |
| --- | ---: |
| Input rows | 32 |
| Checked rows | 32 |
| Rows with errors | 0 |
| Rows with warnings | 32 |
| Tokenizer status | `direct_tokenizers_loaded_after_transformers_import_failed:ImportError:Error importing numpy: you should not try to import numpy from         its source directory; please exit the numpy source tree, and relaunch         your python interpreter from there.` |
| Official template ok | 32 |
| Official template failed | 0 |

## Lengths

| Render | Mean | P50 | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| chars | 15236.6 | 14409 | 22567 | 34336 |
| tokens | 4874.7 | 4447 | 7093 | 11812 |

## Loss Target Coverage

| Metric | Value |
| --- | ---: |
| Mean target char ratio | 0.4043 |
| Min target char ratio | 0.2170 |
| Max target char ratio | 0.5639 |
| Mean target token ratio | 0.3507 |

## Error And Warning Types

| Type | Count |
| --- | ---: |
| `conversation_starts_with_assistant` | 32 |

## Per-row Check

| Sample | Task | Split | Msgs | Target spans | Chars | Errors | Warnings |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `sft_success_task1_trial1` | 1 | train | 16 | 12 | 10646 | - | `conversation_starts_with_assistant` |
| `sft_success_task1_trial2` | 1 | train | 18 | 13 | 11846 | - | `conversation_starts_with_assistant` |
| `sft_success_task1_trial3` | 1 | train | 16 | 12 | 10531 | - | `conversation_starts_with_assistant` |
| `sft_success_task2_trial2` | 2 | test | 32 | 21 | 16366 | - | `conversation_starts_with_assistant` |
| `sft_success_task2_trial3` | 2 | test | 28 | 17 | 13025 | - | `conversation_starts_with_assistant` |
| `sft_success_task12_trial1` | 12 | train | 22 | 15 | 11799 | - | `conversation_starts_with_assistant` |
| `sft_success_task12_trial2` | 12 | train | 24 | 15 | 15298 | - | `conversation_starts_with_assistant` |
| `sft_success_task12_trial3` | 12 | train | 20 | 12 | 11588 | - | `conversation_starts_with_assistant` |
| `sft_success_task15_trial2` | 15 | train | 25 | 16 | 22567 | - | `conversation_starts_with_assistant` |
| `sft_success_task16_trial0` | 16 | test | 20 | 14 | 15740 | - | `conversation_starts_with_assistant` |
| `sft_success_task18_trial2` | 18 | test | 31 | 19 | 18172 | - | `conversation_starts_with_assistant` |
| `sft_success_task20_trial0` | 20 | train | 32 | 16 | 14409 | - | `conversation_starts_with_assistant` |
| `sft_success_task23_trial1` | 23 | train | 38 | 21 | 34336 | - | `conversation_starts_with_assistant` |
| `sft_success_task25_trial0` | 25 | test | 22 | 11 | 10055 | - | `conversation_starts_with_assistant` |
| `sft_success_task25_trial1` | 25 | test | 22 | 12 | 11360 | - | `conversation_starts_with_assistant` |
| `sft_success_task25_trial2` | 25 | test | 24 | 12 | 10407 | - | `conversation_starts_with_assistant` |
| `sft_success_task25_trial3` | 25 | test | 22 | 12 | 11651 | - | `conversation_starts_with_assistant` |
| `sft_success_task27_trial2` | 27 | train | 32 | 18 | 16904 | - | `conversation_starts_with_assistant` |
| `sft_success_task32_trial0` | 32 | test | 45 | 28 | 27538 | - | `conversation_starts_with_assistant` |
| `sft_success_task32_trial1` | 32 | test | 33 | 21 | 20428 | - | `conversation_starts_with_assistant` |
| `sft_success_task32_trial2` | 32 | test | 29 | 19 | 17452 | - | `conversation_starts_with_assistant` |
| `sft_success_task33_trial1` | 33 | train | 32 | 19 | 16438 | - | `conversation_starts_with_assistant` |
| `sft_success_task33_trial2` | 33 | train | 42 | 26 | 23046 | - | `conversation_starts_with_assistant` |
| `sft_success_task34_trial1` | 34 | train | 16 | 9 | 9752 | - | `conversation_starts_with_assistant` |
| `sft_success_task34_trial2` | 34 | train | 16 | 9 | 10302 | - | `conversation_starts_with_assistant` |
| `sft_success_task34_trial3` | 34 | train | 16 | 9 | 10100 | - | `conversation_starts_with_assistant` |
| `sft_success_task37_trial0` | 37 | test | 24 | 17 | 16482 | - | `conversation_starts_with_assistant` |
| `sft_success_task37_trial2` | 37 | test | 28 | 18 | 14171 | - | `conversation_starts_with_assistant` |
| `sft_success_task37_trial3` | 37 | test | 30 | 17 | 13362 | - | `conversation_starts_with_assistant` |
| `sft_success_task38_trial0` | 38 | train | 30 | 18 | 11916 | - | `conversation_starts_with_assistant` |
| `sft_success_task42_trial0` | 42 | train | 23 | 16 | 14549 | - | `conversation_starts_with_assistant` |
| `sft_success_task42_trial2` | 42 | train | 25 | 17 | 15335 | - | `conversation_starts_with_assistant` |

## Notes

- This v1 check always runs a project-native ChatML-style renderer.
- It first tries `transformers`; if that is unavailable or broken, it falls back to `huggingface_hub` + `tokenizers` + Jinja for Qwen tokenizer/template checks.
- Token-level label spans are computed on the project-native rendered text; official Qwen `chat_template` rendering is checked for compatibility.
- The `conversation_starts_with_assistant` warning is expected for tau2 because the agent opens the conversation.
