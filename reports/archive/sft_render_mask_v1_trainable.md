# SFT Render/Mask Check v1

## Summary

| Metric | Value |
| --- | ---: |
| Input rows | 32 |
| Checked rows | 18 |
| Rows with errors | 0 |
| Rows with warnings | 18 |
| Tokenizer status | `direct_tokenizers_loaded_after_transformers_import_failed:ImportError:Error importing numpy: you should not try to import numpy from         its source directory; please exit the numpy source tree, and relaunch         your python interpreter from there.` |
| Official template ok | 18 |
| Official template failed | 0 |

## Lengths

| Render | Mean | P50 | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| chars | 15075.7 | 11916 | 22567 | 34336 |
| tokens | 4845.5 | 3649 | 7093 | 11812 |

## Loss Target Coverage

| Metric | Value |
| --- | ---: |
| Mean target char ratio | 0.3952 |
| Min target char ratio | 0.2170 |
| Max target char ratio | 0.5639 |
| Mean target token ratio | 0.3382 |

## Error And Warning Types

| Type | Count |
| --- | ---: |
| `conversation_starts_with_assistant` | 18 |

## Per-row Check

| Sample | Task | Split | Msgs | Target spans | Chars | Errors | Warnings |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `sft_success_task1_trial1` | 1 | train | 16 | 12 | 10646 | - | `conversation_starts_with_assistant` |
| `sft_success_task1_trial2` | 1 | train | 18 | 13 | 11846 | - | `conversation_starts_with_assistant` |
| `sft_success_task1_trial3` | 1 | train | 16 | 12 | 10531 | - | `conversation_starts_with_assistant` |
| `sft_success_task12_trial1` | 12 | train | 22 | 15 | 11799 | - | `conversation_starts_with_assistant` |
| `sft_success_task12_trial2` | 12 | train | 24 | 15 | 15298 | - | `conversation_starts_with_assistant` |
| `sft_success_task12_trial3` | 12 | train | 20 | 12 | 11588 | - | `conversation_starts_with_assistant` |
| `sft_success_task15_trial2` | 15 | train | 25 | 16 | 22567 | - | `conversation_starts_with_assistant` |
| `sft_success_task20_trial0` | 20 | train | 32 | 16 | 14409 | - | `conversation_starts_with_assistant` |
| `sft_success_task23_trial1` | 23 | train | 38 | 21 | 34336 | - | `conversation_starts_with_assistant` |
| `sft_success_task27_trial2` | 27 | train | 32 | 18 | 16904 | - | `conversation_starts_with_assistant` |
| `sft_success_task33_trial1` | 33 | train | 32 | 19 | 16438 | - | `conversation_starts_with_assistant` |
| `sft_success_task33_trial2` | 33 | train | 42 | 26 | 23046 | - | `conversation_starts_with_assistant` |
| `sft_success_task34_trial1` | 34 | train | 16 | 9 | 9752 | - | `conversation_starts_with_assistant` |
| `sft_success_task34_trial2` | 34 | train | 16 | 9 | 10302 | - | `conversation_starts_with_assistant` |
| `sft_success_task34_trial3` | 34 | train | 16 | 9 | 10100 | - | `conversation_starts_with_assistant` |
| `sft_success_task38_trial0` | 38 | train | 30 | 18 | 11916 | - | `conversation_starts_with_assistant` |
| `sft_success_task42_trial0` | 42 | train | 23 | 16 | 14549 | - | `conversation_starts_with_assistant` |
| `sft_success_task42_trial2` | 42 | train | 25 | 17 | 15335 | - | `conversation_starts_with_assistant` |

## Notes

- This v1 check always runs a project-native ChatML-style renderer.
- It first tries `transformers`; if that is unavailable or broken, it falls back to `huggingface_hub` + `tokenizers` + Jinja for Qwen tokenizer/template checks.
- Token-level label spans are computed on the project-native rendered text; official Qwen `chat_template` rendering is checked for compatibility.
- The `conversation_starts_with_assistant` warning is expected for tau2 because the agent opens the conversation.
