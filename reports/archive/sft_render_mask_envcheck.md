# SFT Render/Mask Check v1

## Summary

| Metric | Value |
| --- | ---: |
| Input rows | 32 |
| Checked rows | 3 |
| Rows with errors | 0 |
| Rows with warnings | 3 |
| Tokenizer status | `transformers_loaded` |
| Official template ok | 3 |
| Official template failed | 0 |

## Lengths

| Render | Mean | P50 | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| chars | 11007.7 | 10646 | 11846 | 11846 |
| tokens | 3487.7 | 3410 | 3649 | 3649 |

## Loss Target Coverage

| Metric | Value |
| --- | ---: |
| Mean target char ratio | 0.3521 |
| Min target char ratio | 0.3318 |
| Max target char ratio | 0.3903 |
| Mean target token ratio | 0.2511 |

## Error And Warning Types

| Type | Count |
| --- | ---: |
| `conversation_starts_with_assistant` | 3 |

## Per-row Check

| Sample | Task | Split | Msgs | Target spans | Chars | Errors | Warnings |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `sft_success_task1_trial1` | 1 | train | 16 | 12 | 10646 | - | `conversation_starts_with_assistant` |
| `sft_success_task1_trial2` | 1 | train | 18 | 13 | 11846 | - | `conversation_starts_with_assistant` |
| `sft_success_task1_trial3` | 1 | train | 16 | 12 | 10531 | - | `conversation_starts_with_assistant` |

## Notes

- This v1 check always runs a project-native ChatML-style renderer.
- It first tries `transformers`; if that is unavailable or broken, it falls back to `huggingface_hub` + `tokenizers` + Jinja for Qwen tokenizer/template checks.
- Token-level label spans are computed on the project-native rendered text; official Qwen `chat_template` rendering is checked for compatibility.
- The `conversation_starts_with_assistant` warning is expected for tau2 because the agent opens the conversation.
