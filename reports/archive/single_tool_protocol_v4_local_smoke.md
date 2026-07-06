# SFT Smoke Test v1

## Status

- Status: `OK`
- Model init: `tiny-random`
- Tokenizer: `models\Qwen2.5-0.5B-Instruct`
- Device: `cpu`
- Max sequence length: `1536`
- Train sample ids: `None`
- Valid sample ids: `None`
- Shuffle: `True`
- Total parameters: `9904960`
- Trainable parameters: `9904960`
- Output dir: `outputs/single_tool_protocol_v4_local_smoke`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 2 | `12` | 84.5 | 85 | 20.0 | 0 |
| valid | 1 | `23` | 41.0 | 41 | 35.0 | 0 |

## Training

- Steps: `2`
- First train loss: `11.9737`
- Final train loss: `11.8497`
- Min train loss: `11.8497`
- Max train loss: `11.9737`
- Final rolling mean: `11.9117`
- Valid loss before: `11.8781`
- Valid loss after: `11.8318`
- Elapsed seconds: `0.5`
- Max CUDA memory MB: `None`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 11.8781 |
| 1 | 11.8475 |
| 2 | 11.8318 |

## Loss Trace

```text
1: 11.9737
2: 11.8497
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- The `tiny-random` model is intentionally small and randomly initialized; it is not a quality result.
- Use `--model-init pretrained --pretrained-model Qwen/Qwen2.5-0.5B-Instruct` for the next realism check.
