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
- Output dir: `outputs/slot_grounded_action_prefix_v3_local_smoke`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 2 | `12` | 114.5 | 120 | 47.0 | 0 |
| valid | 1 | `23` | 41.0 | 41 | 32.0 | 0 |

## Training

- Steps: `2`
- First train loss: `11.8873`
- Final train loss: `11.8286`
- Min train loss: `11.8286`
- Max train loss: `11.8873`
- Final rolling mean: `11.8579`
- Valid loss before: `11.8731`
- Valid loss after: `11.8413`
- Elapsed seconds: `0.3`
- Max CUDA memory MB: `None`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 11.8731 |
| 1 | 11.8527 |
| 2 | 11.8413 |

## Loss Trace

```text
1: 11.8873
2: 11.8286
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- The `tiny-random` model is intentionally small and randomly initialized; it is not a quality result.
- Use `--model-init pretrained --pretrained-model Qwen/Qwen2.5-0.5B-Instruct` for the next realism check.
