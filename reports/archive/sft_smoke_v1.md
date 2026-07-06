# SFT Smoke Test v1

## Status

- Status: `OK`
- Model init: `tiny-random`
- Tokenizer: `Qwen/Qwen2.5-7B-Instruct`
- Device: `cuda`
- Max sequence length: `1024`
- Output dir: `outputs/sft_smoke`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 4 | `1, 12` | 1024.0 | 1024 | 238.2 | 4 |
| valid | 2 | `15, 23` | 1024.0 | 1024 | 230.0 | 2 |

## Training

- Steps: `8`
- First train loss: `11.9461`
- Final train loss: `11.6952`
- Valid loss before: `11.9414`
- Valid loss after: `11.7530`
- Elapsed seconds: `0.7`
- Max CUDA memory MB: `2517.89453125`

## Loss Trace

```text
1: 11.9461
2: 11.9125
3: 11.8615
4: 11.8379
5: 11.7520
6: 11.7591
7: 11.6286
8: 11.6952
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- The default `tiny-random` model is intentionally small and randomly initialized; it is not a quality result.
- Use `--model-init pretrained --pretrained-model Qwen/Qwen2.5-0.5B-Instruct` for the next realism check.
