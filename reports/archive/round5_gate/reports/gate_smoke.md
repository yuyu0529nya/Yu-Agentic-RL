# SFT Small-Scale Train v1

## Status

- Status: `OK`
- Model init: `pretrained`
- Tokenizer: `$HOME/agentic-rl/models/qwen25-7b-instruct`
- Device: `cuda`
- Max sequence length: `2048`
- Train sample ids: `None`
- Valid sample ids: `None`
- Shuffle: `True`
- Total parameters: `7655986688`
- Trainable parameters: `40370176`
- Output dir: `outputs/gate_smoke`
- LoRA: `r=16, alpha=32, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 81 | `1, 12, 20, 27, 34, 38, 42` | 910.4 | 1998 | 2.0 | 0 |
| valid | 20 | `15, 23, 33` | 975.3 | 2043 | 2.0 | 0 |

## Training

- Steps: `20`
- First train loss: `14.7844`
- Final train loss: `3.6895`
- Min train loss: `2.2342`
- Max train loss: `25.1009`
- Final rolling mean: `6.4044`
- Valid loss before: `19.0046`
- Valid loss after: `1.8056`
- Elapsed seconds: `9.1`
- Max CUDA memory MB: `19406.73974609375`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 19.0046 |
| 20 | 1.8056 |

## Loss Trace

```text
1: 14.7844
2: 17.9503
3: 17.6338
4: 25.1009
5: 17.7511
6: 22.3185
7: 14.9652
8: 20.7455
9: 10.1628
10: 17.9250
11: 13.1485
12: 9.7531
13: 7.3701
14: 2.2342
15: 8.0560
16: 3.3787
17: 4.0191
18: 8.3077
19: 4.0873
20: 3.6895
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
