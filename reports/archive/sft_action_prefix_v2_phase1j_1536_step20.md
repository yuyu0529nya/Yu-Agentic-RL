# SFT Small-Scale Train v1

## Status

- Status: `OK`
- Model init: `pretrained`
- Tokenizer: `.\models\Qwen2.5-0.5B-Instruct`
- Device: `cuda`
- Max sequence length: `1536`
- Train sample ids: `None`
- Valid sample ids: `None`
- Shuffle: `True`
- Total parameters: `498431872`
- Trainable parameters: `4399104`
- Output dir: `outputs/sft_action_prefix_v2_phase1j_1536_step20`
- LoRA: `r=8, alpha=16, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 60 | `1, 12, 20, 27, 34, 38, 42` | 872.4 | 1528 | 62.0 | 0 |
| valid | 32 | `15, 23, 33` | 965.9 | 1488 | 91.6 | 0 |

## Training

- Steps: `20`
- First train loss: `1.3406`
- Final train loss: `1.0620`
- Min train loss: `0.0867`
- Max train loss: `2.5777`
- Final rolling mean: `1.2006`
- Valid loss before: `2.1104`
- Valid loss after: `1.0496`
- Elapsed seconds: `44.8`
- Max CUDA memory MB: `4109.349609375`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 2.1104 |
| 10 | 1.4272 |
| 20 | 1.0496 |

## Loss Trace

```text
1: 1.3406
2: 2.5777
3: 1.2954
4: 1.4644
5: 0.4132
6: 0.5338
7: 1.6308
8: 0.4891
9: 1.7366
10: 1.3629
11: 1.0315
12: 1.2078
13: 0.0867
14: 1.2062
15: 1.1708
16: 1.5908
17: 0.8572
18: 0.9720
19: 1.5211
20: 1.0620
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
