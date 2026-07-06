# SFT Small-Scale Train v1

## Status

- Status: `OK`
- Model init: `pretrained`
- Tokenizer: `.\models\Qwen2.5-0.5B-Instruct`
- Device: `cuda`
- Max sequence length: `2048`
- Train sample ids: `None`
- Valid sample ids: `None`
- Shuffle: `True`
- Total parameters: `498431872`
- Trainable parameters: `4399104`
- Output dir: `outputs/sft_qwen25_05b_lora_phase1f_2k_best_step20`
- LoRA: `r=8, alpha=16, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 14 | `1, 12, 20, 27, 34, 38, 42` | 2048.0 | 2048 | 467.9 | 14 |
| valid | 4 | `15, 23, 33` | 2048.0 | 2048 | 689.5 | 4 |

## Training

- Steps: `20`
- First train loss: `1.1699`
- Final train loss: `0.2833`
- Min train loss: `0.2833`
- Max train loss: `1.4948`
- Final rolling mean: `0.9309`
- Valid loss before: `1.2835`
- Valid loss after: `0.9649`
- Elapsed seconds: `59.5`
- Max CUDA memory MB: `5266.33154296875`

## Eval Trace

| Step | Valid loss |
| ---: | ---: |
| 0 | 1.2835 |
| 10 | 1.0097 |
| 20 | 0.9649 |

## Loss Trace

```text
1: 1.1699
2: 0.7297
3: 1.1509
4: 1.0895
5: 1.4948
6: 0.9346
7: 1.1782
8: 1.3477
9: 0.4485
10: 0.7897
11: 1.4155
12: 1.3658
13: 0.5183
14: 0.4880
15: 1.2577
16: 0.9198
17: 1.2964
18: 1.1185
19: 0.6461
20: 0.2833
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
