# SFT Smoke Test v1

## Status

- Status: `OK`
- Model init: `pretrained`
- Tokenizer: `.\models\Qwen2.5-0.5B-Instruct`
- Device: `cuda`
- Max sequence length: `1024`
- Total parameters: `496232320`
- Trainable parameters: `2199552`
- Output dir: `outputs/sft_qwen25_05b_lora_1024`
- LoRA: `r=4, alpha=8, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 4 | `1, 12` | 1024.0 | 1024 | 238.2 | 4 |
| valid | 2 | `15, 23` | 1024.0 | 1024 | 230.0 | 2 |

## Training

- Steps: `8`
- First train loss: `1.5422`
- Final train loss: `1.5171`
- Valid loss before: `1.2879`
- Valid loss after: `1.1069`
- Elapsed seconds: `5.4`
- Max CUDA memory MB: `3116.62451171875`

## Loss Trace

```text
1: 1.5422
2: 1.1233
3: 1.6906
4: 1.9168
5: 1.2858
6: 1.8407
7: 0.9333
8: 1.5171
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
