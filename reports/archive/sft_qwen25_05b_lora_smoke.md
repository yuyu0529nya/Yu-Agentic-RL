# SFT Smoke Test v1

## Status

- Status: `OK`
- Model init: `pretrained`
- Tokenizer: `.\models\Qwen2.5-0.5B-Instruct`
- Device: `cuda`
- Max sequence length: `512`
- Total parameters: `496232320`
- Trainable parameters: `2199552`
- Output dir: `outputs/sft_qwen25_05b_lora_smoke`
- LoRA: `r=4, alpha=8, dropout=0.05`
- LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

## Data

| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| train | 2 | `12` | 512.0 | 512 | 110.0 | 2 |
| valid | 1 | `15` | 512.0 | 512 | 106.0 | 1 |

## Training

- Steps: `4`
- First train loss: `1.8644`
- Final train loss: `1.7969`
- Valid loss before: `1.5887`
- Valid loss after: `1.4308`
- Elapsed seconds: `2.2`
- Max CUDA memory MB: `2053.10693359375`

## Loss Trace

```text
1: 1.8644
2: 1.9947
3: 1.5359
4: 1.7969
```

## Notes

- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.
- This run uses real pretrained model weights, not a random toy model.
- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.
