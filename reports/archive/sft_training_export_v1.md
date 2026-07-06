# SFT Training Export v1

## Source

- Input SFT data: `data/sft/tau2_airline_sft_v1.jsonl`
- Render/mask report: `reports/sft_render_mask_v1.json`
- Export format: `tau2_airline_sft_training_v1`
- Training rows include `messages`, `loss_mask`, `loss_policy`, and metadata only.
- Full task/evaluation criteria are intentionally not exported into training rows.

## Outputs

- train: `data/train/tau2_airline_sft_train.jsonl`
- valid: `data/train/tau2_airline_sft_valid.jsonl`
- heldout: `data/train/tau2_airline_sft_heldout.jsonl`

## Split Summary

| Split | Rows | Tasks | Mean tokens | P90 | Max | Target ratio | Tool calls | Write calls |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 14 | `1:3, 12:3, 20:1, 27:1, 34:3, 38:1, 42:2` | 3894.3 | 5388 | 5457 | 0.3372 | 86 | 8 |
| valid | 4 | `15:1, 23:1, 33:2` | 8174.8 | 11812 | 11812 | 0.3416 | 37 | 13 |
| heldout | 14 | `2:2, 16:1, 18:1, 25:4, 32:3, 37:3` | 4912.2 | 6576 | 8189 | 0.3667 | 101 | 22 |

## Validation Split

- Validation task ids: `15, 23, 33`
- Validation is grouped by task id, so no task appears in both train and valid.
- Official test tasks are exported only to heldout.

## Sequence Length Recommendation

- Max Qwen2.5 token count across exported splits: `11812`
- Recommended first SFT `max_seq_len`: `16384`
- A 4K context would truncate several trajectories. Use 8K only for filtered short experiments; use 16K for the full first SFT run.

## Validation

No structural export errors.

## Training Notes

- Use only `train.jsonl` for fitting.
- Use `valid.jsonl` for loss/format monitoring.
- Use `heldout.jsonl` for offline inspection or post-training evaluation, not fitting.
- Loss should be applied only where `loss_mask` is true: assistant content and assistant tool calls.
- The current Windows global Python is 32-bit and has a broken NumPy import; create a fresh 64-bit training environment before running SFT.
