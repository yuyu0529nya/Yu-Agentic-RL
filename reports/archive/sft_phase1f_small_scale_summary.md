# Phase 1F Small-Scale SFT Summary

## Goal

Run a real pretrained model SFT experiment after the single-sample mask overfit gold test. This phase checks whether the current SFT data, assistant-only labels, Qwen model loading, LoRA training, validation monitoring, and checkpoint saving work together on the full v1 train split.

## Setup

| Item | Value |
| --- | --- |
| Base model | `Qwen2.5-0.5B-Instruct` |
| Training method | LoRA SFT |
| Train rows | 14 |
| Valid rows | 4 |
| Context length | 2048 |
| LoRA | `r=8, alpha=16, dropout=0.05` |
| Trainable params | 4,399,104 / 498,431,872 |
| GPU | RTX 3060 Laptop GPU, 6GB |
| Max CUDA memory | ~5266 MB |

All train and valid examples were truncated to 2048 tokens. This is a local-resource experiment, not the final long-trajectory setting.

## Runs

| Run | Steps | Train Loss | Valid Loss | Best Valid | Checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| Long run | 120 | 1.1699 -> 0.0133 | 1.2835 -> 1.4752 | 0.9091 @ step 20 | `outputs/sft_qwen25_05b_lora_phase1f_2k/checkpoint` |
| Early-stop run | 20 | 1.1699 -> 0.2833 | 1.2835 -> 0.9649 | 0.9649 @ step 20 | `outputs/sft_qwen25_05b_lora_phase1f_2k_best_step20/checkpoint` |

## Interpretation

- The model learns the training samples quickly: train loss drops from about 1.17 to near 0.01 in the 120-step run.
- Validation improves early, then degrades: the long run reaches best validation near step 20, then overfits.
- The saved early-stop checkpoint is the safer local adapter for follow-up inspection.
- The overfitting is expected: v1 has only 14 train trajectories, all truncated to 2K, and the valid split contains different task families.

## Artifacts

- Full 120-step report: `reports/sft_qwen25_05b_lora_phase1f_2k.md`
- Early-stop report: `reports/sft_qwen25_05b_lora_phase1f_2k_best_step20.md`
- Gold mask report: `reports/sft_mask_overfit_gold.md`
- Early-stop adapter: `outputs/sft_qwen25_05b_lora_phase1f_2k_best_step20/checkpoint`
- Full-run adapter: `outputs/sft_qwen25_05b_lora_phase1f_2k/checkpoint`

## Next

The next useful step is not just more training. We should evaluate the early-stop adapter on heldout trajectories or build an inference wrapper that loads the LoRA adapter and compares base vs SFT behavior on a few airline prompts. For a final long-trajectory SFT, use 8K-16K context on a rented GPU.
