# Phase2F Single-Tool Protocol v4 4090 Results

## Status

OK. Ran Qwen2.5-7B-Instruct 4-bit QLoRA SFT on AutoDL RTX 4090D for Single-Tool Protocol v4.

Run time: 2026-06-15 22:47-22:52 CST  
Remote GPU: NVIDIA GeForce RTX 4090 D, 24 GB  
Remote shutdown: not requested; instance left running.

## Goal

Phase2E Slot-Grounded v3 proved that action-prefix SFT can teach the model to emit useful tool-call content, but it still failed the runtime protocol boundary:

- Protocol wrapper rate: 0.000
- Single-call rate: 0.083
- Multi-call rate: 0.625
- Tool-name accuracy: 0.708

Phase2F v4 changes the supervision target to exactly one executable `<tool_call>...</tool_call>` span per sample.

## Dataset

Single-tool protocol v4, max sequence length 1536:

| Split | Kept rows | Rejected rows | Notes |
| --- | ---: | ---: | --- |
| train | 42 | 7 | rejected multi-call targets |
| valid | 15 | 5 | rejected multi-call targets |
| heldout | 38 | 13 | behavior eval sampled 32 probes |

Validation errors: 0

## Training

| Metric | Value |
| --- | ---: |
| Steps | 160 |
| LoRA trainable params | 40,370,176 |
| First train loss | 7.3060 |
| Final train loss | 0.0001 |
| Valid loss before | 1.9999 |
| Valid loss after | 0.2489 |
| Elapsed seconds | 136.0 |
| Peak CUDA memory | 12,024.8 MB |

The training objective is healthy: valid loss drops by 87.6%, and peak memory is comfortably within 4090D capacity.

## Heldout Behavior

32 heldout next-tool-call probes, deterministic generation.

| Model | Tool-call rate | Protocol wrapper rate | Single-call rate | Multi-call rate | Tool-name acc | Arg exact acc | Clean single exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen2.5-7B | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| v4 SFT | 0.875 | 0.875 | 0.875 | 0.000 | 0.750 | 0.750 | 0.750 |

## Interpretation

This is a strong positive result. Phase2F directly fixes the main Phase2E failure mode:

- v3 learned tool content but not executable protocol.
- v4 learns executable wrapper and one-call boundary.
- v4 removes multi-call over-generation on this proxy eval.

The remaining failures are concentrated in harder decision points:

- some turn-0 prompts still produce generic text instead of a tool call;
- task 18 late-stage write/calculate decisions are still confused;
- one task 25 search case chooses a plausible but wrong evidence-gathering action.

So v4 is not the final agent yet, but it is a real algorithmic improvement over v3 and a much better SFT base for full tau2 rollout.

## Artifacts

- Training report: `reports/sft_single_tool_protocol_v4_qwen25_7b_qlora_1536.md`
- Behavior report: `reports/behavior_sft_single_tool_protocol_v4_qwen25_7b_qlora_1536.md`
- Pulled artifact directory: `autodl_artifacts/phase2f_single_tool_protocol_v4_4090_20260615_224720`
- Full remote artifact tarball: `autodl_artifacts/phase2f_single_tool_protocol_v4_4090_20260615_224720/yuyu_single_tool_protocol_v4_4090_20260615_224720.tar.gz`

## Next Step

Run full tau2 rollout comparison:

- Base Qwen2.5-7B pass^1 on target tasks.
- Slot-Grounded v3 pass^1 on the same tasks.
- Single-Tool Protocol v4 pass^1 on the same tasks.

If v4 improves full pass^1, scale dataset sampling. If behavior improves but pass^1 does not, the next algorithm step should be constrained tool decoding or mixed SFT with natural assistant turns, not blindly adding more epochs.
