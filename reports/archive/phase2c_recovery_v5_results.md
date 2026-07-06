# Phase 2C Recovery-Prefix v5 Results

## Summary

Recovery-Prefix v5 successfully trained, but did not improve targeted tau2 task success.

The important result is diagnostic: targeted SFT improved offline next-action behavior, yet full interactive tau2 still failed because the model copied or invented the wrong task slot values at runtime.

## Artifacts

- Local artifact folder: `autodl_artifacts/phase2c_recovery_v5_4090_20260615_1113`
- Adapter: `autodl_artifacts/phase2c_recovery_v5_4090_20260615_1113/outputs__sft_recovery_prefix_v5_qwen25_7b_qlora_2048__checkpoint__adapter_model.safetensors`
- Training report: `autodl_artifacts/phase2c_recovery_v5_4090_20260615_1113/reports__sft_recovery_prefix_v5_qwen25_7b_qlora_2048.md`
- Behavior report: `autodl_artifacts/phase2c_recovery_v5_4090_20260615_1113/reports__behavior_sft_recovery_prefix_v5_qwen25_7b_qlora_2048.md`
- Tau2 target results: `autodl_artifacts/phase2c_recovery_v5_4090_20260615_1113/third_party__tau2-bench__data__simulations__airline_qwen25_7b_sft_phase2c_recovery_v5_target_3task_4090_20260615_111020__results.json`

## Training

- Model: `Qwen2.5-7B-Instruct`
- Method: 4-bit QLoRA SFT
- Adapter: `sft_recovery_prefix_v5_qwen25_7b_qlora_2048`
- Steps: `650`
- Trainable params: `40,370,176`
- Selected train examples under 2048 tokens: `811`
- Valid examples: `62`
- Max CUDA memory: `13,398 MB` reported by the trainer
- Wall time: `579.8s`
- Valid loss: `3.4027 -> 0.5078`

## Offline Behavior Eval

Heldout next-action behavior improved over base:

| Model | Tool-call rate | Single-call rate | Tool-name acc | Arg exact acc | Clean single exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.031 | 0.031 | 0.031 | 0.031 | 0.031 |
| Recovery-Prefix v5 | 0.438 | 0.438 | 0.281 | 0.156 | 0.156 |

Interpretation: v5 learned to emit executable single tool calls more often, but the free-form policy is still not reliable enough for full tau2 task success.

## Tau2 Targeted Eval

Run: `phase2c_recovery_v5_target_3task_4090_20260615_111020`

Configuration:

- Agent: `Qwen2.5-7B + Recovery-Prefix v5 LoRA`
- User simulator: `GLM-5.1`
- Tasks: `18,25,44`
- Trials: `1`
- `MAX_MODEL_LEN=12288`
- `AGENT_MAX_TOKENS=128`
- Stop sequence: `</tool_call>`
- Result: `pass^1 = 0.0000`

| Task | Reward | DB | Action reward | Main observed failure |
| ---: | ---: | --- | --- | --- |
| 18 | 0.0 | mismatch | 0/5 | Model hallucinated wrong user id `sara_doe_496`, then transferred instead of using the provided user id. |
| 25 | 0.0 | mismatch | 0/1 | Model repeated identity/reservation verification and never called `book_reservation`. |
| 44 | 0.0 | mismatch | 0/19 | Model hallucinated wrong user id `sara_doe_496`, then transferred before inspecting reservations or searching durations. |

## Diagnosis

v5 fixed part of the local next-action problem but not the full interactive control problem.

The failures are now less about tool-call syntax and more about state grounding:

- The model can learn a desired tool family offline.
- In multi-turn tau2, it still fails to copy the current user's concrete identifiers.
- Once the first slot is wrong, the environment returns `not found`, and the model falls into transfer/refusal behavior.
- For task 25, the model has a verification loop: it asks for evidence that is already present instead of committing to the booking action after confirmation.

## Decision

Do not continue blindly scaling the same Recovery-Prefix SFT.

Next algorithm step should be one of:

1. `Slot-Grounded Action Decoder`: before each tool call, constrain or validate arguments against values present in dialogue/tool state.
2. `PRM Rerank`: sample multiple trajectories and rank them with explicit checks for copied user id, reservation id coverage, payment arithmetic, and write-after-confirmation.
3. `Tool-call constrained decoding`: restrict generation to one valid JSON tool call with schema and argument post-validation, instead of relying on free-form chat generation.

Recommended next move: implement a lightweight slot validator plus PRM-rerank for tasks `18,25,44`, because this directly targets the new failure mode without spending another training run first.
