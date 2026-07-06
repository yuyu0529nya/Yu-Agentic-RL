# Phase 1J Action-Prefix Diagnosis And 4090 Plan

## Current Status

Local 3060 validation is complete enough to justify preparing a rented-GPU run.

- Dataset route: Action-Prefix SFT, one sample per assistant tool-call turn.
- Local model: Qwen2.5-0.5B-Instruct.
- Local training: LoRA SFT, 1536-token budget, 20 steps.
- Local safety: peak CUDA memory was about 4109 MB, and no residual training process is running.

## Artifacts

- v2 dataset report: `reports/action_prefix_dataset_v2.md`
- v2 train data: `data/action_prefix/tau2_airline_action_prefix_v2_train.jsonl`
- v2 valid data: `data/action_prefix/tau2_airline_action_prefix_v2_valid.jsonl`
- v2 heldout data: `data/action_prefix/tau2_airline_action_prefix_v2_heldout.jsonl`
- v2 0.5B checkpoint: `outputs/sft_action_prefix_v2_phase1j_1536_step20/checkpoint`
- behavior report: `reports/sft_behavior_eval_phase1j_action_prefix_v2.md`
- safe NLL report: `reports/sft_adapter_eval_phase1j_action_prefix_v2_safe.md`

## Local Results

### Training

| Run | Model | Data | Max seq | Steps | Valid before | Valid after |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Action-Prefix v2 | Qwen2.5-0.5B | v2 1536 | 1536 | 20 | 2.1104 | 1.0496 |

The v2 dataset is learnable: teacher-forced validation loss drops clearly.

### Free-Form Behavior

| Model | Tool-call rate | Tool-name acc | Exact call acc |
| --- | ---: | ---: | ---: |
| base | 0.042 | 0.042 | 0.000 |
| action_prefix_v1 | 0.542 | 0.333 | 0.250 |
| action_prefix_v2 | 0.333 | 0.083 | 0.083 |

v1 remains the best local free-form behavior adapter.

### Safe Heldout NLL Probe

The timeout-prone full NLL evaluation was replaced with a safe mode using `--splits` and `--max-rows-per-split`.

| Model | Heldout rows | Weighted loss | Delta vs base |
| --- | ---: | ---: | ---: |
| base | 12 | 1.2214 | 0.0000 |
| action_prefix_v1 | 12 | 0.6409 | -0.5805 |
| action_prefix_v2 | 12 | 0.7585 | -0.4630 |

Both adapters improve teacher-forced NLL, but v1 is stronger than v2 on this local probe.

## Diagnosis

The main finding is:

> Lower teacher-forced loss does not guarantee better free-form tool-call behavior.

The likely causes are:

- v2 trims long prefixes aggressively to fit 1536 tokens, which removes some evidence needed for later tool decisions.
- The 0.5B model is too weak to reliably convert partial evidence into well-formed tool calls.
- Free-form decoding allows the model to answer in natural language instead of emitting a strict tool call.
- The dataset is still small: 60 train action-prefix samples is enough for a pipeline check, not enough for robust behavior.

## Decision

Do not keep pushing long local runs on the 3060. The local machine has done its job:

- data format validated
- mask/target validation passed
- LoRA training passed
- behavior eval exposed a real gap
- safe NLL eval now works

Next useful scale-up is a single RTX 4090 24G run with 7B QLoRA.

## 4090 First Run

Preferred command:

```bash
pip install -r requirements-4090.txt
bash scripts/run_action_prefix_sft_4090.sh
```

Default settings:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- quantization: 4-bit NF4 QLoRA
- context budget: 2048 tokens
- steps: 160
- LoRA rank: 16
- batch size: 1
- gradient checkpointing: on

Optional larger-context run:

```bash
MAX_SAMPLE_TOKENS=3072 STEPS=160 bash scripts/run_action_prefix_sft_4090.sh
```

Run 2048 first. Only try 3072 after the 2048 run confirms memory headroom.

## Success Criteria

The rented-GPU run is useful if it achieves at least one of:

- behavior tool-name accuracy beats local v1 baseline of 0.333 on the 24-probe heldout eval
- exact call accuracy beats local v1 baseline of 0.250
- heldout masked NLL improves over base by at least 40 percent without behavior collapse

If Qwen2.5-7B QLoRA still lowers NLL but does not improve behavior, the next algorithmic step is constrained tool decoding or a tool-call-only target format.

## Next Research Step After 4090

If the 7B SFT run improves behavior, move to:

- larger tau2 sampling for more action-prefix data
- PRM-rerank over sampled tool-call candidates
- preference-pair construction for DPO/IPO

If it does not improve behavior, move to:

- constrained decoding
- separating natural-language assistant content from tool-call JSON
- tool-call schema normalization and invalid-tool penalties
