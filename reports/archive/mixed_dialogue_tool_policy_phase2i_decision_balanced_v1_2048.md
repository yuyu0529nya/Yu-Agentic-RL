# Phase2I Decision-Balanced Mixed Policy Dataset v1

## Goal

Keep the Phase2H tool-call protocol gains while reducing over-calling on assistant text turns. The training split repeats assistant text targets and downsamples protocol-only tool targets; valid and heldout splits keep the original Phase2H distribution for unbiased behavior checks.

## Outputs

- train: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048_train.jsonl`
- valid: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048_valid.jsonl`
- heldout: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048_heldout.jsonl`
- train rejected: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048_train_rejected.jsonl`
- valid rejected: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048_valid_rejected.jsonl`
- heldout rejected: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048_heldout_rejected.jsonl`
- train balance dropped: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048_train_balance_dropped.jsonl`

## Configuration

- Tokenizer model: `/root/autodl-tmp/models/qwen25-7b-instruct`
- Tokenizer stats enabled: `True`
- Max sample tokens: `2048`
- Include protocol variants: `True`
- Train text repeat: `2`
- Train protocol keep ratio: `0.5`
- Balance seed: `11`
- Tool targets are still required to be grounded in the online prefix.
- Only `train` is decision-balanced; `valid` and `heldout` are not repeated or downsampled.

## Decision Mix

| Split | Stage | Rows | Text | Tool | Text share | Tool share | Sample types |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | raw clean | 192 | 80 | 112 | 0.417 | 0.583 | `mixed_policy_protocol_tool:44, mixed_policy_sequential_tool:24, mixed_policy_single_tool:44, mixed_policy_text:80` |
| train | balanced | 250 | 160 | 90 | 0.640 | 0.360 | `mixed_policy_protocol_tool:22, mixed_policy_sequential_tool:24, mixed_policy_single_tool:44, mixed_policy_text:160` |
| valid | raw clean | 82 | 34 | 48 | 0.415 | 0.585 | `mixed_policy_protocol_tool:19, mixed_policy_sequential_tool:10, mixed_policy_single_tool:19, mixed_policy_text:34` |
| heldout | raw clean | 233 | 108 | 125 | 0.464 | 0.536 | `mixed_policy_protocol_tool:43, mixed_policy_sequential_tool:39, mixed_policy_single_tool:43, mixed_policy_text:108` |

## Train Balance Stats

- Input rows: `192`
- Output rows: `250`
- Protocol rows downsampled: `22`
- Balance roles: `{'assistant_text_original': 80, 'assistant_text_repeat': 80, 'protocol_tool_kept': 22, 'tool_original': 68}`

## Token Summary

| Split | Rows | Rejected | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 250 | 27 | 1042.6 | 1975 | 2048 | 128.9 | 103 |
| valid | 82 | 16 | 1331.6 | 2012 | 2041 | 131.0 | 55 |
| heldout | 233 | 36 | 1165.4 | 1997 | 2048 | 106.2 | 106 |

## Rejections

### train

| Reason | Count |
| --- | ---: |
| `target_not_grounded` | 27 |

### valid

| Reason | Count |
| --- | ---: |
| `target_not_grounded` | 16 |

### heldout

| Reason | Count |
| --- | ---: |
| `target_not_grounded` | 36 |

## Validation

No structural/token validation errors on kept samples.

## Training Use

- Train Qwen2.5-7B QLoRA from the base model on the balanced train split.
- Run mixed-policy behavior eval before any full tau2 pass.
- Desired behavior movement: keep tool-name/argument exact accuracy close to Phase2H while raising text no-tool rate.
