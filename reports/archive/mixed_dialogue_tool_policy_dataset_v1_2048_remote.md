# Mixed Dialogue Tool Policy Dataset v1

## Goal

Train the agent on full assistant policy decisions, not only next tool-call protocol. The dataset includes natural assistant text, grounded single-tool calls, and sequentialized single-tool targets converted from gold parallel tool-call turns.

## Outputs

- train: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_train.jsonl`
- valid: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_valid.jsonl`
- heldout: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_heldout.jsonl`
- train rejected: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_train_rejected.jsonl`
- valid rejected: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_valid_rejected.jsonl`
- heldout rejected: `data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_heldout_rejected.jsonl`

## Configuration

- Tokenizer model: `/root/autodl-tmp/models/qwen25-7b-instruct`
- Tokenizer stats enabled: `True`
- Max sample tokens: `2048`
- Include protocol variants: `True`
- Tool targets are required to be grounded in the online prefix.
- Parallel gold tool turns are converted to sequential one-call targets.

## Summary

| Split | Rows | Rejected | Target actions | Sample types | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | 192 | 27 | `assistant_text:80, tool_call:112` | `mixed_policy_protocol_tool:44, mixed_policy_sequential_tool:24, mixed_policy_single_tool:44, mixed_policy_text:80` | 1030.9 | 1975 | 2048 | 97.9 | 63 |
| valid | 82 | 16 | `assistant_text:34, tool_call:48` | `mixed_policy_protocol_tool:19, mixed_policy_sequential_tool:10, mixed_policy_single_tool:19, mixed_policy_text:34` | 1331.6 | 2012 | 2041 | 131.0 | 55 |
| heldout | 233 | 36 | `assistant_text:108, tool_call:125` | `mixed_policy_protocol_tool:43, mixed_policy_sequential_tool:39, mixed_policy_single_tool:43, mixed_policy_text:108` | 1165.4 | 1997 | 2048 | 106.2 | 106 |

## Tool Coverage

| Split | Tools | Source call counts |
| --- | --- | --- |
| train | `cancel_reservation:4, get_flight_status:10, get_reservation_details:42, get_user_details:28, list_all_airports:2, search_direct_flight:18, search_onestop_flight:2, transfer_to_human_agents:4, update_reservation_baggages:2` | `0:80, 1:88, 2:4, 5:12, 7:8` |
| valid | `cancel_reservation:2, get_reservation_details:4, get_user_details:8, search_direct_flight:14, search_onestop_flight:6, update_reservation_baggages:4, update_reservation_flights:10` | `0:34, 1:38, 2:10` |
| heldout | `book_reservation:4, calculate:8, get_flight_status:8, get_reservation_details:45, get_user_details:28, search_direct_flight:23, search_onestop_flight:1, update_reservation_flights:8` | `0:108, 1:86, 2:12, 3:3, 4:20, 6:4` |

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

- This is the Phase2H candidate dataset after Phase2G showed no full tau2 gain from protocol-only v4.
- Train Qwen2.5-7B QLoRA with this dataset, then run mixed-policy behavior eval before another full tau2 pass.
- Do not call the phase successful unless full tau2 pass^1 improves, even if teacher-forced loss drops.
