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

- Tokenizer model: `models/Qwen2.5-0.5B-Instruct`
- Tokenizer stats enabled: `False`
- Max sample tokens: `2048`
- Include protocol variants: `True`
- Tool targets are required to be grounded in the online prefix.
- Parallel gold tool turns are converted to sequential one-call targets.

## Summary

| Split | Rows | Rejected | Target actions | Sample types | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | 219 | 0 | `assistant_text:80, tool_call:139` | `mixed_policy_protocol_tool:53, mixed_policy_sequential_tool:33, mixed_policy_single_tool:53, mixed_policy_text:80` | 0.0 | 0 | 0 | 0.0 | 0 |
| valid | 94 | 4 | `assistant_text:34, tool_call:60` | `mixed_policy_protocol_tool:25, mixed_policy_sequential_tool:10, mixed_policy_single_tool:25, mixed_policy_text:34` | 0.0 | 0 | 0 | 0.0 | 0 |
| heldout | 269 | 0 | `assistant_text:108, tool_call:161` | `mixed_policy_protocol_tool:60, mixed_policy_sequential_tool:41, mixed_policy_single_tool:60, mixed_policy_text:108` | 0.0 | 0 | 0 | 0.0 | 0 |

## Tool Coverage

| Split | Tools | Source call counts |
| --- | --- | --- |
| train | `book_reservation:2, cancel_reservation:4, get_flight_status:10, get_reservation_details:63, get_user_details:28, list_all_airports:2, search_direct_flight:18, search_onestop_flight:2, transfer_to_human_agents:4, update_reservation_baggages:6` | `0:80, 1:106, 2:4, 5:15, 7:14` |
| valid | `book_reservation:6, cancel_reservation:2, get_reservation_details:4, get_user_details:10, search_direct_flight:14, search_onestop_flight:6, update_reservation_baggages:4, update_reservation_flights:14` | `0:34, 1:50, 2:10` |
| heldout | `book_reservation:12, calculate:8, get_flight_status:8, get_reservation_details:49, get_user_details:28, search_direct_flight:23, search_onestop_flight:1, update_reservation_flights:32` | `0:108, 1:120, 2:12, 3:3, 4:20, 6:6` |

## Rejections

### train

No rejected rows.

### valid

| Reason | Count |
| --- | ---: |
| `target_not_grounded` | 4 |

### heldout

No rejected rows.

## Validation

No structural/token validation errors on kept samples.

## Training Use

- This is the Phase2H candidate dataset after Phase2G showed no full tau2 gain from protocol-only v4.
- Train Qwen2.5-7B QLoRA with this dataset, then run mixed-policy behavior eval before another full tau2 pass.
- Do not call the phase successful unless full tau2 pass^1 improves, even if teacher-forced loss drops.
