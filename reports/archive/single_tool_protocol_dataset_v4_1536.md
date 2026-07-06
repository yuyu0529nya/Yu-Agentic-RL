# Single-Tool Protocol Dataset v4

## Goal

Combine Phase2E slot-grounded supervision with the earlier executable tool-call protocol. Each target is exactly one grounded tool call with empty assistant text, and loss covers the `<tool_call>` wrapper plus JSON payload.

## Outputs

- train: `data/tool_call_protocol/tau2_airline_single_tool_protocol_v4_1536_train.jsonl`
- valid: `data/tool_call_protocol/tau2_airline_single_tool_protocol_v4_1536_valid.jsonl`
- heldout: `data/tool_call_protocol/tau2_airline_single_tool_protocol_v4_1536_heldout.jsonl`
- train rejected: `data/tool_call_protocol/tau2_airline_single_tool_protocol_v4_1536_train_rejected.jsonl`
- valid rejected: `data/tool_call_protocol/tau2_airline_single_tool_protocol_v4_1536_valid_rejected.jsonl`
- heldout rejected: `data/tool_call_protocol/tau2_airline_single_tool_protocol_v4_1536_heldout_rejected.jsonl`

## Configuration

- Tokenizer model: `models/Qwen2.5-0.5B-Instruct`
- Max sample tokens: `1536`
- Source: slot-grounded action-prefix v3 kept samples
- Target assistant content: removed
- Target tool calls: exactly one
- Loss: `assistant_tool_call_wrappers=True`, `assistant_tool_calls=True`, `assistant_content=False`

## Summary

| Split | Rows | Rejected | Tasks | Tools | Mean tokens | P90 tokens | Max tokens | Mean target tokens |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| train | 42 | 7 | `1:3, 12:9, 20:4, 27:4, 34:12, 38:8, 42:2` | `get_flight_status:5, get_reservation_details:10, get_user_details:14, list_all_airports:1, search_direct_flight:9, search_onestop_flight:1, transfer_to_human_agents:2` | 724.6 | 1311 | 1528 | 33.6 |
| valid | 15 | 5 | `15:1, 23:7, 33:7` | `cancel_reservation:1, get_reservation_details:1, get_user_details:3, search_direct_flight:4, search_onestop_flight:2, update_reservation_flights:4` | 894.5 | 1474 | 1476 | 45.7 |
| heldout | 38 | 13 | `2:6, 16:1, 18:2, 25:12, 32:7, 37:10` | `calculate:3, get_flight_status:2, get_reservation_details:9, get_user_details:14, search_direct_flight:9, update_reservation_flights:1` | 791.6 | 1408 | 1516 | 28.2 |

## Rejections

### train

| Reason | Count |
| --- | ---: |
| `target_tool_call_count_2` | 2 |
| `target_tool_call_count_5` | 3 |
| `target_tool_call_count_7` | 2 |

### valid

| Reason | Count |
| --- | ---: |
| `target_tool_call_count_2` | 5 |

### heldout

| Reason | Count |
| --- | ---: |
| `target_tool_call_count_2` | 6 |
| `target_tool_call_count_3` | 1 |
| `target_tool_call_count_4` | 5 |
| `target_tool_call_count_6` | 1 |

## Validation

No structural/token validation errors.

## Intended Next Run

- Train Qwen2.5-7B QLoRA on this dataset.
- Evaluate with `--stop-sequence </tool_call>` and small `--max-new-tokens` such as 64 or 96.
- Target: keep tool-name accuracy high while raising single-call rate.
