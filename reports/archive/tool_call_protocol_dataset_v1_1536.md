# Tool-Call Protocol Dataset v1

## Goal

Build executable tool-call supervision samples for Qwen/tau2. Each target assistant turn contains no natural language and exactly one tool call. The loss covers the `<tool_call>` protocol wrapper and JSON payload.

## Outputs

- train: `data/tool_call_protocol/tau2_airline_tool_call_protocol_v1_1536_train.jsonl`
- valid: `data/tool_call_protocol/tau2_airline_tool_call_protocol_v1_1536_valid.jsonl`
- heldout: `data/tool_call_protocol/tau2_airline_tool_call_protocol_v1_1536_heldout.jsonl`

## Configuration

- Tokenizer model: `models/Qwen2.5-0.5B-Instruct`
- Max sample tokens: `1536`
- Target assistant content: removed
- Target tool calls: exactly one
- Loss: `assistant_tool_call_wrappers=True`, `assistant_tool_calls=True`, `assistant_content=False`

## Summary

| Split | Rows | Tasks | Tools | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed rows | Mean dropped prefix msgs | Mean removed content chars |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 53 | `1:3, 12:12, 20:5, 27:11, 34:12, 38:8, 42:2` | `book_reservation:1, get_flight_status:5, get_reservation_details:17, get_user_details:14, list_all_airports:1, search_direct_flight:9, search_onestop_flight:1, transfer_to_human_agents:2, update_reservation_baggages:3` | 848.0 | 1413 | 1528 | 35.6 | 19 | 3.5 | 102.2 |
| valid | 27 | `15:2, 23:10, 33:15` | `book_reservation:3, cancel_reservation:1, get_reservation_details:3, get_user_details:5, search_direct_flight:4, search_onestop_flight:2, update_reservation_baggages:2, update_reservation_flights:7` | 962.7 | 1474 | 1488 | 67.0 | 20 | 12.9 | 80.4 |
| heldout | 60 | `2:6, 16:2, 18:7, 25:18, 32:14, 37:13` | `book_reservation:6, calculate:3, get_flight_status:2, get_reservation_details:10, get_user_details:14, search_direct_flight:9, update_reservation_flights:16` | 958.8 | 1466 | 1536 | 59.0 | 36 | 8.3 | 49.4 |

## Validation

No structural/token validation errors.

## Why This Replaces Action-Prefix v2

Action-Prefix v2 improved offline next-tool-call probes but the 5-task tau2 run showed protocol drift: the model emitted JSON as normal assistant text, so no tools were executed. This dataset explicitly trains the executable tool-call protocol.
