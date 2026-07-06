# Decision-Prefix Dataset v1

## Goal

Train the model to choose the next assistant action: either ask/confirm in natural language or emit exactly one executable tool call. This addresses the protocol SFT failure mode where the model can format tools but keeps calling tools when it should first collect evidence.

## Outputs

- train: `data/decision_prefix/tau2_airline_decision_prefix_v1_2048_train.jsonl`
- valid: `data/decision_prefix/tau2_airline_decision_prefix_v1_2048_valid.jsonl`
- heldout: `data/decision_prefix/tau2_airline_decision_prefix_v1_2048_heldout.jsonl`

## Configuration

- Tokenizer model: `/root/autodl-tmp/models/qwen25-7b-instruct`
- Max sample tokens: `2048`
- Include initial greeting: `False`
- Max target content chars: `1200`
- Tool targets: content removed, one tool call per sample, wrapper loss enabled
- Text targets: content loss enabled, tool-call loss disabled

## Summary

| Split | Rows | Actions | Tasks | Tools | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed rows | Mean dropped prefix msgs |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 139 | `assistant_text:53, tool_call:86` | `1:25, 12:29, 20:15, 27:13, 34:14, 38:14, 42:29` | `book_reservation:1, cancel_reservation:4, get_flight_status:5, get_reservation_details:46, get_user_details:14, list_all_airports:1, search_direct_flight:9, search_onestop_flight:1, transfer_to_human_agents:2, update_reservation_baggages:3` | 1057.3 | 1950 | 2036 | 87.4 | 54 | 3.8 |
| valid | 62 | `assistant_text:25, tool_call:37` | `15:13, 23:15, 33:34` | `book_reservation:3, cancel_reservation:1, get_reservation_details:5, get_user_details:5, search_direct_flight:10, search_onestop_flight:4, update_reservation_baggages:2, update_reservation_flights:7` | 1345.5 | 2012 | 2041 | 131.5 | 42 | 10.3 |
| heldout | 191 | `assistant_text:90, tool_call:101` | `2:29, 16:10, 18:17, 25:41, 32:53, 37:41` | `book_reservation:6, calculate:5, get_flight_status:6, get_reservation_details:39, get_user_details:14, search_direct_flight:14, search_onestop_flight:1, update_reservation_flights:16` | 1231.7 | 2012 | 2048 | 116.6 | 104 | 6.5 |

## Skipped

- train: `initial_greeting:14, text_target_too_long:13`
- valid: `initial_greeting:4, text_target_too_long:5`
- heldout: `initial_greeting:14, text_target_too_long:4`

## Validation

No structural/token validation errors.

## Training Use

- Train this after the tool-call protocol adapter as the first behavior-policy adapter.
- Evaluate with offline behavior probes for both `assistant_text` and `tool_call`, then run tau2 SFT-only smoke with `AGENT_STOP_SEQUENCE=</tool_call>`.
