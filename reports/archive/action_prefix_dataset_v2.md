# Action-Prefix Dataset v2

## Goal

Create one next-tool-call SFT sample per assistant tool-call turn, with message-level context trimming so each sample fits the local training token budget while preserving the target tool call.

## Outputs

- train: `data/action_prefix/tau2_airline_action_prefix_v2_train.jsonl`
- valid: `data/action_prefix/tau2_airline_action_prefix_v2_valid.jsonl`
- heldout: `data/action_prefix/tau2_airline_action_prefix_v2_heldout.jsonl`

## Configuration

- Tokenizer model: `models/Qwen2.5-0.5B-Instruct`
- Max sample tokens: `1536`
- Trim strategy: `message_suffix_target_preserving`

## Summary

| Split | Rows | Tasks | Tools | Mean tokens | Max tokens | Mean target tokens | Trimmed rows | Mean dropped prefix msgs | Max dropped prefix msgs |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 60 | `1:6, 12:12, 20:5, 27:11, 34:12, 38:8, 42:6` | `book_reservation:1, cancel_reservation:4, get_flight_status:5, get_reservation_details:46, get_user_details:14, list_all_airports:1, search_direct_flight:9, search_onestop_flight:1, transfer_to_human_agents:2, update_reservation_baggages:3` | 872.4 | 1528 | 62.0 | 21 | 3.6 | 20 |
| valid | 32 | `15:5, 23:10, 33:17` | `book_reservation:3, cancel_reservation:1, get_reservation_details:5, get_user_details:5, search_direct_flight:10, search_onestop_flight:4, update_reservation_baggages:2, update_reservation_flights:7` | 965.9 | 1488 | 91.6 | 22 | 11.7 | 30 |
| heldout | 73 | `2:8, 16:4, 18:8, 25:18, 32:17, 37:18` | `book_reservation:6, calculate:5, get_flight_status:6, get_reservation_details:39, get_user_details:14, search_direct_flight:14, search_onestop_flight:1, update_reservation_flights:16` | 951.6 | 1536 | 75.7 | 40 | 7.5 | 35 |

## Validation

No structural/token validation errors.

## Training Use

- Train with `--max-seq-len` equal to the v2 token budget and `--truncation-side right`; no target should be lost because samples are pre-trimmed.
- This is the preferred local 3060 route before renting GPU: prove next-tool-call behavior improves on v2, then scale model/context.
