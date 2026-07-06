# Action-Prefix Dataset v1

## Goal

Convert successful tau2 airline trajectories into dense next-tool-call supervision samples. Each sample contains a conversation prefix plus exactly one target assistant tool-call turn.

## Outputs

- train: `data/action_prefix/tau2_airline_action_prefix_train.jsonl`
- valid: `data/action_prefix/tau2_airline_action_prefix_valid.jsonl`
- heldout: `data/action_prefix/tau2_airline_action_prefix_heldout.jsonl`

## Summary

| Split | Rows | Tasks | Tools | Mean prefix msgs | Max prefix msgs | Mean tokens | P90 tokens | Max tokens | Mean target tokens |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 60 | `1:6, 12:12, 20:5, 27:11, 34:12, 38:8, 42:6` | `book_reservation:1, cancel_reservation:4, get_flight_status:5, get_reservation_details:46, get_user_details:14, list_all_airports:1, search_direct_flight:9, search_onestop_flight:1, transfer_to_human_agents:2, update_reservation_baggages:3` | 9.7 | 28 | 1443.6 | 3282 | 4473 | 62.0 |
| valid | 32 | `15:5, 23:10, 33:17` | `book_reservation:3, cancel_reservation:1, get_reservation_details:5, get_user_details:5, search_direct_flight:10, search_onestop_flight:4, update_reservation_baggages:2, update_reservation_flights:7` | 17.1 | 38 | 3775.9 | 7699 | 10862 | 91.6 |
| heldout | 73 | `2:8, 16:4, 18:8, 25:18, 32:17, 37:18` | `book_reservation:6, calculate:5, get_flight_status:6, get_reservation_details:39, get_user_details:14, search_direct_flight:14, search_onestop_flight:1, update_reservation_flights:16` | 13.8 | 41 | 2310.2 | 5093 | 7410 | 75.7 |

## Validation

No structural validation errors.

## Training Use

- Use the train split for action-prefix SFT.
- Keep valid and heldout for behavior evaluation.
- The current local 3060 can run 2K-context smoke experiments; larger context or 7B/8B models should wait until this action-prefix route shows a behavior gain.
