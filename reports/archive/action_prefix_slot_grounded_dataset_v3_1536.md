# Slot-Grounded Action-Prefix Dataset v3

## Goal

Build action-prefix SFT samples where the target tool-call arguments are grounded in the online prefix. This directly targets the Phase2D failure mode: hallucinated `user_id` and `reservation_id` arguments.

## Outputs

- train: `data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_train.jsonl`
- valid: `data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_valid.jsonl`
- heldout: `data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_heldout.jsonl`
- train rejected: `data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_train_rejected.jsonl`
- valid rejected: `data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_valid_rejected.jsonl`
- heldout rejected: `data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_heldout_rejected.jsonl`

## Configuration

- Tokenizer model: `models/Qwen2.5-0.5B-Instruct`
- Max sample tokens: `1536`
- Base route: action-prefix v2 trimming plus target-only slot-grounding validation

## Summary

| Split | Candidates | Kept | Rejected | Keep rate | Mean tokens | Max tokens | Mean target tokens | Trimmed rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 60 | 49 | 11 | 0.817 | 772.0 | 1528 | 61.0 | 10 |
| valid | 32 | 20 | 12 | 0.625 | 909.8 | 1479 | 77.0 | 12 |
| heldout | 73 | 51 | 22 | 0.699 | 839.8 | 1529 | 60.7 | 18 |

## Rejection Issues

### train

| Issue | Count |
| --- | ---: |
| `ungrounded_payment_id` | 5 |
| `ungrounded_reservation_id` | 10 |
| `ungrounded_user_id` | 1 |

### valid

| Issue | Count |
| --- | ---: |
| `ungrounded_payment_id` | 9 |
| `ungrounded_reservation_id` | 7 |
| `ungrounded_user_id` | 3 |

### heldout

| Issue | Count |
| --- | ---: |
| `ungrounded_payment_id` | 25 |
| `ungrounded_reservation_id` | 15 |
| `ungrounded_user_id` | 5 |

## Kept Task Coverage

| Split | Tasks | Tools |
| --- | --- | --- |
| train | `1:6, 12:9, 20:4, 27:4, 34:12, 38:8, 42:6` | `cancel_reservation:4, get_flight_status:5, get_reservation_details:39, get_user_details:14, list_all_airports:1, search_direct_flight:9, search_onestop_flight:1, transfer_to_human_agents:2` |
| valid | `15:4, 23:7, 33:9` | `cancel_reservation:1, get_reservation_details:3, get_user_details:3, search_direct_flight:10, search_onestop_flight:4, update_reservation_flights:4` |
| heldout | `2:8, 16:3, 18:3, 25:12, 32:10, 37:15` | `calculate:5, get_flight_status:6, get_reservation_details:38, get_user_details:14, search_direct_flight:14, search_onestop_flight:1, update_reservation_flights:1` |

## Validation

No structural/token validation errors on kept samples.

## Training Use

- Train on the kept train split first.
- Use the rejected files as hard negatives or future correction targets, not as SFT positives.
- After training, rerun the Phase2D targeted tasks and check whether slot-grounding issues drop before expecting pass@4 to rise.
