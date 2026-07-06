# Recovery-Prefix Dataset v1

## Goal

Mix online failure corrections into decision-prefix SFT. The corrections target evidence discipline: ask for identity before tools, do not call a tool while asking a question, and stop guessing after tool `not found` errors.

## Outputs

- train: `data/recovery_prefix/tau2_airline_recovery_prefix_v2_2048_train.jsonl`
- valid: `data/recovery_prefix/tau2_airline_recovery_prefix_v2_2048_valid.jsonl`
- heldout: `data/recovery_prefix/tau2_airline_recovery_prefix_v2_2048_heldout.jsonl`
- corrections: `data/recovery_prefix/tau2_airline_recovery_prefix_v2_2048_corrections.jsonl`

## Summary

- Base train rows: `139`
- Raw correction rows: `31`
- Oversample factor: `12`
- Final train rows: `511`

| Split | Rows | Sample types | Recovery families | Tasks | Mean messages | Max messages |
| --- | ---: | --- | --- | --- | ---: | ---: |
| train | 511 | `decision_prefix_text:53, decision_prefix_tool_call:86, recovery_prefix_text:300, recovery_prefix_tool_call:72` | `ask_identity_before_tool:24, confirm_before_write:12, no_tool_while_asking:24, reservation_not_found_stop_guessing:228, route_disambiguation_check_all_reservations:12, route_disambiguation_check_remaining_reservations:24, route_disambiguation_gold_chain:24, user_not_found_stop_guessing:12, write_after_explicit_confirmation:12` | `1:25, 12:29, 16:228, 2:144, 20:15, 27:13, 34:14, 38:14, 42:29` | 17.2 | 41 |
| valid | 62 | `decision_prefix_text:25, decision_prefix_tool_call:37` | `` | `15:13, 23:15, 33:34` | 7.8 | 15 |
| heldout | 191 | `decision_prefix_text:90, decision_prefix_tool_call:101` | `` | `16:10, 18:17, 2:29, 25:41, 32:53, 37:41` | 8.4 | 16 |
| corrections | 31 | `recovery_prefix_text:25, recovery_prefix_tool_call:6` | `ask_identity_before_tool:2, confirm_before_write:1, no_tool_while_asking:2, reservation_not_found_stop_guessing:19, route_disambiguation_check_all_reservations:1, route_disambiguation_check_remaining_reservations:2, route_disambiguation_gold_chain:2, user_not_found_stop_guessing:1, write_after_explicit_confirmation:1` | `16:19, 2:12` | 20.7 | 41 |

## Caveat

This is online correction data from failed task 2/16 runs, so it must not be reported as clean heldout generalization. Use it to validate the recovery mechanism, then evaluate on separate tasks.
