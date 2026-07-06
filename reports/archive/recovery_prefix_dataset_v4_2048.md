# Recovery-Prefix Dataset v1

## Goal

Mix online failure corrections into decision-prefix SFT. The corrections target evidence discipline: ask for identity before tools, do not call a tool while asking a question, and stop guessing after tool `not found` errors.

## Outputs

- train: `data/recovery_prefix/tau2_airline_recovery_prefix_v4_2048_train.jsonl`
- valid: `data/recovery_prefix/tau2_airline_recovery_prefix_v4_2048_valid.jsonl`
- heldout: `data/recovery_prefix/tau2_airline_recovery_prefix_v4_2048_heldout.jsonl`
- corrections: `data/recovery_prefix/tau2_airline_recovery_prefix_v4_2048_corrections.jsonl`

## Summary

- Base train rows: `139`
- Raw correction rows: `62`
- Oversample factor: `16`
- Final train rows: `1131`

| Split | Rows | Sample types | Recovery families | Tasks | Mean messages | Max messages |
| --- | ---: | --- | --- | --- | ---: | ---: |
| train | 1131 | `decision_prefix_text:53, decision_prefix_tool_call:86, recovery_prefix_text:608, recovery_prefix_tool_call:384` | `ask_identity_before_tool:32, change_existing_reservation_not_book_new:112, cheapest_economy_calculation_confirm_before_write:128, confirm_before_write:16, date_change_policy_search_not_refuse:128, no_tool_while_asking:32, refund_answer_no_unknown_tool:80, reservation_not_found_stop_guessing:304, route_disambiguation_check_all_reservations:64, route_disambiguation_check_remaining_reservations:32, route_disambiguation_gold_chain:32, user_not_found_stop_guessing:16, write_after_explicit_confirmation:16` | `1:25, 12:29, 16:800, 2:192, 20:15, 27:13, 34:14, 38:14, 42:29` | 18.8 | 41 |
| valid | 62 | `decision_prefix_text:25, decision_prefix_tool_call:37` | `` | `15:13, 23:15, 33:34` | 7.8 | 15 |
| heldout | 191 | `decision_prefix_text:90, decision_prefix_tool_call:101` | `` | `16:10, 18:17, 2:29, 25:41, 32:53, 37:41` | 8.4 | 16 |
| corrections | 62 | `recovery_prefix_text:38, recovery_prefix_tool_call:24` | `ask_identity_before_tool:2, change_existing_reservation_not_book_new:7, cheapest_economy_calculation_confirm_before_write:8, confirm_before_write:1, date_change_policy_search_not_refuse:8, no_tool_while_asking:2, refund_answer_no_unknown_tool:5, reservation_not_found_stop_guessing:19, route_disambiguation_check_all_reservations:4, route_disambiguation_check_remaining_reservations:2, route_disambiguation_gold_chain:2, user_not_found_stop_guessing:1, write_after_explicit_confirmation:1` | `16:50, 2:12` | 20.3 | 41 |

## Caveat

This is online correction data from failed task 2/16 runs, so it must not be reported as clean heldout generalization. Use it to validate the recovery mechanism, then evaluate on separate tasks.
