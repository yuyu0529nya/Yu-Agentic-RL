# Recovery-Prefix Dataset

## Goal

Mix online failure corrections and synthetic gold chains into decision-prefix SFT. The corrections target evidence discipline, route/date disambiguation, payment calculation, duration-based search, and confirmation-before-write behavior.

## Outputs

- train: `data/recovery_prefix/tau2_airline_recovery_prefix_v5_2048_train.jsonl`
- valid: `data/recovery_prefix/tau2_airline_recovery_prefix_v5_2048_valid.jsonl`
- heldout: `data/recovery_prefix/tau2_airline_recovery_prefix_v5_2048_heldout.jsonl`
- corrections: `data/recovery_prefix/tau2_airline_recovery_prefix_v5_2048_corrections.jsonl`

## Summary

- Base train rows: `139`
- Raw correction rows: `126`
- Oversample factor: `16`
- Final train rows: `2155`

| Split | Rows | Sample types | Recovery families | Tasks | Mean messages | Max messages |
| --- | ---: | --- | --- | --- | ---: | ---: |
| train | 2155 | `decision_prefix_text:53, decision_prefix_tool_call:86, recovery_prefix_text:832, recovery_prefix_tool_call:1184` | `all_reservations_lookup_from_user_id:48, ask_identity_before_tool:32, book_after_payment_confirmation:64, booking_payment_plan_single_passenger:176, bulk_downgrade_write_after_confirmation:80, change_existing_reservation_not_book_new:144, cheapest_economy_calculation_confirm_before_write:128, confirm_before_write:16, confirm_bulk_downgrade_before_writes:16, confirm_duration_based_upgrades_before_writes:16, date_change_policy_search_not_refuse:80, duration_based_upgrade_write_after_confirmation:48, duration_requires_search_direct:368, inspect_all_reservations_for_bulk_change:96, inspect_all_reservations_for_duration_rules:80, no_tool_while_asking:32, refund_answer_no_unknown_tool:112, reservation_not_found_stop_guessing:304, route_disambiguation_check_all_reservations:64, route_disambiguation_check_remaining_reservations:32, route_disambiguation_gold_chain:32, search_matching_itinerary_before_booking:16, user_not_found_stop_guessing:16, write_after_explicit_confirmation:16` | `1:25, 12:29, 16:816, 18:240, 2:192, 20:15, 25:256, 27:13, 34:14, 38:14, 42:29, 44:512` | 20.4 | 41 |
| valid | 62 | `decision_prefix_text:25, decision_prefix_tool_call:37` | `` | `15:13, 23:15, 33:34` | 7.8 | 15 |
| heldout | 191 | `decision_prefix_text:90, decision_prefix_tool_call:101` | `` | `16:10, 18:17, 2:29, 25:41, 32:53, 37:41` | 8.4 | 16 |
| corrections | 126 | `recovery_prefix_text:52, recovery_prefix_tool_call:74` | `all_reservations_lookup_from_user_id:3, ask_identity_before_tool:2, book_after_payment_confirmation:4, booking_payment_plan_single_passenger:11, bulk_downgrade_write_after_confirmation:5, change_existing_reservation_not_book_new:9, cheapest_economy_calculation_confirm_before_write:8, confirm_before_write:1, confirm_bulk_downgrade_before_writes:1, confirm_duration_based_upgrades_before_writes:1, date_change_policy_search_not_refuse:5, duration_based_upgrade_write_after_confirmation:3, duration_requires_search_direct:23, inspect_all_reservations_for_bulk_change:6, inspect_all_reservations_for_duration_rules:5, no_tool_while_asking:2, refund_answer_no_unknown_tool:7, reservation_not_found_stop_guessing:19, route_disambiguation_check_all_reservations:4, route_disambiguation_check_remaining_reservations:2, route_disambiguation_gold_chain:2, search_matching_itinerary_before_booking:1, user_not_found_stop_guessing:1, write_after_explicit_confirmation:1` | `16:51, 18:15, 2:12, 25:16, 44:32` | 21.3 | 41 |

## Caveat

This is targeted recovery data from failed and diagnostic tau2 runs, so it must not be reported as clean heldout generalization. Use it to train the recovery mechanism, then evaluate on separate regression and heldout tasks.
