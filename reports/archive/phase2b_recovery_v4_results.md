# Phase 2B Recovery-Prefix v4 Results

## Summary

Recovery-Prefix v4 targeted the task16 policy mistake found in v3: the model incorrectly claimed that changing a reservation from May 23 to May 24 was not allowed. v4 adds a new correction family, `date_change_policy_search_not_refuse`, teaching the model to search the next-day ATL to PHL Economy options instead of refusing or transferring.

## Data

- Dataset: `data/recovery_prefix/tau2_airline_recovery_prefix_v4_2048_train.jsonl`
- Base train rows: `139`
- Raw correction rows: `62`
- Oversample factor: `16`
- Final train rows: `1131`
- New correction family: `date_change_policy_search_not_refuse: 8`

## Training

- Model: `Qwen2.5-7B-Instruct`
- Method: 4-bit QLoRA SFT
- Steps: `650`
- LoRA trainable params: `40,370,176`
- Valid loss: `3.4027 -> 0.4907`
- Output adapter: `/root/autodl-tmp/yuyu/outputs/sft_recovery_prefix_v4_qwen25_7b_qlora_2048/checkpoint`

## Behavior Eval

Heldout next-tool-call behavior improved:

| Model | Tool-name acc | Arg exact acc | Single-call rate |
| --- | ---: | ---: | ---: |
| Base | 0.031 | 0.031 | 0.031 |
| Recovery-Prefix v3 | 0.156 | 0.156 | 0.156 |
| Recovery-Prefix v4 | 0.250 | 0.250 | 0.250 |

Interpretation: v4 learns more executable tool-call behavior than v3, but free-form tool generation is still weak enough that constrained decoding or more action-prefix data remains necessary.

## Tau2 Results

### Targeted 2-Task Eval

Run: `phase2b_recovery_v4_allguards_sftonly_2task_4090_20260614_2207`

| Task | Result | Notes |
| ---: | --- | --- |
| 2 | pass | DB match |
| 16 | pass | Correct `update_reservation_flights`; DB match |

Overall: `pass^1 = 1.0000`

### Wider No-Task16 Eval

Run: `phase2b_recovery_v4_allguards_sftonly_4task_no16_4090_20260614_2219`

| Task | Result | Failure family |
| ---: | --- | --- |
| 2 | pass | No DB write needed; reward passed despite action-check mismatch |
| 18 | fail | `all_reservations_lookup_from_user_id`: model refused to look up reservations from verified user id |
| 25 | fail | `booking_payment_plan`: model got stuck in price/payment calculation and never booked |
| 44 | fail | `duration_requires_search_direct`: model used `get_flight_status` instead of `search_direct_flight` for schedule/duration evidence |

Overall: `pass^1 = 0.2500`

## Notes

An attempted 5-task run with task16 included hit a 16K context limit when using `max_tokens=160`. Reducing to `max_tokens=96` avoided overflow but changed task16 behavior, so it is not counted as a clean comparison. Clean interpretation should combine:

- task16 targeted eval at `max_tokens=160`: passed
- wider no-task16 eval at `max_tokens=160`: 1/4 passed

## Next Failure Families

1. `all_reservations_lookup_from_user_id`
   - Target task: 18
   - Desired behavior: if the user gives a user id and asks to modify all reservations, call `get_user_details` first, then inspect listed reservation IDs instead of refusing.

2. `duration_requires_search_direct`
   - Target task: 44
   - Desired behavior: for flight duration/schedule comparisons, call `search_direct_flight(origin, destination, date)` to retrieve scheduled times, not `get_flight_status`.

3. `booking_payment_plan`
   - Target task: 25
   - Desired behavior: keep passenger count grounded in the user request, calculate total fare and available payment balances correctly, then call `book_reservation` only after explicit confirmation.

## Decision

v4 is a real targeted improvement, not a general solution. The next useful training round should be Recovery-Prefix v5 with the three families above, followed by:

- targeted eval on `18,25,44`
- regression eval on `2,16`
- then a larger pass^1 sweep only if targeted eval improves
