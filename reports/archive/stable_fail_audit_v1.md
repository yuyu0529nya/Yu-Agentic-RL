# Stable-Fail Audit v1: tau2 airline failed-21 N=4

## Scope

This audit analyzes the five tasks that remained unsolved after GLM-5.1 N=4 sampling:

- Stable-fail tasks: `7, 14, 21, 29, 44`
- Source run: `airline_failed21_n4_timeout300_v2_merged`
- Samples inspected: 20 trajectories
- Shared property: all five tasks have `0/4` task-level success, so PRM rerank cannot solve them without better candidate trajectories.

The goal is to convert failures into SFT and future GRPO training targets.

## Executive Summary

| Task | Main Failure Family | What the model fails to learn | SFT Data Need |
| ---: | --- | --- | --- |
| 7 | Communication target missed after correct actions | Completes DB actions but fails to explicitly communicate required total `1628` | Teach post-action checklist and mid-conversation new-intent handling |
| 14 | Payment planning and basic-economy policy | Uses too many certificates, updates basic economy, or miscalculates Mastercard charge | Teach cancel-and-rebook plus max-one-certificate payment planning |
| 21 | Temporal optimization plus baggage/payment details | Selects wrong return itinerary, wrong gift card, or wrong free baggage count | Teach constrained itinerary search and exact write arguments |
| 29 | Policy-gated rebooking | Tries to update destination-changing reservation or books with wrong payment/insurance | Teach when to cancel-and-book instead of update |
| 44 | Tool affordance and multi-reservation planning | Uses flight status instead of schedule search, defers, or cancels/updates wrong reservations | Teach duration evidence collection before any write |

These five failures are qualitatively different. A useful SFT set should cover all five instead of only adding more generic successful traces.

## Task 7

### User Goal

The user wants to cancel upcoming reservations `XEHM4B` and `59XX6W`.

Important twist:

- If either reservation is basic economy, the user asks to upgrade to business first and then cancel.
- Use the credit card ending in `2135`.
- After the third agent message, the user introduces a new intent: check other upcoming flights and report their total cost.

Reference success requires:

- `get_reservation_details(XEHM4B)`
- `get_reservation_details(59XX6W)`
- `update_reservation_flights(XEHM4B, business, HAT005/HAT178, credit_card_2408938)`
- `cancel_reservation(XEHM4B)`
- `cancel_reservation(59XX6W)`
- Communicate `1628`

### Observed Failures

| Trial | DB | Action Match | Communicate `1628` | Pattern |
| ---: | --- | --- | --- | --- |
| 0 | match | 5/5 | no | Correct writes, wrong final communication |
| 1 | match | 5/5 | no | Correct writes, reports `708` instead of required total |
| 2 | mismatch | 2/5 | no | Transfers instead of completing upgrade/cancel |
| 3 | match | 5/5 | no | Correct writes, reports `708` instead of required total |

The key point: three of four trajectories are not action failures. They are communication failures after successful DB updates.

### Common Failure Point

The model handles the cancellation plan but does not maintain a post-action checklist for the user's inserted information request. It reports an alternate total such as `708` or focuses on refunds instead of explicitly communicating the expected total `1628`.

### Failure Family

- `communication_target_miss`
- `mid_conversation_intent_forgetting`
- Partial `policy_workaround_execution`

### SFT Data Requirement

Create traces where the assistant:

1. Executes the upgrade-and-cancel sequence.
2. Keeps a running list of unresolved user intents.
3. After writes, explicitly answers the inserted "other upcoming flights total" request.
4. Verifies the final response contains the required numeric answer.

PRM-Lite note: current process scoring under-penalizes this case because DB/action success masks the communication miss. Add or strengthen a `missing_required_communication` component.

## Task 14

### User Goal

The user wants:

1. Sum of gift card balances.
2. Sum of certificate balances.
3. Change a basic-economy reservation to the cheapest business round trip on the same dates.
4. If basic economy cannot be modified, cancel and book a new reservation.
5. Use certificates as much as allowed, then gift cards, then Mastercard.
6. Only one certificate can be used.
7. Book only if Mastercard charge is under `$2000`.

Reference success requires:

- Communicate gift cards total: `327`
- Communicate certificate total: `1000`
- Cancel `K1NW8N`
- Book business flights `HAT023`, `HAT204`, `HAT100`
- Payment split:
  - `certificate_3765853`: `$500`
  - `gift_card_8020792`: `$198`
  - `gift_card_6136092`: `$129`
  - `credit_card_2198526`: `$1786`
- Communicate `$1786`

### Observed Failures

| Trial | Main Tool Pattern | Communicate `1786` | Main Error |
| ---: | --- | --- | --- |
| 0 | cancel + book | no | Uses all certificates, charges Mastercard `$1286` |
| 1 | update basic economy + transfer | no | Attempts invalid update-only strategy |
| 2 | update basic economy + transfer | no | Treats upgrade cost as `$2046`, says Mastercard `$719` |
| 3 | cancel + book twice | no | Uses all certificates and wrong flight/payment plan |

### Common Failure Point

The model knows it should use stored value, but fails the business rule "max one certificate" and then propagates the wrong arithmetic into payment planning. It also sometimes chooses `update_reservation_flights` on a basic-economy reservation even though the correct strategy is cancel-and-book.

### Failure Family

- `payment_planning_error`
- `calculation_error`
- `basic_economy_update_attempt`
- `cancel_rebook_plan_not_closed`

### SFT Data Requirement

Create traces that explicitly teach:

1. Read balances and compute totals.
2. Apply the max-one-certificate constraint before arithmetic.
3. Detect basic-economy non-modifiability.
4. Cancel first, then book the replacement.
5. Compute `2613 - 500 - 198 - 129 = 1786`.
6. Say the Mastercard charge before booking.
7. Use the exact payment method ids and amounts in `book_reservation`.

PRM-Lite note: current `payment_planning_error` and `calculation_trace_inconsistent` tags are useful and should stay.

## Task 21

### User Goal

The user wants to change the return flights for reservation `OBUT9V`.

Constraints:

- Houston to Denver trip, departure date May 27.
- Return must also be on May 27.
- Choose the fastest return trip including stopover time.
- Stay in economy.
- Add one more checked bag.
- Use the gift card with the smallest balance.

Reference success requires:

- `update_reservation_flights(OBUT9V, economy, HAT078/HAT118/HAT290/HAT175, gift_card_6276644)`
- `update_reservation_baggages(OBUT9V, total_baggages=2, nonfree_baggages=0, gift_card_6276644)`

### Observed Failures

| Trial | Return Flights Chosen | Payment | Baggage | Main Error |
| ---: | --- | --- | --- | --- |
| 0 | HAT229/HAT266 then HAT084/HAT266 | `gift_card_7480005` | nonfree `0` | Wrong return itinerary and wrong gift card |
| 1 | HAT084/HAT266 | `gift_card_6276644` | nonfree `1` | Wrong return itinerary and wrong baggage fee |
| 2 | HAT084/HAT266 | `gift_card_7480005` | nonfree `0` | Wrong return itinerary and wrong gift card |
| 3 | HAT084/HAT266 | mixed gift cards | nonfree `1` | Wrong return itinerary, wrong baggage, repeated write |

### Common Failure Point

The model consistently fails the constrained route optimization. It finds same-day return options but chooses `HAT084/HAT266` or related alternatives instead of the expected fastest valid return `HAT290/HAT175`. It also inconsistently reasons about membership baggage allowance and smallest-balance gift card.

### Failure Family

- `temporal_route_optimization_error`
- `object_selection_error`
- `payment_method_selection_error`
- `baggage_policy_error`

### SFT Data Requirement

Create traces that teach:

1. Identify target reservation by user and trip description.
2. Search return options from `DEN` to `IAH` on May 27.
3. Compute total duration including stopover.
4. Verify return starts after outbound trip context.
5. Select `HAT290/HAT175`.
6. Choose `gift_card_6276644` as the smallest-balance gift card.
7. Set `nonfree_baggages=0` when membership/policy makes both bags free.

PRM-Lite note: current PRM only sees reference mismatch. A future scorer should add route-duration and payment-method selection diagnostics.

## Task 29

### User Goal

The user wants to change a DTW-LGA round trip to nonstop DTW-JFK and JFK-DTW flights on the same dates.

Constraints:

- Reservation `VA5SGQ`.
- Economy, not basic economy.
- Early flights arriving before 7am.
- If prices are shown and the user is asked, choose `HAT169` and `HAT033`.
- Add 1 checked bag.
- Because destination changes are not modifiable, correct path is cancel and book.

Reference success requires:

- `get_reservation_details(VA5SGQ)`
- `cancel_reservation(VA5SGQ)`
- `book_reservation` with:
  - flights `HAT169` on 2024-05-17 and `HAT033` on 2024-05-19
  - payment `credit_card_8003957`, amount `$282`
  - `total_baggages=1`
  - `nonfree_baggages=0`
  - `insurance=no`

### Observed Failures

| Trial | Main Tool Pattern | Main Error |
| ---: | --- | --- |
| 0 | update existing reservation + update baggage | Uses update instead of cancel-and-book |
| 1 | update existing reservation + update baggage | Uses update instead of cancel-and-book, wrong baggage cost |
| 2 | cancel + book | Wrong payment card/amount and adds insurance |
| 3 | search only + transfer | Refuses/defers instead of cancel-and-book |

### Common Failure Point

The model does not reliably apply the policy gate: destination changes cannot be handled as a normal reservation update. Even when it chooses cancel-and-book in trial 2, it does not preserve the exact payment and insurance requirements.

### Failure Family

- `policy_gate_missed`
- `cancel_rebook_plan_not_closed`
- `unexpected_write_tool`
- `payment_argument_error`

### SFT Data Requirement

Create traces that teach:

1. Destination change means update is invalid.
2. Inform user that cancellation and new booking are required.
3. Cancel `VA5SGQ`.
4. Book `HAT169/HAT033` as a new reservation.
5. Preserve user preferences: economy, one checked bag, no insurance.
6. Use `credit_card_8003957` with amount `$282`.

PRM-Lite note: current unexpected-write penalty correctly catches update-only failures. It should also detect wrong payment card/insurance in near-correct cancel-book attempts.

## Task 44

### User Goal

The user wants the agent to inspect all future reservations and:

- Cancel future reservations containing any flights longer than 4 hours, when cancellation is allowed.
- Upgrade flights/reservations that are under or equal to 3 hours including layovers, where possible.
- Tell the user total business-upgrade cost before upgrading.
- Do not upgrade then cancel.
- The user is healthy, so health-related cancellation logic should not be used.

Reference success requires a long evidence chain:

- Read user and five reservation details.
- Use `search_direct_flight` on relevant route/date pairs to obtain schedule evidence.
- Upgrade:
  - `NM1VX1` to business with `HAT300/HAT208`
  - `H8Q05L` to business with `HAT268`
  - `KC18K6` to business with `HAT300/HAT215`
- Do not cancel `S61CZX`.
- Communicate total upgrade cost around `$1380-$1390`.

### Observed Failures

| Trial | Main Pattern | Main Error |
| ---: | --- | --- |
| 0 | reads reservations, uses `get_flight_status`, transfers | Wrong tool for duration evidence |
| 1 | reads reservations, uses `get_flight_status`, stops/defers | Wrong tool and premature deferral |
| 2 | reads reservations only, defers | Does not search schedule evidence |
| 3 | reads more evidence, writes | Cancels `KC18K6` and `S61CZX`, upgrades only two reservations, misses `KC18K6` upgrade |

### Common Failure Point

The model does not know which tool provides duration/schedule evidence. It uses `get_flight_status`, which cannot answer duration. When it does eventually write, it writes before completing the full classification table and cancels reservations that should not be cancelled.

### Failure Family

- `tool_affordance_error`
- `incomplete_evidence`
- `multi_reservation_planning_error`
- `premature_write`

### SFT Data Requirement

Create traces that teach:

1. `get_flight_status` is not enough for duration.
2. Use schedule-returning search tools for each route/date.
3. Build a table of every reservation and every leg.
4. Classify each reservation before any write:
   - cancel allowed or not
   - upgrade allowed or not
   - total upgrade charge
5. Communicate total upgrade cost before write calls.
6. Execute only the expected upgrades and avoid cancellation without policy basis.

PRM-Lite note: current `wrong_tool_for_schedule_lookup` and `premature_deferral` components are correctly identifying the problem. This task is a strong regression test for tool-affordance reward design.

## Cross-Task Training Targets

| Training Target | Covered Tasks | Why It Matters |
| --- | --- | --- |
| Maintain unresolved user intents | 7 | Correct DB writes are insufficient if required information is not communicated. |
| Policy-gated write planning | 14, 29, 44 | The model must choose update vs cancel-and-book vs no cancellation before writing. |
| Exact payment planning | 14, 21, 29 | Correct tool name is not enough; payment ids and amounts must match. |
| Schedule and duration reasoning | 21, 44 | Long-horizon airline tasks often require sorting by time, not just finding available flights. |
| Multi-object state table before writes | 7, 21, 44 | The model must track many reservations/flights and avoid acting on the wrong object. |
| Final numeric communication | 7, 14, 44 | A trajectory can fail solely because the final answer omits a required number. |

## Recommended SFT Seed Design

Build `airline_agent_sft_v1` with three data types:

1. Positive successful trajectories from split tasks in the N=4 run.
2. Repaired stable-fail trajectories for `7,14,21,29,44`.
3. Contrastive snippets that show the wrong action and the corrected action.

Minimum repaired examples:

| Task | Repaired Example Focus |
| ---: | --- |
| 7 | Correct actions plus final communication of `1628` |
| 14 | Cancel-and-book with one certificate and Mastercard `$1786` |
| 21 | Select `HAT290/HAT175`, `gift_card_6276644`, `nonfree_baggages=0` |
| 29 | Cancel `VA5SGQ`, book `HAT169/HAT033`, no insurance, card `$282` |
| 44 | Schedule-search evidence table, total upgrade cost, three upgrades, no invalid cancellations |

## PRM Follow-Up

After this audit, PRM ablation should test whether the following rule families are independently useful:

- `reference_action_mismatch`
- `premature_write`
- `tool_affordance_error`
- `payment_planning_error`
- `calculation_trace_inconsistent`
- new: `missing_required_communication`
- new: `route_duration_selection_error`
- new: `wrong_payment_method_or_amount`

Expected Phase 0 conclusion:

1. PRM-rerank solves tasks with at least one good candidate.
2. Stable-fail tasks expose missing policy, arithmetic, route-optimization, and tool-affordance skills.
3. The next phase should be SFT data construction, not GRPO yet.

## Artifacts Used

- `third_party/tau2-bench/data/simulations/airline_failed21_n4_timeout300_v2_merged/results.json`
- `reports/airline_failed21_n4_timeout300_v2_merged.summary.json`
- `reports/airline_failed21_n4_timeout300_v2_merged.prm_v2.json`
- `reports/airline_failed21_n4_timeout300_v2_merged.rerank_v2_full.json`
