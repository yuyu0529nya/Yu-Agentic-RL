# GLM 5.1 Failure Case Study: Tasks 44, 2, 37, 23

Run: `airline_baseline_50_anthropic_glm_5_1_50tasks_1trials`

These four tasks had some of the lowest PRM-Lite process scores in the 50-task GLM baseline. They are useful because they represent different long-horizon agent failure modes.

## Overview

| Task | Reward | Process | Main Tags | Short Diagnosis |
| --- | ---: | ---: | --- | --- |
| 44 | 0.0 | -13.0 | `action_mismatch`, `incomplete_evidence`, `communication_db_gap` | Used the wrong tool for flight duration and gave up instead of searching schedules and upgrading eligible reservations. |
| 2 | 0.0 | -9.0 | `incomplete_evidence`, `object_selection_error`, `premature_write`, `compensation_policy_error` | Picked the wrong "most recent" reservation and issued forbidden compensation. |
| 37 | 0.0 | -6.5 | `temporal_policy_error`, `premature_write`, `incomplete_evidence` | Correctly refused one cancellation, but incorrectly cancelled a past reservation. |
| 23 | 0.0 | -5.5 | `action_mismatch`, `premature_write`, `incomplete_evidence` | Chose an update flow instead of the expected cancel-and-rebook flow; also made arithmetic/payment mistakes. |

## Task 44: Wrong Tool Affordance For Schedule Reasoning

Task goal:

- User wants to cancel future reservations containing any flights longer than 4 hours.
- For flights 3 hours or less, including layovers, user wants upgrade to business where possible.
- Agent should infer durations, mention total upgrade cost around `$1380-$1390`, and upgrade `NM1VX1`, `H8Q05L`, and `KC18K6`.
- Agent should not cancel `S61CZX`.

Actual behavior:

- GLM correctly called `get_user_details`.
- GLM correctly fetched all five reservations.
- Then it called `get_flight_status` for several flights.
- `get_flight_status` only returns statuses like `available`, not schedule times.
- The model concluded it could not access flight durations and suggested transfer to a human agent.
- It never called `search_direct_flight`, which returns scheduled departure and arrival times.
- It never upgraded any eligible reservation.

Failure type:

- Tool affordance misunderstanding.
- Evidence retrieval incomplete.
- Over-deferral: gave up although the needed information was available through a different tool.

PRM-Lite v0 caught:

- `reference_action_mismatch`: expected 19 actions, only 6 matched.
- `incomplete_evidence`.
- `communication_db_gap`.

PRM-Lite v1 should add:

- `wrong_tool_for_schedule_lookup`: if the task requires duration/schedule and the model calls `get_flight_status` but not `search_direct_flight` / `search_onestop_flight`.
- `premature_deferral`: transfer/refusal after partial tool use when other available tools can answer the query.

## Task 2: Wrong Object Selection + Forbidden Compensation

Task goal:

- User starts booking a SF to NY flight, then switches topic to a delayed flight in the "most recent reservation".
- User will incorrectly claim there were 3 passengers.
- Agent should verify the actual reservation and passenger count.
- Agent should check that the flight was indeed delayed.
- Agent should not offer a `$50` certificate because the user does not want to change or cancel the reservation.

Actual behavior:

- GLM retrieved user details.
- User profile had two reservations: `SDZQKO`, `4OG6T3`.
- GLM directly selected `4OG6T3` as the "most recent" reservation.
- It did not fetch `SDZQKO`, so it never compared `created_at` across all candidate reservations.
- It checked flight statuses for `4OG6T3` and found one delayed segment.
- It correctly corrected the user's passenger-count claim from 3 to 1.
- It then offered and issued a `$50` travel certificate.

Failure type:

- Incomplete evidence before object selection.
- Premature write action.
- Compensation policy error.

PRM-Lite v0 caught:

- Missing candidate reservation `SDZQKO`.
- `send_certificate` was unexpected.
- Certificate was forbidden by task assertions.
- Delayed-flight certificate was issued without a prior change/cancellation action.

PRM-Lite v1 should add:

- `latest_object_requires_full_sort`: for words like latest/recent/last, require fetching all candidate records and comparing timestamps.
- `compensation_requires_user_goal`: delayed-flight compensation requires not only delay verification, but the right user intent and policy precondition.

## Task 37: Cancellation Eligibility Error On Past Reservation

Task goal:

- User wants to cancel `IFOYYZ` and `NQNU5R`.
- User wants to upgrade `M20IZO` to business.
- Correct behavior:
  - Do not cancel `IFOYYZ`: basic economy, no insurance, outside 24h.
  - Do not cancel `NQNU5R`: flights already in the past.
  - Upgrade `M20IZO` to business.

Actual behavior:

- GLM retrieved user details and the three requested reservations.
- It correctly refused to cancel `IFOYYZ`.
- It incorrectly said `NQNU5R` can be cancelled because it is business class.
- It ignored that `NQNU5R` flights were on `2024-05-13` and `2024-05-14`, while current benchmark time is `2024-05-15 15:00:00 EST`.
- It called `cancel_reservation(NQNU5R)`.
- It also upgraded `M20IZO`.

Failure type:

- Temporal policy reasoning failure.
- Premature write action.
- Missing flight-date precondition before cancellation.

PRM-Lite v0 caught:

- Unexpected `cancel_reservation`.
- `cancel_after_24h_without_policy_basis`.
- Action mismatch because expected search actions were missing.

Important nuance:

- The main error is not "business class can never be cancelled"; the model overgeneralized "business can be cancelled" and missed the stronger constraint that already-flown/past reservations cannot be cancelled.

PRM-Lite v1 should add:

- `past_flight_cancel_attempt`: penalize cancellation when all relevant flight dates are before current time.
- `policy_precedence_error`: detect when the model applies a permissive rule while ignoring a stricter blocking rule.

## Task 23: Complex Payment Optimization And Wrong Strategy

Task goal:

- User wants totals for gift cards and certificates.
- Then wants to change recent reservation `K1NW8N` to cheapest business round trip with same dates.
- Because of one-certificate-per-reservation policy, the expected solution is:
  - Cancel `K1NW8N`.
  - Book three separate business round-trip reservations, one per passenger, using one certificate per reservation.
  - Communicate Mastercard charge of `$1286`.

Actual behavior:

- GLM correctly communicated:
  - Gift cards total: `$327`.
  - Certificates total: `$1000`.
- It searched direct and one-stop flights.
- It made several arithmetic errors:
  - Gave inconsistent Mastercard charges: `$1940`, `$2440`, then recalculated again.
  - Used `calculate`, but on the wrong expression and wrong price basis.
- It rejected the separate-reservation strategy for a while.
- It finally processed a single `update_reservation_flights` on `K1NW8N`.
- It did not cancel `K1NW8N`.
- It did not book the three expected separate reservations.
- It failed to communicate the required `$1286` Mastercard charge.

Failure type:

- Long-horizon planning failure.
- Payment optimization failure.
- Arithmetic inconsistency.
- Wrong write strategy: update instead of cancel-and-rebook.

PRM-Lite v0 caught:

- Expected writes were all missing.
- Unexpected `update_reservation_flights`.
- Communication failed for `$1286`.

PRM-Lite v1 should add:

- `payment_constraint_optimization_error`: one-certificate-per-reservation should trigger comparing single-reservation update vs split bookings.
- `calculation_trace_inconsistent`: penalize contradictory monetary totals across turns.
- `expected_write_plan_missing`: if task requires cancel+book, penalize update-only plans before final write.

## Cross-case Taxonomy Update

| Failure Category | Seen In | What It Means |
| --- | --- | --- |
| Tool affordance misunderstanding | 44 | Model chose a tool that cannot answer the required question. |
| Incomplete evidence | 44, 2, 37, 23 | Model acted or stopped before gathering enough records/tool outputs. |
| Object selection error | 2 | Model selected an entity without comparing all candidates. |
| Temporal policy error | 37, also task 1 | Model missed exact time/date constraints. |
| Policy precedence error | 37 | Model applied a permissive rule while ignoring a blocking rule. |
| Premature write action | 2, 37, 23 | Model wrote to DB before all preconditions were satisfied. |
| Long-horizon payment planning error | 23 | Model failed to optimize under payment constraints over many turns. |
| Communication-DB gap | all four | Model sounded helpful, but final DB state was wrong. |

## Next PRM-Lite v1 Rules

1. Schedule/duration tasks must use schedule-returning search tools, not only status tools.
2. "Latest/last/recent" object selection requires checking all candidate records and comparing `created_at`.
3. Cancellation requires checking:
   - booking age,
   - insurance,
   - airline-cancelled status,
   - whether the flight has already departed.
4. Certificate for delayed flights requires:
   - delay confirmed,
   - user intent compatible with policy,
   - change/cancellation precondition when required.
5. Payment optimization tasks require explicit comparison of candidate plans and stable arithmetic.

