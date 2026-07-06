# Baseline Analysis: DeepSeek Chat vs GLM 5.1 on tau2 Airline

Date: 2026-05-19

## Runs

| Run | Agent/User model | Tasks | Pass^1 | DB match | Communicate |
| --- | --- | ---: | ---: | ---: | ---: |
| `airline_debug_deepseek_chat_5tasks` | `deepseek/deepseek-chat` | 5 | 0.80 | 4/5 | 5/5 |
| `airline_debug_glm51_5tasks` | `anthropic/glm-5.1` | 5 | 0.60 | 3/5 | 5/5 |

Both models achieved full communication score on all five tasks. Failures came from backend state changes, not conversational politeness or task closure.

## Per-task Comparison

| Task | DeepSeek | GLM 5.1 | Main observation |
| --- | ---: | ---: | --- |
| 0 | Pass | Pass | Both handled the simple cancellation/refusal case. |
| 1 | Fail | Fail | Both incorrectly approved a cancellation that was outside the 24-hour refund window. |
| 2 | Pass | Fail | GLM selected the wrong "most recent" reservation and issued an inappropriate certificate. |
| 3 | Pass | Pass | Both completed the read/check/communicate workflow. |
| 4 | Pass | Pass | Both completed the multi-reservation lookup workflow. |

## Failure 1: Temporal Policy Reasoning

Task 1 tests whether the agent can refuse a cancellation that has been implicitly "approved" by a phone representative.

Ground truth:

- Target reservation: `Q69X3R`
- Created at: `2024-05-14T09:52:38`
- Benchmark current time: around `2024-05-15 15:00:00 EST`
- The booking is more than 24 hours old, so the agent should not approve cancellation/refund.

Observed behavior:

- DeepSeek and GLM both retrieved the correct reservation.
- Both treated "May 14 to May 15" as if it were automatically within 24 hours.
- Both then called `cancel_reservation`.
- Result: `COMMUNICATE=1.0`, `DB=0.0`.

Diagnosis:

This is not a tool-use failure. It is a policy arithmetic failure: the model failed to compute exact elapsed time and over-relied on coarse date wording.

## Failure 2: Incomplete Evidence Gathering

Task 2 tests topic switching plus claim verification. The user first asks to book a flight, then pivots to a delayed flight in the "most recent reservation."

Ground truth task expectations include:

- `get_user_details(noah_muller_9847)`
- `get_reservation_details(SDZQKO)`
- `get_reservation_details(4OG6T3)`
- The agent should verify the actual delayed flight and should not offer a $50 certificate unless policy permits it.

Observed GLM behavior:

- GLM retrieved user details.
- It directly selected reservation `4OG6T3` as the "most recent" reservation.
- It did not fully compare all relevant reservations by `created_at`.
- It accepted the wrong context and later issued `send_certificate(amount=50)`.
- Result: `COMMUNICATE=1.0`, `DB=0.0`.

Diagnosis:

This is an evidence-completeness failure. The model can call tools, but it does not know when it has enough evidence to safely act.

## Quantitative Notes

| Metric | DeepSeek | GLM 5.1 |
| --- | ---: | ---: |
| Avg reward | 0.80 | 0.60 |
| Successful tasks | 4 | 3 |
| Failed tasks | 1 | 2 |
| Read/action match | 13/13 | 12/13 |
| Communication score | 5/5 | 5/5 |

GLM was also slower in this run. Some LiteLLM cost logs show provider-mapping warnings, so the reported `$0.0000` cost should not be trusted as real billing information.

## What This Means For The Project

The baseline already reveals two useful training targets:

1. Temporal rule checking: compute exact elapsed time before refund/cancellation actions.
2. Evidence completeness: before acting on "latest", "most recent", "delayed", "eligible", or "approved", gather all needed records and compare them explicitly.

These are good candidates for a small but meaningful post-training dataset:

- Positive trajectories where the agent computes time and refuses correctly.
- Negative trajectories where a model prematurely acts.
- Process rewards for checking all relevant reservations before write tools.
- Penalties for write actions after incomplete evidence.

## Next Step

Build a failure taxonomy table over more tasks:

| Category | Example | Reward signal idea |
| --- | --- | --- |
| Temporal policy error | Task 1 | Reward exact timestamp comparison before cancellation. |
| Incomplete retrieval | Task 2 | Reward checking all candidate reservations before selecting one. |
| Premature write action | Task 1/2 | Penalize write tools when required preconditions are missing. |
| Communication-only success illusion | All failed cases | Track DB reward separately from communication reward. |

