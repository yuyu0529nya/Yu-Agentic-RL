# Phase 2D: Slot-Grounded PRM Validation

Date: 2026-06-15

## Goal

Phase 2C showed that Recovery-Prefix v5 can learn offline next-action behavior, but targeted tau2 eval still failed on tasks 18, 25, and 44. The most important live failure was not generic tool selection. It was slot hallucination:

- The model called tools with an ungrounded `user_id` such as `sara_doe_496`.
- After the invalid tool result, it escalated or transferred instead of recovering.
- In task 25, the model did not hallucinate a slot, but looped instead of completing the booking flow.

Phase 2D adds an explicit slot-grounding validator and wires it into PRM-Lite so rerank/training analysis can separate "tool choice" from "argument grounding".

## Implemented

- Added `scripts/slot_grounding_validator.py`.
- Integrated slot-grounding penalties into `scripts/process_reward_scorer.py`.
- Added PRM ablations in `scripts/prm_ablation_tau2.py`:
  - `no_slot_grounding`
  - `slot_grounding_only`
- Added a minimal rerank fixture:
  - `tests/fixtures/slot_grounding_rerank_results.json`

## Validator Behavior

The validator tracks grounded slots from prior visible conversation state:

- User-provided identifiers.
- Successful tool outputs.
- Reservation/payment ids observed before later write actions.

It flags:

- `ungrounded_user_id`
- `ungrounded_reservation_id`
- `ungrounded_payment_id`
- `ask_identifier_and_call_tool_same_turn`
- `transfer_after_ungrounded_slot_error`

`flight_number` is observed but not hard-penalized for update/book calls in v1, because passing tau2 trajectories can contain flight numbers only implied by backend search results. This avoids false positives on successful simulations.

## Verification

### 1. Syntax Check

Command:

```powershell
python -m py_compile scripts\slot_grounding_validator.py scripts\process_reward_scorer.py scripts\prm_rerank_tau2.py scripts\prm_ablation_tau2.py
```

Result: passed.

### 2. Phase 2C Failed Targeted Run

Input:

```text
autodl_artifacts/phase2c_recovery_v5_4090_20260615_1113/...target_3task...results.json
```

Output:

```text
reports/phase2c_slot_grounding_v5_target_3task.json
reports/phase2c_slot_grounding_v5_target_3task.md
reports/phase2c_prm_v5_target_3task.json
reports/phase2c_prm_v5_target_3task.md
reports/phase2c_prm_v5_target_3task.csv
```

Slot validator summary:

| Metric | Value |
|---|---:|
| simulations | 3 |
| checked tool calls | 4 |
| simulations with issues | 2 |
| total issues | 6 |
| ungrounded_user_id | 2 |
| ask_identifier_and_call_tool_same_turn | 2 |
| transfer_after_ungrounded_slot_error | 2 |

PRM summary:

| Metric | Value |
|---|---:|
| avg reward | 0.0000 |
| avg process score | -13.8333 |
| slot_grounding_error sims | 2 |
| ungrounded_user_id sims | 2 |

Interpretation:

- Tasks 18 and 44 are now correctly diagnosed as slot-grounding failures.
- Task 25 is not a slot hallucination case; it remains a separate flow-completion/recovery failure.

### 3. Phase 2B Passing Smoke Run

Input:

```text
autodl_artifacts/phase2b_recovery_v4_allguards_sftonly_2task_4090_20260614_2207/...2task...results.json
```

Output:

```text
reports/phase2c_slot_grounding_v4_pass_2task.json
reports/phase2c_slot_grounding_v4_pass_2task.md
```

Slot validator summary:

| Metric | Value |
|---|---:|
| simulations | 2 |
| checked tool calls | 7 |
| simulations with issues | 0 |
| total issues | 0 |

Interpretation:

- The validator no longer false-penalizes known passing trajectories.

### 4. Rerank Fixture

Input:

```text
tests/fixtures/slot_grounding_rerank_results.json
```

The fixture contains two candidates for the same task:

- Trial 0: asks for user id but also calls `get_user_details` with hallucinated `sara_doe_496`, then transfers.
- Trial 1: waits for user-provided `omar_davis_3817`, then calls `get_user_details` with a grounded argument.

Validator summary:

| Metric | Value |
|---|---:|
| simulations | 2 |
| checked tool calls | 3 |
| simulations with issues | 1 |
| total issues | 3 |

PRM-rerank result:

| Metric | Value |
|---|---:|
| first_trial_pass | 0.0000 |
| sample_success_rate | 0.5000 |
| oracle_pass@N | 1.0000 |
| prm_rerank_pass@N | 1.0000 |
| gain_vs_first | +1.0000 |

Selected candidate:

```text
trial = 1
reward = 1.0
process_score = -0.5
```

Interpretation:

- Slot-grounded PRM can choose the grounded candidate over the hallucinated candidate when both are present in the sample set.

## Conclusion

Phase 2D is complete as a local verification step.

We now have a stronger diagnostic and rerank signal:

- It catches the exact live regression seen in Phase 2C.
- It avoids false positives on known passing simulations.
- It changes PRM-rerank behavior in the intended direction on a controlled fixture.

The next useful GPU step is not more blind SFT. It is a targeted tau2 rerank evaluation with multiple samples per task:

- Run Qwen2.5-7B base and Recovery-Prefix v5 with `N=4` or `N=8`.
- Compare first-sample pass^1, oracle pass@N, and slot-grounded PRM rerank pass@N.
- If oracle pass@N is high but rerank pass@N is low, improve PRM.
- If oracle pass@N is also low, collect more recovery/slot-grounding training data.
