# Slot Grounding Report: fixtures

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 2 |
| Checked tool calls | 3 |
| Simulations with issues | 1 |
| Total issues | 3 |

## Issue Counts

| Issue | Count |
| --- | ---: |
| `ask_identifier_and_call_tool_same_turn` | 1 |
| `transfer_after_ungrounded_slot_error` | 1 |
| `ungrounded_user_id` | 1 |

## Per Simulation

| Task | Trial | Tool calls | Issues | Details |
| ---: | ---: | ---: | ---: | --- |
| 18 | 0 | 2 | 3 | `ungrounded_user_id` get_user_details: user_id=sara_doe_496<br>`ask_identifier_and_call_tool_same_turn` get_user_details: user_id=sara_doe_496<br>`transfer_after_ungrounded_slot_error` transfer_to_human_agents: user_id=sara_doe_496 |
| 18 | 1 | 1 | 0 | - |
