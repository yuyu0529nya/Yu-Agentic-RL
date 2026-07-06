# Slot Grounding Report: phase2c_recovery_v5_4090_20260615_1113

## Summary

| Metric | Value |
| --- | ---: |
| Simulations | 3 |
| Checked tool calls | 4 |
| Simulations with issues | 2 |
| Total issues | 6 |

## Issue Counts

| Issue | Count |
| --- | ---: |
| `ask_identifier_and_call_tool_same_turn` | 2 |
| `transfer_after_ungrounded_slot_error` | 2 |
| `ungrounded_user_id` | 2 |

## Per Simulation

| Task | Trial | Tool calls | Issues | Details |
| ---: | ---: | ---: | ---: | --- |
| 18 | 0 | 2 | 3 | `ungrounded_user_id` get_user_details: user_id=sara_doe_496<br>`ask_identifier_and_call_tool_same_turn` get_user_details: user_id=sara_doe_496<br>`transfer_after_ungrounded_slot_error` transfer_to_human_agents: user_id=sara_doe_496 |
| 25 | 0 | 0 | 0 | - |
| 44 | 0 | 2 | 3 | `ungrounded_user_id` get_user_details: user_id=sara_doe_496<br>`ask_identifier_and_call_tool_same_turn` get_user_details: user_id=sara_doe_496<br>`transfer_after_ungrounded_slot_error` transfer_to_human_agents: user_id=sara_doe_496 |
