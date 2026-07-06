# Decision Gate Behavior Evaluation

## Goal

Evaluate a decision-only adapter that predicts the next assistant action label: `assistant_text` or `tool_call`.

## Setup

- Base model: `models/qwen25-7b-instruct`
- Data: `data/decision_gate/tau2_airline_decision_gate_v1_2048_heldout.jsonl`
- Probes: `20`
- Max sequence length: `2048`
- Max new tokens: `8`
- Device: `cuda`

## Summary

| Model | Accuracy | Text recall | Tool recall | Unknown | Predicted labels |
| --- | ---: | ---: | ---: | ---: | --- |
| `base` | 0.000 | 0.000 | 0.000 | 1.000 | `{'unknown': 20}` |
| `gate_smoke` | 0.300 | 0.000 | 0.857 | 0.550 | `{'tool_call': 9, 'unknown': 11}` |

## Confusion

- `base`: `{'assistant_text->unknown': 13, 'tool_call->unknown': 7}`
- `gate_smoke`: `{'assistant_text->tool_call': 3, 'assistant_text->unknown': 10, 'tool_call->tool_call': 6, 'tool_call->unknown': 1}`

## Per-Probe Results

### base

| Probe | Task | Target | Pred | Match | Generated preview |
| --- | ---: | --- | --- | ---: | --- |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn2_text` | 2 | `assistant_text` | `unknown` | N | `Sure, I can help you with that` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn4_text` | 2 | `assistant_text` | `unknown` | N | `Great, Noah! With your user ID` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn6_text` | 2 | `assistant_text` | `unknown` | N | `I understand how frustrating travel disruptions can be` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn8_single_tool` | 2 | `tool_call` | `unknown` | N | `I understand how frustrating that must be.` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn10_seq_call0` | 2 | `tool_call` | `unknown` | N | `Thank you for providing that information. Based` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn10_seq_call1` | 2 | `tool_call` | `unknown` | N | `It looks like the reservation with the delayed` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn13_text` | 2 | `assistant_text` | `unknown` | N | `It looks like the reservation with the delayed` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn15_text` | 2 | `assistant_text` | `unknown` | N | `Thank you for the additional information. Based` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call0` | 2 | `tool_call` | `unknown` | N | `I understand how frustrating it can be when` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call1` | 2 | `tool_call` | `unknown` | N | `I've checked the status of the flight` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call2` | 2 | `tool_call` | `unknown` | N | `It seems that the flight statuses for both` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call3` | 2 | `tool_call` | `unknown` | N | `It seems that the flight statuses for all` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn22_text` | 2 | `assistant_text` | `unknown` | N | `It appears that all the flights in your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn24_text` | 2 | `assistant_text` | `unknown` | N | `I completely understand your frustration with flight delays` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn26_text` | 2 | `assistant_text` | `unknown` | N | `Absolutely, it's important for airlines to` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn28_text` | 2 | `assistant_text` | `unknown` | N | `Absolutely, I understand your perspective and the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn30_text` | 2 | `assistant_text` | `unknown` | N | `You're welcome! I appreciate your understanding` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn2_text` | 2 | `assistant_text` | `unknown` | N | `Sure, I can help you with that` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn4_text` | 2 | `assistant_text` | `unknown` | N | `Great, Noah! With your user ID` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn6_text` | 2 | `assistant_text` | `unknown` | N | `I understand how frustrating it must be to` |

### gate_smoke

| Probe | Task | Target | Pred | Match | Generated preview |
| --- | ---: | --- | --- | ---: | --- |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn2_text` | 2 | `assistant_text` | `unknown` | N | `Sure, I can help you with that` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn4_text` | 2 | `assistant_text` | `unknown` | N | `Great, Noah! With your user ID` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn6_text` | 2 | `assistant_text` | `unknown` | N | `I understand how frustrating travel disruptions can be` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn8_single_tool` | 2 | `tool_call` | `unknown` | N | `Unfortunately, without the reservation ID, it` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn10_seq_call0` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=call_99` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn10_seq_call1` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=call_8e` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn13_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call_id=call_8e` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn15_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call_id=call_8e` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call0` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=call_8e` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call1` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id: call_07` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call2` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=call_07` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call3` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id: call_8a` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn22_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call_id: call_07` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn24_text` | 2 | `assistant_text` | `unknown` | N | `Absolutely, flight delays can be incredibly disruptive` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn26_text` | 2 | `assistant_text` | `unknown` | N | `Absolutely, it's important for airlines to` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn28_text` | 2 | `assistant_text` | `unknown` | N | `Absolutely, I understand your perspective and the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn30_text` | 2 | `assistant_text` | `unknown` | N | `You're welcome! I'm glad you` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn2_text` | 2 | `assistant_text` | `unknown` | N | `Sure, I can help you with that` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn4_text` | 2 | `assistant_text` | `unknown` | N | `Great, Noah! With your user ID` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn6_text` | 2 | `assistant_text` | `unknown` | N | `I understand how frustrating it can be when` |

## Interpretation

- If text recall is low, the agent still over-calls tools.
- If tool recall is low, the gate blocks useful tool calls and the downstream tool policy cannot help.
- The target is a high-recall gate on both classes before combining it with Phase2H tool generation.
