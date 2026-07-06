# Decision Gate Behavior Evaluation

## Goal

Evaluate a decision-only adapter that predicts the next assistant action label: `assistant_text` or `tool_call`.

## Setup

- Base model: `models/qwen25-7b-instruct`
- Data: `data/decision_gate/tau2_airline_decision_gate_v1_2048_heldout.jsonl`
- Probes: `195`
- Max sequence length: `2048`
- Max new tokens: `8`
- Device: `cuda`

## Summary

| Model | Accuracy | Text recall | Tool recall | Unknown | Predicted labels |
| --- | ---: | ---: | ---: | ---: | --- |
| `base` | 0.015 | 0.000 | 0.030 | 0.985 | `{'tool_call': 3, 'unknown': 192}` |
| `gate_repro_s27` | 0.610 | 0.415 | 0.792 | 0.000 | `{'assistant_text': 60, 'tool_call': 135}` |

## Confusion

- `base`: `{'assistant_text->unknown': 94, 'tool_call->tool_call': 3, 'tool_call->unknown': 98}`
- `gate_repro_s27`: `{'assistant_text->assistant_text': 39, 'assistant_text->tool_call': 55, 'tool_call->assistant_text': 21, 'tool_call->tool_call': 80}`

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
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn8_single_tool` | 2 | `tool_call` | `unknown` | N | `Of course, I can help you try` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn10_single_tool` | 2 | `tool_call` | `unknown` | N | `Thank you for providing that information. Based` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn12_single_tool` | 2 | `tool_call` | `unknown` | N | `It looks like the reservation SDZQ` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn14_text` | 2 | `assistant_text` | `unknown` | N | `It looks like the reservation with ID `` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn16_text` | 2 | `assistant_text` | `unknown` | N | `I apologize for the inconvenience caused by the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn18_single_tool` | 2 | `tool_call` | `unknown` | N | `Thank you for clarifying. I understand` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn20_single_tool` | 2 | `tool_call` | `unknown` | N | `I checked the status of the outbound flight` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn22_text` | 2 | `assistant_text` | `unknown` | N | `I checked the status of your flights on` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn24_text` | 2 | `assistant_text` | `unknown` | N | `I completely understand your feelings about the delay` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn26_text` | 2 | `assistant_text` | `unknown` | N | `Absolutely, I understand your concerns and appreciate` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn2_text` | 16 | `assistant_text` | `unknown` | N | `Of course, I can help you with` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn4_single_tool` | 16 | `tool_call` | `unknown` | N | `Thank you for providing your user ID,` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn6_seq_call0` | 16 | `tool_call` | `unknown` | N | `I found your reservation details. You have` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn6_seq_call1` | 16 | `tool_call` | `unknown` | N | `It looks like your reservation M05` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn9_seq_call0` | 16 | `tool_call` | `unknown` | N | `It looks like the reservation M05` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn9_seq_call1` | 16 | `tool_call` | `unknown` | N | `It seems there are no direct flights available` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn12_text` | 16 | `assistant_text` | `unknown` | N | `Based on the provided flight data, here` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn14_text` | 16 | `assistant_text` | `unknown` | N | `Great choice! Let's proceed with Option` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn16_single_tool` | 16 | `tool_call` | `unknown` | N | `Great! I have confirmed and processed the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn18_text` | 16 | `assistant_text` | `unknown` | N | `The reservation has been successfully updated to the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn2_text` | 18 | `assistant_text` | `unknown` | N | `Of course, I can help you with` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn4_single_tool` | 18 | `tool_call` | `unknown` | N | `Sure, Omar, I can assist you` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call0` | 18 | `tool_call` | `unknown` | N | `Thank you for providing your user ID and` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call1` | 18 | `tool_call` | `unknown` | N | `It looks like you have a one-way` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call2` | 18 | `tool_call` | `unknown` | N | `It looks like one of your reservations is` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call3` | 18 | `tool_call` | `unknown` | N | `It looks like you have several reservations,` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call4` | 18 | `tool_call` | `tool_call` | Y | `<tool_call>` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call5` | 18 | `tool_call` | `tool_call` | Y | `<tool_call>` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn13_text` | 18 | `assistant_text` | `unknown` | N | `Here is a summary of the reservations made` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn15_text` | 18 | `assistant_text` | `unknown` | N | `Understood, Omar. I will proceed` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn17_single_tool` | 18 | `tool_call` | `unknown` | N | `Sure, I can calculate the savings for` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn19_single_tool` | 18 | `tool_call` | `unknown` | N | `I've successfully downgraded the first reservation` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn21_single_tool` | 18 | `tool_call` | `unknown` | N | `I've successfully downgraded the first two` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn23_single_tool` | 18 | `tool_call` | `tool_call` | Y | `<tool_call>` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn25_single_tool` | 18 | `tool_call` | `unknown` | N | `I have processed the downgrades for all` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn27_single_tool` | 18 | `tool_call` | `unknown` | N | `The reservation with ID `BOH1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn29_text` | 18 | `assistant_text` | `unknown` | N | `The total amount paid for the reservations is` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn2_text` | 25 | `assistant_text` | `unknown` | N | `Of course! To help you duplicate your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn4_single_tool` | 25 | `tool_call` | `unknown` | N | `Great, thank you for providing your user` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn6_single_tool` | 25 | `tool_call` | `unknown` | N | `I've retrieved your details, Ivan.` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn8_text` | 25 | `assistant_text` | `unknown` | N | `Based on your current reservation details, here` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn10_text` | 25 | `assistant_text` | `unknown` | N | `Sure, I can help with that.` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn12_text` | 25 | `assistant_text` | `unknown` | N | `Sure, I can proceed with using the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn14_single_tool` | 25 | `tool_call` | `unknown` | N | `Great! I've confirmed and booked the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn16_text` | 25 | `assistant_text` | `unknown` | N | `It looks like there was a miscal` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn18_single_tool` | 25 | `tool_call` | `unknown` | N | `Great! I will proceed with booking the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn20_text` | 25 | `assistant_text` | `unknown` | N | `The reservation has been successfully booked! Here` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn2_text` | 25 | `assistant_text` | `unknown` | N | `Of course! To help you duplicate your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn4_single_tool` | 25 | `tool_call` | `unknown` | N | `Great, thank you for providing your user` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn6_single_tool` | 25 | `tool_call` | `unknown` | N | `Based on the information provided, your current` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn8_text` | 25 | `assistant_text` | `unknown` | N | `Based on the details of your current reservation` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn10_single_tool` | 25 | `tool_call` | `unknown` | N | `Got it! Let's proceed step by` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn12_single_tool` | 25 | `tool_call` | `unknown` | N | `Based on the available flights, here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn14_text` | 25 | `assistant_text` | `unknown` | N | `Based on the available flights, here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn16_text` | 25 | `assistant_text` | `unknown` | N | `Got it! Let's break down the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn18_single_tool` | 25 | `tool_call` | `unknown` | N | `Great! I will proceed with booking the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn20_text` | 25 | `assistant_text` | `unknown` | N | `The reservation has been successfully booked! Here` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn2_text` | 25 | `assistant_text` | `unknown` | N | `Of course! To help you duplicate your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn4_single_tool` | 25 | `tool_call` | `unknown` | N | `Great, thank you for providing your user` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn6_single_tool` | 25 | `tool_call` | `unknown` | N | `I've retrieved your details, Ivan.` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn8_text` | 25 | `assistant_text` | `unknown` | N | `Based on your current reservation details, here` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn10_text` | 25 | `assistant_text` | `unknown` | N | `Got it! Let's proceed with the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn12_text` | 25 | `assistant_text` | `unknown` | N | `Sure, that works! Here’s how` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn14_single_tool` | 25 | `tool_call` | `unknown` | N | `Great! I'll proceed with the booking` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn16_text` | 25 | `assistant_text` | `unknown` | N | `It looks like there was an error because` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn18_text` | 25 | `assistant_text` | `unknown` | N | `Great! I'll proceed with the booking` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn20_single_tool` | 25 | `tool_call` | `unknown` | N | `Great! I will proceed with the booking` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn22_text` | 25 | `assistant_text` | `unknown` | N | `The reservation has been successfully booked! Here` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn2_text` | 25 | `assistant_text` | `unknown` | N | `Of course! To help you duplicate your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn4_single_tool` | 25 | `tool_call` | `unknown` | N | `Great, thank you for providing your user` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn6_single_tool` | 25 | `tool_call` | `unknown` | N | `Based on the information provided, your current` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn8_text` | 25 | `assistant_text` | `unknown` | N | `Based on your current reservation details, here` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn10_single_tool` | 25 | `tool_call` | `unknown` | N | `Got it! Let's proceed step by` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn12_single_tool` | 25 | `tool_call` | `unknown` | N | `Based on the available flights, here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn14_text` | 25 | `assistant_text` | `unknown` | N | `Based on the available flights, here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn16_text` | 25 | `assistant_text` | `unknown` | N | `Sure, I can proceed with booking the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn18_single_tool` | 25 | `tool_call` | `unknown` | N | `Great! I will now proceed with booking` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn20_text` | 25 | `assistant_text` | `unknown` | N | `The reservation has been successfully booked for Ivan` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn2_text` | 32 | `assistant_text` | `unknown` | N | `I understand that this situation is urgent and` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn4_single_tool` | 32 | `tool_call` | `unknown` | N | `Of course, Ivan, I understand that` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn6_seq_call0` | 32 | `tool_call` | `unknown` | N | `I've found your reservation details, Ivan` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn6_seq_call1` | 32 | `tool_call` | `unknown` | N | `It looks like your reservation OWZ4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn6_seq_call2` | 32 | `tool_call` | `unknown` | N | `It seems that the reservation you provided (` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn6_seq_call3` | 32 | `tool_call` | `unknown` | N | `It looks like none of the reservations I` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn11_text` | 32 | `assistant_text` | `unknown` | N | `I've checked all your reservations, and` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn13_single_tool` | 32 | `tool_call` | `unknown` | N | `Sure, I can help you upgrade your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn15_text` | 32 | `assistant_text` | `unknown` | N | `I've found a direct flight option for` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn17_text` | 32 | `assistant_text` | `unknown` | N | `Understood. Let's proceed with upgrading` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn19_single_tool` | 32 | `tool_call` | `unknown` | N | `Thank you for clarifying. Let's` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn21_text` | 32 | `assistant_text` | `unknown` | N | `I apologize for the confusion. It appears` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn23_text` | 32 | `assistant_text` | `unknown` | N | `Understood. Let's handle both changes` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn25_text` | 32 | `assistant_text` | `unknown` | N | `Understood. Let's proceed with the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn27_text` | 32 | `assistant_text` | `unknown` | N | `Understood. Let's proceed with the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn29_single_tool` | 32 | `tool_call` | `unknown` | N | `Great! I've processed the upgrade from` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn31_text` | 32 | `assistant_text` | `unknown` | N | `I apologize for the inconvenience. It seems` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn33_single_tool` | 32 | `tool_call` | `unknown` | N | `Sure, I'll proceed with the upgrade` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn35_text` | 32 | `assistant_text` | `unknown` | N | `The upgrade to economy class on the current` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn37_text` | 32 | `assistant_text` | `unknown` | N | `I apologize for the confusion. Let's` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn39_text` | 32 | `assistant_text` | `unknown` | N | `Sure, I understand your frustration, and` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn41_single_tool` | 32 | `tool_call` | `unknown` | N | `Sure, I will proceed with the flight` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn43_text` | 32 | `assistant_text` | `unknown` | N | `**✅ Step 2 Complete: Non` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn2_text` | 32 | `assistant_text` | `unknown` | N | `I understand that this situation is urgent and` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn4_single_tool` | 32 | `tool_call` | `unknown` | N | `Of course, I understand that this is` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn6_seq_call0` | 32 | `tool_call` | `unknown` | N | `I've found your reservation details, Ivan` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn6_seq_call1` | 32 | `tool_call` | `unknown` | N | `It looks like your reservation OWZ4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn6_seq_call2` | 32 | `tool_call` | `unknown` | N | `It seems there are no reservations matching your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn6_seq_call3` | 32 | `tool_call` | `unknown` | N | `It looks like none of the reservations I` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn11_text` | 32 | `assistant_text` | `unknown` | N | `I've found your reservation for the flight` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn13_text` | 32 | `assistant_text` | `unknown` | N | `Certainly, upgrading your ticket to regular economy` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn15_text` | 32 | `assistant_text` | `unknown` | N | `Certainly, let's try to upgrade your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn17_single_tool` | 32 | `tool_call` | `unknown` | N | `Certainly, I will proceed with the upgrade` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn19_text` | 32 | `assistant_text` | `unknown` | N | `I found a direct flight available on May` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn21_text` | 32 | `assistant_text` | `unknown` | N | `Understood. Let's proceed with the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn23_single_tool` | 32 | `tool_call` | `unknown` | N | `Great! I will now proceed with the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn25_text` | 32 | `assistant_text` | `unknown` | N | `The cabin upgrade has been successfully processed for` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn27_text` | 32 | `assistant_text` | `unknown` | N | `Yes, that's correct! The new` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn29_single_tool` | 32 | `tool_call` | `unknown` | N | `Great! I will now attempt to change` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn31_text` | 32 | `assistant_text` | `unknown` | N | `**The flight change to the nonstop` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn2_text` | 32 | `assistant_text` | `unknown` | N | `I understand that this situation is urgent and` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn4_single_tool` | 32 | `tool_call` | `unknown` | N | `Of course, I understand that this is` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn6_seq_call0` | 32 | `tool_call` | `unknown` | N | `I've found your reservations. Here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn6_seq_call1` | 32 | `tool_call` | `unknown` | N | `It looks like your reservation OWZ4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn6_seq_call2` | 32 | `tool_call` | `unknown` | N | `It looks like the reservation you're looking` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn6_seq_call3` | 32 | `tool_call` | `unknown` | N | `It looks like none of the reservations listed` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn11_text` | 32 | `assistant_text` | `unknown` | N | `I've checked all your reservations, and` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn13_text` | 32 | `assistant_text` | `unknown` | N | `Sure, I can help you upgrade your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn15_text` | 32 | `assistant_text` | `unknown` | N | `Sure, I'll proceed with upgrading your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn17_single_tool` | 32 | `tool_call` | `unknown` | N | `Great! I've upgraded your cabin class` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn19_single_tool` | 32 | `tool_call` | `unknown` | N | `The cabin class for your reservation OWZ` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn21_text` | 32 | `assistant_text` | `unknown` | N | `I found a nonstop flight option for` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn23_text` | 32 | `assistant_text` | `unknown` | N | `Absolutely, I'll provide you with the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn25_single_tool` | 32 | `tool_call` | `unknown` | N | `I'll proceed with changing your flights to` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn27_text` | 32 | `assistant_text` | `unknown` | N | `The update to your reservation has been completed` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn2_text` | 37 | `assistant_text` | `unknown` | N | `Of course, I can help you with` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn4_single_tool` | 37 | `tool_call` | `unknown` | N | `Great, thank you for providing your user` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn6_seq_call0` | 37 | `tool_call` | `unknown` | N | `I've found your reservations. Here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn6_seq_call1` | 37 | `tool_call` | `unknown` | N | `Here are the details for reservation M2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn6_seq_call2` | 37 | `tool_call` | `unknown` | N | `Here are the details of your reservations:` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn6_seq_call3` | 37 | `tool_call` | `unknown` | N | `Here are the details of your reservations:` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn11_text` | 37 | `assistant_text` | `unknown` | N | `Here are the details of your four reservations` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn13_seq_call0` | 37 | `tool_call` | `unknown` | N | `Sure, I understand your request. Let` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn13_seq_call1` | 37 | `tool_call` | `unknown` | N | `Based on the available direct flights on May` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn16_text` | 37 | `assistant_text` | `unknown` | N | `### Upgrade Options for M20IZ` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn18_text` | 37 | `assistant_text` | `unknown` | N | `Great! I'll proceed with upgrading your` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn20_single_tool` | 37 | `tool_call` | `unknown` | N | `Great! I will now proceed with upgrading` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn22_text` | 37 | `assistant_text` | `unknown` | N | `The reservation M20IZO has` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn2_text` | 37 | `assistant_text` | `unknown` | N | `Of course, I'd be happy to` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn4_single_tool` | 37 | `tool_call` | `unknown` | N | `Great, thank you for providing your user` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn6_text` | 37 | `assistant_text` | `unknown` | N | `I've retrieved your account details, A` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn8_seq_call0` | 37 | `tool_call` | `unknown` | N | `Got it! You want to cancel reservations` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn8_seq_call1` | 37 | `tool_call` | `unknown` | N | `Here are the details for the reservation I` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn8_seq_call2` | 37 | `tool_call` | `unknown` | N | `Here are the details for the reservations you` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn12_text` | 37 | `assistant_text` | `unknown` | N | `Here are the details for the reservations you` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn14_seq_call0` | 37 | `tool_call` | `unknown` | N | `Great! I will proceed with searching for` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn14_seq_call1` | 37 | `tool_call` | `unknown` | N | `I found the available direct flights from JFK` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn17_seq_call0` | 37 | `tool_call` | `unknown` | N | `Based on the available flights, here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn17_seq_call1` | 37 | `tool_call` | `unknown` | N | `The cost to upgrade your reservation M2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn20_text` | 37 | `assistant_text` | `unknown` | N | `Based on the available flights, here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn22_text` | 37 | `assistant_text` | `unknown` | N | `Great! I will now proceed with upgrading` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn24_single_tool` | 37 | `tool_call` | `unknown` | N | `Great! I will now proceed with upgrading` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn26_text` | 37 | `assistant_text` | `unknown` | N | `The upgrade has been successfully processed. Here` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn2_text` | 37 | `assistant_text` | `unknown` | N | `Of course, I can help you with` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn4_single_tool` | 37 | `tool_call` | `unknown` | N | `Great, thank you for providing your user` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn6_text` | 37 | `assistant_text` | `unknown` | N | `Based on the information provided, here are` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn8_single_tool` | 37 | `tool_call` | `unknown` | N | `Great! I will proceed with canceling` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn10_single_tool` | 37 | `tool_call` | `unknown` | N | `Great! Here are the details for the` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn12_single_tool` | 37 | `tool_call` | `unknown` | N | `It looks like there was a mix-up` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn14_text` | 37 | `assistant_text` | `unknown` | N | `Great! Here are the details for each` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn16_single_tool` | 37 | `tool_call` | `unknown` | N | `Great! To upgrade reservation M20` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn18_single_tool` | 37 | `tool_call` | `unknown` | N | `Based on the available flight information for May` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn20_single_tool` | 37 | `tool_call` | `unknown` | N | `Based on the available flights on May` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn22_single_tool` | 37 | `tool_call` | `unknown` | N | `The upgrade cost for both flights from economy` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn24_text` | 37 | `assistant_text` | `unknown` | N | `The total cost to upgrade both flights to` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn26_single_tool` | 37 | `tool_call` | `unknown` | N | `Great! I will proceed with the upgrade` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn28_text` | 37 | `assistant_text` | `unknown` | N | `Your reservation M20IZO has` |

### gate_repro_s27

| Probe | Task | Target | Pred | Match | Generated preview |
| --- | ---: | --- | --- | ---: | --- |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn2_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn4_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn6_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_textassistant_textassistant_textassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn8_single_tool` | 2 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn10_seq_call0` | 2 | `tool_call` | `tool_call` | Y | `tool_call tool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn10_seq_call1` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_8` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn13_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call_idtool_call_idtool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn15_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call0` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call1` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_0` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call2` | 2 | `tool_call` | `tool_call` | Y | `tool_call_texttool_call_texttool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn17_seq_call3` | 2 | `tool_call` | `tool_call` | Y | `tool_call_texttool_call_id=tool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn22_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn24_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_textassistant_texttool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn26_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn28_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial2_turn30_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_text="assistant_text":tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn2_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn4_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn6_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_textassistant_textassistant_textassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn8_single_tool` | 2 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn10_single_tool` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn12_single_tool` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_7` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn14_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call_texttool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn16_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_5` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn18_single_tool` | 2 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn20_single_tool` | 2 | `tool_call` | `tool_call` | Y | `tool_call_texttool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn22_text` | 2 | `assistant_text` | `tool_call` | N | `tool_call_texttool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn24_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_text=assistant_textassistant_textassistant` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task2_trial3_turn26_text` | 2 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn2_text` | 16 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn4_single_tool` | 16 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn6_seq_call0` | 16 | `tool_call` | `tool_call` | Y | `tool_call_id=get_reservation_details tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn6_seq_call1` | 16 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn9_seq_call0` | 16 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn9_seq_call1` | 16 | `tool_call` | `tool_call` | Y | `tool_call_texttool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn12_text` | 16 | `assistant_text` | `assistant_text` | Y | `assistant_text = """` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn14_text` | 16 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn16_single_tool` | 16 | `tool_call` | `tool_call` | Y | `tool_callassistant_texttool_callassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task16_trial0_turn18_text` | 16 | `assistant_text` | `tool_call` | N | `tool_call tool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn2_text` | 18 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn4_single_tool` | 18 | `tool_call` | `tool_call` | Y | `tool_call(text="tool_call texttool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call0` | 18 | `tool_call` | `tool_call` | Y | `tool_call_idtool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call1` | 18 | `tool_call` | `tool_call` | Y | `tool_calltool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call2` | 18 | `tool_call` | `tool_call` | Y | `tool_calltool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call3` | 18 | `tool_call` | `tool_call` | Y | `tool_call_texttool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call4` | 18 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn6_seq_call5` | 18 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn13_text` | 18 | `assistant_text` | `assistant_text` | Y | `assistant_text=assistant_textassistant_textassistant` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn15_text` | 18 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn17_single_tool` | 18 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn19_single_tool` | 18 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_9` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn21_single_tool` | 18 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn23_single_tool` | 18 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn25_single_tool` | 18 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn27_single_tool` | 18 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_ids=[tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task18_trial2_turn29_text` | 18 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_b0` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn2_text` | 25 | `assistant_text` | `assistant_text` | Y | `assistant_text/plainassistant_text/plainassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn4_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn6_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn8_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_d2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn10_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn12_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn14_single_tool` | 25 | `tool_call` | `assistant_text` | N | `assistant_text="assistant_texttool_call_id` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn16_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_texttool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn18_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_d8` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial0_turn20_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call tool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn2_text` | 25 | `assistant_text` | `assistant_text` | Y | `assistant_text/plainassistant_text/plainassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn4_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn6_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_idtool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn8_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn10_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn12_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_idtool_call_idtool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn14_text` | 25 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn16_text` | 25 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn18_single_tool` | 25 | `tool_call` | `assistant_text` | N | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial1_turn20_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_status_texttool_call_status_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn2_text` | 25 | `assistant_text` | `assistant_text` | Y | `assistant_text/plainassistant_text/plainassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn4_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn6_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_idtool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn8_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn10_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn12_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn14_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn16_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_texttool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn18_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_f0` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn20_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_f0` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial2_turn22_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call tool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn2_text` | 25 | `assistant_text` | `assistant_text` | Y | `assistant_text/plainassistant_text/plainassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn4_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn6_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn8_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn10_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn12_single_tool` | 25 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn14_text` | 25 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn16_text` | 25 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn18_single_tool` | 25 | `tool_call` | `assistant_text` | N | `assistant_text="assistant_text":assistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task25_trial3_turn20_text` | 25 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_f2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn2_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn4_single_tool` | 32 | `tool_call` | `tool_call` | Y | `tool_call(text="tool_call_texttool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn6_seq_call0` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn6_seq_call1` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn6_seq_call2` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn6_seq_call3` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn11_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn13_single_tool` | 32 | `tool_call` | `assistant_text` | N | `assistant_text=assistant_textassistant_texttool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn15_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn17_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn19_single_tool` | 32 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn21_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_textassistant_texttool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn23_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_text=assistant_textassistant_texttool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn25_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn27_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn29_single_tool` | 32 | `tool_call` | `assistant_text` | N | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn31_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_9` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn33_single_tool` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn35_text` | 32 | `assistant_text` | `tool_call` | N | `tool_calltool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn37_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_9` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn39_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn41_single_tool` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial0_turn43_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_result_texttooltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn2_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn4_single_tool` | 32 | `tool_call` | `tool_call` | Y | `tool_call(text="tool_call_texttool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn6_seq_call0` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_0` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn6_seq_call1` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_9` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn6_seq_call2` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn6_seq_call3` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_9` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn11_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_idtool_call_idtool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn13_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn15_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn17_single_tool` | 32 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn19_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_idtool_call_id` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn21_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn23_single_tool` | 32 | `tool_call` | `assistant_text` | N | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn25_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_5` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn27_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn29_single_tool` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial1_turn31_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_texttool_call_texttool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn2_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn4_single_tool` | 32 | `tool_call` | `tool_call` | Y | `tool_call(text="tool_call_texttool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn6_seq_call0` | 32 | `tool_call` | `tool_call` | Y | `tool_call_idtool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn6_seq_call1` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn6_seq_call2` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_6` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn6_seq_call3` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_7` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn11_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_idtool_call_idtool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn13_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_text=assistant_textassistant_texttool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn15_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn17_single_tool` | 32 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn19_single_tool` | 32 | `tool_call` | `assistant_text` | N | `assistant_text=assistant_texttool_call_id` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn21_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_f5` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn23_text` | 32 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn25_single_tool` | 32 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task32_trial2_turn27_text` | 32 | `assistant_text` | `assistant_text` | Y | `assistant_text:assistant_texttool_call_status` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn2_text` | 37 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn4_single_tool` | 37 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn6_seq_call0` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn6_seq_call1` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_eca` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn6_seq_call2` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_eca` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn6_seq_call3` | 37 | `tool_call` | `tool_call` | Y | `tool_call_texttool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn11_text` | 37 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn13_seq_call0` | 37 | `tool_call` | `assistant_text` | N | `assistant_text=assistant_textassistant_texttool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn13_seq_call1` | 37 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_idtool_call_id` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn16_text` | 37 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn18_text` | 37 | `assistant_text` | `assistant_text` | Y | `assistant_text="assistant_text":assistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn20_single_tool` | 37 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial0_turn22_text` | 37 | `assistant_text` | `tool_call` | N | `tool_call tool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn2_text` | 37 | `assistant_text` | `assistant_text` | Y | `assistant_textassistant_textassistant_textassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn4_single_tool` | 37 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn6_text` | 37 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn8_seq_call0` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn8_seq_call1` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn8_seq_call2` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn12_text` | 37 | `assistant_text` | `tool_call` | N | `tool_call_idtool_call_idtool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn14_seq_call0` | 37 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_id=tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn14_seq_call1` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_4` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn17_seq_call0` | 37 | `tool_call` | `assistant_text` | N | `assistant_texttool_call_idtool_call_id` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn17_seq_call1` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_0` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn20_text` | 37 | `assistant_text` | `assistant_text` | Y | `assistant_text:assistant_textassistant_textassistant` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn22_text` | 37 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn24_single_tool` | 37 | `tool_call` | `assistant_text` | N | `assistant_text="assistant_texttool_call_id` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial2_turn26_text` | 37 | `assistant_text` | `assistant_text` | Y | `assistant_text="assistant_text":tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn2_text` | 37 | `assistant_text` | `tool_call` | N | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn4_single_tool` | 37 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn6_text` | 37 | `assistant_text` | `tool_call` | N | `tool_call_idtool_calltool_calltool` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn8_single_tool` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn10_single_tool` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn12_single_tool` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_2` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn14_text` | 37 | `assistant_text` | `assistant_text` | Y | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn16_single_tool` | 37 | `tool_call` | `tool_call` | Y | `tool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn18_single_tool` | 37 | `tool_call` | `tool_call` | Y | `tool_call_id=tool_call_1` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn20_single_tool` | 37 | `tool_call` | `assistant_text` | N | `assistant_texttool_calltool_calltool_call` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn22_single_tool` | 37 | `tool_call` | `assistant_text` | N | `assistant_textassistant_textassistant_textassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn24_text` | 37 | `assistant_text` | `assistant_text` | Y | `assistant_textassistant_textassistant_textassistant_text` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn26_single_tool` | 37 | `tool_call` | `assistant_text` | N | `assistant_text="assistant_texttool_call_id` |
| `decision_gate_heldout_mixed_policy_heldout_sft_success_task37_trial3_turn28_text` | 37 | `assistant_text` | `tool_call` | N | `tool_call_id=tool_call_1` |

## Interpretation

- If text recall is low, the agent still over-calls tools.
- If tool recall is low, the gate blocks useful tool calls and the downstream tool policy cannot help.
- The target is a high-recall gate on both classes before combining it with Phase2H tool generation.
