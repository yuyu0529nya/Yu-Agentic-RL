# Phase 1H SFT Behavior Evaluation

## Goal

Compare base and SFT adapters on heldout next-tool-call behavior, using deterministic generation from the same decision-point prompts.

## Key Findings

- `sft_single_tool_protocol_v4_qwen25_7b_qlora_1536`: tool-name accuracy 0.000 -> 0.750; exact call accuracy 0.000 -> 0.750; single-call rate 0.000 -> 0.875.
- Best free-form behavior adapter: `sft_single_tool_protocol_v4_qwen25_7b_qlora_1536`.
- Free-form generation is still weak; this motivates action-prefix SFT or constrained tool decoding.

## Setup

- Base model: `/root/autodl-tmp/models/qwen25-7b-instruct`
- Data: `data/tool_call_protocol/tau2_airline_single_tool_protocol_v4_1536_heldout.jsonl`
- Probes: `32`
- Max sequence length: `1536`
- Max new tokens: `96`
- Stop sequences: `['</tool_call>']`
- Device: `cuda`

## Summary

| Model | Tool-call rate | Protocol wrapper rate | Single-call rate | Multi-call rate | Tool-name acc | Arg exact acc | Clean single exact | Mean calls | JSON failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| `sft_single_tool_protocol_v4_qwen25_7b_qlora_1536` | 0.875 | 0.875 | 0.875 | 0.000 | 0.750 | 0.750 | 0.750 | 0.88 | 0 |

## Delta Vs Base

| Model | Tool-name acc delta | Arg exact delta | Single-call delta | Clean single exact delta |
| --- | ---: | ---: | ---: | ---: |
| `sft_single_tool_protocol_v4_qwen25_7b_qlora_1536` | +0.750 | +0.750 | +0.875 | +0.750 |

## Per-Probe Results

### base

| Probe | Task | Ref tool | Pred tool | # Calls | Wrapped | Tool ok | Arg exact | Generated preview |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial2_turn8::turn8` | 2 | `get_user_details` | `-` | 0 | N | N | N | `I understand how frustrating that must be. Unfortunately, without the reservation ID, I won't be able to look up your specific booking. However, we can try a few other methods to f...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn8::turn8` | 2 | `get_user_details` | `-` | 0 | N | N | N | `Of course, I can help you try to locate your reservation. To do that, I'll need some additional information such as your name, contact email, phone number, and the travel dates for...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn10::turn8` | 2 | `get_user_details` | `-` | 0 | N | N | N | `Of course, I can help you try to locate your reservation. To do that, I'll need some additional information such as your name, contact email, phone number, and the travel dates for...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn10::turn10` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Thank you for providing that information. Based on your profile, your most recent reservation ID is `SDZQKO`. Now, let's check the status of your flight with reservation ID `SDZQKO...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn12::turn5` | 2 | `get_user_details` | `-` | 0 | N | N | N | `I understand how frustrating it can be to lose track of reservation numbers. Unfortunately, without the reservation number, I won't be able to look up your specific booking. Howeve...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn12::turn7` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `It looks like I found your recent reservation under the ID `SDZQKO`. Let's check the status of your flight with this reservation number. Would you like to proceed with checking the...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn12::turn9` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `It looks like the reservation with ID SDZQKO is the one you're referring to. The flight details are as follows: - **Round Trip** - **Cabin Class**: Basic Economy - **Flights**: - L...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn18::turn0` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Sure, I'm here to help! What would you like assistance with? Whether it's a specific question, a task you need help with, or some information you're looking for, feel free to ask. ...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn18::turn6` | 2 | `get_flight_status` | `-` | 0 | N | N | N | `Thank you for clarifying. I understand how disappointing and frustrating it can be to experience a significant delay at the beginning of a trip. Let’s review the details and see wh...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn20::turn0` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Sure, I'm here to help! What would you like assistance with? Whether it's a specific question, a task you need help with, or some information you're looking for, feel free to ask. ...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn20::turn6` | 2 | `get_flight_status` | `-` | 0 | N | N | N | `Thank you for clarifying. I understand how disappointing and frustrating it can be to experience a significant delay at the beginning of a trip. Let’s review the details and see wh...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn20::turn8` | 2 | `get_flight_status` | `-` | 0 | N | N | N | `The outbound flight HAT006 from Boston (BOS) to Seattle (SEA) on May 11, 2024, landed late. According to the records, the flight experienced a significant delay, causing you to be ...` |
| `single_tool_protocol_v4_heldout_sft_success_task16_trial0_turn4::turn4` | 16 | `get_user_details` | `-` | 0 | N | N | N | `Thank you for providing your user ID, Aarav Garcia. Let's proceed with looking up your reservation. To do this, I'll need to access the system with your user ID. However, since I d...` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn4::turn4` | 18 | `get_user_details` | `-` | 0 | N | N | N | `Sure, Omar, I can assist you with that. To proceed, I'll need the details of the specific flights you want to downgrade. Could you please provide me with the flight numbers and dat...` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn27::turn0` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `Sure, I'm here to help! What would you like assistance with? Whether it's a specific question, a task you need help with, or some information you're looking for, feel free to ask. ...` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn27::turn2` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `The reservation with ID `X7BYG1` has been successfully updated with the new flights. Here are the details of the updated reservation: - **Reservation ID**: X7BYG1 - **User ID**: om...` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn27::turn4` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `The reservation with ID `EQ1G6C` has been successfully updated with the new flights and payment details. Here is a summary of the changes: - **Reservation ID**: EQ1G6C - **User ID*...` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn27::turn6` | 18 | `calculate` | `-` | 0 | N | N | N | `The reservation with ID `BOH180` has been successfully updated with the new flight details and payment information. Here is a summary of the changes: - **Reservation ID**: BOH180 -...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial0_turn4::turn4` | 25 | `get_user_details` | `-` | 0 | N | N | N | `Great, thank you for providing your user ID, Ivan_Muller_7015! To ensure that the new reservation is exactly the same as your current one, I'll need some additional details such as...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial0_turn6::turn4` | 25 | `get_user_details` | `-` | 0 | N | N | N | `Great, thank you for providing your user ID, Ivan_Muller_7015! To ensure that the new reservation is exactly the same as your current one, I'll need some additional details such as...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial0_turn6::turn6` | 25 | `get_reservation_details` | `-` | 0 | N | N | N | `I've retrieved your details, Ivan. Your current reservation is G72NSF. To create an identical reservation for your friend, I'll need some additional information such as the date, t...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn4::turn4` | 25 | `get_user_details` | `-` | 0 | N | N | N | `Great, thank you for providing your user ID, Ivan_Muller_7015. Let's review and confirm the details of your current reservation so we can create an identical one for your friend. C...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn6::turn4` | 25 | `get_user_details` | `-` | 0 | N | N | N | `Great, thank you for providing your user ID, Ivan_Muller_7015. Let's review and confirm the details of your current reservation so we can create an identical one for your friend. C...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn6::turn6` | 25 | `get_reservation_details` | `-` | 0 | N | N | N | `Based on the information provided, your current reservation details are as follows: - **Reservation ID:** G72NSF - **Passenger Name:** Ivan Muller - **Address:** 256 Pine Lane, Sui...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn10::turn4` | 25 | `get_user_details` | `-` | 0 | N | N | N | `Great, thank you for providing your user ID, Ivan_Muller_7015. Let's review and confirm the details of your current reservation so we can create an identical one for your friend. C...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn10::turn6` | 25 | `get_reservation_details` | `-` | 0 | N | N | N | `Based on the information provided, your current reservation details are as follows: - **Reservation ID:** G72NSF - **Passenger Name:** Ivan Muller - **Address:** 256 Pine Lane, Sui...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn10::turn10` | 25 | `search_direct_flight` | `-` | 0 | N | N | N | `Got it! Let's proceed step by step. ### Passenger Details Your friend's name is Ivan Smith, and his information is already in your user profile. We can use that. ### Baggage and In...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn12::turn0` | 25 | `get_reservation_details` | `-` | 0 | N | N | N | `Sure, I'm here to help! What would you like assistance with? Whether it's a specific question, a task you need help with, or some information you're looking for, feel free to ask. ...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn12::turn4` | 25 | `search_direct_flight` | `-` | 0 | N | N | N | `Sure, let's proceed step-by-step. ### Passenger Details Based on your information, we have: - **First Name**: Ivan - **Last Name**: Smith We'll need to retrieve his date of birth f...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn12::turn6` | 25 | `search_direct_flight` | `-` | 0 | N | N | N | `Based on the available direct flights, here are the options for your friend's flight from Detroit (DTW) to Phoenix (PHX) on May 17th, 2024: 1. **Flight HAT035**: - Departure Time: ...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial2_turn4::turn4` | 25 | `get_user_details` | `-` | 0 | N | N | N | `Great, thank you for providing your user ID, Ivan_Muller_7015! To ensure that the new reservation is exactly the same as your current one, I'll need some additional details such as...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial2_turn6::turn4` | 25 | `get_user_details` | `-` | 0 | N | N | N | `Great, thank you for providing your user ID, Ivan_Muller_7015! To ensure that the new reservation is exactly the same as your current one, I'll need some additional details such as...` |

### sft_single_tool_protocol_v4_qwen25_7b_qlora_1536

| Probe | Task | Ref tool | Pred tool | # Calls | Wrapped | Tool ok | Arg exact | Generated preview |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial2_turn8::turn8` | 2 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"noah_muller_9847"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn8::turn8` | 2 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"noah_muller_9847"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn10::turn8` | 2 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"noah_muller_9847"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn10::turn10` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"SDZQKO"},"name":"get_reservation_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn12::turn5` | 2 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"noah_muller_9847"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn12::turn7` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"SDZQKO"},"name":"get_reservation_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn12::turn9` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"4OG6T3"},"name":"get_reservation_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn18::turn0` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Given the following context: The United States has a federal system of government, which means that power is divided between a central government and state governments. The central...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn18::turn6` | 2 | `get_flight_status` | `get_flight_status` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"date":"2024-05-11","flight_number":"HAT006"},"name":"get_flight_status"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn20::turn0` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Given the following context: The United States has a federal system of government, which means that power is divided between a central government and state governments. The central...` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn20::turn6` | 2 | `get_flight_status` | `get_flight_status` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"date":"2024-05-11","flight_number":"HAT006"},"name":"get_flight_status"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task2_trial3_turn20::turn8` | 2 | `get_flight_status` | `get_flight_status` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"date":"2024-05-11","flight_number":"HAT018"},"name":"get_flight_status"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task16_trial0_turn4::turn4` | 16 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"aarav_garcia_1177"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn4::turn4` | 18 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"omar_davis_3817"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn27::turn0` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `Given the following context: The United States has a federal system of government, which means that power is divided between a central government and state governments. The central...` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn27::turn2` | 18 | `update_reservation_flights` | `get_user_details` | 1 | Y | N | N | `<tool_call> {"arguments":{"user_id":"omar_davis_3817"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn27::turn4` | 18 | `update_reservation_flights` | `get_flight_status` | 1 | Y | N | N | `<tool_call> {"arguments":{"date":"2024-05-24","flight_number":"HAT232"},"name":"get_flight_status"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task18_trial2_turn27::turn6` | 18 | `calculate` | `get_flight_status` | 1 | Y | N | N | `<tool_call> {"arguments":{"date":"2024-05-24","flight_number":"HAT232"},"name":"get_flight_status"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial0_turn4::turn4` | 25 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"ivan_muller_7015"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial0_turn6::turn4` | 25 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"ivan_muller_7015"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial0_turn6::turn6` | 25 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"G72NSF"},"name":"get_reservation_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn4::turn4` | 25 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"ivan_muller_7015"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn6::turn4` | 25 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"ivan_muller_7015"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn6::turn6` | 25 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"G72NSF"},"name":"get_reservation_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn10::turn4` | 25 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"ivan_muller_7015"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn10::turn6` | 25 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"G72NSF"},"name":"get_reservation_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn10::turn10` | 25 | `search_direct_flight` | `search_direct_flight` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"date":"2024-05-17","destination":"PHX","origin":"DTW"},"name":"search_direct_flight"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn12::turn0` | 25 | `get_reservation_details` | `-` | 0 | N | N | N | `Given the following context: The United States has a federal system of government, which means that power is divided between a central government and state governments. The central...` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn12::turn4` | 25 | `search_direct_flight` | `get_user_details` | 1 | Y | N | N | `<tool_call> {"arguments":{"user_id":"ivan_muller_7015"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial1_turn12::turn6` | 25 | `search_direct_flight` | `search_direct_flight` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"date":"2024-05-17","destination":"SEA","origin":"PHX"},"name":"search_direct_flight"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial2_turn4::turn4` | 25 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"ivan_muller_7015"},"name":"get_user_details"} </tool_call>` |
| `single_tool_protocol_v4_heldout_sft_success_task25_trial2_turn6::turn4` | 25 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"ivan_muller_7015"},"name":"get_user_details"} </tool_call>` |

## Interpretation

- This is a behavior-level proxy focused on next tool-call generation, not full tau2 pass rate.
- A weak adapter can lower teacher-forced NLL without improving free-form tool-call generation.
- If tool-name accuracy improves but protocol wrapper rate stays low, the model is learning JSON content without learning executable tool-call format.
- If tool-name accuracy improves but single-call rate stays low, the model is learning the right action family without learning the runtime stop boundary.
- If generation metrics are poor, the next step is to train on shorter action-decision prefixes or evaluate with constrained decoding.
