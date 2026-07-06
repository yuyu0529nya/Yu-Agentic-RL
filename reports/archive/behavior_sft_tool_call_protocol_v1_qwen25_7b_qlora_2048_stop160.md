# Phase 1H SFT Behavior Evaluation

## Goal

Compare base and SFT adapters on heldout next-tool-call behavior, using deterministic generation from the same decision-point prompts.

## Key Findings

- `sft_tool_call_protocol_v1_qwen25_7b_qlora_2048`: tool-name accuracy 0.094 -> 0.688; exact call accuracy 0.031 -> 0.438; single-call rate 0.094 -> 0.875.
- Best free-form behavior adapter: `sft_tool_call_protocol_v1_qwen25_7b_qlora_2048`.
- Free-form generation is still weak; this motivates action-prefix SFT or constrained tool decoding.

## Setup

- Base model: `/root/autodl-tmp/models/qwen25-7b-instruct`
- Data: `data/tool_call_protocol/tau2_airline_tool_call_protocol_v1_2048_heldout.jsonl`
- Probes: `32`
- Max sequence length: `2048`
- Max new tokens: `160`
- Stop sequences: `['</tool_call>']`
- Device: `cuda`

## Summary

| Model | Tool-call rate | Protocol wrapper rate | Single-call rate | Multi-call rate | Tool-name acc | Arg exact acc | Clean single exact | Mean calls | JSON failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.094 | 0.094 | 0.094 | 0.000 | 0.094 | 0.031 | 0.031 | 0.09 | 0 |
| `sft_tool_call_protocol_v1_qwen25_7b_qlora_2048` | 0.875 | 0.875 | 0.875 | 0.000 | 0.688 | 0.438 | 0.438 | 0.88 | 0 |

## Delta Vs Base

| Model | Tool-name acc delta | Arg exact delta | Single-call delta | Clean single exact delta |
| --- | ---: | ---: | ---: | ---: |
| `sft_tool_call_protocol_v1_qwen25_7b_qlora_2048` | +0.594 | +0.406 | +0.781 | +0.406 |

## Per-Probe Results

### base

| Probe | Task | Ref tool | Pred tool | # Calls | Wrapped | Tool ok | Arg exact | Generated preview |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `tool_call_protocol_heldout_sft_success_task2_trial2_turn8_call0::turn8` | 2 | `get_user_details` | `-` | 0 | N | N | N | `I understand how frustrating that must be. Unfortunately, without the reservation ID, I won't be able to look up your specific booking. However, we can try a few other methods to f...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn8_call0::turn8` | 2 | `get_user_details` | `-` | 0 | N | N | N | `Of course, I can help you try to locate your reservation. To do that, I'll need some additional information such as your name, contact email, phone number, and the travel dates for...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn10_call0::turn8` | 2 | `get_user_details` | `-` | 0 | N | N | N | `Of course, I can help you try to locate your reservation. To do that, I'll need some additional information such as your name, contact email, phone number, and the travel dates for...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn10_call0::turn10` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Thank you for providing that information. Based on your profile, your most recent reservation ID is `SDZQKO`. Now, let's check the status of your flight with reservation ID `SDZQKO...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn12_call0::turn8` | 2 | `get_user_details` | `-` | 0 | N | N | N | `Of course, I can help you try to locate your reservation. To do that, I'll need some additional information such as your name, contact email, phone number, and the travel dates for...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn12_call0::turn10` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Thank you for providing that information. Based on your profile, your most recent reservation ID is `SDZQKO`. Now, let's check the status of your flight with reservation ID `SDZQKO...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn12_call0::turn12` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `It looks like the reservation with ID SDZQKO is for a round trip from Los Angeles (LAX) to Newark (EWR) and then to Miami (MIA). However, there's no status information provided for...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn18_call0::turn0` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Sure, I'm here to help! What would you like assistance with? Whether it's a specific question, a task you need help with, or some information you're looking for, feel free to ask. ...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn18_call0::turn2` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Based on the reservation details for SDZQKO, there is no information provided about a delayed flight. The reservation includes a round trip from LAX to MIA and back, with specific ...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn18_call0::turn8` | 2 | `get_flight_status` | `-` | 0 | N | N | N | `Thank you for clarifying. I understand how disappointing and frustrating it can be to experience a significant delay at the beginning of a trip. Let’s see what steps we can take to...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn20_call0::turn0` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Sure, I'm here to help! What would you like assistance with? Whether it's a specific question, a task you need help with, or some information you're looking for, feel free to ask. ...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn20_call0::turn2` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | `Based on the reservation details for SDZQKO, there is no information provided about a delayed flight. The reservation includes a round trip from LAX to MIA and back, with specific ...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn20_call0::turn8` | 2 | `get_flight_status` | `-` | 0 | N | N | N | `Thank you for clarifying. I understand how disappointing and frustrating it can be to experience a significant delay at the beginning of a trip. Let’s see what steps we can take to...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn20_call0::turn10` | 2 | `get_flight_status` | `-` | 0 | N | N | N | `I checked the status of the outbound flight HAT006 from Boston (BOS) to Seattle (SEA) on May 11, 2024. The flight landed as expected, but it seems there might have been a delay in ...` |
| `tool_call_protocol_heldout_sft_success_task16_trial0_turn4_call0::turn4` | 16 | `get_user_details` | `-` | 0 | N | N | N | `Thank you for providing your user ID, Aarav Garcia. Let's proceed with looking up your reservation. To do this, I'll need to access the system with your user ID. However, since I d...` |
| `tool_call_protocol_heldout_sft_success_task16_trial0_turn16_call0::turn4` | 16 | `update_reservation_flights` | `-` | 0 | N | N | N | `Great! I've processed the change to your reservation. Here are the updated details: **New Reservation:** - Date: May 24, 2024 - Route: Atlanta → LaGuardia → Philadelphia - Cabin: E...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn4_call0::turn4` | 18 | `get_user_details` | `-` | 0 | N | N | N | `Sure, Omar, I can assist you with that. To proceed, I'll need the details of the specific flights you want to downgrade. Could you please provide me with the flight numbers and dat...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn17_call0::turn4` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `Sure, I can help with that. Let's break down the savings for each reservation based on the typical fare differences between business class and economy class. Note that the exact am...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn19_call0::turn4` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `Sure, I can help with that. Let's break down the savings for each reservation based on the typical fare differences between business class and economy class. Note that the exact am...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn19_call0::turn6` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `I've successfully downgraded the first reservation, JG7FMM, from business class to economy class. Here are the details: - **Original Flight**: MCO to CLT on May 21 - **New Flights*...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn21_call0::turn4` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `Sure, I can help with that. Let's break down the savings for each reservation based on the typical fare differences between business class and economy class. Note that the exact am...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn21_call0::turn6` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `I've successfully downgraded the first reservation, JG7FMM, from business class to economy class. Here are the details: - **Original Flight**: MCO to CLT on May 21 - **New Flights*...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn21_call0::turn8` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-24","flight_number":"HAT012"},{"date":"2024-05-24","flight_number":"HAT014"}],"payment_id":"credit_card_2929...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn23_call0::turn2` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `Sure, I can help with that. Let's break down the savings based on the original fares and the new economy class fares. However, since we don't have the exact original fares, I'll as...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn23_call0::turn4` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `I've successfully downgraded the first reservation (JG7FMM) to economy class. Here are the details: - **Original Flight**: Business Class - **New Flights**: - HAT028 (MCO to BOS) -...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn23_call0::turn6` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `I've successfully updated the first two reservations. Now, let's continue with the remaining ones: ### Reservation 3: X7BYG1 **Original Flight Details:** - Date: 2024-05-25 - Fligh...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn23_call0::turn8` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-25","flight_number":"HAT120"},{"date":"2024-05-25","flight_number":"HAT116"}],"payment_id":"gift_card_348193...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn25_call0::turn0` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `Sure, I'm here to help! What would you like assistance with? Whether it's a specific question, a task you need help with, or some information you're looking for, feel free to ask. ...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn25_call0::turn2` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `The reservation has been updated with the new flight details. Here is a summary of the changes: - **Reservation ID**: 2FBBAH - **User ID**: omar_davis_3817 - **Origin**: DEN (Denve...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn25_call0::turn4` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `The reservation with ID `X7BYG1` has been updated successfully. Here are the details of the updated reservation: - **Reservation ID**: X7BYG1 - **User ID**: omar_davis_3817 - **Ori...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn25_call0::turn6` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `The reservation with ID `EQ1G6C` has been updated successfully. Here are the details of the updated reservation: - **Reservation ID**: EQ1G6C - **User ID**: omar_davis_3817 - **Ori...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn27_call0::turn0` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | `Sure, I'm here to help! What would you like assistance with? Whether it's a specific question, a task you need help with, or some information you're looking for, feel free to ask. ...` |

### sft_tool_call_protocol_v1_qwen25_7b_qlora_2048

| Probe | Task | Ref tool | Pred tool | # Calls | Wrapped | Tool ok | Arg exact | Generated preview |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| `tool_call_protocol_heldout_sft_success_task2_trial2_turn8_call0::turn8` | 2 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"noah_muller_9847"},"name":"get_user_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn8_call0::turn8` | 2 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"noah_muller_9847"},"name":"get_user_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn10_call0::turn8` | 2 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"noah_muller_9847"},"name":"get_user_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn10_call0::turn10` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"SDZQKO"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn12_call0::turn8` | 2 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"noah_muller_9847"},"name":"get_user_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn12_call0::turn10` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"SDZQKO"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn12_call0::turn12` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"4OG6T3"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn18_call0::turn0` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | ````python import pandas as pd import numpy as np from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.linear_model imp...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn18_call0::turn2` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"4OG6T3"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn18_call0::turn8` | 2 | `get_flight_status` | `get_flight_status` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"date":"2024-05-11","flight_number":"HAT006"},"name":"get_flight_status"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn20_call0::turn0` | 2 | `get_reservation_details` | `-` | 0 | N | N | N | ````python import pandas as pd import numpy as np from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.linear_model imp...` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn20_call0::turn2` | 2 | `get_reservation_details` | `get_reservation_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"reservation_id":"4OG6T3"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn20_call0::turn8` | 2 | `get_flight_status` | `get_flight_status` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"date":"2024-05-11","flight_number":"HAT006"},"name":"get_flight_status"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task2_trial3_turn20_call0::turn10` | 2 | `get_flight_status` | `get_flight_status` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"date":"2024-05-11","flight_number":"HAT018"},"name":"get_flight_status"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task16_trial0_turn4_call0::turn4` | 16 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"aarav_garcia_1177"},"name":"get_user_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task16_trial0_turn16_call0::turn4` | 16 | `update_reservation_flights` | `get_reservation_details` | 1 | Y | N | N | `<tool_call> {"arguments":{"reservation_id":"M05KNL"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn4_call0::turn4` | 18 | `get_user_details` | `get_user_details` | 1 | Y | Y | Y | `<tool_call> {"arguments":{"user_id":"omar_davis_3817"},"name":"get_user_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn17_call0::turn4` | 18 | `update_reservation_flights` | `get_reservation_details` | 1 | Y | N | N | `<tool_call> {"arguments":{"reservation_id":"JG7FMM"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn19_call0::turn4` | 18 | `update_reservation_flights` | `get_reservation_details` | 1 | Y | N | N | `<tool_call> {"arguments":{"reservation_id":"JG7FMM"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn19_call0::turn6` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-28","flight_number":"HAT428"},{"date":"2024-05-29","flight_number":"HAT429"},{"date":"2024-05-30","flight_nu...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn21_call0::turn4` | 18 | `update_reservation_flights` | `get_reservation_details` | 1 | Y | N | N | `<tool_call> {"arguments":{"reservation_id":"JG7FMM"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn21_call0::turn6` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-28","flight_number":"HAT428"},{"date":"2024-05-29","flight_number":"HAT429"},{"date":"2024-05-30","flight_nu...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn21_call0::turn8` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-24","flight_number":"HAT124"},{"date":"2024-05-24","flight_number":"HAT125"}],"payment_id":"credit_card_2929...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn23_call0::turn2` | 18 | `update_reservation_flights` | `get_reservation_details` | 1 | Y | N | N | `<tool_call> {"arguments":{"reservation_id":"JG7FMM"},"name":"get_reservation_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn23_call0::turn4` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-21","flight_number":"HAT028"},{"date":"2024-05-21","flight_number":"HAT277"}],"payment_id":"gift_card_423486...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn23_call0::turn6` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-22","flight_number":"HAT123"},{"date":"2024-05-22","flight_number":"HAT245"}],"payment_id":"credit_card_2929...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn23_call0::turn8` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-25","flight_number":"HAT124"},{"date":"2024-05-25","flight_number":"HAT128"}],"payment_id":"gift_card_348193...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn25_call0::turn0` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | ````python import pandas as pd import numpy as np from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.linear_model imp...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn25_call0::turn2` | 18 | `update_reservation_flights` | `get_user_details` | 1 | Y | N | N | `<tool_call> {"arguments":{"user_id":"omar_davis_3817"},"name":"get_user_details"} </tool_call>` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn25_call0::turn4` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"business","flights":[{"date":"2024-05-28","flight_number":"HAT456"},{"date":"2024-05-28","flight_number":"HAT345"},{"date":"2024-05-30","flight_n...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn25_call0::turn6` | 18 | `update_reservation_flights` | `update_reservation_flights` | 1 | Y | Y | N | `<tool_call> {"arguments":{"cabin":"economy","flights":[{"date":"2024-05-25","flight_number":"HAT123"},{"date":"2024-05-25","flight_number":"HAT124"}],"payment_id":"credit_card_4567...` |
| `tool_call_protocol_heldout_sft_success_task18_trial2_turn27_call0::turn0` | 18 | `update_reservation_flights` | `-` | 0 | N | N | N | ````python import pandas as pd import numpy as np from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.linear_model imp...` |

## Interpretation

- This is a behavior-level proxy focused on next tool-call generation, not full tau2 pass rate.
- A weak adapter can lower teacher-forced NLL without improving free-form tool-call generation.
- If tool-name accuracy improves but protocol wrapper rate stays low, the model is learning JSON content without learning executable tool-call format.
- If tool-name accuracy improves but single-call rate stays low, the model is learning the right action family without learning the runtime stop boundary.
- If generation metrics are poor, the next step is to train on shorter action-decision prefixes or evaluate with constrained decoding.
