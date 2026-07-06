# Phase2E Slot-Grounded Action-Prefix SFT on 4090D

Run tag: `slot_grounded_v3_1536_20260615_201005`

Local artifact directory:

`autodl_artifacts/slot_grounded_v3_1536_4090_20260615_201005`

Remote artifact archive:

`yuyu_slot_grounded_action_prefix_artifacts_20260615_201013.tar.gz`

## Goal

Train Qwen2.5-7B-Instruct with QLoRA on slot-grounded action-prefix samples.

This run directly follows the Phase2D finding: Recovery-v5 generated hallucinated `user_id`, `reservation_id`, and `payment_id` values on stable-fail tasks.

## Dataset

Max sample tokens: `1536`

| Split | Candidates | Kept | Rejected | Keep rate | Main rejection issues |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 60 | 49 | 11 | 0.817 | `ungrounded_reservation_id=10`, `ungrounded_payment_id=5`, `ungrounded_user_id=1` |
| valid | 32 | 20 | 12 | 0.625 | `ungrounded_payment_id=9`, `ungrounded_reservation_id=7`, `ungrounded_user_id=3` |
| heldout | 73 | 51 | 22 | 0.699 | `ungrounded_payment_id=25`, `ungrounded_reservation_id=15`, `ungrounded_user_id=5` |

The kept train/valid/heldout samples all passed target-slot grounding validation.

## Training

| Item | Value |
| --- | ---: |
| Base model | `Qwen2.5-7B-Instruct` |
| Trainable params | 40,370,176 |
| Steps | 160 |
| Max seq len | 1536 |
| First train loss | 3.2847 |
| Final train loss | 0.000045 |
| Min train loss | 0.000007 |
| Valid loss before | 1.5305 |
| Valid loss after | 0.3621 |
| Peak CUDA memory | 12020 MB |

The adapter clearly learned the slot-grounded action-prefix supervision under teacher forcing.

## Behavior Proxy

Evaluation used 24 heldout next-tool-call probes from the slot-grounded v3 heldout set.

| Model | Tool-call rate | Tool-name acc | Arg exact acc | Single-call rate | Multi-call rate | Mean generated calls | JSON failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Qwen2.5-7B | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0 |
| Slot-grounded v3 SFT | 0.708 | 0.708 | 0.708 | 0.083 | 0.625 | 1.54 | 0 |

## Interpretation

This is a real improvement over base on the offline behavior proxy: the model now often emits the right tool family and exact arguments on heldout action-prefix probes.

However, the run also exposes the next bottleneck:

- The model often emits multiple tool calls when the probe expects a single next action.
- The generated calls are parseable JSON, but not wrapped in the expected `<tool_call>...</tool_call>` protocol.
- This is not ready to declare a tau2 pass-rate win until we rerun full environment evaluation.

## Next Step

The next algorithmic step should be protocol/boundary control, not just more SFT:

1. Add a tool-call-only target format with explicit `<tool_call>` wrapper.
2. Add a single-action stop boundary so one prompt produces one executable action.
3. Rerun the same behavior proxy.
4. Only then rerun tau2 targeted tasks `18,25,44` with the new adapter.

Success criterion for the next iteration:

- keep `tool_name_accuracy` near or above `0.70`
- raise `single_call_rate` far above `0.083`
- make protocol wrapper rate nonzero or switch serving/parser to the exact generated JSON protocol
