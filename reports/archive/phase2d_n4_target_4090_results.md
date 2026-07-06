# Phase2D Targeted N=4 Eval on 4090D

Run tag: `phase2d_n4_target_20260615_1220`

Local artifact directory:

`autodl_artifacts/phase2d_n4_target_4090_phase2d_n4_target_20260615_1220`

## Setup

- Domain: `tau2 airline`
- Tasks: `18,25,44`
- Samples: `N=4` per task, 12 simulations per model
- Policy base: `Qwen2.5-7B-Instruct`
- SFT adapter: `Recovery-v5 QLoRA`
- User simulator: `GLM-5.1`
- Serving: vLLM OpenAI-compatible server
- Final vLLM settings:
  - eager mode enabled
  - `max_model_len=16384`
  - `max_tokens=512`
  - `temperature=0.7`
  - `top_p=0.95`

Earlier attempts showed two useful infrastructure constraints:

- vLLM torch.compile failed on this runtime, so eager mode was required.
- 8K and 12K context were too small for these long airline trajectories; 16K was required.

## Main Result

| Model | Simulations | Pass^1 | Pass^4 | Oracle pass@4 | PRM rerank pass@4 | DB match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B base | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 / 11 |
| Recovery-v5 SFT | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 / 7 |

Both models failed all sampled trajectories on tasks `18`, `25`, and `44`.

The key point is `oracle pass@4 = 0`. PRM-rerank cannot improve a task when no successful candidate exists in the sampled set.

## Termination

| Model | user_stop | too_many_errors | infrastructure_error |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B base | 11 | 0 | 1 |
| Recovery-v5 SFT | 7 | 4 | 1 |

Recovery-v5 produced more unstable runs, including 4 `too_many_errors` terminations.

## Slot Grounding

| Model | Checked tool calls | Simulations with issues | Total issues |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B base | 103 | 0 | 0 |
| Recovery-v5 SFT | 64 | 10 | 52 |

Recovery-v5 issue breakdown:

| Issue | Count |
| --- | ---: |
| `ungrounded_reservation_id` | 38 |
| `ungrounded_user_id` | 7 |
| `ask_identifier_and_call_tool_same_turn` | 6 |
| `transfer_after_ungrounded_slot_error` | 1 |

Interpretation: the current recovery-prefix SFT does not teach reliable slot grounding. It may even encourage hallucinated user or reservation identifiers on stable-fail tasks.

## Conclusion

This is a useful negative result.

The current Recovery-v5 adapter is not enough for long-horizon tau2 airline failures. Since sampled candidates contain no successes, reranking is not the bottleneck. The bottleneck is trajectory generation quality, especially grounded tool arguments and policy-correct recovery behavior.

## Next Direction

Do not keep scaling blind recovery-prefix SFT yet.

Recommended next step:

1. Build a slot-grounded action-prefix dataset.
2. Filter or generate training targets where every tool argument is grounded in prior user text or tool results.
3. Add an offline validator to reject ungrounded `user_id` and `reservation_id` samples.
4. Re-train a small adapter and rerun the same targeted N=4 benchmark.

Success criterion for the next iteration:

- slot grounding issues drop sharply on Recovery-v5 successor
- at least one of tasks `18`, `25`, `44` has `oracle pass@4 > 0`
- only after that should PRM-rerank be expected to help
