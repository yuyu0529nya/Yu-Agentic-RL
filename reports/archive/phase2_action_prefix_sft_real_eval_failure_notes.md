# Phase 2A Failure Notes: Action-Prefix SFT Real Eval

Run tag:

```text
phase2a_clean12k_5task_5090d_20260614_180900
```

## Headline

Action-Prefix SFT v2 improved offline next-tool-call behavior, but it did not
improve full tau2 airline `pass^1` on the 5-task real eval slice.

```text
Qwen2.5-7B base:             pass^1 = 0.20
Qwen2.5-7B + Action-Prefix:  pass^1 = 0.20
```

The important failure is not only low pass rate. The SFT model often emits
tool-call-shaped JSON as ordinary assistant text, so tau2/vLLM does not execute
the tools.

## Evidence

Base model:

```text
task 2:  real_tool_calls=6, action_match=2/3, reward=1.0
task 16: real_tool_calls=6, action_match=0/1, reward=0.0
task 18: real_tool_calls=9, action_match=0/0, reward=0.0, max_steps
task 25: real_tool_calls=3, action_match=0/1, reward=0.0
task 44: context-window infrastructure error
```

SFT model:

```text
task 2:  real_tool_calls=0, action_match=0/3, reward=1.0
task 16: real_tool_calls=0, action_match=0/1, reward=0.0
task 18: real_tool_calls=0, action_match=0/5, reward=0.0
task 25: real_tool_calls=0, action_match=0/1, reward=0.0
task 44: real_tool_calls=0, action_match=0/19, reward=0.0
```

Typical SFT output pattern:

```text
I can help with that...
{"arguments":{"user_id":"..."},"name":"get_user_details"}
{"arguments":{"reservation_id":"..."},"name":"get_reservation_details"}
```

This is semantically close to a tool call, but it is not a real OpenAI/vLLM
`tool_calls` message. The environment treats it as normal text.

## Failure Taxonomy v1

| family | symptom | consequence |
| --- | --- | --- |
| Protocol drift | Tool JSON appears in assistant text instead of `tool_calls` | Tools are not executed |
| Multi-call dumping | SFT emits many JSON snippets in one assistant turn | Parser cannot route one clean action |
| Length truncation | Several SFT turns stop at `finish_reason=length` | Tool JSON is incomplete or repeated |
| Evidence bypass | Model writes plausible next actions without reading state | DB writes fail or never happen |
| Long-horizon collapse | Hard tasks require many reads/writes across reservations | Base loops; SFT text-dumps calls |

## Interpretation

The current Action-Prefix SFT target is too loose for the real runtime. It
teaches the model the surface form of a tool action, but not the executable
message protocol required by vLLM and tau2.

This explains the mismatch:

```text
offline next-tool-call metric improves
but
end-to-end pass^1 does not improve
```

## Next Algorithmic Step

Do not scale the same SFT recipe blindly.

Next version should be `Tool-Call-Only / Protocol SFT`:

- train only turns whose target is one executable tool call;
- render the assistant target in the exact format expected by the serving stack;
- mask natural language around the tool call;
- reject examples where the target contains multiple dumped tool JSON snippets;
- evaluate both offline protocol accuracy and real tau2 pass^1.

Candidate follow-up experiments:

```text
E1: protocol-only SFT target, one tool call per sample
E2: constrained tool decoding during vLLM serving
E3: PRM rerank over N sampled trajectories, with protocol-validity reward
E4: 16K clean rerun for task 44, only after protocol issue is fixed
```

## 2026-06-14 Follow-Up: Protocol SFT v1

Follow-up report:

```text
reports/phase2b_tool_call_protocol_sft_4090_summary.md
```

Protocol SFT v1 confirmed the diagnosis:

```text
base wrapper rate:        0.094
protocol SFT wrapper:     0.875
base first-tool acc:      0.094
protocol SFT first-tool:  0.688
base first-call exact:    0.031
protocol SFT exact:       0.438
```

The remaining blocker changed from "not executable protocol" to "weak stop
boundary": protocol SFT often emits multiple `<tool_call>` blocks in one
assistant response.

Next step is therefore not another broad Action-Prefix run. It should be a
single-tool-stop variant: protocol target plus stop sequence / shorter decoding
and strict single-call metrics before the next tau2 pass-rate run.
