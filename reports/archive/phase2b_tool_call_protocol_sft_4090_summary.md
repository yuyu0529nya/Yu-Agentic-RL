# Phase 2B: Tool-Call Protocol SFT 4090 Summary

Run tag:

```text
tool_call_protocol_v1_4090_20260614_184600
```

## Goal

Fix the main Phase 2A failure: Action-Prefix SFT learned tool-call-shaped JSON,
but tau2/vLLM treated it as ordinary assistant text instead of executable
`tool_calls`.

This run changed the target to protocol-style tool calls:

- assistant target has no natural-language content;
- each sample targets exactly one assistant tool call;
- loss covers the `<tool_call>` wrapper and JSON payload;
- Qwen2.5-7B-Instruct is trained with 4-bit QLoRA on one RTX 4090D.

## Setup

```text
model: Qwen2.5-7B-Instruct
method: QLoRA SFT
steps: 160
LoRA rank: 16
max_seq_len: 2048
train rows: 53
valid rows: 27
heldout behavior probes: 32
peak training memory: 13.4 GB
```

Artifact directory:

```text
autodl_artifacts/tool_call_protocol_v1_4090_20260614_184600/
```

Key reports:

```text
reports/tool_call_protocol_dataset_v1_2048.md
reports/sft_tool_call_protocol_v1_qwen25_7b_qlora_2048.md
reports/behavior_sft_tool_call_protocol_v1_qwen25_7b_qlora_2048.md
```

## Training Result

The protocol dataset is clearly learnable.

```text
valid loss before: 2.5809
valid loss after:  0.1528
first train loss:  1.2460
final train loss:  0.0003
```

This proves the 7B LoRA path, masking, tokenizer rendering, and 4090
environment are working.

## Behavior Result

Heldout next-tool-call behavior improved strongly versus base:

| Model | Tool-call rate | Protocol wrapper rate | First-tool acc | First-call exact acc |
| --- | ---: | ---: | ---: | ---: |
| base | 0.094 | 0.094 | 0.094 | 0.031 |
| protocol SFT | 0.875 | 0.875 | 0.688 | 0.438 |

Stricter runtime-readiness metrics expose the remaining problem:

| Model | Single-call rate | Multi-call rate | Clean single exact rate | Mean generated calls |
| --- | ---: | ---: | ---: | ---: |
| base | 0.094 | 0.000 | 0.031 | 0.09 |
| protocol SFT | 0.281 | 0.594 | 0.031 | 3.19 |

## Interpretation

This run is a real upgrade over Action-Prefix SFT v2.

The model now learns the executable protocol format: wrapper rate jumps from
near-zero behavior to 0.875 on heldout probes, and first-tool accuracy rises to
0.688. So the Phase 2A diagnosis was correct: target format mattered.

But the run is not yet ready for end-to-end tau2 pass-rate claims. The model
often keeps generating additional `<tool_call>` blocks after the first correct
one. In a runtime that expects one next action, this can still break execution
or produce unsafe extra writes.

The new failure family is:

```text
Protocol learned, stop boundary weak.
```

## Decision

Do not blindly scale this exact recipe yet.

The next algorithmic step should be `Single-Tool-Stop SFT / Decoding`:

- keep the protocol-only target;
- reduce `max_new_tokens` during behavior and tau2 serving;
- add an explicit stop sequence after `</tool_call>`;
- optionally add negative examples or a loss term that penalizes repeated tool
  blocks;
- rerun behavior evaluation with single-call metrics;
- only then run tau2 pass^1.

## Next Run Candidate

Short eval-side fix first:

```text
max_new_tokens: 64 or 96
stop sequence: </tool_call>
metric target:
  protocol wrapper rate >= 0.80
  first-tool acc >= 0.60
  single-call rate >= 0.70
  clean single exact rate > base
```

If that works, run a small tau2 slice:

```text
Qwen2.5-7B base
vs
Qwen2.5-7B + Tool-Call Protocol SFT
tasks: 2,16,18,25,44
context: 12K or 16K
```

## Status

The 4090 instance was left running intentionally for inspection and reuse.
No automatic shutdown was issued for this run.
