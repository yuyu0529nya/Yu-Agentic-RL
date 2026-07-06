# Phase2F Single-Tool Protocol v4 Local Prep

## Goal

Phase2E showed that slot-grounded action-prefix SFT improved heldout tool behavior:

- tool-name accuracy: `0.000 -> 0.708`
- argument exact accuracy: `0.000 -> 0.708`

But it also exposed the next bottleneck:

- protocol wrapper rate: `0.000`
- single-call rate: `0.083`
- multi-call rate: `0.625`

Phase2F prepares a dataset and run path for the next training experiment: keep slot-grounding, but train executable single-tool protocol targets.

## What Changed

New builder:

`scripts/build_single_tool_protocol_dataset.py`

It reads the clean slot-grounded action-prefix v3 data and converts only the final target turn:

- target assistant content is removed
- target must contain exactly one tool call
- target tool call must already be slot-grounded
- loss covers `<tool_call>`, JSON payload, and `</tool_call>`
- multi-call target rows are rejected instead of used as positives

## Dataset v4

Output stem:

`data/tool_call_protocol/tau2_airline_single_tool_protocol_v4_1536`

| Split | Rows | Rejected | Main reason |
| --- | ---: | ---: | --- |
| train | 42 | 7 | multi-call target |
| valid | 15 | 5 | multi-call target |
| heldout | 38 | 13 | multi-call target |

Validation:

- structural/token errors: `0`
- all kept rows have exactly one target tool call
- all kept rows have empty assistant target content
- all kept rows enable `assistant_tool_call_wrappers=True`
- target token count is nonzero for every kept row

Dataset report:

`reports/single_tool_protocol_dataset_v4_1536.md`

## Smoke Test

Local CPU tiny-random smoke:

`reports/single_tool_protocol_v4_local_smoke.md`

Result:

- train examples: `2`
- valid examples: `1`
- steps: `2`
- first loss: `11.9737`
- final loss: `11.8497`
- valid loss: `11.8781 -> 11.8318`

This is not a quality result. It only proves rendering, masking, target spans, and training plumbing work.

## 4090 Run Entry

Scripts:

- `scripts/run_single_tool_protocol_sft_4090.sh`
- `scripts/run_single_tool_protocol_sft_4090_autodl.sh`

Default settings:

- base model: `Qwen2.5-7B-Instruct`
- max seq len: `1536`
- steps: `160`
- QLoRA rank: `16`
- behavior eval max new tokens: `96`
- behavior eval stop sequence: `</tool_call>`

AutoDL behavior:

- default `AUTO_SHUTDOWN=0`
- set `AUTO_SHUTDOWN=1` only when explicitly requested

## Next Success Criteria

The next 4090 run should be judged by behavior metrics before tau2 pass rate:

| Metric | Target |
| --- | ---: |
| protocol wrapper rate | >= 0.80 |
| tool-name accuracy | >= 0.60 |
| single-call rate | >= 0.70 |
| clean single exact rate | above base |

If this works, then run a small tau2 environment slice with the new adapter.
