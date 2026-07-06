# Phase2I Decision-Balanced Mixed Policy 5090D Results

## Status

- Status: completed
- Run time: 2026-06-21
- GPU: RTX 5090D
- Base model: Qwen2.5-7B-Instruct
- Method: 4-bit QLoRA SFT
- Goal: reduce Phase2H over-calling while keeping tool-call behavior strong.

## Local Artifacts

- Run log: `E:\yuyu\autodl_artifacts\phase2i_decision_balanced_5090d_20260621_232029\phase2i_decision_balanced_5090d_20260621_232029.log`
- SFT metrics: `E:\yuyu\autodl_artifacts\phase2i_decision_balanced_5090d_20260621_232029\sft_metrics.json`
- Behavior summary: `E:\yuyu\autodl_artifacts\phase2i_decision_balanced_5090d_20260621_232029\behavior_summary.json`
- Full artifact tarball: `E:\yuyu\autodl_artifacts\phase2i_decision_balanced_5090d_20260621_232029\yuyu_phase2i_decision_balanced_5090d_20260621_232029.tar.gz`

## Dataset

| Split | Raw rows | Final rows | Assistant text | Tool call | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 192 | 250 | 160 | 90 | 27 |
| valid | 82 | 82 | 34 | 48 | 16 |
| heldout | 233 | 233 | 108 | 125 | 36 |

Training split changes:

- Repeated assistant-text rows: text targets were repeated to reduce over-calling.
- Downsampled protocol-only tool rows: protocol-only tool variants were kept at 50%.
- Valid and heldout stayed unchanged for comparable behavior evaluation.

## Training

| Metric | Value |
| --- | ---: |
| Trainable params | 40,370,176 |
| Steps | 240 |
| Max sequence length | 2048 |
| First train loss | 1.4203 |
| Final train loss | 0.9554 |
| Final rolling mean | 0.1743 |
| Valid loss before | 2.3680 |
| Valid loss after | 0.4120 |
| Max CUDA memory MB | 13,431.6 |

Training itself is healthy: validation loss drops strongly. The main question is behavior transfer.

## Behavior Eval

Reported metrics on 64 heldout probes:

| Model | Action type acc | Text no-tool | Tool-call rate | Wrapper | Tool-name acc | Arg exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.500 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase2H | 0.734 | 0.719 | 0.750 | 0.750 | 0.750 | 0.750 |
| Phase2I | 0.703 | 0.812 | 0.594 | 0.594 | 0.531 | 0.531 |

Deduplicated metrics, merging single-tool and protocol duplicate probes:

| Model | N | Action type acc | Text no-tool | Tool-call rate | Wrapper | Tool-name acc | Arg exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 55 | 0.582 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase2I | 55 | 0.746 | 0.812 | 0.652 | 0.652 | 0.565 | 0.565 |

## Interpretation

Phase2I partially worked.

What improved:

- Text no-tool improved from Phase2H 0.719 to Phase2I 0.812.
- This means the model over-calls tools less often on turns that should be normal assistant text.
- Action type accuracy is comparable after deduplication: 0.746.

What regressed:

- Tool-name and argument exact accuracy dropped from Phase2H 0.750 to Phase2I 0.531 reported / 0.565 dedup.
- Tool-call rate also dropped, so the model became more conservative but less reliable when a tool is actually needed.

## Verdict

Phase2I is a useful diagnostic, not the final recipe.

It proves the over-calling problem is data-distribution sensitive, but text repetition plus protocol downsampling is too blunt. The next design should separate the decision problem from the tool-generation problem:

- decision head/data: decide `assistant_text` vs `tool_call`
- tool policy data: generate exact tool name and arguments
- decoding constraint: when a tool call is selected, force valid `<tool_call>{json}</tool_call>` structure

Recommended next phase: Phase2J, a two-stage decision-gated tool policy.
