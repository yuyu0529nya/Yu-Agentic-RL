# Phase2J Decision-Gated Tool Policy Plan

## Why Phase2J

Phase2H and Phase2I show a clear tradeoff:

| Phase | Text no-tool | Tool-name acc | Arg exact | Main lesson |
| --- | ---: | ---: | ---: | --- |
| Phase2H | 0.719 | 0.750 | 0.750 | strong tools, too much over-calling |
| Phase2I | 0.812 | 0.531 | 0.531 | fewer false tool calls, weaker tools |

This means one mixed SFT target is fighting itself. The model must learn two different skills at once:

1. decide whether the next assistant action is text or tool;
2. if it is tool, generate the exact tool name and arguments.

Phase2J should separate those two skills.

## Core Idea

Use a decision gate before tool generation:

```text
conversation prefix
        |
        v
decision gate: assistant_text or tool_call
        |
        +-- assistant_text -> normal assistant response
        |
        +-- tool_call -> constrained tool-call generator
```

The gate should reduce false tool calls. The generator should preserve Phase2H's stronger tool accuracy.

## First Experiment

Build an offline gated behavior evaluator.

Inputs:

- Phase2H behavior predictions
- Phase2I behavior predictions
- heldout labels

Compare four policies:

| Policy | Description |
| --- | --- |
| base | original Qwen2.5-7B |
| phase2h | one-stage Phase2H adapter |
| phase2i | one-stage Phase2I adapter |
| oracle_gate_h_tool | use gold action decision; when tool is needed, use Phase2H tool prediction |

The oracle gate is not a deployable model. It gives the upper bound: if a good gate exists, how much can we recover?

## Success Criteria

Phase2J is worth training only if the upper bound is clearly better than both Phase2H and Phase2I:

- text no-tool close to 1.000
- tool-name/arg exact close to Phase2H 0.750
- action type accuracy above 0.800

If oracle gating looks strong, train a real gate next.

## Real Gate Candidate

Train a small decision-only adapter:

- input: conversation prefix
- target: one token/string label, either `assistant_text` or `tool_call`
- no tool JSON generation in this adapter

Then inference becomes:

1. gate model predicts action type;
2. if `assistant_text`, use base or mixed text model;
3. if `tool_call`, use Phase2H-style tool generator with constrained decoding.

## Why This Is Better Than Another Data Ratio Run

Changing text/tool ratios improved one metric and damaged another. A gate makes the tradeoff explicit. It also gives a cleaner algorithm story:

- failure diagnosis: mixed SFT creates decision/tool conflict;
- solution: decompose action decision and tool generation;
- evaluation: compare one-stage vs gated policy;
- next step: full tau2 rollout with the best gated policy.

## Next Files To Add

- `scripts/analyze_gated_policy_upper_bound.py`
- `reports/phase2j_gated_policy_upper_bound.md`

The script should read local behavior summaries and create a table comparing base, Phase2H, Phase2I, and oracle-gated Phase2H.
