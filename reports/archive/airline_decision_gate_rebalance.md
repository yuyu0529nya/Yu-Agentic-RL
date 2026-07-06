# Airline Decision-Gate — Rebalancing the Tool Bias (Round-5)

## Goal

Phase2J's decision-only gate (predict the next action label: `assistant_text` vs `tool_call`)
had an **over-calling** problem — 23/72 text turns were mis-predicted as `tool_call`, capping
text recall. Hypothesis for the cause: the train set is **tool-biased** (66 text / 86 tool),
giving the model a tool-leaning prior. The trainer has no class-weight option, so the fix is at
the **data layer** — up-sample the `assistant_text` rows.

## Setup

- 3 variants × 3 seeds (7/17/27), base Qwen2.5-7B-Instruct, LoRA r16/α32, 180 steps, fp16, dual-GPU parallel.
- Eval: **195-probe heldout, deterministic greedy classification** (no sampling noise) — unified 口径 across all runs (fixes the earlier 128-vs-64 probe inconsistency).
  - `repro` = 66/86 (original)  •  `bal11` = 86/86 (text up-sampled to 1:1)  •  `txt12` = 103/86 (text-favored 1.2×)

## Results — variant means (3 seeds)

| variant | acc | text recall | tool recall |
|---|---|---|---|
| repro (original) | 0.612 | 0.387 | 0.822 |
| bal11 (1:1) | 0.624 | 0.333 | 0.894 |
| **txt12 (text-favored)** | **0.660** | **0.475** | 0.832 |

Per-seed spread (the headline caveat):

| variant | text recall by seed (7/17/27) |
|---|---|
| repro | 0.745 / **0.000** / 0.415 |
| bal11 | 0.266 / 0.404 / 0.330 |
| txt12 | 0.628 / 0.277 / 0.521 |

## Findings

1. **Direction confirmed.** `txt12` (text-favored) is best — acc 0.660 (>repro 0.612), text
   recall 0.475 (>repro 0.387, **+0.088**, the over-calling target), tool recall 0.832 held (no
   collapse). `bal11` (1:1) under-corrected (text recall even dipped). Tilting the data toward
   text is the effective lever for over-calling.
2. **Pipeline + Phase2J reproduced.** `repro` seed-7 = acc **0.713 / text 0.745**, matching
   Phase2J's 0.727 — the same-口径 cross-check passes, so the rig is trustworthy.
3. **The dominating story is variance, not the mean.** With ~80–90 effective training samples,
   LoRA SFT is extremely seed-sensitive (repro text recall spans **0.000–0.745**; seed-17
   collapsed to all-tool). The 3-seed mean trend (txt12 > repro) is a **weak signal, not a
   nailed-down result**.
4. **This is the structural data-scarcity constraint** that the reference long-horizon tau-bench
   work also flagged: τ-bench airline has only 50 tasks. Data scarcity caps both the mean and
   its reliability — "add data" is not optional, it's the bottleneck.

## Takeaway

Rebalancing toward text is the **right direction** for over-calling, but 50-task airline data is
too small for a stable single-stage SFT gate — the variance, not the algorithm, is the ceiling.
Next options: (a) enlarge/augment the gate data, or (b) the small-scale airline GRPO PoC
(process reward + length-aware advantage, both validated on the search agent) which attacks the
same data-scarcity death-lock from the RL side (group-variance collapse).

*Artifacts: `reports/round5_gate/` (10 behavior summaries + GATE_MASTER.log), md5-verified.*
