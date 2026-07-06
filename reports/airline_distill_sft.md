# Airline Teacher-Distillation SFT — first significant win on the airline line

**TL;DR.** Distilling successful teacher trajectories (collected through our own tau2 harness)
into Qwen2.5-7B via long-sequence QLoRA SFT raised held-out pass^1 from **0.18 → 0.38**
(+0.20, paired bootstrap CI [+0.13, +0.27], n=20 tasks × 5 trials, significant) — the first
significant positive result on the airline line after three rigorous nulls. The gain required
pushing training loss well below the 3-epoch default (1.06 → 0.74 at 9 epochs); at 3 epochs the
same data produced a flat result (+0.04, ns). Behavioral diffing shows the mechanism: the
distilled policy reads state ~2.3× more before acting, halves reckless write-calls, transfers
out-of-policy requests ~3× more often, and communicates ~35% more per turn — reversing the exact
failure modes our earlier SFT attempts exhibited.

## 1. Why distillation (motivation from our own diagnoses)

Three prior results boxed us in:

- **GRPO-from-base is gradient-starved.** In the controlled 2×2 ablation (vanilla /
  process-reward / length-aware-advantage / combined; `abl_4way_dual_0701`), no configuration
  beat base 0.23: with base success ~20%, most 30-task × N=4 rollout groups come back all-fail →
  zero intra-group variance → no learning signal.
- **Self-RFT is skill-bounded.** Our own successful rollouts NEVER call
  `update_reservation_flights/passengers` — precisely the tools held-out tasks require
  (documented in `grpo_rl_phase_findings.md`). You cannot bootstrap skills the policy never
  exhibits.
- **Naive SFT on tool fragments hurts.** M2: single-turn tool-biased SFT collapsed held-out
  0.28 → 0.10-0.12 (over-calling, under-communicating).

Teacher distillation attacks all three: a strong API model (deepseek-class) plays the agent
**through the same tau2 harness** (same system prompt, tool schemas, dialogue machinery), we keep
only reward=1 trajectories, and clone complete multi-turn behavior — injecting the missing skills
that RL can later amplify.

## 2. Data

- Teacher success **137/180 = 76%** on the 30 RL-train tasks (base policy: ~20%); weak tasks
  re-collected with 8 extra trials each → **30/30 tasks covered**.
- Final distill set: **114 trajectories** (success-only, per-task dedup by action signature,
  cap 4/task). **51% contain write actions**: `update_reservation_flights` ×56, `book` ×21,
  `cancel` ×20, `baggages` ×16, `passengers` ×8 — exactly the self-RFT blind spot.
- Collection is CPU+API only (no GPU): ~20 min for 180 sims, cost in the
  single-digit-RMB range.

## 3. Engineering: 20K-token trajectories on one 32GB card

Teacher trajectories are long (p50 = 5.1K, max = 19.2K tokens). Three stacked OOMs had to be
fixed before the first step ran — each caught by a 5-minute smoke on the 5 longest trajectories
before committing the full run:

1. **Eager attention** materializes the L×L score matrix (28 heads × 19218² fp32 = one 38.5GB
   allocation) → `attn_implementation="sdpa"`.
2. **SDPA silently fell back to its math backend** (which also materializes L×L) because the
   loss passed an explicit all-ones attention mask; at batch=1 there is no padding, so dropping
   the mask restores the fused causal fast path.
3. **The loss head materialized full-sequence vocab logits in fp32 twice**
   (`.float()` + `log_softmax` ≈ 2 × 11.7GB at 19K tokens) → run the backbone for hidden states
   only, then compute lm-head + cross-entropy per 4096-token chunk under
   `torch.utils.checkpoint` (logits recomputed in backward; full-sequence logits never exist).

Result: peak 29.3GB / 31.4GB, 114/114 trajectories rendered and trained with zero truncation.
A CPU-only pre-flight (tokenizer-level render of the real dataset) had already caught that the
trainer's default 4096-token right-truncation would silently cut the late write-action turns
from 67/114 examples — found and fixed before any GPU time was spent.

## 4. Results (held-out 20 tasks × 5 trials, self-hosted deterministic user-sim, USER_TEMP=0,
fixed eval seed, same-night base; paired bootstrap CI over tasks)

| checkpoint | train loss | held-out pass^1 | Δ vs base | 95% CI | train-fit pass^1 (10 hard train tasks) |
|---|---|---|---|---|---|
| base | — | 0.180 | — | — | 0.060 |
| SFT 3 epochs (45 steps) | 1.061 | 0.220 | +0.040 | [−0.03, +0.13] ns | 0.080 (ns) |
| **SFT 9 epochs (135 steps)** | **0.743** | **0.380** | **+0.200** | **[+0.13, +0.27] ★** | **0.140 (CI [+0.02, +0.14] ★)** |

The 3-epoch row is the finding that almost fooled us: loss "looked trained" (1.25 → 1.06) yet
behavior barely moved. Behavior cloning has a **training-depth threshold** — the capability jump
only appeared after pushing loss into the ~0.7 regime. (Loss was still descending at −0.02/epoch
when we stopped; headroom likely remains.)

## 5. Mechanism: what the policy actually learned (behavioral diff over the same 100 eval sims)

| behavior | base | e3 | e9 |
|---|---|---|---|
| read calls (reservation/flight/user lookups) | 291 | 313 | **669** |
| write calls / sim | 1.89 | 1.47 | **1.00** |
| tool-error messages | 201 | 146 | **137** |
| transfers to human | 13 | 17 | **38** |
| text chars / non-tool turn | 336 | 236 | **454** |

The distilled policy **reads before it writes** (2.3× more state lookups), **stopped spraying
writes** (base emitted 57 `update_reservation_passengers` calls across 100 sims, mostly wrong),
**learned that some requests should be refused/transferred** (several airline tasks are
policy-refusal tasks), and **communicates substantially more** per turn. This is the exact
inverse of our M2 failure profile (over-calling + terse turns) — evidence that cloning complete
successful conversations transfers *policy*, not just formatting.

## 6. Caveats (honest)

- n=20 held-out tasks; CIs are wide even when significant. Single seed, single run.
- Same-night protocol: base re-measured in the same session (run-to-run drift of the
  temp-0.5 agent across nights is ~1 SE; cross-night comparisons are not valid).
- Teacher dialogues used an API user-simulator; evals use the self-hosted 7B user-sim — a mild
  train/eval style mismatch that did not prevent the gain but may cap it.
- Train-fit slice (0.14) remains far below the teacher (0.76) — the student has not saturated
  its teacher; more depth and/or RL should close part of the gap.

## 7. Next

1. Continue SFT (e15) — loss slope suggests unexhausted headroom.
2. **GRPO from the SFT checkpoint**: at ~0.38 success, rollout groups become outcome-mixed →
   intra-group variance returns → the RL phase that was gradient-starved from base becomes
   trainable. This is the designed second act: distillation raises the floor, RL optimizes on it.

## Recompute any number

```
# pass^1 (any eval file)
python3 -c "import json;rs=[json.loads(l)['reward'] for l in open(F)];print(sum(r>=1-1e-6 for r in rs),'/',len(rs))"
# paired CI table
python3 scripts/grpo/tau2_eval_analyze.py autodl_artifacts/distill_sft_0702/base_eval.jsonl \
  e3=autodl_artifacts/distill_sft_0702/distill_eval.jsonl \
  e9=autodl_artifacts/distill_sft_0702_ext/distill_ext_eval.jsonl
```
