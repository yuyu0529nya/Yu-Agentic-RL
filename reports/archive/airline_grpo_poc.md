# Airline GRPO PoC — End-to-End RL on tau2-bench (Round-6)

## Goal

The airline line had stalled at SFT (decision-gate) with a negative full-tau2 result. Round-6
asks: can we run **actual GRPO RL on tau2 airline** on our own dual-GPU box, porting the
process-reward + length-aware-advantage recipe validated on the search agent?

## Setup

- **Self-built single-card GRPO** (`collect_rollouts` → `grpo_update`), NOT verl.
- **Independent usersim**: base 7B resident on GPU0; policy on GPU1 (vLLM up during rollout,
  stopped before the QLoRA update to free the card). Both served via vLLM hermes tool-calling.
- DOMAIN airline, train 8 tasks, held-out eval 6 tasks (greedy pass^1), N=6 rollouts/task,
  3 iters, reward = outcome + 0.3·process (prm_lite), LATA on, LR 1e-5, QLoRA 4-bit.

## Results

| iter | held-out pass^1 | train loss | usable rollouts | mean·|adv| |
|---|---|---|---|---|
| base | 0.500 (3/6) | — | — | — |
| 1 | 0.500 (3/6) | 1.08 | 18 (9+/9−) | 0.52 |
| **2** | **0.667 (4/6)** | 0.62 | 24 (11+/13−) | 0.60 |
| 3 | 0.500 (3/6) | 0.34 | 24 (11+/13−) | 0.54 |

## Findings

1. **The full RL loop runs end-to-end on the box.** tau2 rollout (multi-turn tool-calling +
   independent usersim) → trajectory + outcome reward → group-relative advantage (LATA +
   process reward) → QLoRA update → adapter hot-reload → next iter. This is the airline line's
   **first working RL loop** (previously stalled at SFT).
2. **Process reward keeps breaking group saturation.** Usable rollouts (groups with BOTH a
   success and a failure → nonzero advantage) held at 18–24 / 48 and *rose* with training —
   exactly the death-lock (all-0/all-1 group → std=0 → no gradient) that the long-horizon
   reference work flagged as the core obstacle. Our process signal kept intra-group variance alive.
3. **Train loss falls monotonically (1.08→0.62→0.34) but held-out pass^1 does not climb**
   (0.5 / 0.5 / 0.667 / 0.5). The 6-task held-out is coarse (±1 task = ±0.167), so the iter-2
   bump is within noise. This is the textbook "train reward up, generalization flat" of
   small-data GRPO — our train pool is **8 tasks**; the reference work hit the same wall at 40.

## Takeaway

An honest, negative-leaning result with a strong engineering/method core: **airline GRPO is now
a working, reproducible loop** on a single dual-GPU box, with an independent usersim and the
process-reward + LATA recipe — but 8-task data is far too small to show a stable held-out gain,
consistent with the structural data-scarcity constraint of tau2 airline (only 50 tasks total).

**Next (to turn this from PoC into a result):** scale the train pool to the full 30 tasks,
use finer eval (all 20 held-out × multi-trial pass^1 with paired CI), run more iters, and do a
vanilla-GRPO vs prm_lite vs prm_lite+LATA ablation to isolate the method's effect from the noise.

*Artifacts: `reports/round6_airline_grpo/` (master.log + eval_iter0–3.jsonl), md5-verified
`307d910d…`. GPU restored to borrow-time state (trap auto-stopped both vLLMs).*
