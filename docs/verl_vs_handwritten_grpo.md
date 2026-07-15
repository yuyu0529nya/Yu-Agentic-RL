# GRPO, two ways: my from-scratch trainer vs. veRL

I hand-wrote the entire GRPO loop (`scripts/grpo/grpo_update.py`) to understand the mechanics,
then reproduced the same loop on **veRL** (ByteDance's FSDP + vLLM RL framework) to see how an
industrial system makes the same choices. This note is the point-by-point comparison. All claims
are backed by the source of both implementations.

**One-line framing.** veRL is *online PPO-clip-style GRPO* (rollout and update in one process,
importance-sampling ratio + clipping, FSDP + async vLLM). My version is an *offline
REINFORCE-with-group-baseline* GRPO core (collect rollouts, then run one advantage-weighted NLL
pass under QLoRA — no ratio, no clip). I deliberately stripped the PPO shell and kept only the
part that makes GRPO "G" (the group-relative baseline), then added a few mechanisms of my own.

---

## 1. Group-relative advantage `(r − mean)/(std + ε)`

Same math on both sides: **outcome-level** (one scalar reward per sequence) → standardize within
the group → broadcast the scalar back onto every response token (so a GRPO advantage is constant
along the response).

- **veRL** (`core_algos.py`, `compute_grpo_outcome_advantage`): sums token-level reward to a
  per-sequence scalar, groups by `uid`, standardizes with `ε = 1e-6`; singleton groups get
  `mean=0, std=1`; `norm_adv_by_std_in_grpo=False` degrades to Dr. GRPO (subtract mean only).
- **Mine** (`grpo_update.py`): groups by `task_id`, `a = (r_shaped − m)/(sd + denom_eps)`, with
  two deliberate departures — (1) **`denom_eps = 0.25`**, used as a *minimum effective
  denominator / temperature* so tiny shaped-reward gaps aren't blown up to unit scale when
  `sd ≈ 0`; (2) **advantage hard-clipped to ±3** (veRL does not clip on the advantage side — it
  clips on the loss side via the ratio).

## 2. Length-aware advantage (LATA, `√L`) — unique to my version

My length normalization lives on the loss side: `norm = √L` (LATA) vs. `norm = L` (vanilla),
loss `= (−adv · Σlogp)/norm`. Intuition: with `÷L`, per-token gradient on long trajectories
decays as `1/L`, which trains the model to *give up faster*; `√L` decays only as `1/√L`, keeping
the incentive to reason across many turns.

veRL's closest knob is `loss_agg_mode` with four settings: `token-mean` (÷ global),
`seq-mean-token-mean` (÷L ≈ my vanilla), `seq-mean-token-sum` (÷1), `…sum-norm` (÷ const).
**`√L` sits between `÷L` and `÷1`; veRL's enum cannot express it.** (I also have a
Turn-Discounted advantage that up-weights early tokens — orthogonal to LATA, also absent in veRL.)

## 3. Policy loss + KL

- **veRL = PPO-clip (+ dual-clip):** `ratio = exp(logp − old_logp)`, two-sided clip, extra
  dual-clip when `adv < 0`. Genuinely on/off-policy tolerant, so it can reuse samples over
  multiple PPO epochs. KL on two independent paths: KL-in-reward (adaptive β controller, folded
  into reward *before* advantage) and KL-in-loss (`kl_loss_type = low_var_kl`).
- **Mine = plain policy gradient (no ratio, no clip):** loss `= (−adv · Σlogp)/norm`. Stability
  comes from advantage clip ±3, grad clip, and `denom_eps` instead of a ratio clip. KL is
  loss-side only and optional (`--kl-coef`, default 0).
- **KL estimator is identical on both sides — Schulman k3** (`exp(kl) − kl − 1`), non-negative and
  low-variance. veRL additionally offers a straight-through unbiased-gradient variant (`k3+`).
- **Reference model:** I use `with model.disable_adapter():` — the *same* model with LoRA turned
  off is the reference, **zero extra memory**. veRL spins up a separate frozen FSDP reference
  worker.

**Sharp point.** veRL's clip protects against policy drift between rollout and update
(off-policyness). I collect and immediately update once per batch — near-strictly on-policy — so I
*can* drop the clip; the price is lower sample efficiency (each rollout is used once). Being able
to say *why* the clip exists, and when you can omit it, is the point.

## 4. Token mask — different problems entirely

- **Mine = render-twice-diff** (`grpo_update.py`, `render_with_assistant_mask`): my training data
  is *offline tau2 multi-turn tool-agent trajectories*, where assistant turns are scattered
  between user/tool turns. I render `messages[:i]` and `messages[:i+1]`, diff the token lengths to
  locate exactly the tokens the policy generated, with a prefix-consistency check and an
  `assert_render_sane` guard that catches transformers-v5's silent all-zero mask (which is why the
  repo pins transformers 4.57.x).
- **veRL:** rollouts are self-produced, so the mask (`response_mask`) is known at generation time.
  The training forward instead spends effort on **remove_padding / flash-attn varlen** (a
  throughput optimization), not on recovering which tokens to train.

## 5. Outcome-variance gating — mine has it, veRL doesn't

- **Mine:** when a group's binary outcomes are all identical (all pass / all fail, `pstdev = 0`),
  I zero the whole group's advantage and then **drop it** (`abs(adv) < 1e-8` filter). The
  motivation is concrete: under process-shaping, all-fail groups *fabricate* a phantom advantage
  correlated with length (~63% length-aligned), training the model to give up — the mechanistic
  cause of an observed 0.35 → 0.20 collapse.
- **veRL:** an all-identical group standardizes to advantage ≈ 0 naturally, **but the samples stay
  in the batch** (contribute zero loss, never dropped); singletons even keep their raw score.
- This is the same problem DAPO solves with **"dynamic sampling"** — I arrived at it independently
  by diagnosing the failure, and can explain the *why*, not just the *what*.
- **Measured, 2026-07-15 (honest negative result).** I ported this gate into veRL's standard
  trainer (`USE_DYNAMIC_SAMPLING`) and ran a single-variable A/B on tau2-airline from a fixed SFT
  checkpoint: **gating OFF → val 0.5625, gating ON → 0.4125.** On this task the gate *hurts*:
  at ~20% success most groups are all-fail, so dropping them removes 54–75% of the batch and
  starves the gradient — the sample-count loss outweighs the denoising. The earlier flat curve I
  had blamed on this was actually an LR problem (lr too small, `grad_norm≈0.05`). The gate's
  implementation is still correct (11 unit tests; kills the phantom advantage exactly as designed)
  — it just doesn't transfer to a low-success-rate task. See
  [`../verl-integration/results_20260712/RESULTS_ab_gating.md`](../verl-integration/results_20260712/RESULTS_ab_gating.md).

## 6. Memory / parallelism

|  | Mine | veRL |
|---|---|---|
| Scale | 4-bit NF4 base + LoRA r=16 (train adapter only) | full-parameter FSDP shards (bf16), optional LoRA |
| Parallel | **single-node, dual-GPU**: user-sim / rollout server on one card, QLoRA update on the other | multi-GPU FSDP + Ulysses sequence parallel |
| Rollout | **offline, decoupled**: collect → JSONL → update (two processes, two steps) | **online, integrated**: async rollout inside `fit`, update immediately |

My long-sequence memory engineering (the highest-signal part of the code) — running 20K-token
trajectories on a 32GB card — is hand-built and commented: (1) don't pass an attention_mask on a
single unpadded sequence, or SDPA falls off the flash path and materializes a full `L×L` fp32
matrix (28 heads × 19218² ≈ 38.5GB in one alloc); (2) never materialize full-sequence logits
(`[19K, 152K]`) — run the backbone to hidden states, then compute `lm_head + CE` in 4096-token
chunks under `torch.utils.checkpoint`; (3) `attn_implementation="sdpa"`. veRL absorbs all of this
into the framework (flash-varlen + FSDP offload + chunked loss).

## 7. What each has that the other doesn't

**veRL has, mine doesn't:** PPO ratio + clip + dual-clip, adaptive KL controller, a menu of
advantage variants (RLOO, REINFORCE++, ReMax, GPG, …) and policy losses (GSPO, CISPO, KL-cov, …),
k3+ unbiased KL gradient, remove_padding / Ulysses / FSDP-Megatron / Ray orchestration, online
async rollout.

**Mine has, veRL doesn't:** LATA (`√L`), Turn-Discounted advantage, explicit outcome-variance
gating + phantom-advantage guard, `denom_eps`-as-temperature + advantage hard-clip, LoRA-disable
reference model (zero extra memory), render self-test + pinned transformers version, one-flag RFT
mode, and the whole pipeline for recovering assistant masks from offline multi-turn trajectories.

## 8. Where this sits in the 2026 landscape

GRPO is no longer the single SOTA — DAPO, GSPO, and Dr. GRPO are the well-known refinements, but
they are *minor refinements of GRPO with the same core idea* (groups of rollouts as the baseline).
My work already touches these frontier concerns independently: **outcome-variance gating ≈ DAPO
dynamic sampling**, **LATA ≈ the length-bias problem that Dr. GRPO targets**, and the code already
exposes `norm_adv_by_std_in_grpo` (the Dr. GRPO std-normalization debate). The natural next step is
to run DAPO/GSPO in veRL and add them to this comparison.

---

*Source anchors — veRL: `core_algos.py` (advantage / agg_loss / vanilla policy loss / low_var_kl),
`ray_trainer.py` (apply_kl_penalty / compute_advantage / fit), `workers/utils/losses.py`,
`workers/engine/fsdp/transformer_impl.py` (remove_padding). Mine: `scripts/grpo/grpo_update.py`
(advantage+gate / mask / loss+LATA+td / k3 KL / QLoRA memory), `prm_lite_reward.py`,
`collect_rollouts.py`.*
