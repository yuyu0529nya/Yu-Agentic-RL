# Pre-registration — GSM8K algorithm bake-off & the dynamic-sampling confound

**Committed before any of these runs started.** Written on 2026-07-17, after the tau2-airline A/B
(2 seeds × 2 arms) failed to reproduce its own single-seed result and forced a rewrite of what I
thought "dynamic sampling" was doing.

Every outcome below is recorded as-run, including the ones that make my own trick look useless.

---

## 1. What went wrong last time, and why this experiment exists

On tau2-airline I measured "binary-outcome gating hurts" from **one seed** (0.4125 vs 0.5625,
−0.15). A second seed collapsed it to −0.0125. The gated arm's own cross-seed spread (0.125)
exceeded the between-arm gap (0.081), so at n=2 there was no effect to explain.

Reading the source explained why there could not have been one:

1. veRL's GRPO advantage is `(score − mean)/(std + eps)` with `norm_adv_by_std_in_grpo=true`.
   An all-same-outcome group has `std = 0`, so its advantage is **exactly 0**. Those rows already
   contribute nothing to the gradient.
2. My implementation zeroed their `response_mask`. veRL's default `loss_agg_mode=token-mean`
   divides by `loss_mask.sum()`, so this left the numerator alone and **shrank the denominator** —
   scaling the loss by `1/live_frac`, measured at **2.4×–24×**, varying step to step.
3. Adam is ~invariant to a global gradient rescale, so most of (2) was absorbed. Measured:
   `pg_loss` ↑4–17× and `grad_norm` ↑2–3×, but **`ppo_kl` identical across arms**.

**So "mask zeroing" is not a filter, it is a time-varying learning-rate multiplier, and any A/B
built on it is confounded with LR.** This experiment removes that confound.

## 2. Platform, fixed in advance

| | |
|---|---|
| task | GSM8K (deterministic exact-match reward → **no user simulator, no API cost, low noise**) |
| model | `Qwen2.5-1.5B-Instruct` |
| trainer | veRL, single GPU, LoRA, derived from the already-working `run_gsm8k_grpo_smoke.sh` |
| reward | flexible answer extraction (`custom_reward_function`), **not** veRL's default `strict` |
| seeds | **`{42, 123}`**, both arms of every comparison, fixed here |
| metric | `val-core/openai/gsm8k/acc/mean@1` on the held-out test split |
| eval cadence | `val_before_train=True`, `test_freq=5` |

**Gate on the baseline before spending anything else:** a plain GRPO arm must reach
**≥ 0.50 val accuracy**. A previous run on the company machine scored **0.0796–0.0895** because
veRL's default `strict` extractor requires `#### <number>`, which an *Instruct* model does not
emit unprompted. If the baseline does not clear 0.50, the reward wiring is broken and **no other
arm is run** — every number would be measuring the extractor, not the algorithm.

## 3. Arms

Single-variable throughout: every arm is derived from the same working script and changes only the
keys listed. Same seeds, same data, same everything else.

| # | arm | gate signal | drop mechanism | what changes |
|---|---|---|---|---|
| ① | **GRPO** (baseline) | — | — | nothing |
| ② | **DAPO** | reward std | physical + resample | clip 0.2/0.28, token-mean, no KL, **+ filter_groups** |
| ③ | **GSPO** | — | — | `policy_loss.loss_mode=gspo` (sequence-level ratio) |
| ④ | **Dr.GRPO** | — | — | `norm_adv_by_std_in_grpo=false` |
| ⑤ | **gated-mask** | binary outcome | **mask zeroing** | my tau2 implementation, unchanged |
| ⑥ | **gated-drop** | binary outcome | **physical + resample** | same gate as ⑤, DAPO's mechanism |

Arms ②/⑤/⑥ use `group_filter.py` (19 CPU unit tests passing, committed alongside this).

## 4. The comparisons this buys, and what each isolates

| comparison | isolates | pre-registered prediction |
|---|---|---|
| **⑤ vs ⑥** | **the denominator confound** (mask vs physical, gate held fixed) | ⑥ ≠ ⑤ if the confound mattered; ⑥ ≈ ⑤ if Adam really absorbed it |
| **⑥ vs ①** | **does dropping dead groups help at all**, un-confounded | *unknown — this is the actual open question* |
| **② vs ⑥** | **gate signal** (std vs outcome), mechanism held fixed | **should be ~identical** — see §5 |
| ①②③④ | the algorithm bake-off | no prediction registered |

## 5. A built-in falsification check

Unit test `test_binary_gates_agree` proves that under a **binary** reward the std-gate and the
outcome-gate make the *same* decision on every group (all-fail ⇒ std = 0 ⇒ both call it dead).

GSM8K's reward is binary. **Therefore arms ② and ⑥ should produce near-identical training
dynamics.** If they diverge beyond noise, the implementation is wrong — not the theory. I am
registering this in advance as a bug detector, not as a result.

Corollary, also registered in advance: **this experiment cannot show binary-outcome gating beating
std-gating.** Under a binary reward there is no phantom advantage to kill, so the gate's one real
differentiator is invisible here. Demonstrating it requires a **shaped/PRM reward**, which is out
of scope for this run and is named here so a later positive result cannot be presented as if it
had been predicted after the fact.

## 6. Noise discipline

- **Two seeds per arm.** Any difference smaller than the between-seed spread of the *same* arm is
  reported as "not resolved at n=2", never as an effect. This is the exact mistake that produced
  the −0.15 phantom on tau2.
- GSM8K's test split is n≈1319, so per-run sampling noise is far smaller than tau2's ±0.05 — that
  is the main reason this platform was chosen over tau2 for algorithm comparison.
- `group_filter` records `live_frac`, `loss_scale_if_masked`, and `generation_overhead` every step,
  so the confound's size is logged rather than inferred afterwards.

## 7. What would make me report a negative result

Stated in advance so it cannot be renegotiated later:

- **⑥ ≈ ①** → dropping dead groups does nothing on this task, and my trick has no effect even
  after removing the confound. Reported as such.
- **⑤ ≈ ⑥** → the LR confound was absorbed by Adam and never mattered. Reported as such; the tau2
  conclusion stands unchanged.
- **⑥ < ①** → the gate actively hurts once the confound is removed. Reported as such.
- **② diverges from ⑥** → implementation bug (see §5). The run is discarded, not published.

## 8. Cost

~6 arms × 2 seeds. Single GPU, no API. Auto-shutdown on completion via the existing watchdog
pattern. Expected order **¥30–60** of rented GPU time; the number is small enough that "we ran out
of budget" is not an acceptable reason to skip an arm or a seed.
