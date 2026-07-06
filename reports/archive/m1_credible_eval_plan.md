# M1: Credible End-to-End Tau2 Eval — Plan + Key Finding

## Goal

Replace the noisy 5-task pass^1 (where a 1-task change = 20%) with a statistically
credible, leakage-aware end-to-end tau2 airline pass^1 over all 50 tasks, so that
M2 (does the SFT stack move pass^1?) has an honest baseline to compare against.

## Eval harness (how end-to-end pass^1 is produced)

- Agent = local Qwen2.5-7B served by **vLLM** (OpenAI-compatible, localhost:8000). Free GPU time.
- User simulator = **`anthropic/glm-5.1` via API** -> the cost driver (scales with tasks x turns).
- Driver: `scripts/run_tau2_base_vs_sft_vllm_autodl.sh` (single) / `run_phase2g_full_tau2_compare_autodl.sh` (compare).
- pass^1 via `scripts/summarize_tau2_results.py`; NEW seen/unseen split via `scripts/summarize_tau2_seen_unseen.py`.

## Task set + seen/unseen split

Airline = 50 tasks, ids `"0".."49"` (no built-in split). Canonical split file:
`scripts/m1_eval_task_split.json`.

- **seen (10)** = union of ALL SFT train/valid task ids across Phase2H mixed-policy v1,
  Phase2I decision-balanced, and Phase2J decision gate (all used the same set):
  `1,12,15,20,23,27,33,34,38,42`.
- **unseen (40)** = the remaining 40 tasks.

## KEY FINDING (from re-analyzing the existing GLM-5.1 50-task baseline, no new spend)

`python scripts/summarize_tau2_seen_unseen.py third_party/tau2-bench/data/simulations/airline_baseline_50_anthropic_glm_5_1_50tasks_1trials`

| Split | pass^1 | successes/trials |
| --- | ---: | ---: |
| overall | 0.5800 | 29/50 |
| seen (SFT train/valid) | **0.0000** | 0/10 |
| unseen | 0.7250 | 29/40 |

**The 10 seen tasks all fail at baseline (0/10); the 40 unseen score 0.725.**
The SFT trained exclusively on the *hardest* tasks (consistent with the project's
failed21 / stable-fail history), NOT a random sample.

Consequences:
1. **seen/unseen is confounded with difficulty** — do NOT read seen-vs-unseen as
   memorization-vs-generalization. Seen tasks are intrinsically harder.
2. The old Phase2G 5-task slice (`2,16,18,25,44`) is **entirely unseen + easier**.
   base = v3 = v4 = 0.20 there partly because it measured transfer to tasks the SFT
   never targeted, not the tasks it actually learned.
3. The seen-10 tasks were **never in any end-to-end eval slice** until now.

## Revised eval design (M1 -> M2)

1. **M1 baseline**: run base Qwen2.5-7B on all 50 tasks, NUM_TRIALS=1, report
   overall + seen + unseen pass^1. (The GLM-5.1 table above is a strong-agent
   reference, NOT the Qwen base — still must run Qwen base.)
2. **M2 comparison**: evaluate the SFT stack (decision gate + Phase2H tool generator)
   on the SAME 50 tasks. Compare **paired, same-task** base-vs-SFT, reported
   separately for seen-10 (where the SFT has direct signal -> biggest expected lift)
   and unseen-40 (transfer). Headline = paired delta, with the difficulty confound
   stated explicitly.
3. Optionally NUM_TRIALS>=2 later for pass^k stability; start with 1 trial x 50 tasks.

## Cost / time estimate

- ~34 messages/task ≈ ~17 GLM-5.1 user-sim calls/task -> ~850 calls for 50 tasks.
- Rough API cost: ~¥10–50 (depends on GLM tier; smoke will pin it down).
- Wall-clock: sequential (max-concurrency 1) ≈ 1.5–4 h. Bump vLLM `--enforce-eager`
  + max-concurrency 4–8 + MAX_NUM_SEQS 4–8 -> target ~20–40 min. (Runner currently
  hardcodes `--max-concurrency 1`; add a MAX_CONCURRENCY env var before the full run.)

## BLOCKER: vLLM cannot start on the current 5090D image

vLLM 0.23.0 requires torch==2.11.0 (installed, matched), but torch 2.11.0+cu130's own
inductor is internally inconsistent: `torch._inductor.kernel.mm_scaled` imports
`scaled_mm_configs` from `mm_common`, which is absent ->
`ImportError: cannot import name 'scaled_mm_configs'`. Triggered at import via a
module-level `@torch.compile` in `vllm/utils/deep_gemm.py`, so `--enforce-eager` and
other runtime flags cannot dodge it. Training (Phase2J) is unaffected (no vLLM).

Fix candidates (deferred per user choice 2026-06-22):
1. `pip install --force-reinstall --no-deps torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0`
   from the cu130 index to repair the corrupted files (same version -> low risk to the
   peft/bitsandbytes training env; ~2–4GB download, pytorch cu130 index may be slow from CN).
2. Build an isolated eval venv just for vLLM serving (safest isolation, more setup).
3. Pin a different known-good vLLM+torch combo.

The 2-task smoke (2026-06-22) validated everything UP TO vLLM start: launch script,
`.env` credential sourcing (GLM keys present), and the tau2 runner are all OK.

## Validated tooling (works now, pre-fix)

- `scripts/m1_eval_task_split.json` — canonical 50-task seen/unseen split.
- `scripts/summarize_tau2_seen_unseen.py` — leakage-aware pass^1; reuses
  `summarize_tau2_results.py` parsing; tested on the 50-task GLM baseline (overall
  0.5800 matches the original summarizer).

## Success criteria

- M1 done: a base Qwen2.5-7B pass^1 over 50 tasks with seen/unseen breakdown, where a
  single-task change does not flip the headline.
- M2 worth a full run only after the decision gate clearly beats Phase2I on
  same-口径 action selection; then look for a paired same-task pass^1 lift (esp. on seen-10).
