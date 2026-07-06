# NEXT RUN (2026-07-03 night) — e15 continuation, then GRPO-from-SFT

STATUS: e9 distill = VERIFIED WIN (held-out 0.18→0.38, CI [+0.13,+0.27]★; report
`reports/airline_distill_sft.md`). All run scripts now flock-guarded (ssh-replay safe).
LAUNCH RULE: nohup wrapper logs use `>>` APPEND (an overwrite destroyed crash diagnostics 07-03).

## Tonight step 1 — continue SFT e9→e15 (~1.7h incl. eval)
```
cd $HOME/agentic-rl && RUN=distill_sft_0703_e15 \
  ADAPTER_IN=outputs/distill_sft_0702_ext/distill_adapter_ext \
  BASE_DIR=outputs/distill_sft_0702 LABEL=distill_e15 \
  setsid nohup bash scripts/grpo/continue_distill_sft.sh >> logs/distill_e15.log 2>&1 &
```
Read: if e15 held-out ≥ e9+0.03 → e15 becomes the GRPO warm start; if flat/worse → e9 is it.

## Tonight step 2 — GRPO-from-SFT, single arm first (~3h)
```
cd $HOME/agentic-rl && RUN=grpo_from_sft_0703 METHODS=vanilla \
  INIT_ADAPTER=<best adapter from step 1> N=6 ITERS=3 \
  setsid nohup bash scripts/grpo/run_grpo_ablation_4way.sh >> logs/grpo_from_sft.log 2>&1 &
```
The script now self-evals the warm-start checkpoint (init_eval.jsonl) after base eval, so the
curve has its anchor: base → init(SFT) → it1..3. Watch: intra-group variance in the collect
lines — at ~0.38 start the groups should finally be outcome-MIXED (the whole point).

---

# (done 07-02/03) — airline DISTILLATION: teacher trajectories → SFT → (then GRPO-from-SFT)

Post-4-way verdict (abl_4way_dual_0701, VERIFIED): no GRPO config beats base 0.23 at n=20; the
reference's gains ride on an SFT-warm-start + 250-step budget + a collapsed-vanilla comparator we
don't have. Biggest single lever for OUR pass^1 = raise the policy floor via TEACHER DISTILLATION
(deepseek succeeds where self-RFT is skill-bounded — our own successes NEVER call
update_reservation_flights/passengers, which held-out tasks need; see grpo_rl_phase_findings).

## Daytime (NO GPU): collect teacher trajectories on the GPU box (CPU + API only)
```
# smoke (2 tasks × 2 trials, pennies):
TRAIN_TASK_IDS=2,3 TRIALS=2 MAXC=2 RUN=teacher_smoke_0702 bash scripts/grpo/run_teacher_collect.sh
# full (30 train tasks × 6 trials ≈ 180 sims):
RUN=teacher_ds_0702 bash scripts/grpo/run_teacher_collect.sh
```
Needs DEEPSEEK_API_KEY in $HOME/agentic-rl/.env. Output: outputs/<RUN>/distill_dataset.jsonl
(success-only, dedup, cap 4/task). Teacher = agent deepseek-chat + usersim deepseek (temp 0.7 both,
diversity for TRAINING data — eval protocol stays self-hosted/deterministic, unaffected).

## Tonight (GPU): SFT from the distill set + same-protocol paired eval
```
DATASET=outputs/teacher_ds_0702/distill_dataset.jsonl RUN=distill_sft_0702 \
  bash scripts/grpo/run_distill_sft_company.sh
```
Protocol identical to the 4-way (usersim GPU0 / policy GPU1, USER_TEMP=0, EVAL_SEED=900, 5-trial,
32768 ctx, caches on /data). Reports: held-out 20 paired-vs-base + a 10-task TRAIN-FIT slice with
its own base reference (SFT must lift train-fit first; if train-fit doesn't move, the SFT is inert;
if train-fit jumps but held-out doesn't, it's a transfer/coverage question — both are findings).
GO/NO-GO for the next night: held-out distill ≥ base + ~0.05 → run GRPO-from-SFT (ADAPTER_INIT,
more iters, N=8) — needs a small run_grpo script tweak to accept an initial adapter (TODO).

---

# (older) NEXT RUN — airline GRPO improvement (after R4 null result)

Diagnosis (from the 55-agent workflow, verified on the real R4 jsonl):
1. **Measurement blindness** — 20×1 eval, 10/20 tasks unsolvable by anyone; the 0.35→0.20→0.25→0.35 curve is inside the ±0.15 noise band (McNemar p≥0.25). On the discriminating **live-10**, base is actually **0.70**.
2. **Gradient starvation + phantom gradient** — 11/15 train groups zero binary variance; prm_lite fabricated phantom advantage on all-fail groups (usable=90/90 was hiding this). **Deeper root bug now fixed**: prm_lite read tau2's *flat* tool_calls as OpenAI-*nested* → every tool was unclassified → only length penalties fired (= "reward shorter failures").
3. **Train/eval mismatch** — train was temp1.0/256tok (84–112 broken tool calls/iter) vs eval temp0/512tok (0 broken).

## SEARCH-AGENT RLVR (flagship, built 2026-06-25 — Agent+RL centerpiece)
Multi-turn agent searches a LOCAL BM25 corpus, verifiable exact-match QA reward, on-policy GRPO.
Reuses grpo_update.py + summarize_eval.py. Files (all logic self-tested, no GPU): `qa_reward.py`
(EM/F1), `search_retriever.py` (pure-stdlib BM25 + HotpotQA corpus builder), `search_agent.py`
(multi-turn loop: `<search>q</search>` → BM25 passages → `<answer>`; `--mock` self-tests
orchestration), `run_search_agent.sh` (base eval → 4-iter on-policy collect+GRPO → adapter eval →
summarize). Run (validate on 1.5B first, then POLICY_MODEL=7B for the flagship):
```
RUN=search_r1 HF_ENDPOINT=https://hf-mirror.com WORKDIR=/root/autodl-tmp/yuyu \
  PYBIN=/root/miniconda3/bin/python EXTRA_PATH=/root/miniconda3/bin \
  POLICY_MODEL=/root/autodl-tmp/models/qwen25-1.5b-instruct bash scripts/grpo/run_search_agent.sh
```
CAVEAT (only thing untested locally — no datasets/net): `datasets` 5.0 may have dropped the
`hotpot_qa` loader script. If `load_dataset("hotpot_qa","distractor",...)` fails, load the parquet
directly from the hub (hf-mirror) or swap to a parquet-native multihop QA set; adjust
`search_retriever.build_hotpotqa_corpus`. Everything else is tested.

## R5 UPDATE (2026-06-24) — read first
Ran the full plan (clean base + RFT + gated GRPO). **All within noise, McNemar p=1.0** — no significant win. The gated GRPO worked MECHANICALLY (curriculum 2,3,8,13,19 + N=12 → usable 48–60/60 real gradient, no phantom). **Dominant blocker = the DEEPSEEK USER-SIMULATOR is stochastic**: same base/temp0 gave 0.35/0.70 (R4) vs 0.20/0.30 (R5) across runs. So single-run pass^1 is not reproducible; cross-run base-vs-adapter is unfair.
**FIX (built + tested): variance-controlled paired eval.** `collect_rollouts.py --user-temperature 0` (env `USER_TEMP`) pins the user-sim temp via tau2 `--user-llm-args`; `summarize_eval.py` now adds a PAIRED per-task rate test (bootstrap 95% CI + sign test). One command:
```
ADAPTER_PATH=outputs/rft_airline/adapter TAG=rft_paired \
  WORKDIR=/root/autodl-tmp/yuyu PYBIN=/root/miniconda3/bin/python EXTRA_PATH=/root/miniconda3/bin \
  USER_TEMP=0.0 NUM_TRIALS=8 bash scripts/grpo/run_paired_eval.sh
```
This gives a TRUSTWORTHY delta (it won't manufacture a win — it tells you if any real effect exists above the now-reduced noise). To actually MOVE the metric still needs scope-up: more held-out tasks (20→50) for power, and broader training data (full trajectories covering the write-tools the live-10 needs).

## Code changes already made & tested locally
- `collect_rollouts.py`: `--max-tokens` default 256→**768**; `--no-auto-resume` (fresh eval); `--assert-no-broken-calls` hygiene gate.
- `grpo_update.py`: **advantage gate** (drop all-same-binary-outcome groups), denom `(sd+0.25)` + clip ±3; **`--rft`** mode (advantage=+1, mean-NLL, no LATA); `--gate/--no-gate`.
- `prm_lite_reward.py`: **fixed flat/nested tool_calls** (the big one), **revived entity extraction** (was dead `{}`), per-`tool_call_id` error pairing, length threshold 8→16.
- `base/adapter_eval_retail.sh`: `NUM_TRIALS`/`TEMP`/`MAX_TOKENS` env, `--no-auto-resume`, adapter eval `SAVE_TAG`.
- new `summarize_eval.py` (per-task pass^1 + live-10 headline + paired McNemar), `build_rft_dataset.py`, `run_rft.sh`.

## AutoDL env (prepend to every command)
```
export WORKDIR=/root/autodl-tmp/yuyu PYBIN=/root/miniconda3/bin/python EXTRA_PATH=/root/miniconda3/bin \
       POLICY_MODEL=/root/autodl-tmp/models/qwen25-7b-instruct DOMAIN=airline USER_LLM=deepseek/deepseek-chat
```

## STEP 0 (do FIRST, ~15–30 min) — trustworthy base, multi-trial
```
NUM_TRIALS=5 TEMP=0.7 MAX_TOKENS=768 TASKS=0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46 \
  bash scripts/grpo/base_eval_retail.sh
python scripts/grpo/summarize_eval.py --eval outputs/base_airline_eval.jsonl
```
→ trust the **LIVE-10 pass^1 ± SE** as the real baseline (expect ~0.70). Everything below is measured the SAME way (temp0.7×5).

## PATH A (recommended first capability shot) — RFT/STaR, no new rollouts
```
bash scripts/grpo/run_rft.sh        # build 33-ex dataset → SFT from base → eval → paired summary
```
Success = on live-10, net `fail→succ > succ→fail` vs base with McNemar p<0.10. Skill-bounded: the 42 successes never call update_reservation_flights/passengers.

## PATH B — gated GRPO (binary, clean contrast)
```
RUN=grpo_airline_r5 TRAIN_TASK_IDS=2,3,8,13,19 N=12 ITERS=3 \
  REWARD_MODE=binary LATA=1 LR=5e-6 CHAIN=0 TEMP=1.0 AGENT_MAX_TOKENS=768 MAX_CONCURRENCY=8 \
  bash scripts/grpo/run_grpo_min.sh
# then eval each adapter at the SAME decoding as base:
for it in 1 2 3; do NUM_TRIALS=5 TEMP=0.7 SAVE_TAG=r5_iter$it \
  ADAPTER_PATH=outputs/grpo_airline_r5/adapter_iter$it \
  TASKS=0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46 bash scripts/grpo/adapter_eval_retail.sh
  python scripts/grpo/summarize_eval.py --eval outputs/adapter_airline_r5_iter$it.jsonl --base outputs/base_airline_eval.jsonl; done
```
`gate` is on by default → only mixed-outcome groups (2,3,13,19) give gradient; CHAIN=0 branches each iter from base (no drift). prm_lite is now honest but mildly length-biased → prefer `REWARD_MODE=binary` for the headline; use prm_lite only as an ablation.

## DO NOT
- trust 1-trial / overall-20 pass^1 (use live-10 + multi-trial + McNemar).
- raise max_steps (40 never hit), add a naive write-bonus (fires 74% in FAILED rollouts), or add a terseness rule (base wins are MORE verbose).
- expect the max_tokens fix to lift *eval* (eval was already truncation-free) — its value is the *training* signal.

CRUX caveat: train ids and held-out ids have ZERO overlap → every gain is cross-task transfer (unproven).
