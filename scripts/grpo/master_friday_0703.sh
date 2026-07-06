#!/usr/bin/env bash
set -uo pipefail
# FRIDAY-NIGHT MAX PIPELINE (07-03): squeeze everything into one unattended run.
#   A. continue SFT e9 -> e15 (+6 epochs) + same-night eval            (~1.7h)
#   B. auto-pick warm start: e15 only if held-out >= e9 + 2 successes  (tie -> e9, earlier stop)
#   C. GRPO-from-SFT FULL 2x2 (vanilla, prmlata, prmonly, lataonly), N=6, ITERS=3,
#      init-anchored curve. Ordered so the two headline arms land first (~02:30);
#      prmonly/lataonly are the pre-dawn bonus. Each arm's evals save incrementally.
# GPU box: NO auto-shutdown; 4-way trap frees GPUs at exit. All child scripts flock-guarded.
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; cd "$WORKDIR"
OUT="outputs/friday_0703"; mkdir -p "$OUT" logs
exec 9>"$OUT/.run.lock"
flock -n 9 || { echo "[fri] another friday pipeline is running — refusing (replay guard)"; exit 9; }
exec > >(tee -a "$OUT/master.log") 2>&1

E9_ADP="outputs/distill_sft_0702_ext/distill_adapter_ext"
E9_PASS=38   # verified 07-03: 38/100 held-out
PY="$WORKDIR/venv/bin/python"

echo "######## FRIDAY PIPELINE START $(date) ########"

echo "======== STEP A: continue SFT e9 -> e15 $(date +%H:%M) ========"
RUN=distill_sft_0703_e15 ADAPTER_IN="$E9_ADP" BASE_DIR=outputs/distill_sft_0702 LABEL=distill_e15 \
  bash scripts/grpo/continue_distill_sft.sh || echo "[fri] e15 step FAILED — falling back to e9"

BEST="$E9_ADP"; BEST_NAME=e9
E15_EVAL="outputs/distill_sft_0703_e15/distill_ext_eval.jsonl"
E15_ADP="outputs/distill_sft_0703_e15/distill_adapter_ext"
if [ -s "$E15_EVAL" ] && [ -d "$E15_ADP" ]; then
  E15_PASS=$($PY -c "import json;rs=[json.loads(l)['reward'] for l in open('$E15_EVAL')];print(sum(1 for r in rs if r>=1-1e-6))")
  echo "[fri] e15 held-out successes: $E15_PASS/100 (e9 was $E9_PASS/100)"
  if [ "$E15_PASS" -ge "$((E9_PASS + 2))" ]; then BEST="$E15_ADP"; BEST_NAME=e15; fi
else
  echo "[fri] e15 eval/adapter missing — using e9"
fi
echo "======== STEP B verdict: GRPO warm start = $BEST_NAME ($BEST) $(date +%H:%M) ========"

echo "======== STEP C: GRPO-from-SFT full 2x2 (headline arms first) $(date +%H:%M) ========"
RUN=grpo_from_sft_0703 METHODS="vanilla prmlata prmonly lataonly" \
  INIT_ADAPTER="$BEST" N=6 ITERS=3 \
  bash scripts/grpo/run_grpo_ablation_4way.sh || echo "[fri] GRPO step ended nonzero"

echo "FRIDAY_ALLDONE $(date) (warm start was $BEST_NAME)"
