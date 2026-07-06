#!/usr/bin/env bash
set -uo pipefail
# SATURDAY-NIGHT PIPELINE (07-04): does the bigger/stronger teacher set (mega-212) break the
# 114-example distill ceiling (0.38), and does GRPO climb even higher from the new floor?
#   STEP 1: SFT on mega-212 (EPOCHS=8) + same-session eval (held-out + train-fit + paired)
#   STEP 2: GRPO-from-mega-SFT 2x2 headline arms (vanilla, prmlata), init-anchored curve
# EXPORT the box knobs so children inherit them (the WORKDIR/TAU2_CMD bugs on 07-04 were exactly
# this — master ran on /data but children defaulted to stale /home + `uv run tau2`).
export WORKDIR=$HOME/agentic-rl
export PYBIN=$WORKDIR/venv/bin/python
export EXTRA_PATH=$WORKDIR/venv/bin
export POLICY_MODEL=$WORKDIR/models/qwen25-7b-instruct
export TAU2_CMD=tau2
export POLICY_GPU=0 USER_GPU=1 GPU_UTIL=0.70 MAXC=3
cd "$WORKDIR"
OUT="outputs/saturday_0704"; mkdir -p "$OUT" logs
exec 9>"$OUT/.run.lock"
flock -n 9 || { echo "[sat] another saturday pipeline is running — refusing"; exit 9; }
exec > >(tee -a "$OUT/master.log") 2>&1

MEGA="outputs/teacher_mega_0704/distill_dataset_final.jsonl"
echo "######## SATURDAY PIPELINE START $(date) mega=$(wc -l < $MEGA) examples ########"
[ -s "$MEGA" ] || { echo "[sat] mega dataset missing"; exit 2; }

echo "======== STEP 1: SFT on mega-212 (EPOCHS=8) $(date +%H:%M) ========"
DATASET="$MEGA" RUN=distill_mega_0704 EPOCHS="${MEGA_EPOCHS:-8}" LR=1e-5 \
  bash scripts/grpo/run_distill_sft_company.sh || echo "[sat] mega SFT step ended nonzero"

MEGA_ADP="outputs/distill_mega_0704/distill_adapter"
if [ ! -d "$MEGA_ADP" ]; then echo "[sat] mega adapter missing — abort GRPO"; echo "SATURDAY_ALLDONE $(date)"; exit 2; fi
MEGA_PASS=$($PYBIN -c "import json;rs=[json.loads(l)['reward'] for l in open('outputs/distill_mega_0704/distill_eval.jsonl')];print(sum(1 for r in rs if r>=1-1e-6),'/',len(rs))" 2>/dev/null || echo "?")
echo "======== STEP 1 done: mega-SFT held-out = $MEGA_PASS $(date +%H:%M) ========"

echo "======== STEP 2: GRPO-from-mega-SFT (vanilla + prmlata) $(date +%H:%M) ========"
RUN=grpo_from_mega_0704 METHODS="vanilla prmlata" INIT_ADAPTER="$MEGA_ADP" N=6 ITERS=3 \
  bash scripts/grpo/run_grpo_ablation_4way.sh || echo "[sat] GRPO-from-mega step ended nonzero"

echo "SATURDAY_ALLDONE $(date)"
