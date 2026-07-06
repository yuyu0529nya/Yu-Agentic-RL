#!/bin/bash
# Round-4 (company 2x5090, Sunday): DATA-PARALLEL multi-experiment scheduling — both cards
# run a full experiment each, concurrently (~100% utilization vs round-3's serial ~50%).
# Goal: stabilize the best config (LATA+process reward, which peaked 0.503 but dipped at iter3)
# and a small grid, by trying 4 variants. NO auto-shutdown (company box).
set -u
WORKDIR=$HOME/agentic-rl; PYBIN=$WORKDIR/venv/bin/python
cd "$WORKDIR" || { echo "set WORKDIR"; exit 9; }
mkdir -p outputs
exec >> outputs/PARALLEL_R4.log 2>&1
echo "######## PARALLEL_R4 START $(date) ########"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
CMN="POLICY_MODEL=$WORKDIR/models/qwen25-7b-instruct WORKDIR=$WORKDIR PYBIN=$PYBIN \
EXTRA_PATH=$WORKDIR/venv/bin HF_HOME=$WORKDIR/hf-cache HF_ENDPOINT=https://hf-mirror.com \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True REWARD=f1 LATA=1 N=8 LR=5e-6 ITERS=6 \
N_TRAIN_Q=128 N_EVAL_Q=300 GRPO_SEQ=2560 BATCH=2 PROG=10 EVAL_TRIALS=1 GPU_UTIL=0.78"

# All 4 variants branch off the best config (LATA + process reward, beta 0.3). Two cards in parallel:
# GPU0 :18001 runs kl01 then lr3 ; GPU1 :18000 runs b015 then n12.
( env $CMN CUDA_VISIBLE_DEVICES=0 PORT=18001 RUN=p_kl01 PROC_BETA=0.3 KL_COEF=0.01 bash scripts/grpo/run_one_card.sh
  env $CMN CUDA_VISIBLE_DEVICES=0 PORT=18001 RUN=p_lr3  PROC_BETA=0.3 LR=3e-6     bash scripts/grpo/run_one_card.sh
) > outputs/g0_r4.log 2>&1 &
G0=$!
( env $CMN CUDA_VISIBLE_DEVICES=1 PORT=18000 RUN=p_b015 PROC_BETA=0.15           bash scripts/grpo/run_one_card.sh
  env $CMN CUDA_VISIBLE_DEVICES=1 PORT=18000 RUN=p_n12  PROC_BETA=0.3 N=12        bash scripts/grpo/run_one_card.sh
) > outputs/g1_r4.log 2>&1 &
G1=$!
wait $G0; wait $G1
echo "######## PARALLEL_R4 ALLDONE $(date) — 4 variants done (p_kl01,p_lr3,p_b015,p_n12) ########"
