#!/bin/bash
# Round-4 SERIAL stabilization (company 2x5090): 4 variants around the best config
# (LATA + process reward, which peaked 0.503 but dipped at iter3). Uses the round-3-PROVEN
# dual-decoupled script (vLLM resident GPU1 + train GPU0 — no serve/stop churn, no OOM trap).
# Serial = one variant at a time (double-GPU ~50% util, but stable). NO auto-shutdown.
set -u
WORKDIR=$HOME/agentic-rl; PYBIN=$WORKDIR/venv/bin/python
cd "$WORKDIR" || { echo "set WORKDIR"; exit 9; }
mkdir -p outputs
exec >> outputs/MASTER_R4S.log 2>&1
echo "######## MASTER_R4S START $(date) ########"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
CMN="POLICY_MODEL=$WORKDIR/models/qwen25-7b-instruct WORKDIR=$WORKDIR PYBIN=$PYBIN \
EXTRA_PATH=$WORKDIR/venv/bin HF_HOME=$WORKDIR/hf-cache HF_ENDPOINT=https://hf-mirror.com \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True REWARD=f1 LATA=1 N=8 LR=5e-6 ITERS=6 \
N_TRAIN_Q=128 N_EVAL_Q=300 GRPO_SEQ=2560 BATCH=2 PROG=10 EVAL_TRIALS=1 PORT=18000 VLLM_GPU=1 TRAIN_GPU=0"

run_dual() {  # $1=RUN rest=KEY=VAL
  local R=$1; shift
  echo "==== DUAL $R $(date) ===="
  env $CMN RUN=$R "$@" bash scripts/grpo/run_search_agent_dual.sh || echo "==== $R FAILED $(date) ===="
}
# 4 variants to stabilize LATA+process-reward (baseline lata_proc: peak 0.503, iter3 dip to 0.333):
run_dual v_kl01  PROC_BETA=0.3  KL_COEF=0.01     # + light KL anchor (kill the dip?)
run_dual v_lr3   PROC_BETA=0.3  LR=3e-6          # gentler LR (smoother?)
run_dual v_b015  PROC_BETA=0.15                  # weaker process weight (less over-push?)
run_dual v_n12   PROC_BETA=0.3  N=12             # bigger group (steadier gradient?)
echo "######## MASTER_R4S ALLDONE $(date) ########"
