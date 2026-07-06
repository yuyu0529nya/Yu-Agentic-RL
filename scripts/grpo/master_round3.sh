#!/bin/bash
# Round-3 weekend matrix on the company 2x5090 (Sunday: teammate won't preempt -> safe to borrow GPU0).
# Dual-GPU: vLLM resident on GPU1, training borrows GPU0. Single-trial per-iter eval for fast curves;
# a multi-trial nail-down of the best checkpoint is a separate follow-up step. NO auto-shutdown
# (company box). Cleanup is done by the user at night via ~/yuyu/cleanup.sh (PURGE=1).
#
# Defaults below are tuned for the company box; override via env if needed.
set -u
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; PYBIN="${PYBIN:-$WORKDIR/venv/bin/python}"
cd "$WORKDIR" || { echo "ERROR: set WORKDIR to the repo root (has scripts/grpo/)"; exit 9; }
mkdir -p outputs
exec > >(tee -a outputs/MASTER_R3.log) 2>&1
echo "######## MASTER_R3 START $(date) ########"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMN="N=8 LR=5e-6 GRPO_SEQ=2560 PROG=10 N_TRAIN_Q=128 N_EVAL_Q=300 MAX_MODEL_LEN=4096 \
PORT=${PORT:-18000} VLLM_GPU=${VLLM_GPU:-1} TRAIN_GPU=${TRAIN_GPU:-0} EVAL_TRIALS=${EVAL_TRIALS:-1} \
WORKDIR=$WORKDIR PYBIN=$PYBIN POLICY_MODEL=${POLICY_MODEL:?set POLICY_MODEL to the 7B-Instruct path} \
HF_HOME=${HF_HOME:-$WORKDIR/hf-cache} HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
EXTRA_PATH=${EXTRA_PATH:-$WORKDIR/venv/bin} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

run_dual() {  # $1=RUN  rest=KEY=VAL ; BATCH=2 (borrowing GPU0, F1 rollouts are long) with a BATCH=1 fallback
  local R=$1; shift
  echo "==== DUAL $R $(date) (BATCH=2) ===="
  env $CMN RUN=$R BATCH=2 "$@" bash scripts/grpo/run_search_agent_dual.sh \
    || { echo "==== $R BATCH=2 failed -> retry BATCH=1 ===="; \
         env $CMN RUN=$R BATCH=1 "$@" bash scripts/grpo/run_search_agent_dual.sh \
         || echo "==== $R FAILED both batch sizes ===="; }
}

# NOTE: dual_f1lata (the length-aware run) already ran separately. Matrix continues with the rest:
# B) length-aware x dense process reward — does combining the two winning levers beat either alone?
run_dual dual_lata_proc  REWARD=f1 LATA=1 PROC_BETA=0.3 ITERS=6
# C) length-aware + a gentle KL anchor
run_dual dual_lata_kl    REWARD=f1 LATA=1 KL_COEF=0.02  ITERS=6
# D/E) KL sweep — can a gentler anchor stay stable WITHOUT capping the peak (vs 0.05 -> 0.437)?
run_dual dual_kl02       REWARD=f1 KL_COEF=0.02         ITERS=8
run_dual dual_kl01       REWARD=f1 KL_COEF=0.01         ITERS=8
echo "######## MASTER_R3 ALLDONE $(date) — no auto-shutdown (company box); user PURGE-cleans at night ########"
