#!/usr/bin/env bash
# ==============================================================================
# FLAGSHIP on 1x H800 80GB — SFT'd-7B → veRL GRPO, tau2 airline.
# Single 80GB card hosts BOTH the local 7B user simulator AND the 7B policy
# (GRPO + colocated vLLM rollout). This drops every 2x32GB compromise:
#   - NO tensor parallel (single card, no TP=2 hang, no NCCL hack)
#   - NO OpenRouter (local usersim, free, self-hosted, low latency)
#   - generous vLLM util + bigger batch (room to spare)
#   - bf16 LoRA on the SFT-warmed 7B (no quantization needed)
#
# Memory on the one card (~80GB):
#   usersim 7B (util 0.22 ≈ 18GB) + policy FSDP 14GB + policy vLLM (util 0.5 ≈
#   40GB) ≈ 72GB. Fits with headroom.
#
# Usage (open GPU first):  bash run_h800_sft_grpo.sh
# ==============================================================================
set -euo pipefail
ROOT=/root/autodl-tmp/verl-work
INTEG=$ROOT/tau2_integration
cd "$INTEG"

echo "==== [1/3] launching local 7B usersim (util 0.22, ~18GB of the 80GB card) ===="
USERSIM_UTIL=${USERSIM_UTIL:-0.22} USERSIM_GPU=0 \
  nohup bash serve_usersim_7b.sh > "$ROOT/usersim_h800.log" 2>&1 &
echo "usersim pid=$!"

echo "==== [2/3] waiting for usersim to be ready ===="
for i in $(seq 1 48); do
  if curl -sf http://127.0.0.1:18001/v1/models >/dev/null 2>&1; then
    echo "usersim READY (${i}0s)"; break
  fi
  sleep 10
done
curl -sf http://127.0.0.1:18001/v1/models >/dev/null 2>&1 || { echo "[ABORT] usersim never came up; see usersim_h800.log"; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "==== [3/3] launching SFT'd-7B veRL GRPO on the same card ===="
GPU=0 \
POLICY_MODEL=${POLICY_MODEL:-$ROOT/models/qwen25-7b-sft-airline} \
TRAIN_BS=${TRAIN_BS:-16} ROLLOUT_N=${ROLLOUT_N:-8} PPO_MINI=${PPO_MINI:-8} \
GPU_UTIL=${GPU_UTIL:-0.5} MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240} MAX_RESP=${MAX_RESP:-3072} \
LR=${LR:-2e-6} EPOCHS=${EPOCHS:-20} SAVE_FREQ=${SAVE_FREQ:--1} TEST_FREQ=${TEST_FREQ:-5} \
RESUME_MODE=disable PARAM_OFFLOAD=False PRECHECK_MAX=30000 \
EXP_NAME=${EXP_NAME:-tau2_airline_7b_sft_grpo} \
  bash run_tau2_grpo_7b.sh
