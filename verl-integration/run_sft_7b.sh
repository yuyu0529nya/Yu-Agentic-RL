#!/usr/bin/env bash
# ==============================================================================
# Stage 1/2: Distillation SFT (QLoRA 4-bit) of Qwen2.5-7B on tau2 airline
# teacher trajectories — the warm-up the resume's Baseline→SFT→GRPO needs and
# that last night's raw-3B GRPO was missing.
#
# Reuses the hand-written trainer grpo_update.py in --rft mode (rejection FT:
# SFT on reward=1 teacher trajectories, assistant-token-masked NLL). It loads
# the 7B base in 4-bit nf4 (BitsAndBytes) → ~4GB, so 7B fits one 5090.
#
# Single GPU, offline (no usersim / no rollout). Output = a LoRA adapter to
# seed the veRL GRPO stage.
#
# Usage (open GPU first):  bash run_sft_7b.sh
# ==============================================================================
set -xeuo pipefail

ROOT=/root/autodl-tmp/verl-work
VENV=/root/autodl-tmp/venv-verl
SFT=$ROOT/sft_assets

export TMPDIR=$ROOT/tmp
export HF_HOME=$ROOT/.hfhome
export HF_ENDPOINT=https://hf-mirror.com
export TRITON_CACHE_DIR=$ROOT/.triton
export XDG_CACHE_HOME=$ROOT/.xdgcache
mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES=${GPU:-0}

# precheck: chosen GPU mostly free
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES} | tr -d " ")
echo "[precheck] GPU ${CUDA_VISIBLE_DEVICES} used ${used} MiB"
if [ "${used}" -gt 4000 ]; then echo "[ABORT] GPU busy (${used}MiB)."; exit 1; fi

source "$VENV/bin/activate"
cd "$SFT"

POLICY_MODEL=${POLICY_MODEL:-$ROOT/models/qwen25-7b-instruct}
DATASET=${DATASET:-$SFT/distill_mega_212.jsonl}      # 212 teacher trajectories (reward=1)
OUT_ADAPTER=${OUT_ADAPTER:-$ROOT/ckpts/sft_airline_7b_adapter}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-5}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-20480}                    # airline traj up to ~19k tokens
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-32}

mkdir -p "$(dirname "$OUT_ADAPTER")"

echo "######## SFT (QLoRA --rft) START $(date) ########"
echo "  base=$POLICY_MODEL  data=$DATASET  epochs=$EPOCHS lr=$LR max_seq=$MAX_SEQ_LEN"

python3 grpo_update.py --rft \
    --rollouts "$DATASET" \
    --base-model "$POLICY_MODEL" \
    --out-adapter "$OUT_ADAPTER" \
    --domain airline \
    --reward-mode binary \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    2>&1 | tee "$ROOT/sft_airline_7b.log"

echo "######## SFT DONE $(date)  adapter -> $OUT_ADAPTER ########"
ls -la "$OUT_ADAPTER" 2>/dev/null
