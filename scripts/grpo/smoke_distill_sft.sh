#!/usr/bin/env bash
set -uo pipefail
# 5-minute GPU smoke for tonight's distill SFT: train 1 epoch on the 5 LONGEST teacher
# trajectories (incl. the 19218-token max) at --max-seq-len 20480 — validates VRAM and the
# RFT path on long teacher data BEFORE committing to the full run. Mirrors
# run_distill_sft_company.sh env exactly.
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; PYBIN="${PYBIN:-$WORKDIR/venv/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-$WORKDIR/venv/bin}:$HOME/.local/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKDIR/.cache}" TMPDIR="${TMPDIR:-$WORKDIR/tmp}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORKDIR/.cache/vllm}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKDIR/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKDIR/.cache/torchinductor}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" outputs/distill_smoke

POLICY_MODEL="${POLICY_MODEL:-$WORKDIR/models/qwen25-7b-instruct}"
echo "[smoke] SFT 1 epoch on 5 longest teacher trajectories, max_seq_len=20480, GPU${POLICY_GPU:-1}"
CUDA_VISIBLE_DEVICES="${POLICY_GPU:-1}" $PYBIN scripts/grpo/grpo_update.py --rft \
  --rollouts outputs/teacher_ds_0702/smoke5_longest.jsonl \
  --base-model "$POLICY_MODEL" --out-adapter outputs/distill_smoke/adapter \
  --domain airline --reward-mode binary --epochs 1 --lr 1e-5 --max-seq-len 20480 \
  --progress-every 1
rc=$?
echo "[smoke] exit=$rc peak VRAM:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
[ $rc -eq 0 ] && echo SMOKE_DISTILL_OK || echo SMOKE_DISTILL_FAILED
