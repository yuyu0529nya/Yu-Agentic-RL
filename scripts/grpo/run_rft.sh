#!/usr/bin/env bash
set -euo pipefail
# RFT / STaR path (R4-improvement plan rank-3): the cheapest capability shot to beat base.
# Build a success-only dataset from EXISTING rollouts -> plain SFT from BASE -> multi-trial eval.
# No new rollout collection needed. Portable via env (AutoDL defaults).
#   AutoDL: WORKDIR=/root/autodl-tmp/yuyu PYBIN=/root/miniconda3/bin/python EXTRA_PATH=/root/miniconda3/bin
WORKDIR="${WORKDIR:-/root/autodl-tmp/yuyu}"; PYBIN="${PYBIN:-/root/miniconda3/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-/root/miniconda3/bin}:/usr/local/bin:${PATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
set -a; for f in .env .remote_eval_env .tau2_secret_env; do [ -f "$f" ] && . "./$f"; done; set +a

POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/qwen25-7b-instruct}"
DOMAIN="${DOMAIN:-airline}"
ROLLOUT_GLOB="${ROLLOUT_GLOB:-outputs/grpo_airline_r4_20260624_135504/rollouts_iter*.jsonl}"
RUN="${RUN:-rft_airline}"
CAP="${CAP:-8}"; EPOCHS="${EPOCHS:-3}"; LR="${LR:-1e-5}"
EVAL_TASKS="${EVAL_TASKS:-0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46}"
export NUM_TRIALS="${NUM_TRIALS:-5}" TEMP="${TEMP:-0.7}" MAX_TOKENS="${MAX_TOKENS:-768}"
OUT="outputs/$RUN"; mkdir -p "$OUT"

echo "[rft] 1/3 build success-only dataset"
$PYBIN scripts/grpo/build_rft_dataset.py --rollouts $ROLLOUT_GLOB --out "$OUT/rft_dataset.jsonl" --cap-per-task "$CAP"

echo "[rft] 2/3 plain SFT from BASE (grpo_update --rft: advantage=+1, no LATA, no group baseline)"
$PYBIN scripts/grpo/grpo_update.py --rft --rollouts "$OUT/rft_dataset.jsonl" \
  --base-model "$POLICY_MODEL" --out-adapter "$OUT/adapter" --domain "$DOMAIN" \
  --reward-mode binary --epochs "$EPOCHS" --lr "$LR"

echo "[rft] 3/3 multi-trial eval + paired summary vs base"
EXTRA_PATH="${EXTRA_PATH:-/root/miniconda3/bin}" WORKDIR="$WORKDIR" PYBIN="$PYBIN" POLICY_MODEL="$POLICY_MODEL" \
  DOMAIN="$DOMAIN" TASKS="$EVAL_TASKS" SAVE_TAG="rft" ADAPTER_PATH="$OUT/adapter" \
  bash scripts/grpo/adapter_eval_retail.sh
$PYBIN scripts/grpo/summarize_eval.py --eval "outputs/adapter_${DOMAIN}_rft.jsonl" \
  --base "outputs/base_${DOMAIN}_eval.jsonl" || echo "[rft] (run base multi-trial eval first for the paired test)"
echo "ALLDONE_RFT RUN=$RUN"
