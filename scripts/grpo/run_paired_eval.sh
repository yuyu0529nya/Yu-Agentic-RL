#!/usr/bin/env bash
set -euo pipefail
# Variance-controlled PAIRED eval — tames the user-simulator noise that dominated R4/R5.
# Pins the user-sim temperature (USER_TEMP, default 0) AND runs N trials/task, then compares
# an adapter vs base with a PAIRED per-task rate test (bootstrap CI + sign test).
# Usage: ADAPTER_PATH=outputs/<run>/adapter TAG=mytag bash scripts/grpo/run_paired_eval.sh
#   AutoDL: WORKDIR=/root/autodl-tmp/yuyu PYBIN=/root/miniconda3/bin/python EXTRA_PATH=/root/miniconda3/bin
WORKDIR="${WORKDIR:-/root/autodl-tmp/yuyu}"; PYBIN="${PYBIN:-/root/miniconda3/bin/python}"
cd "$WORKDIR"
export EXTRA_PATH="${EXTRA_PATH:-/root/miniconda3/bin}"
export POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/qwen25-7b-instruct}"
export DOMAIN="${DOMAIN:-airline}" USER_LLM="${USER_LLM:-deepseek/deepseek-chat}"
export NUM_TRIALS="${NUM_TRIALS:-8}" TEMP="${TEMP:-0.0}" USER_TEMP="${USER_TEMP:-0.0}"
export MAX_TOKENS="${MAX_TOKENS:-768}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
export TASKS="${TASKS:-0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46}"
ADAPTER_PATH="${ADAPTER_PATH:?need ADAPTER_PATH (adapter to test vs base)}"; export ADAPTER_PATH
TAG="${TAG:-paired}"

echo "######## BASE eval (user-temp=$USER_TEMP, N=$NUM_TRIALS, agent temp=$TEMP) $(date +%H:%M:%S) ########"
bash scripts/grpo/base_eval_retail.sh
echo "######## ADAPTER eval ($ADAPTER_PATH) $(date +%H:%M:%S) ########"
SAVE_TAG="$TAG" bash scripts/grpo/adapter_eval_retail.sh
echo "######## PAIRED summary ########"
"$PYBIN" scripts/grpo/summarize_eval.py --eval "outputs/adapter_${DOMAIN}_${TAG}.jsonl" --base "outputs/base_${DOMAIN}_eval.jsonl"
echo "ALLDONE_PAIRED $(date +%H:%M:%S)"
