#!/usr/bin/env bash
set -euo pipefail
# Base policy pass^1 baseline on a tau2 domain (start vLLM base -> collect 1 trial/task).
# Portable: WORKDIR, PYBIN, POLICY_MODEL, DOMAIN, TASKS, USER_LLM via env.
WORKDIR="${WORKDIR:-/root/yuyu}"; PYBIN="${PYBIN:-/usr/bin/python3}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-/usr/bin}:/root/.local/bin:/usr/local/bin:${PATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
set -a; [ -f .env ] && . ./.env; set +a
MODEL="${POLICY_MODEL:-/root/yuyu/models/qwen25-7b-instruct}"
DOMAIN="${DOMAIN:-retail}"
TASKS="${TASKS:-$($PYBIN -c 'print(",".join(str(i) for i in range(30)))')}"
USER_LLM="${USER_LLM:-deepseek/deepseek-chat}"
NUM_TRIALS="${NUM_TRIALS:-1}"; TEMP="${TEMP:-0.0}"; MAX_TOKENS="${MAX_TOKENS:-768}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
USER_TEMP="${USER_TEMP:-0}"  # pin user-sim to greedy: the dominant eval-noise source (R5)
export OPENAI_API_BASE=http://127.0.0.1:8000/v1 OPENAI_API_KEY=dummy
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}" TAU2_CMD="${TAU2_CMD:-tau2}"
mkdir -p outputs/vllm_logs
pkill -9 -f vllm.entrypoints 2>/dev/null || true   # kill stale api_server...
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null || true; done  # ...AND any orphaned EngineCore holding the GPU
sleep 3
echo "[base_eval] start vLLM base ($MODEL)"
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
$PYBIN -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name policy \
  --host 127.0.0.1 --port 8000 --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.9 \
  --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager \
  --no-enable-flashinfer-autotune \
  > outputs/vllm_logs/base_eval.log 2>&1 &
VP=$!
trap 'kill "$VP" 2>/dev/null || true; wait "$VP" 2>/dev/null || true' EXIT   # never orphan vLLM, even on set -e abort
for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q policy && { echo "[base_eval] vLLM ready"; break; }; sleep 5; done
echo "[base_eval] collect 1-trial on $DOMAIN tasks: $TASKS"
$PYBIN scripts/grpo/collect_rollouts.py --domain "$DOMAIN" --served-model policy --user-llm "$USER_LLM" \
  --task-ids "$TASKS" --num-trials "$NUM_TRIALS" --temperature "$TEMP" --max-tokens "$MAX_TOKENS" --max-steps 40 \
  --user-temperature "$USER_TEMP" --no-auto-resume \
  --max-concurrency "$MAX_CONCURRENCY" --save-to "base_${DOMAIN}_eval" --out "outputs/base_${DOMAIN}_eval.jsonl"
kill "$VP" 2>/dev/null || true; wait "$VP" 2>/dev/null || true
echo "[base_eval] the '[collect] reward: success X/N' line above == base pass^1 on $DOMAIN"
echo BASE_EVAL_DONE
