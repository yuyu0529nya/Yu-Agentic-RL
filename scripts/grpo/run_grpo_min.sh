#!/usr/bin/env bash
set -euo pipefail
# Minimal single-GPU GRPO driver for tau2 (alternate rollout <-> train).
# Portable across machines via env: WORKDIR, PYBIN, POLICY_MODEL, EXTRA_PATH.
#   AutoDL:  WORKDIR=/root/autodl-tmp/yuyu PYBIN=/root/miniconda3/bin/python
#   106.75:  WORKDIR=/root/yuyu PYBIN=/usr/bin/python3
# Memory trick: vLLM is UP only during rollout, then STOPPED before the QLoRA
# update, so a 7B policy fits one 24-32GB GPU. User-sim = deepseek (fast, no rate limit).

WORKDIR="${WORKDIR:-/root/autodl-tmp/yuyu}"
PYBIN="${PYBIN:-python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-/root/miniconda3/bin}:/root/.local/bin:/usr/local/bin:${PATH}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache}"
# bitsandbytes needs libnvJitLink.so.13 from nvidia/*/lib (works for venv/conda/system)
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
set -a; for f in .env .remote_eval_env .tau2_secret_env; do [ -f "$f" ] && . "./$f"; done; set +a

RUN="${RUN:-grpo_retail_r1_$(date +%Y%m%d_%H%M%S)}"
DOMAIN="${DOMAIN:-retail}"
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/qwen25-7b-instruct}"
TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-0,1,2,3,4,5,6,7}"
N="${N:-6}"
ITERS="${ITERS:-2}"
LR="${LR:-1e-5}"
REWARD_MODE="${REWARD_MODE:-prm_lite}"
LATA="${LATA:-0}"
CHAIN="${CHAIN:-1}"   # 1 = chain adapter iter->iter (default); 0 = branch each iter from BASE (no cross-iter drift)
TEMP="${TEMP:-1.0}"
MAX_STEPS="${MAX_STEPS:-40}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"  # 32768: 16384 overflowed long airline tasks -> truncated rollouts / spurious 0
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT="${PORT:-8000}"; HOST="${HOST:-127.0.0.1}"
export OPENAI_API_BASE="http://${HOST}:${PORT}/v1" OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export USER_LLM="${USER_LLM:-deepseek/deepseek-chat}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
export TAU2_CMD="${TAU2_CMD:-tau2}"

OUT="outputs/${RUN}"; mkdir -p "$OUT" outputs/vllm_logs
CUR_ADAPTER=""
VLLM_PID=""

start_vllm() {
  local adapter="$1" log="$2"; stop_vllm || true
  # agent always calls model "policy": base served as "policy", or LoRA named "policy" (distinct base name to avoid conflict)
  local name_args=(--served-model-name policy)
  [ -n "$adapter" ] && name_args=(--served-model-name basemodel --enable-lora --lora-modules "policy=${adapter}")
  echo "[driver] start vLLM (adapter='${adapter:-<base>}')"
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 $PYBIN -m vllm.entrypoints.openai.api_server \
    --model "$POLICY_MODEL" \
    --host "$HOST" --port "$PORT" --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 4 \
    --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    "${name_args[@]}" > "$log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 120); do
    if $PYBIN - <<PY 2>/dev/null
import json,urllib.request,sys
d=json.loads(urllib.request.urlopen("${OPENAI_API_BASE}/models",timeout=5).read())
sys.exit(0 if any(m.get("id")=="policy" for m in d.get("data",[])) else 1)
PY
    then echo "[driver] vLLM ready"; return 0; fi
    sleep 5
  done
  echo "[driver] vLLM failed; see $log" >&2; tail -n 30 "$log" >&2; exit 4
}
stop_vllm() {
  [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
  [ -n "$VLLM_PID" ] && wait "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""; sleep 3
}
trap stop_vllm EXIT

for ((it=1; it<=ITERS; it++)); do
  echo "==================== GRPO ITER $it / $ITERS ===================="
  ROLL="$OUT/rollouts_iter${it}.jsonl"; SAVE="grpo_${RUN}_iter${it}"; NEW_ADAPTER="$OUT/adapter_iter${it}"
  start_vllm "$CUR_ADAPTER" "outputs/vllm_logs/${SAVE}.log"
  $PYBIN scripts/grpo/collect_rollouts.py \
    --domain "$DOMAIN" --served-model policy --user-llm "$USER_LLM" --task-ids "$TRAIN_TASK_IDS" \
    --num-trials "$N" --temperature "$TEMP" --max-tokens "$AGENT_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" --max-concurrency "$MAX_CONCURRENCY" --seed "$it" --save-to "$SAVE" --out "$ROLL"
  stop_vllm
  lata_flag=(); [ "$LATA" = "1" ] && lata_flag=(--lata)
  in_flag=(); [ -n "$CUR_ADAPTER" ] && [ "$CHAIN" = "1" ] && in_flag=(--adapter-in "$CUR_ADAPTER")
  $PYBIN scripts/grpo/grpo_update.py \
    --rollouts "$ROLL" --base-model "$POLICY_MODEL" --out-adapter "$NEW_ADAPTER" \
    --domain "$DOMAIN" --reward-mode "$REWARD_MODE" --lr "$LR" "${lata_flag[@]}" "${in_flag[@]}"
  CUR_ADAPTER="$NEW_ADAPTER"; echo "[driver] iter $it done -> $CUR_ADAPTER"
done
echo "[driver] GRPO finished. Final adapter: $CUR_ADAPTER"
