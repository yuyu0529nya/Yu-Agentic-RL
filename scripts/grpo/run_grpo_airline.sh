#!/usr/bin/env bash
set -uo pipefail
# Real airline GRPO with an INDEPENDENT usersim.
# GPU0: base 7B usersim (resident, served-model=usersim). GPU1: policy (per-iter rollout<->train,
# vLLM up during collect, stopped before QLoRA update to free the card). prm_lite + LATA.
# base eval + per-iter held-out eval. NO auto-shutdown.
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; PYBIN="${PYBIN:-$WORKDIR/venv/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-$WORKDIR/venv/bin}:$HOME/.local/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
export TAU2_CMD="${TAU2_CMD:-uv run tau2}" OPENAI_API_KEY=dummy

RUN="${RUN:-grpo_airline_real}"
POLICY_MODEL="${POLICY_MODEL:-$WORKDIR/models/qwen25-7b-instruct}"
DOMAIN=airline
TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-2,3,4,7,8,9,12,13}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0,1,5,6,10,11}"
N="${N:-6}"; ITERS="${ITERS:-3}"; LR="${LR:-1e-5}"
REWARD_MODE="${REWARD_MODE:-prm_lite}"; LATA="${LATA:-1}"
TEMP="${TEMP:-1.0}"; MAX_STEPS="${MAX_STEPS:-40}"; AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"; GPU_UTIL="${GPU_UTIL:-0.80}"  # 32768: 16384 overflowed long tasks -> spurious 0
USER_TEMP="${USER_TEMP:-0}"; EVAL_SEED="${EVAL_SEED:-900}"  # pin user-sim (dominant noise) + fixed eval seed (fair paired compare)
POLICY_GPU="${POLICY_GPU:-1}"; USER_GPU="${USER_GPU:-0}"
PPORT="${PPORT:-18000}"; UPORT="${UPORT:-18001}"; HOST=127.0.0.1
OUT="outputs/$RUN"; mkdir -p "$OUT" outputs/vllm_logs
exec >> "$OUT/master.log" 2>&1
UVP=""; PVP=""

serve_user() {
  echo "[real] serve usersim vLLM GPU$USER_GPU :$UPORT"
  CUDA_VISIBLE_DEVICES=$USER_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" --served-model-name usersim \
    --host $HOST --port $UPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    > outputs/vllm_logs/${RUN}_usersim.log 2>&1 &
  UVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && { echo "[real] usersim ready"; return 0; }; sleep 5; done
  echo "[real] usersim FAILED"; tail -25 outputs/vllm_logs/${RUN}_usersim.log; exit 4
}
serve_policy() {
  local adapter="$1"; stop_policy
  local nm=(--served-model-name policy); [ -n "$adapter" ] && nm=(--served-model-name basemodel --enable-lora --lora-modules "policy=$adapter")
  echo "[real] serve policy vLLM GPU$POLICY_GPU :$PPORT (adapter='${adapter:-<base>}')"
  CUDA_VISIBLE_DEVICES=$POLICY_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" \
    --host $HOST --port $PPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 4 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    "${nm[@]}" > outputs/vllm_logs/${RUN}_policy.log 2>&1 &
  PVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy && { echo "[real] policy ready"; return 0; }; sleep 5; done
  echo "[real] policy FAILED"; tail -25 outputs/vllm_logs/${RUN}_policy.log; exit 4
}
stop_policy() { [ -n "$PVP" ] && { pkill -9 -P "$PVP" 2>/dev/null; kill -9 "$PVP" 2>/dev/null; wait "$PVP" 2>/dev/null; }; PVP=""; sleep 4; }
stop_user()  { [ -n "$UVP" ] && { pkill -9 -P "$UVP" 2>/dev/null; kill -9 "$UVP" 2>/dev/null; wait "$UVP" 2>/dev/null; }; UVP=""; }
trap 'stop_policy; stop_user' EXIT

collect() {  # tasks N temp save out seed
  $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
    --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
    --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" --max-concurrency 2 --seed "$6" --user-temperature "$USER_TEMP" --save-to "$4" --out "$5" --no-auto-resume || echo "collect $4 nonzero"
}
passrate() { $PYBIN -c "import json;rs=[json.loads(l)['reward'] for l in open('$1')];print(f'pass^1={sum(1 for r in rs if r>=1-1e-6)}/{len(rs)} mean={sum(rs)/max(len(rs),1):.3f}')" 2>&1; }

echo "######## GRPO_AIRLINE_REAL START $(date) RUN=$RUN train=$TRAIN_TASK_IDS eval=$EVAL_TASK_IDS N=$N ITERS=$ITERS reward=$REWARD_MODE lata=$LATA ########"
serve_user
# base eval
serve_policy ""
collect "$EVAL_TASK_IDS" 1 0.0 "grpo_${RUN}_ev0" "$OUT/eval_iter0.jsonl" "$EVAL_SEED"
stop_policy
echo "[real] BASE eval: $(passrate "$OUT/eval_iter0.jsonl")"
CUR=""
for ((it=1; it<=ITERS; it++)); do
  echo "==================== ITER $it/$ITERS $(date +%H:%M:%S) ===================="
  ROLL="$OUT/rollouts_iter${it}.jsonl"; ADP="$OUT/adapter_iter${it}"
  serve_policy "$CUR"
  collect "$TRAIN_TASK_IDS" "$N" "$TEMP" "grpo_${RUN}_tr${it}" "$ROLL" "$it"
  stop_policy
  lata_flag=(); [ "$LATA" = "1" ] && lata_flag=(--lata)
  in_flag=(); [ -n "$CUR" ] && in_flag=(--adapter-in "$CUR")
  CUDA_VISIBLE_DEVICES=$POLICY_GPU $PYBIN scripts/grpo/grpo_update.py --rollouts "$ROLL" --base-model "$POLICY_MODEL" \
    --out-adapter "$ADP" --domain $DOMAIN --reward-mode "$REWARD_MODE" --lr "$LR" "${lata_flag[@]}" "${in_flag[@]}" || { echo "iter $it update FAILED"; break; }
  CUR="$ADP"
  serve_policy "$CUR"
  collect "$EVAL_TASK_IDS" 1 0.0 "grpo_${RUN}_ev${it}" "$OUT/eval_iter${it}.jsonl" "$EVAL_SEED"
  stop_policy
  echo "[real] ITER $it eval: $(passrate "$OUT/eval_iter${it}.jsonl")"
done
echo "######## GRPO_AIRLINE_REAL ALLDONE $(date) ########"
for ((i=0; i<=ITERS; i++)); do [ -f "$OUT/eval_iter${i}.jsonl" ] && echo "iter$i: $(passrate "$OUT/eval_iter${i}.jsonl")"; done
