#!/usr/bin/env bash
set -uo pipefail
# LINE 4 — multi-trial NAIL-DOWN of the weekend's best airline checkpoints.
# One same-session usersim instance (base 7B, USER_TEMP=0, constant seed) evaluates base + the
# 3 headline adapters at N_TRIALS=10 (vs 5 before) on the 20 held-out tasks, then paired
# bootstrap CI over tasks. Higher trials/task -> tighter per-task rate -> tighter CI on the
# already-significant deltas. NO training, eval-only.
export WORKDIR=$HOME/agentic-rl
export PYBIN=$WORKDIR/venv/bin/python
export EXTRA_PATH=$WORKDIR/venv/bin
export POLICY_MODEL=$WORKDIR/models/qwen25-7b-instruct
export TAU2_CMD=tau2
cd "$WORKDIR"
export PATH="$EXTRA_PATH:$HOME/.local/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORKDIR/hf-cache"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
export OPENAI_API_KEY=dummy
export XDG_CACHE_HOME="$WORKDIR/.cache" TMPDIR="$WORKDIR/tmp"
export VLLM_CACHE_ROOT="$WORKDIR/.cache/vllm" TRITON_CACHE_DIR="$WORKDIR/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$WORKDIR/.cache/torchinductor"
mkdir -p "$TMPDIR"

RUN="${RUN:-nail_down_0705}"
DOMAIN=airline
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46}"
EVAL_TRIALS="${EVAL_TRIALS:-10}"; EVAL_TEMP="${EVAL_TEMP:-0.5}"
USER_TEMP="${USER_TEMP:-0}"; EVAL_SEED="${EVAL_SEED:-900}"
MAX_STEPS=40; AGENT_MAX_TOKENS=768; MAX_MODEL_LEN=32768; GPU_UTIL="${GPU_UTIL:-0.70}"; MAXC="${MAXC:-3}"
POLICY_GPU="${POLICY_GPU:-0}"; USER_GPU="${USER_GPU:-1}"
PPORT="${PPORT:-18000}"; UPORT="${UPORT:-18001}"; HOST=127.0.0.1
# checkpoints to nail: label:adapterpath  (empty path = base)
CKPTS=(
  "base:"
  "e9_sft:outputs/distill_sft_0702_ext/distill_adapter_ext"
  "line1_prmlata:outputs/grpo_from_sft_0703c/prmlata_adapter_it3"
  "mega_vanilla:outputs/grpo_from_mega_0704/vanilla_adapter_it3"
)
OUT="outputs/$RUN"; mkdir -p "$OUT" outputs/vllm_logs
exec 9>"$OUT/.run.lock"; flock -n 9 || { echo "[nail] already running — refusing"; exit 9; }
exec > >(tee -a "$OUT/master.log") 2>&1
UVP=""; PVP=""; CUR_ADAPTER=""

serve_user() {
  echo "[nail] serve usersim GPU$USER_GPU :$UPORT"
  CUDA_VISIBLE_DEVICES=$USER_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" --served-model-name usersim \
    --host $HOST --port $UPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    >> outputs/vllm_logs/${RUN}_usersim.log 2>&1 &
  UVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && { echo "[nail] usersim ready"; return 0; }; sleep 5; done
  echo "[nail] usersim FAILED"; tail -25 outputs/vllm_logs/${RUN}_usersim.log; exit 4
}
serve_policy() {
  local adapter="$1"; CUR_ADAPTER="$adapter"; stop_policy
  local nm=(--served-model-name policy); [ -n "$adapter" ] && nm=(--served-model-name basemodel --enable-lora --lora-modules "policy=$adapter")
  CUDA_VISIBLE_DEVICES=$POLICY_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" \
    --host $HOST --port $PPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    "${nm[@]}" >> outputs/vllm_logs/${RUN}_policy.log 2>&1 &
  PVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy && { echo "[nail] policy ready (adapter='${adapter:-<base>}')"; return 0; }; sleep 5; done
  echo "[nail] policy FAILED"; tail -25 outputs/vllm_logs/${RUN}_policy.log; exit 4
}
stop_policy() { [ -n "$PVP" ] && { pkill -9 -P "$PVP" 2>/dev/null; kill -9 "$PVP" 2>/dev/null; wait "$PVP" 2>/dev/null; }; PVP=""; sleep 4; }
stop_user()  { [ -n "$UVP" ] && { pkill -9 -P "$UVP" 2>/dev/null; kill -9 "$UVP" 2>/dev/null; wait "$UVP" 2>/dev/null; }; UVP=""; }
ensure_user()   { curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && return 0; echo "[nail] usersim DOWN -> restart"; stop_user; serve_user; }
ensure_policy() { curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy  && return 0; echo "[nail] policy DOWN -> restart"; serve_policy "$CUR_ADAPTER"; }
trap 'stop_policy; stop_user' EXIT

collect() {  # tasks trials temp save out seed
  ensure_user; ensure_policy
  $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
    --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
    --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "$4" --out "$5" --no-auto-resume || echo "collect $4 nonzero"
  if [ ! -s "$5" ]; then
    echo "[nail] WARN collect $4 -> 0 rollouts; restart both + retry once"
    ensure_user; ensure_policy
    $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
      --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
      --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
      --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "${4}_rt" --out "$5" --no-auto-resume || echo "collect $4 retry nonzero"
  fi
}
passrate() { $PYBIN -c "import json;rs=[json.loads(l)['reward'] for l in open('$1')];print(f'pass^1={sum(1 for r in rs if r>=1-1e-6)}/{len(rs)} mean={sum(rs)/max(len(rs),1):.3f}')" 2>&1; }

echo "######## NAIL-DOWN START $(date) trials=$EVAL_TRIALS ckpts=${#CKPTS[@]} ########"
serve_user
ANALYZE=()
for entry in "${CKPTS[@]}"; do
  label="${entry%%:*}"; adp="${entry#*:}"
  echo "----- eval $label (adapter='${adp:-<base>}') $(date +%H:%M) -----"
  serve_policy "$adp"
  collect "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "nail_${label}" "$OUT/${label}_eval.jsonl" "$EVAL_SEED"
  stop_policy
  echo "[nail] $label: $(passrate "$OUT/${label}_eval.jsonl")"
  [ "$label" != "base" ] && [ -s "$OUT/${label}_eval.jsonl" ] && ANALYZE+=("$label=$OUT/${label}_eval.jsonl")
done
echo "######## NAIL-DOWN PAIRED ANALYSIS (n=20 tasks x $EVAL_TRIALS trials) $(date) ########"
$PYBIN scripts/grpo/tau2_eval_analyze.py "$OUT/base_eval.jsonl" "${ANALYZE[@]}" 2>&1 || echo "analyze failed"
echo "NAIL_DOWN_DONE $(date)"
