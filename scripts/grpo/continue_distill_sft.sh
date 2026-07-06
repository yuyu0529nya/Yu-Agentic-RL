#!/usr/bin/env bash
set -uo pipefail
# UNDER-TRAINING TEST (single variable = training budget): continue the distill SFT from
# tonight's adapter (+EPOCHS more, same lr/data/seq), then re-eval under the IDENTICAL
# same-night protocol and compare against tonight's base evals (same usersim instance/seed).
# Rationale: 3 epochs = 45 opt steps ended at loss 1.06 with train-fit FLAT (0.08 vs 0.06) —
# behavior cloning likely hasn't bitten yet; reference used 250-300 steps.
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; PYBIN="${PYBIN:-$WORKDIR/venv/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-$WORKDIR/venv/bin}:$HOME/.local/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
export TAU2_CMD="${TAU2_CMD:-tau2}" OPENAI_API_KEY=dummy
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKDIR/.cache}" TMPDIR="${TMPDIR:-$WORKDIR/tmp}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORKDIR/.cache/vllm}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKDIR/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKDIR/.cache/torchinductor}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR"

RUN="${RUN:-distill_sft_0702_ext}"
PREV="${PREV:-outputs/distill_sft_0702}"           # tonight's run (adapter + same-night base evals)
ADAPTER_IN="${ADAPTER_IN:-$PREV/distill_adapter}"  # override to chain further rounds (e.g. .../distill_adapter_ext)
BASE_DIR="${BASE_DIR:-$PREV}"                      # where the same-night base_eval/base_fit_eval jsonls live
LABEL="${LABEL:-distill_ext}"                      # label for this round in the paired analysis
DATASET="${DATASET:-outputs/teacher_ds_0702/distill_dataset_final.jsonl}"
POLICY_MODEL="${POLICY_MODEL:-$WORKDIR/models/qwen25-7b-instruct}"
DOMAIN=airline
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46}"
FIT_TASK_IDS="${FIT_TASK_IDS:-2,7,12,17,22,27,32,37,42,47}"
EPOCHS="${EPOCHS:-6}"; LR="${LR:-1e-5}"; MAX_SEQ_LEN="${MAX_SEQ_LEN:-20480}"
EVAL_TEMP="${EVAL_TEMP:-0.5}"; EVAL_TRIALS="${EVAL_TRIALS:-5}"
USER_TEMP="${USER_TEMP:-0}"; EVAL_SEED="${EVAL_SEED:-900}"
MAX_STEPS="${MAX_STEPS:-40}"; AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"; GPU_UTIL="${GPU_UTIL:-0.80}"; MAXC="${MAXC:-3}"
POLICY_GPU="${POLICY_GPU:-1}"; USER_GPU="${USER_GPU:-0}"
PPORT="${PPORT:-18000}"; UPORT="${UPORT:-18001}"; HOST=127.0.0.1
OUT="outputs/$RUN"; mkdir -p "$OUT" outputs/vllm_logs
# ssh-replay guard (07-03: a laptop sleep/wake REPLAYED the buffered launch command and started
# a duplicate run): flock refuses a second instance; the lock dies with the process.
exec 9>"$OUT/.run.lock"
flock -n 9 || { echo "[ext] another $RUN instance is already running — refusing (replay guard)"; exit 9; }
exec > >(tee -a "$OUT/master.log") 2>&1
UVP=""; PVP=""; CUR_ADAPTER=""

serve_user() {
  echo "[ext] serve usersim GPU$USER_GPU :$UPORT"
  CUDA_VISIBLE_DEVICES=$USER_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" --served-model-name usersim \
    --host $HOST --port $UPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    >> outputs/vllm_logs/${RUN}_usersim.log 2>&1 &
  UVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && { echo "[ext] usersim ready"; return 0; }; sleep 5; done
  echo "[ext] usersim FAILED"; tail -25 outputs/vllm_logs/${RUN}_usersim.log; exit 4
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
  for _ in $(seq 1 120); do curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy && { echo "[ext] policy ready (adapter='${adapter:-<base>}')"; return 0; }; sleep 5; done
  echo "[ext] policy FAILED"; tail -25 outputs/vllm_logs/${RUN}_policy.log; exit 4
}
stop_policy() { [ -n "$PVP" ] && { pkill -9 -P "$PVP" 2>/dev/null; kill -9 "$PVP" 2>/dev/null; wait "$PVP" 2>/dev/null; }; PVP=""; sleep 4; }
stop_user()  { [ -n "$UVP" ] && { pkill -9 -P "$UVP" 2>/dev/null; kill -9 "$UVP" 2>/dev/null; wait "$UVP" 2>/dev/null; }; UVP=""; }
ensure_user()   { curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && return 0; echo "[ext] usersim DOWN -> restart"; stop_user; serve_user; }
ensure_policy() { curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy  && return 0; echo "[ext] policy DOWN -> restart"; serve_policy "$CUR_ADAPTER"; }
trap 'stop_policy; stop_user' EXIT

collect() {  # tasks trials temp save out seed
  ensure_user; ensure_policy
  $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
    --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
    --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "$4" --out "$5" --no-auto-resume || echo "collect $4 nonzero"
  if [ ! -s "$5" ]; then
    echo "[ext] WARN collect $4 -> 0 rollouts; restart both + retry once"
    ensure_user; ensure_policy
    $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
      --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
      --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
      --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "${4}_rt" --out "$5" --no-auto-resume || echo "collect $4 retry nonzero"
  fi
}
passrate() { $PYBIN -c "import json;rs=[json.loads(l)['reward'] for l in open('$1')];print(f'pass^1={sum(1 for r in rs if r>=1-1e-6)}/{len(rs)} mean={sum(rs)/max(len(rs),1):.3f}')" 2>&1; }

echo "######## DISTILL_EXT START $(date) +$EPOCHS epochs from $ADAPTER_IN ########"
[ -d "$ADAPTER_IN" ] || { echo "[ext] input adapter missing: $ADAPTER_IN"; exit 2; }
[ -s "$BASE_DIR/base_eval.jsonl" ] || { echo "[ext] base_eval missing in $BASE_DIR"; exit 2; }

echo "-------- 1/3 continue SFT (+$EPOCHS epochs, lr=$LR) --------"
ADP="$OUT/distill_adapter_ext"
CUDA_VISIBLE_DEVICES=$POLICY_GPU $PYBIN scripts/grpo/grpo_update.py --rft --rollouts "$DATASET" \
  --base-model "$POLICY_MODEL" --adapter-in "$ADAPTER_IN" --out-adapter "$ADP" \
  --domain $DOMAIN --reward-mode binary --epochs "$EPOCHS" --lr "$LR" --max-seq-len "$MAX_SEQ_LEN" \
  || { echo "[ext] SFT FAILED"; exit 3; }

echo "-------- 2/3 eval extended adapter (same-night protocol) --------"
serve_user
serve_policy "$ADP"
collect "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "ext_ev" "$OUT/distill_ext_eval.jsonl" "$EVAL_SEED"
echo "[ext] EXT held-out eval: $(passrate "$OUT/distill_ext_eval.jsonl")"
collect "$FIT_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "ext_fit" "$OUT/distill_ext_fit_eval.jsonl" "$EVAL_SEED"
echo "[ext] EXT train-fit eval: $(passrate "$OUT/distill_ext_fit_eval.jsonl")"
stop_policy

echo "-------- 3/3 paired analysis vs same-night base --------"
echo "== held-out =="
$PYBIN scripts/grpo/tau2_eval_analyze.py "$BASE_DIR/base_eval.jsonl" \
  "$LABEL=$OUT/distill_ext_eval.jsonl" 2>&1 || echo "analyze failed"
echo "== train-fit =="
$PYBIN scripts/grpo/tau2_eval_analyze.py "$BASE_DIR/base_fit_eval.jsonl" \
  "$LABEL=$OUT/distill_ext_fit_eval.jsonl" 2>&1 || echo "fit analyze failed"
echo "DISTILL_EXT_ALLDONE $(date)"
