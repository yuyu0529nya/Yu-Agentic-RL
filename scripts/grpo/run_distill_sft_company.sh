#!/usr/bin/env bash
set -uo pipefail
# Distillation SFT from teacher trajectories + SAME-PROTOCOL paired eval vs base.
# 2x5090 box. Mirrors run_grpo_ablation_4way.sh conventions exactly:
# usersim resident GPU0, policy GPU1, USER_TEMP=0, EVAL_SEED=900 constant -> every
# checkpoint faces the SAME usersim conversations (fair paired comparison).
#
#   DATASET=outputs/teacher_ds_0702/distill_dataset.jsonl bash scripts/grpo/run_distill_sft_company.sh
# Reuse a prior base eval (same protocol) to save ~1h:
#   BASE_EVAL=outputs/abl_4way_dual_0701/base_eval.jsonl DATASET=... bash scripts/grpo/run_distill_sft_company.sh
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; PYBIN="${PYBIN:-$WORKDIR/venv/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-$WORKDIR/venv/bin}:$HOME/.local/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
export TAU2_CMD="${TAU2_CMD:-tau2}" OPENAI_API_KEY=dummy   # shared /data env: tau2 CLI in venv (NOT uv)
# SIGBUS guard (07-01): shared system disk near-full kills vLLM EngineCore -> keep ALL caches on /data
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKDIR/.cache}" TMPDIR="${TMPDIR:-$WORKDIR/tmp}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORKDIR/.cache/vllm}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKDIR/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKDIR/.cache/torchinductor}"
mkdir -p "$TMPDIR"

RUN="${RUN:-distill_sft_$(date +%m%d)}"
DATASET="${DATASET:?set DATASET=<distill_dataset.jsonl from run_teacher_collect.sh>}"
BASE_EVAL="${BASE_EVAL:-}"       # optional: reuse an existing same-protocol base eval jsonl
POLICY_MODEL="${POLICY_MODEL:-$WORKDIR/models/qwen25-7b-instruct}"
DOMAIN=airline
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46}"
# fitting check: a slice of TRAIN tasks — SFT must visibly lift these before we trust anything
FIT_TASK_IDS="${FIT_TASK_IDS:-2,7,12,17,22,27,32,37,42,47}"
EPOCHS="${EPOCHS:-3}"; LR="${LR:-1e-5}"
# Teacher trajectories are LONG (measured on the real 114-example final set: p50=5124,
# max=19218 tokens) and grpo_update RIGHT-truncates — the default 4096 would cut assistant
# tokens (the late write-action turns!) from 67/114 examples. 20480 covers the entire set.
# If the SFT step OOMs at 20480, fall back to MAX_SEQ_LEN=16384 (damages only ~2 examples).
MAX_SEQ_LEN="${MAX_SEQ_LEN:-20480}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
EVAL_TEMP="${EVAL_TEMP:-0.5}"; EVAL_TRIALS="${EVAL_TRIALS:-5}"
USER_TEMP="${USER_TEMP:-0}"; EVAL_SEED="${EVAL_SEED:-900}"
MAX_STEPS="${MAX_STEPS:-40}"; AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"; GPU_UTIL="${GPU_UTIL:-0.80}"; MAXC="${MAXC:-3}"
POLICY_GPU="${POLICY_GPU:-1}"; USER_GPU="${USER_GPU:-0}"
PPORT="${PPORT:-18000}"; UPORT="${UPORT:-18001}"; HOST=127.0.0.1
OUT="outputs/$RUN"; mkdir -p "$OUT" outputs/vllm_logs
# ssh-replay guard: a laptop sleep/wake can REPLAY the buffered launch command (seen 07-03).
exec 9>"$OUT/.run.lock"
flock -n 9 || { echo "[dst] another $RUN instance is already running — refusing (replay guard)"; exit 9; }
exec > >(tee -a "$OUT/master.log") 2>&1
UVP=""; PVP=""; CUR_ADAPTER=""

serve_user() {
  echo "[dst] serve usersim GPU$USER_GPU :$UPORT"
  CUDA_VISIBLE_DEVICES=$USER_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" --served-model-name usersim \
    --host $HOST --port $UPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    >> outputs/vllm_logs/${RUN}_usersim.log 2>&1 &
  UVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && { echo "[dst] usersim ready"; return 0; }; sleep 5; done
  echo "[dst] usersim FAILED"; tail -25 outputs/vllm_logs/${RUN}_usersim.log; exit 4
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
  for _ in $(seq 1 120); do curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy && { echo "[dst] policy ready (adapter='${adapter:-<base>}')"; return 0; }; sleep 5; done
  echo "[dst] policy FAILED"; tail -25 outputs/vllm_logs/${RUN}_policy.log; exit 4
}
stop_policy() { [ -n "$PVP" ] && { pkill -9 -P "$PVP" 2>/dev/null; kill -9 "$PVP" 2>/dev/null; wait "$PVP" 2>/dev/null; }; PVP=""; sleep 4; }
stop_user()  { [ -n "$UVP" ] && { pkill -9 -P "$UVP" 2>/dev/null; kill -9 "$UVP" 2>/dev/null; wait "$UVP" 2>/dev/null; }; UVP=""; }
ensure_user()   { curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && return 0; echo "[dst] usersim DOWN -> restart $(date +%H:%M:%S)"; stop_user; serve_user; }
ensure_policy() { curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy  && return 0; echo "[dst] policy DOWN -> restart (adapter='${CUR_ADAPTER:-<base>}') $(date +%H:%M:%S)"; serve_policy "$CUR_ADAPTER"; }
trap 'stop_policy; stop_user' EXIT

collect() {  # tasks trials temp save out seed
  ensure_user; ensure_policy
  $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
    --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
    --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "$4" --out "$5" --no-auto-resume || echo "collect $4 nonzero"
  if [ ! -s "$5" ]; then
    echo "[dst] WARN collect $4 -> 0 rollouts; restart both + retry once $(date +%H:%M:%S)"
    ensure_user; ensure_policy
    $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
      --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
      --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
      --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "${4}_rt" --out "$5" --no-auto-resume || echo "collect $4 retry nonzero"
  fi
}
passrate() { $PYBIN -c "import json;rs=[json.loads(l)['reward'] for l in open('$1')];print(f'pass^1={sum(1 for r in rs if r>=1-1e-6)}/{len(rs)} mean={sum(rs)/max(len(rs),1):.3f}')" 2>&1; }

echo "######## DISTILL_SFT START $(date) dataset=$DATASET epochs=$EPOCHS lr=$LR ########"
[ -s "$DATASET" ] || { echo "[dst] dataset missing/empty: $DATASET"; exit 2; }

echo "-------- 1/4 SFT (grpo_update --rft) on GPU$POLICY_GPU --------"
ADP="$OUT/distill_adapter"
CUDA_VISIBLE_DEVICES=$POLICY_GPU $PYBIN scripts/grpo/grpo_update.py --rft --rollouts "$DATASET" \
  --base-model "$POLICY_MODEL" --out-adapter "$ADP" --domain $DOMAIN \
  --reward-mode binary --epochs "$EPOCHS" --lr "$LR" --max-seq-len "$MAX_SEQ_LEN" \
  || { echo "[dst] SFT FAILED"; exit 3; }

serve_user
echo "-------- 2/4 base eval: held-out (reused if BASE_EVAL given) + train-fit reference --------"
NEED_BASE_HELDOUT=1
if [ -n "$BASE_EVAL" ] && [ -s "$BASE_EVAL" ]; then
  cp -f "$BASE_EVAL" "$OUT/base_eval.jsonl"; NEED_BASE_HELDOUT=0
  echo "[dst] reusing base held-out eval: $BASE_EVAL"
fi
serve_policy ""
[ "$NEED_BASE_HELDOUT" = "1" ] && collect "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "dst_base_ev" "$OUT/base_eval.jsonl" "$EVAL_SEED"
collect "$FIT_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "dst_base_fit" "$OUT/base_fit_eval.jsonl" "$EVAL_SEED"
stop_policy
echo "[dst] BASE held-out eval: $(passrate "$OUT/base_eval.jsonl")"
echo "[dst] BASE train-fit eval: $(passrate "$OUT/base_fit_eval.jsonl")"

echo "-------- 3/4 distill-adapter eval: held-out + train-fitting slice --------"
serve_policy "$ADP"
collect "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "dst_ev" "$OUT/distill_eval.jsonl" "$EVAL_SEED"
echo "[dst] DISTILL held-out eval: $(passrate "$OUT/distill_eval.jsonl")"
collect "$FIT_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "dst_fit" "$OUT/distill_fit_eval.jsonl" "$EVAL_SEED"
echo "[dst] DISTILL train-fit eval: $(passrate "$OUT/distill_fit_eval.jsonl")"
stop_policy

echo "-------- 4/4 paired analysis vs base (bootstrap CI over tasks) --------"
echo "== held-out (the headline) =="
$PYBIN scripts/grpo/tau2_eval_analyze.py "$OUT/base_eval.jsonl" distill="$OUT/distill_eval.jsonl" 2>&1 || echo "analyze failed"
echo "== train-fit slice (sanity: SFT must lift these first) =="
$PYBIN scripts/grpo/tau2_eval_analyze.py "$OUT/base_fit_eval.jsonl" distill="$OUT/distill_fit_eval.jsonl" 2>&1 || echo "fit analyze failed"
echo "DISTILL_SFT_ALLDONE $(date)"
