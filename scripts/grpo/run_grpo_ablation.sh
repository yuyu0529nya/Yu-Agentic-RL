#!/usr/bin/env bash
set -uo pipefail
# Airline GRPO ablation: vanilla(binary,no-LATA) vs prm_lite+LATA. Shared usersim + base eval.
# 30 train / 20 held-out tasks, N=4, ITERS=3, eval 2-trial (agent temp 0.5). Paired bootstrap CI.
# GPU0 usersim resident; GPU1 policy (rollout up / stopped before QLoRA update). NO auto-shutdown.
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; PYBIN="${PYBIN:-$WORKDIR/venv/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-$WORKDIR/venv/bin}:$HOME/.local/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
export TAU2_CMD="${TAU2_CMD:-uv run tau2}" OPENAI_API_KEY=dummy

RUN="${RUN:-grpo_ablation}"
POLICY_MODEL="${POLICY_MODEL:-$WORKDIR/models/qwen25-7b-instruct}"
DOMAIN=airline
TRAIN_TASK_IDS="2,3,4,7,8,9,12,13,14,17,18,19,22,23,24,27,28,29,32,33,34,37,38,39,42,43,44,47,48,49"
EVAL_TASK_IDS="0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46"
N="${N:-4}"; ITERS="${ITERS:-3}"; LR="${LR:-1e-5}"
TRAIN_TEMP="${TRAIN_TEMP:-1.0}"; EVAL_TEMP="${EVAL_TEMP:-0.5}"; EVAL_TRIALS="${EVAL_TRIALS:-5}"
# NOISE CONTROL: pin the user-simulator to greedy (USER_TEMP=0) — it is the dominant eval-noise
# source (R5: same base scored 0.35 vs 0.20 across runs). EVAL_SEED is held CONSTANT across base
# and every iter so all checkpoints face the SAME user-sim conversations (fair paired comparison).
USER_TEMP="${USER_TEMP:-0}"; EVAL_SEED="${EVAL_SEED:-900}"
MAX_STEPS="${MAX_STEPS:-40}"; AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"; GPU_UTIL="${GPU_UTIL:-0.80}"; MAXC="${MAXC:-3}"  # 32768: 16384 overflowed long tasks (ContextWindowExceededError -> TOO_MANY_ERRORS -> spurious 0)
POLICY_GPU="${POLICY_GPU:-1}"; USER_GPU="${USER_GPU:-0}"
PPORT="${PPORT:-18000}"; UPORT="${UPORT:-18001}"; HOST=127.0.0.1
OUT="outputs/$RUN"; mkdir -p "$OUT" outputs/vllm_logs
exec >> "$OUT/master.log" 2>&1
UVP=""; PVP=""

serve_user() {
  echo "[abl] serve usersim GPU$USER_GPU :$UPORT"
  CUDA_VISIBLE_DEVICES=$USER_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" --served-model-name usersim \
    --host $HOST --port $UPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    > outputs/vllm_logs/${RUN}_usersim.log 2>&1 &
  UVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && { echo "[abl] usersim ready"; return 0; }; sleep 5; done
  echo "[abl] usersim FAILED"; tail -25 outputs/vllm_logs/${RUN}_usersim.log; exit 4
}
serve_policy() {
  local adapter="$1"; stop_policy
  local nm=(--served-model-name policy); [ -n "$adapter" ] && nm=(--served-model-name basemodel --enable-lora --lora-modules "policy=$adapter")
  CUDA_VISIBLE_DEVICES=$POLICY_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" \
    --host $HOST --port $PPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    "${nm[@]}" > outputs/vllm_logs/${RUN}_policy.log 2>&1 &
  PVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy && { echo "[abl] policy ready (adapter='${adapter:-<base>}')"; return 0; }; sleep 5; done
  echo "[abl] policy FAILED"; tail -25 outputs/vllm_logs/${RUN}_policy.log; exit 4
}
stop_policy() { [ -n "$PVP" ] && { pkill -9 -P "$PVP" 2>/dev/null; kill -9 "$PVP" 2>/dev/null; wait "$PVP" 2>/dev/null; }; PVP=""; sleep 4; }
stop_user()  { [ -n "$UVP" ] && { pkill -9 -P "$UVP" 2>/dev/null; kill -9 "$UVP" 2>/dev/null; wait "$UVP" 2>/dev/null; }; UVP=""; }
trap 'stop_policy; stop_user' EXIT

collect() {  # tasks N temp save out seed
  $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
    --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
    --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "$4" --out "$5" --no-auto-resume || echo "collect $4 nonzero"
}
passrate() { $PYBIN -c "import json;rs=[json.loads(l)['reward'] for l in open('$1')];print(f'pass^1={sum(1 for r in rs if r>=1-1e-6)}/{len(rs)} mean={sum(rs)/max(len(rs),1):.3f}')" 2>&1; }

run_method() {  # name reward_mode lata
  local name="$1" rm="$2" lata="$3" CUR=""
  echo "############### METHOD $name (reward=$rm lata=$lata) $(date +%H:%M:%S) ###############"
  for ((it=1; it<=ITERS; it++)); do
    echo "----- $name ITER $it/$ITERS $(date +%H:%M:%S) -----"
    local ROLL="$OUT/${name}_rollouts_it${it}.jsonl" ADP="$OUT/${name}_adapter_it${it}"
    serve_policy "$CUR"
    collect "$TRAIN_TASK_IDS" "$N" "$TRAIN_TEMP" "abl_${name}_tr${it}" "$ROLL" "$it"
    stop_policy
    local lf=(); [ "$lata" = "1" ] && lf=(--lata)
    local inf=(); [ -n "$CUR" ] && inf=(--adapter-in "$CUR")
    CUDA_VISIBLE_DEVICES=$POLICY_GPU $PYBIN scripts/grpo/grpo_update.py --rollouts "$ROLL" --base-model "$POLICY_MODEL" \
      --out-adapter "$ADP" --domain $DOMAIN --reward-mode "$rm" --lr "$LR" "${lf[@]}" "${inf[@]}" || { echo "$name iter $it update FAILED"; break; }
    CUR="$ADP"
    serve_policy "$CUR"
    collect "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "abl_${name}_ev${it}" "$OUT/${name}_eval_it${it}.jsonl" "$EVAL_SEED"
    stop_policy
    echo "[abl] $name ITER $it eval: $(passrate "$OUT/${name}_eval_it${it}.jsonl")"
  done
  cp -f "$OUT/${name}_eval_it${ITERS}.jsonl" "$OUT/${name}_eval_final.jsonl" 2>/dev/null || true
}

echo "######## GRPO_ABLATION START $(date) train=30 eval=20 N=$N ITERS=$ITERS eval_trials=$EVAL_TRIALS ########"
serve_user
# shared base eval (multi-trial)
serve_policy ""
collect "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "abl_base_ev" "$OUT/base_eval.jsonl" "$EVAL_SEED"
stop_policy
echo "[abl] BASE eval: $(passrate "$OUT/base_eval.jsonl")"
run_method vanilla binary 0
run_method prmlata prm_lite 1
echo "######## GRPO_ABLATION ALLDONE $(date) ########"
echo "====== PAIRED ANALYSIS (multi-trial, bootstrap CI over tasks) ======"
$PYBIN scripts/grpo/tau2_eval_analyze.py "$OUT/base_eval.jsonl" \
  vanilla="$OUT/vanilla_eval_final.jsonl" prmlata="$OUT/prmlata_eval_final.jsonl" 2>&1 || echo "analyze failed"
echo "====== per-iter curves ======"
for nm in vanilla prmlata; do for ((i=1;i<=ITERS;i++)); do f="$OUT/${nm}_eval_it${i}.jsonl"; [ -f "$f" ] && echo "$nm it$i: $(passrate "$f")"; done; done
echo "ABLATION_ANALYSIS_DONE $(date)"
