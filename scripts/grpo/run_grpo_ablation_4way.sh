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
export TAU2_CMD="${TAU2_CMD:-tau2}" OPENAI_API_KEY=dummy  # this box: tau2 in venv (NOT uv) — uv run tau2 silently produces 0 sims here

RUN="${RUN:-grpo_ablation}"
POLICY_MODEL="${POLICY_MODEL:-$WORKDIR/models/qwen25-7b-instruct}"
DOMAIN=airline
TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-2,3,4,7,8,9,12,13,14,17,18,19,22,23,24,27,28,29,32,33,34,37,38,39,42,43,44,47,48,49}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46}"
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
# ssh-replay guard: a laptop sleep/wake can REPLAY the buffered launch command (seen 07-03).
exec 9>"$OUT/.run.lock"
flock -n 9 || { echo "[abl] another $RUN instance is already running — refusing (replay guard)"; exit 9; }
# show progress in the TERMINAL and also keep the full log (run in foreground to watch live;
# or background it and `tail -f $OUT/master.log`). Clean progress: pipe through the grep below.
exec > >(tee -a "$OUT/master.log") 2>&1
UVP=""; PVP=""; CUR_ADAPTER=""

serve_user() {
  echo "[abl] serve usersim GPU$USER_GPU :$UPORT"
  CUDA_VISIBLE_DEVICES=$USER_GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" --served-model-name usersim \
    --host $HOST --port $UPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    >> outputs/vllm_logs/${RUN}_usersim.log 2>&1 &
  UVP=$!
  for _ in $(seq 1 120); do curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && { echo "[abl] usersim ready"; return 0; }; sleep 5; done
  echo "[abl] usersim FAILED"; tail -25 outputs/vllm_logs/${RUN}_usersim.log; exit 4
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
  for _ in $(seq 1 120); do curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy && { echo "[abl] policy ready (adapter='${adapter:-<base>}')"; return 0; }; sleep 5; done
  echo "[abl] policy FAILED"; tail -25 outputs/vllm_logs/${RUN}_policy.log; exit 4
}
stop_policy() { [ -n "$PVP" ] && { pkill -9 -P "$PVP" 2>/dev/null; kill -9 "$PVP" 2>/dev/null; wait "$PVP" 2>/dev/null; }; PVP=""; sleep 4; }
stop_user()  { [ -n "$UVP" ] && { pkill -9 -P "$UVP" 2>/dev/null; kill -9 "$UVP" 2>/dev/null; wait "$UVP" 2>/dev/null; }; UVP=""; }
# ROBUSTNESS (added after 06-30 run died: resident usersim got OOM-killed by a neighbor,
# never restarted -> every later collect silently returned 0 rollouts). Health-check the
# usersim endpoint before each collect; if down, restart it (serve_user aborts loudly if it
# truly cannot come up, instead of a silent multi-hour cascade fail).
ensure_user() {
  curl -sf http://$HOST:$UPORT/v1/models 2>/dev/null | grep -q usersim && return 0
  echo "[abl] usersim endpoint DOWN -> restarting $(date +%H:%M:%S)"
  stop_user; serve_user
}
# Same for the POLICY endpoint. 06-30 re-audit: usersim actually stayed alive (200 OK + clean
# shutdown); the more likely culprit was the policy on GPU1 dying mid-collect under neighbor
# contention. CUR_ADAPTER (set by serve_policy) lets us re-serve the SAME adapter.
ensure_policy() {
  curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q policy && return 0
  echo "[abl] policy endpoint DOWN -> restarting (adapter='${CUR_ADAPTER:-<base>}') $(date +%H:%M:%S)"
  serve_policy "$CUR_ADAPTER"
}
trap 'stop_policy; stop_user' EXIT

collect() {  # tasks N temp save out seed
  ensure_user; ensure_policy   # both endpoints healthy before collecting
  $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
    --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
    --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "$4" --out "$5" --no-auto-resume || echo "collect $4 nonzero"
  # 0 rollouts = an endpoint died mid-collect (contention OOM) -> restart BOTH + retry ONCE
  if [ ! -s "$5" ]; then
    echo "[abl] WARN collect $4 -> 0 rollouts; restart usersim+policy + retry once $(date +%H:%M:%S)"
    ensure_user; ensure_policy
    $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model policy \
      --api-base http://$HOST:$PPORT/v1 --user-llm openai/usersim --user-api-base http://$HOST:$UPORT/v1 \
      --task-ids "$1" --num-trials "$2" --temperature "$3" --max-tokens "$AGENT_MAX_TOKENS" \
      --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$6" --user-temperature "$USER_TEMP" --save-to "${4}_rt" --out "$5" --no-auto-resume || echo "collect $4 retry nonzero"
  fi
}
passrate() { $PYBIN -c "import json;rs=[json.loads(l)['reward'] for l in open('$1')];print(f'pass^1={sum(1 for r in rs if r>=1-1e-6)}/{len(rs)} mean={sum(rs)/max(len(rs),1):.3f}')" 2>&1; }

run_method() {  # name reward_mode lata [td_alpha]
  # INIT_ADAPTER (env, optional): warm-start every method from this adapter (e.g. a distill-SFT
  # checkpoint) instead of raw base. Empty = original from-base behavior.
  local name="$1" rm="$2" lata="$3" td="${4:-0}" seed_off="${5:-0}" CUR="${INIT_ADAPTER:-}"
  echo "############### METHOD $name (reward=$rm lata=$lata td=$td seed_off=$seed_off) $(date +%H:%M:%S) ###############"
  for ((it=1; it<=ITERS; it++)); do
    echo "----- $name ITER $it/$ITERS $(date +%H:%M:%S) -----"
    local ROLL="$OUT/${name}_rollouts_it${it}.jsonl" ADP="$OUT/${name}_adapter_it${it}"
    serve_policy "$CUR"
    collect "$TRAIN_TASK_IDS" "$N" "$TRAIN_TEMP" "abl_${name}_tr${it}" "$ROLL" "$((seed_off + it))"
    stop_policy
    local lf=(); [ "$lata" = "1" ] && lf=(--lata)
    local tf=(); [ "$td" != "0" ] && tf=(--td-alpha "$td")
    local sf=(); [ "$seed_off" != "0" ] && sf=(--seed "$((seed_off + it))")
    local inf=(); [ -n "$CUR" ] && inf=(--adapter-in "$CUR")
    CUDA_VISIBLE_DEVICES=$POLICY_GPU $PYBIN scripts/grpo/grpo_update.py --rollouts "$ROLL" --base-model "$POLICY_MODEL" \
      --out-adapter "$ADP" --domain $DOMAIN --reward-mode "$rm" --lr "$LR" "${lf[@]}" "${tf[@]}" "${sf[@]}" "${inf[@]}" || { echo "$name iter $it update FAILED"; break; }
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
# GRPO-from-SFT: measure the warm-start checkpoint itself (same night, same protocol) so the
# per-iter curve has its true starting anchor — "did RL beat its own SFT start?"
if [ -n "${INIT_ADAPTER:-}" ]; then
  serve_policy "$INIT_ADAPTER"
  collect "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "abl_init_ev" "$OUT/init_eval.jsonl" "$EVAL_SEED"
  stop_policy
  echo "[abl] INIT (warm-start) eval: $(passrate "$OUT/init_eval.jsonl")"
fi
# SAME-DAY 2x2 ablation, dual-GPU, DETERMINISTIC self-hosted usersim (base 7B, USER_TEMP=0):
#   vanilla=binary/noLATA  prmlata=prm_lite/+LATA  prmonly=prm_lite/noLATA  lataonly=binary/+LATA
# METHODS (env): run a subset, e.g. METHODS=vanilla for a single-arm GRPO-from-SFT (+INIT_ADAPTER).
METHODS="${METHODS:-vanilla prmlata prmonly lataonly}"
TD_ALPHA="${TD_ALPHA:-1.05}"   # turn-discount factor for the turndiscount arm
for m in $METHODS; do case "$m" in
  vanilla)      run_method vanilla      binary   0;;
  prmlata)      run_method prmlata      prm_lite 1;;
  prmonly)      run_method prmonly      prm_lite 0;;
  lataonly)     run_method lataonly     binary   1;;
  turndiscount) run_method turndiscount binary   0 "$TD_ALPHA";;   # binary + early-token weighting, no LATA
  lataonly_s*)  run_method "$m"          binary   1 0 "${m##*_s}";; # seed replicate: lataonly_s<seed> (rollout+update RNG offset; eval seed unchanged)
  *) echo "[abl] unknown method: $m (skipped)";;
esac; done
echo "######## GRPO_ABLATION ALLDONE $(date) ########"
echo "====== PAIRED ANALYSIS (multi-trial, bootstrap CI over tasks) ======"
ANALYZE_ARGS=()
[ -s "$OUT/init_eval.jsonl" ] && ANALYZE_ARGS+=("init=$OUT/init_eval.jsonl")
for nm in $METHODS; do f="$OUT/${nm}_eval_final.jsonl"; [ -f "$f" ] && ANALYZE_ARGS+=("$nm=$f"); done
$PYBIN scripts/grpo/tau2_eval_analyze.py "$OUT/base_eval.jsonl" "${ANALYZE_ARGS[@]}" 2>&1 || echo "analyze failed"
echo "====== per-iter curves ======"
for nm in $METHODS; do for ((i=1;i<=ITERS;i++)); do f="$OUT/${nm}_eval_it${i}.jsonl"; [ -f "$f" ] && echo "$nm it$i: $(passrate "$f")"; done; done
echo "ABLATION_ANALYSIS_DONE $(date)"
