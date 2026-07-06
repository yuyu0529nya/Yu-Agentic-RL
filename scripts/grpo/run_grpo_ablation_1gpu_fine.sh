#!/usr/bin/env bash
set -uo pipefail
# SINGLE-GPU airline GRPO ablation: vanilla(binary,no-LATA) vs prm_lite+LATA.
# Faithful 1-card port of run_grpo_ablation.sh: instead of a 2nd GPU for the user-sim,
# ONE vLLM serves base + LoRA together (--enable-lora), so:
#   agent-llm -> served model "policy"   (base + current adapter)  [base cond: "basemodel"]
#   user-llm  -> served model "basemodel"(pure base, FIXED)        via --user-api-base (same endpoint)
# => an INDEPENDENT base-7B user-sim on a single card, NO external API key, zero cost.
# Both arms share the SAME base eval + SAME fixed user-sim (USER_TEMP=0) + SAME constant EVAL_SEED
# so the vanilla-vs-prmlata comparison is paired & fair. Per-iter QLoRA update runs after the
# vLLM is stopped (frees the card), then re-serve for eval.
WORKDIR="${WORKDIR:-/root/autodl-tmp/yuyu}"; PYBIN="${PYBIN:-/root/miniconda3/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-/root/miniconda3/bin}:$HOME/.local/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
export TAU2_CMD="${TAU2_CMD:-tau2}" OPENAI_API_KEY=dummy
# load API keys (deepseek user-sim) from .env if present
[ -f "$WORKDIR/.env" ] && { set -a; . "$WORKDIR/.env"; set +a; }
# USER_BACKEND: selfhost = local base-7B usersim via vLLM base+LoRA multiplex (NO key);
#               deepseek = deepseek/deepseek-chat API (stronger usersim, project-consistent, needs DEEPSEEK_API_KEY)
USER_BACKEND="${USER_BACKEND:-selfhost}"
USER_LLM_DS="${USER_LLM_DS:-deepseek/deepseek-chat}"

RUN="${RUN:-grpo_ablation_1gpu}"
POLICY_MODEL="${POLICY_MODEL:-/root/autodl-tmp/models/qwen25-7b-instruct}"
DOMAIN=airline
TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-2,3,4,7,8,9,12,13,14,17,18,19,22,23,24,27,28,29,32,33,34,37,38,39,42,43,44,47,48,49}"
EVAL_TASK_IDS="${EVAL_TASK_IDS:-0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46}"
N="${N:-4}"; ITERS="${ITERS:-3}"; LR="${LR:-1e-5}"
TRAIN_TEMP="${TRAIN_TEMP:-1.0}"; EVAL_TEMP="${EVAL_TEMP:-0.5}"; EVAL_TRIALS="${EVAL_TRIALS:-5}"
USER_TEMP="${USER_TEMP:-0}"; EVAL_SEED="${EVAL_SEED:-900}"
MAX_STEPS="${MAX_STEPS:-40}"; AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"; GPU_UTIL="${GPU_UTIL:-0.85}"; MAXC="${MAXC:-4}"
GPU="${GPU:-0}"; PPORT="${PPORT:-18000}"; HOST=127.0.0.1
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"; DEADMAN_SECS="${DEADMAN_SECS:-28800}"  # 8h backstop

# ---- SMOKE: tiny end-to-end validation (overrides above, no shutdown) ----
if [ "${SMOKE:-0}" = "1" ]; then
  RUN="${RUN_SMOKE:-grpo_abl_smoke}"; TRAIN_TASK_IDS="2,3"; EVAL_TASK_IDS="0,1"
  N=2; ITERS=1; EVAL_TRIALS=1; AUTO_SHUTDOWN=0
  echo "[abl] *** SMOKE MODE *** train=$TRAIN_TASK_IDS eval=$EVAL_TASK_IDS N=$N ITERS=$ITERS trials=$EVAL_TRIALS"
fi

OUT="outputs/$RUN"; mkdir -p "$OUT" outputs/vllm_logs
exec >> "$OUT/master.log" 2>&1
PVP=""

serve() {  # adapter("" = base only)
  local adapter="$1"; stop_serve
  local lf=()
  if [ -n "$adapter" ]; then
    lf=(--enable-lora --max-lora-rank 32 --lora-modules "policy=$adapter")
  fi
  echo "[abl] serve GPU$GPU :$PPORT  adapter='${adapter:-<base>}'"
  CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 \
    $PYBIN -m vllm.entrypoints.openai.api_server --model "$POLICY_MODEL" --served-model-name basemodel \
    --host $HOST --port $PPORT --dtype auto --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 16 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --no-enable-flashinfer-autotune \
    "${lf[@]}" > outputs/vllm_logs/${RUN}_policy.log 2>&1 &
  PVP=$!
  local want=basemodel; [ -n "$adapter" ] && want=policy
  for _ in $(seq 1 120); do curl -sf http://$HOST:$PPORT/v1/models 2>/dev/null | grep -q "$want" && { echo "[abl] serve ready ($want)"; return 0; }; sleep 5; done
  echo "[abl] serve FAILED"; tail -30 outputs/vllm_logs/${RUN}_policy.log; exit 4
}
stop_serve() { [ -n "$PVP" ] && { pkill -9 -P "$PVP" 2>/dev/null; kill -9 "$PVP" 2>/dev/null; wait "$PVP" 2>/dev/null; }; PVP=""; sleep 4; }
trap 'stop_serve' EXIT

collect() {  # agent_model tasks N temp save out seed
  local ua=(--user-llm openai/basemodel --user-api-base http://$HOST:$PPORT/v1)
  [ "$USER_BACKEND" = "deepseek" ] && ua=(--user-llm "$USER_LLM_DS")
  $PYBIN scripts/grpo/collect_rollouts.py --domain $DOMAIN --served-model "$1" \
    --api-base http://$HOST:$PPORT/v1 "${ua[@]}" \
    --task-ids "$2" --num-trials "$3" --temperature "$4" --max-tokens "$AGENT_MAX_TOKENS" \
    --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" --seed "$7" --user-temperature "$USER_TEMP" \
    --save-to "$5" --out "$6" --no-auto-resume || echo "collect $5 nonzero"
}
passrate() { $PYBIN -c "import json;rs=[json.loads(l)['reward'] for l in open('$1')];print(f'pass^1={sum(1 for r in rs if r>=1-1e-6)}/{len(rs)} mean={sum(rs)/max(len(rs),1):.3f}')" 2>&1; }

run_method() {  # name reward_mode lata
  local name="$1" rm="$2" lata="$3" CUR=""
  echo "############### METHOD $name (reward=$rm lata=$lata) $(date +%H:%M:%S) ###############"
  for ((it=1; it<=ITERS; it++)); do
    echo "----- $name ITER $it/$ITERS $(date +%H:%M:%S) -----"
    local ROLL="$OUT/${name}_rollouts_it${it}.jsonl" ADP="$OUT/${name}_adapter_it${it}"
    local agentm=policy; [ -z "$CUR" ] && agentm=basemodel
    serve "$CUR"
    collect "$agentm" "$TRAIN_TASK_IDS" "$N" "$TRAIN_TEMP" "abl1_${name}_tr${it}" "$ROLL" "$it"
    stop_serve
    local lf=(); [ "$lata" = "1" ] && lf=(--lata)
    local inf=(); [ -n "$CUR" ] && inf=(--adapter-in "$CUR")
    CUDA_VISIBLE_DEVICES=$GPU $PYBIN scripts/grpo/grpo_update.py --rollouts "$ROLL" --base-model "$POLICY_MODEL" \
      --out-adapter "$ADP" --domain $DOMAIN --reward-mode "$rm" --lr "$LR" "${lf[@]}" "${inf[@]}" || { echo "$name iter $it update FAILED"; break; }
    CUR="$ADP"
    serve "$CUR"
    collect policy "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "abl1_${name}_ev${it}" "$OUT/${name}_eval_it${it}.jsonl" "$EVAL_SEED"
    stop_serve
    echo "[abl] $name ITER $it eval: $(passrate "$OUT/${name}_eval_it${it}.jsonl")"
  done
  cp -f "$OUT/${name}_eval_it${ITERS}.jsonl" "$OUT/${name}_eval_final.jsonl" 2>/dev/null || true
}

# deadman backstop so billing stops even if the run hangs (AutoDL: shutdown via PATH)
if [ "$AUTO_SHUTDOWN" = "1" ]; then ( sleep "$DEADMAN_SECS"; echo "[abl] DEADMAN fired"; shutdown -h now ) & fi

echo "######## GRPO_ABLATION_1GPU START $(date) train=$(echo $TRAIN_TASK_IDS|tr , ' '|wc -w) eval=$(echo $EVAL_TASK_IDS|tr , ' '|wc -w) N=$N ITERS=$ITERS eval_trials=$EVAL_TRIALS user_backend=$USER_BACKEND ########"
# shared base eval (multi-trial), agent = pure base
serve ""
collect basemodel "$EVAL_TASK_IDS" "$EVAL_TRIALS" "$EVAL_TEMP" "abl1_base_ev" "$OUT/base_eval.jsonl" "$EVAL_SEED"
stop_serve
echo "[abl] BASE eval: $(passrate "$OUT/base_eval.jsonl")"
# FINE-GRAINED: the two MISSING corners of the 2x2 (vanilla & prmlata done 2026-06-29).
#   prmonly  = prm_lite reward, NO LATA   (isolates the PROCESS-REWARD effect)
#   lataonly = binary reward,   +LATA     (isolates the LATA effect)
run_method prmonly  prm_lite 0
run_method lataonly binary   1
echo "######## GRPO_ABLATION_1GPU ALLDONE $(date) ########"
echo "====== PAIRED ANALYSIS (multi-trial, bootstrap CI over tasks) ======"
echo "[note] base = consistency check vs 2026-06-29 (was 0.190); full 4-way comparison done LOCALLY with yesterday's vanilla/prmlata."
$PYBIN scripts/grpo/tau2_eval_analyze.py "$OUT/base_eval.jsonl" \
  prmonly="$OUT/prmonly_eval_final.jsonl" lataonly="$OUT/lataonly_eval_final.jsonl" 2>&1 || echo "analyze failed"
echo "====== per-iter curves ======"
for nm in prmonly lataonly; do for ((i=1;i<=ITERS;i++)); do f="$OUT/${nm}_eval_it${i}.jsonl"; [ -f "$f" ] && echo "$nm it$i: $(passrate "$f")"; done; done
echo "ABLATION_ANALYSIS_DONE $(date)"
if [ "$AUTO_SHUTDOWN" = "1" ]; then echo "[abl] all done -> shutting down"; sync; shutdown -h now; fi
