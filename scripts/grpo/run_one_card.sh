#!/usr/bin/env bash
set -euo pipefail
# Single-card search-agent RLVR with PER-ITER held-out eval. SHARED-BOX + PARALLEL safe:
# serve() only kills THIS run's vLLM (matched by MODEL+PORT), never other GPUs' procs or a
# teammate's. Run two instances at once, one per card: set CUDA_VISIBLE_DEVICES + PORT + RUN.
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; PYBIN="${PYBIN:-$WORKDIR/venv/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-$WORKDIR/venv/bin}:/usr/local/bin:${PATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="${HF_HOME:-${WORKDIR}/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"
MODEL="${POLICY_MODEL:-$WORKDIR/models/qwen25-7b-instruct}"
RUN="${RUN:-search_one}"; SPLIT="${SPLIT:-validation}"; CONFIG="${CONFIG:-distractor}"
N_TRAIN_Q="${N_TRAIN_Q:-128}"; N_EVAL_Q="${N_EVAL_Q:-300}"
N="${N:-8}"; ITERS="${ITERS:-6}"; LR="${LR:-5e-6}"; TEMP="${TEMP:-1.0}"
BATCH="${BATCH:-2}"; GRPO_SEQ="${GRPO_SEQ:-2560}"; PROG="${PROG:-10}"
REWARD="${REWARD:-f1}"; PROC_BETA="${PROC_BETA:-0}"; KL_COEF="${KL_COEF:-0}"; LATA="${LATA:-0}"
EVAL_TRIALS="${EVAL_TRIALS:-1}"; EVAL_TEMP="${EVAL_TEMP:-0}"
LATA_FLAG=""; [ "$LATA" = "1" ] && LATA_FLAG="--lata"
if [ "$REWARD" = "em" ]; then GRPO_REWARD=binary; else GRPO_REWARD=continuous; fi
if awk "BEGIN{exit !($PROC_BETA > 0)}" 2>/dev/null; then GRPO_REWARD=continuous; fi
MAX_TOKENS="${MAX_TOKENS:-512}"; MAX_SEARCHES="${MAX_SEARCHES:-3}"; TOP_K="${TOP_K:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"; GPU_UTIL="${GPU_UTIL:-0.85}"; MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
PORT="${PORT:-8000}"; HOST=127.0.0.1; export OPENAI_API_BASE="http://${HOST}:${PORT}/v1"
OUT="outputs/${RUN}"; mkdir -p "$OUT" outputs/vllm_logs; VP=""

serve() {  # $1 = adapter path (empty = base). Kills ONLY our own vLLM on THIS port (parallel-safe).
  pkill -9 -f "vllm.entrypoints.openai.api_server --model $MODEL --host $HOST --port $PORT" 2>/dev/null || true
  sleep 2
  local nm=(--served-model-name policy)
  [ -n "$1" ] && nm=(--served-model-name basemodel --enable-lora --lora-modules "policy=$1")
  echo "[$RUN] serve vLLM :$PORT (adapter='${1:-<base>}')"
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_FLASHINFER_AUTOTUNE=0 $PYBIN -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host "$HOST" --port "$PORT" --dtype auto --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 16 --enforce-eager --no-enable-flashinfer-autotune \
    "${nm[@]}" > "outputs/vllm_logs/${RUN}.log" 2>&1 &
  VP=$!
  for _ in $(seq 1 120); do curl -sf "$OPENAI_API_BASE/models" 2>/dev/null | grep -q policy && { echo "[$RUN] vLLM ready"; return 0; }; sleep 5; done
  echo "[$RUN] vLLM FAILED"; tail -n 25 "outputs/vllm_logs/${RUN}.log"; exit 4
}
stop() {  # kill vLLM api server AND its EngineCore worker children, so GPU mem is fully freed before train/next-serve
  if [ -n "$VP" ]; then
    pkill -9 -P "$VP" 2>/dev/null || true   # EngineCore worker(s) are children of the api server
    kill -9 "$VP" 2>/dev/null || true
    wait "$VP" 2>/dev/null || true
  fi
  VP=""; sleep 5
}
trap stop EXIT
SA="$PYBIN scripts/grpo/search_agent.py --config $CONFIG --max-searches $MAX_SEARCHES --top-k $TOP_K --max-tokens $MAX_TOKENS --max-concurrency $MAX_CONCURRENCY"
EV="--eval-trials $EVAL_TRIALS --eval-temp $EVAL_TEMP"

echo "######## $RUN BASE eval $(date +%H:%M:%S) ########"
serve ""
$SA --mode eval --split "$SPLIT" --n-questions "$N_EVAL_Q" $EV --out "$OUT/base_eval.jsonl"
stop
CUR=""
for ((it=1; it<=ITERS; it++)); do
  echo "######## $RUN ITER $it/$ITERS collect (policy='${CUR:-<base>}') $(date +%H:%M:%S) ########"
  serve "${CUR}"
  $SA --mode collect --split train --n-questions "$N_TRAIN_Q" --num-trials "$N" --temperature "$TEMP" --reward-mode "$REWARD" --process-beta "$PROC_BETA" --out "$OUT/rollouts_iter${it}.jsonl"
  stop
  echo "######## $RUN ITER $it GRPO update $(date +%H:%M:%S) ########"
  in=(); [ -n "$CUR" ] && in=(--adapter-in "$CUR")
  $PYBIN scripts/grpo/grpo_update.py --rollouts "$OUT/rollouts_iter${it}.jsonl" --base-model "$MODEL" \
    "${in[@]}" --out-adapter "$OUT/adapter_iter${it}" --reward-mode "$GRPO_REWARD" --gate --lr "$LR" \
    --batch-size "$BATCH" --max-seq-len "$GRPO_SEQ" --progress-every "$PROG" --kl-coef "$KL_COEF" $LATA_FLAG
  CUR="$OUT/adapter_iter${it}"
  echo "######## $RUN ITER $it held-out eval $(date +%H:%M:%S) ########"
  serve "$CUR"
  $SA --mode eval --split "$SPLIT" --n-questions "$N_EVAL_Q" $EV --out "$OUT/iter${it}_eval.jsonl"
  stop
done
cp -f "$OUT/iter${ITERS}_eval.jsonl" "$OUT/adapter_eval.jsonl" 2>/dev/null || true
echo "######## $RUN analyze $(date +%H:%M:%S) ########"
EVALS=""; for ((i=1; i<=ITERS; i++)); do [ -f "$OUT/iter${i}_eval.jsonl" ] && EVALS="$EVALS iter${i}=$OUT/iter${i}_eval.jsonl"; done
$PYBIN scripts/grpo/analyze_search_eval.py --base "$OUT/base_eval.jsonl" --evals $EVALS || true
echo "ALLDONE_ONE RUN=$RUN"
