#!/usr/bin/env bash
# tau2 airline single-task smoke: serve local 7B vLLM (tool-calling) + run 1 task
# with agent AND user simulator both on the same local endpoint (PoC dimensionality cut).
set -uo pipefail
WORKDIR=$HOME/agentic-rl
export PATH="$HOME/.local/bin:$WORKDIR/venv/bin:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
BASE="$WORKDIR/models/qwen25-7b-instruct"
TAU2="$WORKDIR/third_party/tau2-bench"
PYBIN="$WORKDIR/venv/bin/python"
PORT="${PORT:-8000}"; HOST=127.0.0.1; SERVED=qwen-local
TASK="${TASK:-2}"; TRIALS="${TRIALS:-1}"; MAXSTEPS="${MAXSTEPS:-40}"; GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OPENAI_API_KEY=dummy OPENAI_API_BASE="http://$HOST:$PORT/v1"
mkdir -p "$WORKDIR/outputs/vllm_logs"
LD_LIBRARY_PATH="$($PYBIN -c 'import os,glob,nvidia;print(":".join(sorted(glob.glob(os.path.dirname(nvidia.__file__)+"/*/lib"))))' 2>/dev/null):${LD_LIBRARY_PATH:-}"; export LD_LIBRARY_PATH

echo "#### serve vLLM (tool-calling) $(date +%H:%M:%S)"
$PYBIN -m vllm.entrypoints.openai.api_server --model "$BASE" --served-model-name "$SERVED" \
  --host "$HOST" --port "$PORT" --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.80 \
  --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager \
  > "$WORKDIR/outputs/vllm_logs/tau2_smoke.log" 2>&1 &
VP=$!
for i in $(seq 1 100); do curl -sf "http://$HOST:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED" && break; sleep 5; done
if ! curl -sf "http://$HOST:$PORT/v1/models" 2>/dev/null | grep -q "$SERVED"; then
  echo "vLLM FAILED"; tail -25 "$WORKDIR/outputs/vllm_logs/tau2_smoke.log"; kill -9 "$VP" 2>/dev/null; exit 4
fi
echo "vLLM ready $(date +%H:%M:%S)"

AARGS='{"api_base":"http://127.0.0.1:'"$PORT"'/v1","api_key":"dummy","temperature":0.0,"max_tokens":512}'
UARGS='{"api_base":"http://127.0.0.1:'"$PORT"'/v1","api_key":"dummy","temperature":0.7,"max_tokens":512}'
echo "#### tau2 run airline task=$TASK steps=$MAXSTEPS $(date +%H:%M:%S)"
cd "$TAU2"
timeout 900 "$HOME/.local/bin/uv" run tau2 run \
  --domain airline --agent llm_agent --user user_simulator \
  --agent-llm "openai/$SERVED" --user-llm "openai/$SERVED" \
  --agent-llm-args "$AARGS" --user-llm-args "$UARGS" \
  --num-trials "$TRIALS" --max-concurrency 1 --max-steps "$MAXSTEPS" --max-retries 0 \
  --seed 900 --save-to "tau2_smoke_task${TASK}" --task-ids "$TASK" 2>&1 | tail -45
echo "#### TAU2_SMOKE_DONE $(date +%H:%M:%S)"
pkill -9 -P "$VP" 2>/dev/null; kill -9 "$VP" 2>/dev/null
echo "#### results summary ===="
$PYBIN - "$TAU2/data/simulations/tau2_smoke_task${TASK}/results.json" <<'PY' 2>&1 | head -30
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("no results:",e); sys.exit()
sims=d.get("simulations") or d.get("results") or []
print("top keys:",list(d.keys())[:10])
print("n_simulations:",len(sims) if isinstance(sims,list) else "?")
if isinstance(sims,list) and sims:
    s=sims[0]
    print("sim keys:",list(s.keys())[:15])
    print("reward/success:",s.get("reward"),s.get("reward_info"),s.get("success"))
    msgs=s.get("messages") or s.get("trajectory") or []
    print("n_messages:",len(msgs))
PY
echo SMOKE_ALL_END
