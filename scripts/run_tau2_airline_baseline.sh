#!/usr/bin/env bash
set -euo pipefail

AGENT_LLM="anthropic/glm-5.1"
USER_LLM="anthropic/glm-5.1"
AGENT_LLM_ARGS=""
USER_LLM_ARGS=""
NUM_TASKS="5"
NUM_TRIALS="1"
MAX_CONCURRENCY="1"
MAX_STEPS="200"
MAX_RETRIES="0"
RETRY_DELAY="1.0"
SEED="300"
TASK_IDS=""
SAVE_TO="airline_smoke"
VERBOSE_LOGS="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-llm) AGENT_LLM="$2"; shift 2 ;;
    --user-llm) USER_LLM="$2"; shift 2 ;;
    --agent-llm-args) AGENT_LLM_ARGS="$2"; shift 2 ;;
    --user-llm-args) USER_LLM_ARGS="$2"; shift 2 ;;
    --num-tasks) NUM_TASKS="$2"; shift 2 ;;
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --retry-delay) RETRY_DELAY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --task-ids) TASK_IDS="$2"; shift 2 ;;
    --save-to) SAVE_TO="$2"; shift 2 ;;
    --verbose-logs) VERBOSE_LOGS="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAU2_ROOT="$ROOT/third_party/tau2-bench"
REPORTS_DIR="$ROOT/reports"
SUMMARY_SCRIPT="$ROOT/scripts/summarize_tau2_results.py"

if [[ ! -d "$TAU2_ROOT" ]]; then
  echo "tau2-bench checkout not found: $TAU2_ROOT" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it with: brew install uv" >&2
  exit 1
fi

export PYTHONUTF8=1

TAU_ARGS=(
  run tau2 run
  --domain airline
  --agent llm_agent
  --user user_simulator
  --agent-llm "$AGENT_LLM"
  --user-llm "$USER_LLM"
  --num-trials "$NUM_TRIALS"
  --max-concurrency "$MAX_CONCURRENCY"
  --max-steps "$MAX_STEPS"
  --max-retries "$MAX_RETRIES"
  --retry-delay "$RETRY_DELAY"
  --seed "$SEED"
  --save-to "$SAVE_TO"
  --auto-resume
)

if [[ -n "$AGENT_LLM_ARGS" ]]; then
  TAU_ARGS+=(--agent-llm-args "$AGENT_LLM_ARGS")
fi

if [[ -n "$USER_LLM_ARGS" ]]; then
  TAU_ARGS+=(--user-llm-args "$USER_LLM_ARGS")
fi

if [[ -n "$TASK_IDS" ]]; then
  IFS=',' read -r -a TASK_ID_ARRAY <<< "$TASK_IDS"
  TAU_ARGS+=(--task-ids)
  for task_id in "${TASK_ID_ARRAY[@]}"; do
    task_id="$(echo "$task_id" | xargs)"
    [[ -n "$task_id" ]] && TAU_ARGS+=("$task_id")
  done
else
  TAU_ARGS+=(--num-tasks "$NUM_TASKS")
fi

if [[ "$VERBOSE_LOGS" == "1" ]]; then
  TAU_ARGS+=(--verbose-logs --llm-log-mode latest)
fi

echo "Running tau2 airline baseline..."
echo "  agent: $AGENT_LLM"
echo "  user:  $USER_LLM"
echo "  save:  data/simulations/$SAVE_TO"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'uv'
  printf ' %q' "${TAU_ARGS[@]}"
  printf '\n'
  exit 0
fi

(
  cd "$TAU2_ROOT"
  uv "${TAU_ARGS[@]}"
)

RESULTS_PATH="$TAU2_ROOT/data/simulations/$SAVE_TO"
if [[ -d "$RESULTS_PATH" ]]; then
  mkdir -p "$REPORTS_DIR"
  python3 "$SUMMARY_SCRIPT" "$RESULTS_PATH" --out "$REPORTS_DIR/$SAVE_TO.summary.json"
else
  echo "Expected results directory was not found: $RESULTS_PATH" >&2
fi
