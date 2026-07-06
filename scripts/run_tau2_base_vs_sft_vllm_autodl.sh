#!/usr/bin/env bash
set -euo pipefail

# End-to-end tau2 pass^1 comparison for a local OpenAI-compatible vLLM server.
# This evaluator never shuts the instance down. Stop the AutoDL instance from
# the console only when the user explicitly asks for it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TAU2_ROOT="${TAU2_ROOT:-${ROOT}/third_party/tau2-bench}"

export PATH="/root/miniconda3/bin:/root/.local/bin:${PATH}"

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/qwen25-7b-instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-${ROOT}/outputs/sft_action_prefix_v2_qwen25_7b_qlora_2048/checkpoint}"
USER_LLM="${USER_LLM:-anthropic/glm-5.1}"
TASK_IDS="${TASK_IDS:-2,16,18,25,44}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
MAX_RETRIES="${MAX_RETRIES:-0}"
RETRY_DELAY="${RETRY_DELAY:-1.0}"
SEED="${SEED:-900}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-512}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.0}"
AGENT_TOP_P="${AGENT_TOP_P:-}"
AGENT_STOP_SEQUENCE="${AGENT_STOP_SEQUENCE:-}"
AGENT_INCLUDE_STOP_STR="${AGENT_INCLUDE_STOP_STR:-0}"
AGENT_PARALLEL_TOOL_CALLS="${AGENT_PARALLEL_TOOL_CALLS:-false}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
VLLM_PYTHONOPTIMIZE="${VLLM_PYTHONOPTIMIZE:-1}"
VLLM_TORCHDYNAMO_DISABLE="${VLLM_TORCHDYNAMO_DISABLE:-0}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_ENABLE_FLASHINFER_AUTOTUNE="${VLLM_ENABLE_FLASHINFER_AUTOTUNE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
BASE_SERVED_MODEL="${BASE_SERVED_MODEL:-qwen25-7b-base}"
SFT_SERVED_MODEL="${SFT_SERVED_MODEL:-qwen25-7b-action-prefix-sft}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
RUN_BASE="${RUN_BASE:-1}"
RUN_SFT="${RUN_SFT:-1}"

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="http://${HOST}:${PORT}/v1"
export OPENAI_BASE_URL="${OPENAI_API_BASE}"
export TOKENIZERS_PARALLELISM=false

VLLM_PID=""
BASE_SAVE_TO="${BASE_SAVE_TO:-airline_qwen25_7b_base_${RUN_TAG}}"
SFT_SAVE_TO="${SFT_SAVE_TO:-airline_qwen25_7b_sft_${RUN_TAG}}"
COMPARE_STEM="${COMPARE_STEM:-airline_qwen25_7b_base_vs_sft_${RUN_TAG}}"

finish() {
  code=$?
  stop_vllm || true
  echo "[$(date '+%F %T')] EXIT_CODE=${code}"
  echo "[$(date '+%F %T')] Instance left running."
}
trap finish EXIT

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

find_uv() {
  local uv_path
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return
  fi
  if [[ -x "/root/.local/bin/uv" ]]; then
    echo "/root/.local/bin/uv"
    return
  fi
  if [[ -x "/root/miniconda3/bin/uv" ]]; then
    echo "/root/miniconda3/bin/uv"
    return
  fi

  echo "Installing uv." >&2
  python -m pip install -U uv >&2
  hash -r || true

  uv_path="$(command -v uv || true)"
  if [[ -n "${uv_path}" && -x "${uv_path}" ]]; then
    echo "${uv_path}"
    return
  fi
  for uv_path in "/root/.local/bin/uv" "/root/miniconda3/bin/uv"; do
    if [[ -x "${uv_path}" ]]; then
      echo "${uv_path}"
      return
    fi
  done

  local shim="${ROOT}/.cache/uv"
  mkdir -p "$(dirname "${shim}")"
  cat >"${shim}" <<'SH'
#!/usr/bin/env bash
exec python -m uv "$@"
SH
  chmod +x "${shim}"
  echo "${shim}"
}

maybe_install_vllm() {
  if python - <<'PY' >/dev/null 2>&1
import vllm  # noqa: F401
PY
  then
    echo "vLLM already available."
    return
  fi
  if [[ "${INSTALL_VLLM}" != "1" ]]; then
    echo "vLLM is not installed and INSTALL_VLLM=0." >&2
    exit 3
  fi
  echo "Installing vLLM. This can take a while on a fresh image."
  python -m pip install -U "vllm>=0.8.5"
}

check_user_llm_env() {
  if [[ "${USER_LLM}" == anthropic/* ]]; then
    if [[ -z "${ANTHROPIC_API_KEY:-}" && -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
      export ANTHROPIC_API_KEY="${ANTHROPIC_AUTH_TOKEN}"
    fi
    if [[ -z "${ANTHROPIC_API_BASE:-}" && -n "${ANTHROPIC_BASE_URL:-}" ]]; then
      export ANTHROPIC_API_BASE="${ANTHROPIC_BASE_URL}"
    fi
    if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
      echo "Warning: USER_LLM=${USER_LLM}, but ANTHROPIC_AUTH_TOKEN is not set." >&2
    fi
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      echo "Warning: USER_LLM=${USER_LLM}, but ANTHROPIC_API_KEY is not set." >&2
    fi
    if [[ -z "${ANTHROPIC_BASE_URL:-}" && -z "${ANTHROPIC_API_BASE:-}" ]]; then
      echo "Warning: USER_LLM=${USER_LLM}, but ANTHROPIC_BASE_URL/ANTHROPIC_API_BASE is not set." >&2
    fi
  fi
}

stop_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "Stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
  VLLM_PID=""
}

wait_for_vllm() {
  local model_name="$1"
  local deadline=$((SECONDS + 360))
  echo "Waiting for vLLM model=${model_name} on ${OPENAI_API_BASE}"
  until python - "${OPENAI_API_BASE}" "${model_name}" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
model_name = sys.argv[2]
try:
    with urllib.request.urlopen(base_url + "/models", timeout=5) as response:
        data = json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
models = [item.get("id") for item in data.get("data", [])]
raise SystemExit(0 if model_name in models else 1)
PY
  do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for vLLM." >&2
      exit 4
    fi
    sleep 5
  done
  echo "vLLM ready: ${model_name}"
}

start_vllm_base() {
  local served_model="$1"
  local log_path="$2"
  local -a extra_args=()
  if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
    extra_args+=(--enforce-eager)
  fi
  if [[ "${VLLM_ENABLE_FLASHINFER_AUTOTUNE}" == "0" ]]; then
    extra_args+=(--no-enable-flashinfer-autotune)
  fi
  stop_vllm || true
  echo "Starting base vLLM server: ${served_model}"
  TORCHDYNAMO_DISABLE="${VLLM_TORCHDYNAMO_DISABLE}" \
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER}" \
    PYTHONOPTIMIZE="${VLLM_PYTHONOPTIMIZE}" \
    python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --served-model-name "${served_model}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    "${extra_args[@]}" \
    >"${log_path}" 2>&1 &
  VLLM_PID=$!
  wait_for_vllm "${served_model}"
}

start_vllm_sft() {
  local served_model="$1"
  local log_path="$2"
  local -a extra_args=()
  if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
    extra_args+=(--enforce-eager)
  fi
  if [[ "${VLLM_ENABLE_FLASHINFER_AUTOTUNE}" == "0" ]]; then
    extra_args+=(--no-enable-flashinfer-autotune)
  fi
  stop_vllm || true
  echo "Starting SFT LoRA vLLM server: ${served_model}"
  TORCHDYNAMO_DISABLE="${VLLM_TORCHDYNAMO_DISABLE}" \
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER}" \
    PYTHONOPTIMIZE="${VLLM_PYTHONOPTIMIZE}" \
    python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --served-model-name "${BASE_SERVED_MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-lora \
    --lora-modules "${served_model}=${ADAPTER_PATH}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    "${extra_args[@]}" \
    >"${log_path}" 2>&1 &
  VLLM_PID=$!
  wait_for_vllm "${served_model}"
}

task_args() {
  local raw="${TASK_IDS// /,}"
  local IFS=","
  read -ra ids <<< "${raw}"
  for id in "${ids[@]}"; do
    if [[ -n "${id}" ]]; then
      printf '%s\n' "${id}"
    fi
  done
}

run_tau2_eval() {
  local served_model="$1"
  local save_to="$2"
  local uv="$3"
  local -a ids=()
  while IFS= read -r id; do
    ids+=("${id}")
  done < <(task_args)

  local agent_args
  agent_args=$(python - <<PY
import json
agent_args = {
    "api_base": "${OPENAI_API_BASE}",
    "api_key": "${OPENAI_API_KEY}",
    "temperature": float("${AGENT_TEMPERATURE}"),
    "max_tokens": int("${AGENT_MAX_TOKENS}"),
}
top_p = "${AGENT_TOP_P}".strip()
if top_p:
    agent_args["top_p"] = float(top_p)
stop_sequence = "${AGENT_STOP_SEQUENCE}"
if stop_sequence:
    agent_args["stop"] = [stop_sequence]
    if "${AGENT_INCLUDE_STOP_STR}" == "1":
        agent_args["extra_body"] = {"include_stop_str_in_output": True}
parallel_tool_calls = "${AGENT_PARALLEL_TOOL_CALLS}".strip().lower()
if parallel_tool_calls in {"0", "false", "no"}:
    agent_args["parallel_tool_calls"] = False
elif parallel_tool_calls in {"1", "true", "yes"}:
    agent_args["parallel_tool_calls"] = True
print(json.dumps(agent_args))
PY
)

  echo "Running tau2: model=${served_model} save_to=${save_to} tasks=${TASK_IDS}"
  (
    cd "${TAU2_ROOT}"
    timeout "${TIMEOUT_SECONDS}" "${uv}" run tau2 run \
      --domain airline \
      --agent llm_agent \
      --user user_simulator \
      --agent-llm "openai/${served_model}" \
      --user-llm "${USER_LLM}" \
      --agent-llm-args "${agent_args}" \
      --num-trials "${NUM_TRIALS}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --max-steps "${MAX_STEPS}" \
      --max-retries "${MAX_RETRIES}" \
      --retry-delay "${RETRY_DELAY}" \
      --seed "${SEED}" \
      --save-to "${save_to}" \
      --auto-resume \
      --task-ids "${ids[@]}"
  )
}

summarize_run() {
  local save_to="$1"
  local out_json="${ROOT}/reports/${save_to}.summary.json"
  python "${ROOT}/scripts/summarize_tau2_results.py" \
    "${TAU2_ROOT}/data/simulations/${save_to}" \
    --out "${out_json}"
}

compare_runs() {
  python "${ROOT}/scripts/compare_tau2_runs.py" \
    "${TAU2_ROOT}/data/simulations/${BASE_SAVE_TO}" \
    "${TAU2_ROOT}/data/simulations/${SFT_SAVE_TO}" \
    --no-prm \
    --out-csv "${ROOT}/reports/${COMPARE_STEM}.csv" \
    --out-md "${ROOT}/reports/${COMPARE_STEM}.md" \
    --out-json "${ROOT}/reports/${COMPARE_STEM}.json"
}

main() {
  require_path "${TAU2_ROOT}" "tau2-bench checkout"
  require_path "${BASE_MODEL}" "base model"
  require_path "${ADAPTER_PATH}" "SFT LoRA adapter"
  mkdir -p "${ROOT}/reports" "${ROOT}/outputs/vllm_logs"

  echo "Base model: ${BASE_MODEL}"
  echo "Adapter:    ${ADAPTER_PATH}"
  echo "Tasks:      ${TASK_IDS}"
  echo "Trials:     ${NUM_TRIALS}"
  echo "Agent max tokens: ${AGENT_MAX_TOKENS}"
  echo "Agent temperature: ${AGENT_TEMPERATURE}"
  echo "Agent top_p: ${AGENT_TOP_P:-<default>}"
  echo "Agent stop sequence: ${AGENT_STOP_SEQUENCE:-<none>}"
  echo "Agent include stop string: ${AGENT_INCLUDE_STOP_STR}"
  echo "Agent parallel tool calls: ${AGENT_PARALLEL_TOOL_CALLS}"
  echo "vLLM PYTHONOPTIMIZE: ${VLLM_PYTHONOPTIMIZE}"
  echo "vLLM TORCHDYNAMO_DISABLE: ${VLLM_TORCHDYNAMO_DISABLE}"
  echo "vLLM FlashInfer sampler: ${VLLM_USE_FLASHINFER_SAMPLER}"
  echo "vLLM enforce eager: ${VLLM_ENFORCE_EAGER}"
  echo "vLLM FlashInfer autotune: ${VLLM_ENABLE_FLASHINFER_AUTOTUNE}"
  echo "Run base: ${RUN_BASE}"
  echo "Run SFT:  ${RUN_SFT}"

  if [[ "${RUN_BASE}" != "1" && "${RUN_SFT}" != "1" ]]; then
    echo "Nothing to run: set RUN_BASE=1 or RUN_SFT=1." >&2
    exit 2
  fi

  local uv
  uv="$(find_uv)"
  maybe_install_vllm
  check_user_llm_env

  if [[ "${RUN_BASE}" == "1" ]]; then
    start_vllm_base "${BASE_SERVED_MODEL}" "${ROOT}/outputs/vllm_logs/${BASE_SAVE_TO}.log"
    run_tau2_eval "${BASE_SERVED_MODEL}" "${BASE_SAVE_TO}" "${uv}"
    summarize_run "${BASE_SAVE_TO}"
  else
    echo "Skipping base run because RUN_BASE=${RUN_BASE}."
  fi

  if [[ "${RUN_SFT}" == "1" ]]; then
    start_vllm_sft "${SFT_SERVED_MODEL}" "${ROOT}/outputs/vllm_logs/${SFT_SAVE_TO}.log"
    run_tau2_eval "${SFT_SERVED_MODEL}" "${SFT_SAVE_TO}" "${uv}"
    summarize_run "${SFT_SAVE_TO}"
  else
    echo "Skipping SFT run because RUN_SFT=${RUN_SFT}."
  fi

  if [[ "${RUN_BASE}" == "1" && "${RUN_SFT}" == "1" ]]; then
    compare_runs
  else
    echo "Skipping comparison because only one side was run."
  fi
  echo "Done."
  if [[ "${RUN_BASE}" == "1" ]]; then
    echo "Base summary: ${ROOT}/reports/${BASE_SAVE_TO}.summary.json"
  fi
  if [[ "${RUN_SFT}" == "1" ]]; then
    echo "SFT summary:  ${ROOT}/reports/${SFT_SAVE_TO}.summary.json"
  fi
  if [[ "${RUN_BASE}" == "1" && "${RUN_SFT}" == "1" ]]; then
    echo "Compare:      ${ROOT}/reports/${COMPARE_STEM}.md"
  fi
}

main "$@"
