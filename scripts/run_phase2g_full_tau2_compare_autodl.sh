#!/usr/bin/env bash
set -euo pipefail

# Phase2G full tau2 rollout comparison:
#   Base Qwen2.5-7B vs Slot-Grounded v3 vs Single-Tool Protocol v4.
#
# AUTO_SHUTDOWN is kept for compatibility, but shutdown is suppressed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${ROOT}/scripts/run_tau2_base_vs_sft_vllm_autodl.sh"

export PATH="/root/miniconda3/bin:/root/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

RUN_TAG="${RUN_TAG:-phase2g_full_tau2_$(date '+%Y%m%d_%H%M%S')}"
TASK_IDS="${TASK_IDS:-2,16,18,25,44}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
SEED="${SEED:-900}"

BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/qwen25-7b-instruct}"
USER_LLM="${USER_LLM:-anthropic/glm-5.1}"

# 8192 was previously too tight for task 16. 12288 is a safer first pass on a
# 24GB 4090D while still leaving more headroom than 16K.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-128}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.0}"
AGENT_TOP_P="${AGENT_TOP_P:-}"
AGENT_STOP_SEQUENCE="${AGENT_STOP_SEQUENCE:-</tool_call>}"
AGENT_INCLUDE_STOP_STR="${AGENT_INCLUDE_STOP_STR:-1}"
AGENT_PARALLEL_TOOL_CALLS="${AGENT_PARALLEL_TOOL_CALLS:-false}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_BASE="${RUN_BASE:-1}"
RUN_V3="${RUN_V3:-1}"
RUN_V4="${RUN_V4:-1}"

BASE_SERVED_MODEL="${BASE_SERVED_MODEL:-qwen25-7b-base}"
V3_SERVED_MODEL="${V3_SERVED_MODEL:-qwen25-7b-slot-grounded-v3}"
V4_SERVED_MODEL="${V4_SERVED_MODEL:-qwen25-7b-single-tool-v4}"

BASE_SAVE_TO="${BASE_SAVE_TO:-airline_qwen25_7b_base_${RUN_TAG}}"
V3_SAVE_TO="${V3_SAVE_TO:-airline_qwen25_7b_slot_grounded_v3_${RUN_TAG}}"
V4_SAVE_TO="${V4_SAVE_TO:-airline_qwen25_7b_single_tool_v4_${RUN_TAG}}"
COMPARE_STEM="${COMPARE_STEM:-airline_qwen25_7b_base_vs_v3_vs_v4_${RUN_TAG}}"

V3_TARBALL="${V3_TARBALL:-${ROOT}/autodl_artifacts/slot_grounded_v3_1536_4090_20260615_201005/yuyu_slot_grounded_action_prefix_artifacts_20260615_201013.tar.gz}"
V4_TARBALL="${V4_TARBALL:-${ROOT}/autodl_artifacts/phase2f_single_tool_protocol_v4_4090_20260615_224720/yuyu_single_tool_protocol_v4_4090_20260615_224720.tar.gz}"
V3_ADAPTER_PATH="${V3_ADAPTER_PATH:-${ROOT}/outputs/sft_action_prefix_slot_grounded_v3_qwen25_7b_qlora_1536/checkpoint}"
V4_ADAPTER_PATH="${V4_ADAPTER_PATH:-${ROOT}/outputs/sft_single_tool_protocol_v4_qwen25_7b_qlora_1536/checkpoint}"

echo_header() {
  echo
  echo "========== $* =========="
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

adapter_ready() {
  local path="$1"
  [[ -f "${path}/adapter_model.safetensors" && -f "${path}/adapter_config.json" ]]
}

ensure_adapter() {
  local name="$1"
  local tarball="$2"
  local adapter_path="$3"

  if adapter_ready "${adapter_path}"; then
    echo "${name} adapter ready: ${adapter_path}"
    return
  fi

  require_file "${tarball}" "${name} artifact tarball"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1; ${name} adapter would be extracted from ${tarball}"
    return
  fi

  echo "${name} adapter not extracted; unpacking ${tarball}"
  tar -xzf "${tarball}" -C "${ROOT}"

  if ! adapter_ready "${adapter_path}"; then
    echo "${name} adapter still missing after extraction: ${adapter_path}" >&2
    exit 3
  fi
  echo "${name} adapter ready: ${adapter_path}"
}

run_existing_runner() {
  local leg_name="$1"
  local run_base_flag="$2"
  local run_sft_flag="$3"
  local adapter_path="$4"
  local sft_served_model="$5"
  local base_save_to="$6"
  local sft_save_to="$7"

  echo_header "RUN ${leg_name}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1; would run ${leg_name}"
    return
  fi

  BASE_MODEL="${BASE_MODEL}" \
  USER_LLM="${USER_LLM}" \
  TASK_IDS="${TASK_IDS}" \
  NUM_TRIALS="${NUM_TRIALS}" \
  MAX_STEPS="${MAX_STEPS}" \
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
  SEED="${SEED}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS}" \
  AGENT_TEMPERATURE="${AGENT_TEMPERATURE}" \
  AGENT_TOP_P="${AGENT_TOP_P}" \
  AGENT_STOP_SEQUENCE="${AGENT_STOP_SEQUENCE}" \
  AGENT_INCLUDE_STOP_STR="${AGENT_INCLUDE_STOP_STR}" \
  AGENT_PARALLEL_TOOL_CALLS="${AGENT_PARALLEL_TOOL_CALLS}" \
  TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
  INSTALL_VLLM="${INSTALL_VLLM}" \
  RUN_BASE="${run_base_flag}" \
  RUN_SFT="${run_sft_flag}" \
  BASE_SERVED_MODEL="${BASE_SERVED_MODEL}" \
  SFT_SERVED_MODEL="${sft_served_model}" \
  ADAPTER_PATH="${adapter_path}" \
  BASE_SAVE_TO="${base_save_to}" \
  SFT_SAVE_TO="${sft_save_to}" \
  COMPARE_STEM="${COMPARE_STEM}_${leg_name}" \
  bash "${RUNNER}"
}

simulation_path() {
  local save_to="$1"
  echo "${ROOT}/third_party/tau2-bench/data/simulations/${save_to}"
}

require_result() {
  local save_to="$1"
  local path
  path="$(simulation_path "${save_to}")"
  if [[ ! -f "${path}/results.json" ]]; then
    echo "Missing results for ${save_to}: ${path}/results.json" >&2
    exit 4
  fi
}

write_phase2g_report() {
  local compare_json="${ROOT}/reports/${COMPARE_STEM}.json"
  local out_md="${ROOT}/reports/phase2g_full_tau2_compare_${RUN_TAG}.md"

  python - "${compare_json}" "${out_md}" <<'PY'
import json
import os
import sys
from pathlib import Path

compare_json = Path(sys.argv[1])
out_md = Path(sys.argv[2])
rows = json.loads(compare_json.read_text(encoding="utf-8"))

def cell(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")

columns = ["run", "agent_llm", "tasks", "sims", "success_count", "pass^1", "avg_reward", "db_match", "db_mismatch"]
lines = [
    "# Phase2G Full Tau2 Rollout Comparison",
    "",
    "## Configuration",
    "",
    f"- run tag: `{os.environ.get('RUN_TAG', '')}`",
    f"- tasks: `{os.environ.get('TASK_IDS', '')}`",
    f"- trials: `{os.environ.get('NUM_TRIALS', '')}`",
    f"- max steps: `{os.environ.get('MAX_STEPS', '')}`",
    f"- max model len: `{os.environ.get('MAX_MODEL_LEN', '')}`",
    f"- agent max tokens: `{os.environ.get('AGENT_MAX_TOKENS', '')}`",
    f"- agent stop sequence: `{os.environ.get('AGENT_STOP_SEQUENCE', '')}`",
    f"- user simulator: `{os.environ.get('USER_LLM', '')}`",
    "",
    "## Results",
    "",
    "| " + " | ".join(columns) + " |",
    "| " + " | ".join(["---"] * len(columns)) + " |",
]
for row in rows:
    lines.append("| " + " | ".join(cell(row.get(column)) for column in columns) + " |")

lines += [
    "",
    "## Interpretation Checklist",
    "",
    "- If v4 improves pass^1 over both base and v3, Phase2F protocol gains transferred to full tau2.",
    "- If v4 improves wrapper/single-call behavior but pass^1 does not rise, inspect failed trajectories for parser integration, natural-language turns, or policy decisions.",
    "- If base wins on some tasks, compare whether SFT over-specialized to tool-call-only behavior.",
    "",
    "## Source Files",
    "",
    f"- comparison json: `{compare_json}`",
]

out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote: {out_md}")
PY
}

finish() {
  local code=$?
  echo "[$(date '+%F %T')] Phase2G EXIT_CODE=${code}"
  if [[ "${AUTO_SHUTDOWN}" == "1" ]]; then
    echo "AUTO_SHUTDOWN=1 requested, but shutdown is suppressed."
    echo "Shutdown suppressed by no-auto-shutdown policy."
  else
    echo "AUTO_SHUTDOWN=0; instance left running."
  fi
}
trap finish EXIT

main() {
  echo_header "PHASE2G CONFIG"
  echo "Root: ${ROOT}"
  echo "Run tag: ${RUN_TAG}"
  echo "Tasks: ${TASK_IDS}"
  echo "Trials: ${NUM_TRIALS}"
  echo "Base model: ${BASE_MODEL}"
  echo "User LLM: ${USER_LLM}"
  echo "Max model len: ${MAX_MODEL_LEN}"
  echo "Agent max tokens: ${AGENT_MAX_TOKENS}"
  echo "Agent stop sequence: ${AGENT_STOP_SEQUENCE}"
  echo "Run base/v3/v4: ${RUN_BASE}/${RUN_V3}/${RUN_V4}"
  echo "Dry run: ${DRY_RUN}"

  require_file "${RUNNER}" "base-vs-sft runner"
  require_dir "${ROOT}/third_party/tau2-bench" "tau2-bench checkout"
  if [[ "${DRY_RUN}" == "1" && ! -d "${BASE_MODEL}" ]]; then
    echo "DRY_RUN=1; warning: base model is not present yet: ${BASE_MODEL}" >&2
  else
    require_dir "${BASE_MODEL}" "base model"
  fi
  mkdir -p "${ROOT}/reports" "${ROOT}/outputs/vllm_logs"

  if [[ "${RUN_V3}" == "1" || "${RUN_BASE}" == "1" ]]; then
    ensure_adapter "v3" "${V3_TARBALL}" "${V3_ADAPTER_PATH}"
  fi
  if [[ "${RUN_V4}" == "1" ]]; then
    ensure_adapter "v4" "${V4_TARBALL}" "${V4_ADAPTER_PATH}"
  fi

  if [[ "${RUN_BASE}" == "1" ]]; then
    run_existing_runner "base" "1" "0" "${V3_ADAPTER_PATH}" "${V3_SERVED_MODEL}" "${BASE_SAVE_TO}" "${V3_SAVE_TO}"
  fi
  if [[ "${RUN_V3}" == "1" ]]; then
    run_existing_runner "v3" "0" "1" "${V3_ADAPTER_PATH}" "${V3_SERVED_MODEL}" "${BASE_SAVE_TO}" "${V3_SAVE_TO}"
  fi
  if [[ "${RUN_V4}" == "1" ]]; then
    run_existing_runner "v4" "0" "1" "${V4_ADAPTER_PATH}" "${V4_SERVED_MODEL}" "${BASE_SAVE_TO}" "${V4_SAVE_TO}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1; skipping final comparison."
    return
  fi

  local -a compare_paths=()
  if [[ "${RUN_BASE}" == "1" ]]; then
    require_result "${BASE_SAVE_TO}"
    compare_paths+=("$(simulation_path "${BASE_SAVE_TO}")")
  fi
  if [[ "${RUN_V3}" == "1" ]]; then
    require_result "${V3_SAVE_TO}"
    compare_paths+=("$(simulation_path "${V3_SAVE_TO}")")
  fi
  if [[ "${RUN_V4}" == "1" ]]; then
    require_result "${V4_SAVE_TO}"
    compare_paths+=("$(simulation_path "${V4_SAVE_TO}")")
  fi

  if (( ${#compare_paths[@]} < 2 )); then
    echo "Need at least two completed runs for comparison." >&2
    exit 5
  fi

  echo_header "COMPARE"
  python "${ROOT}/scripts/compare_tau2_runs.py" \
    "${compare_paths[@]}" \
    --no-prm \
    --out-csv "${ROOT}/reports/${COMPARE_STEM}.csv" \
    --out-md "${ROOT}/reports/${COMPARE_STEM}.md" \
    --out-json "${ROOT}/reports/${COMPARE_STEM}.json"
  write_phase2g_report

  echo "Done."
  echo "Phase2G report: ${ROOT}/reports/phase2g_full_tau2_compare_${RUN_TAG}.md"
}

main "$@"
