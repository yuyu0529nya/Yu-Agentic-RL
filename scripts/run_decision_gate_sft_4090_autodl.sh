#!/usr/bin/env bash
set -euo pipefail

# AutoDL wrapper for Phase2J decision-only gate SFT.
# This wrapper leaves the instance running after the job.

export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/qwen25-7b-instruct}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-2048}"
STEPS="${STEPS:-180}"
LR="${LR:-1.0e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
EVAL_PROBES="${EVAL_PROBES:-128}"
USE_4BIT="${USE_4BIT:-0}"
INSTALL_TORCH_CU128="${INSTALL_TORCH_CU128:-1}"
VENV_DIR="${VENV_DIR:-/root/yuyu_venv5090}"
TMPDIR="${TMPDIR:-/root/autodl-tmp/yuyu_pip_tmp}"
RUN_VARIANT="fp16_lora"
if [[ "${USE_4BIT}" == "1" ]]; then
  RUN_VARIANT="qlora"
fi
RUN_NAME="sft_decision_gate_v1_qwen25_7b_${RUN_VARIANT}_${MAX_SAMPLE_TOKENS}"
STEM="tau2_airline_decision_gate_v1_${MAX_SAMPLE_TOKENS}"
ARTIFACT_TAR="${ARTIFACT_TAR:-/root/autodl-tmp/yuyu_phase2j_decision_gate_artifacts_$(date '+%Y%m%d_%H%M%S').tar.gz}"
export MODEL_REPO MODEL_DIR MAX_SAMPLE_TOKENS STEPS LR LORA_R LORA_ALPHA MAX_NEW_TOKENS EVAL_PROBES USE_4BIT

package_artifacts() {
  echo "[$(date '+%F %T')] Packaging artifacts to ${ARTIFACT_TAR}"
  tar -czf "${ARTIFACT_TAR}" \
    run_decision_gate_4090.log \
    "reports/decision_gate_dataset_v1_${MAX_SAMPLE_TOKENS}.md" \
    "reports/${RUN_NAME}.md" \
    "reports/behavior_${RUN_NAME}.md" \
    "data/decision_gate/tau2_airline_decision_gate_manifest_v1_${MAX_SAMPLE_TOKENS}.json" \
    "data/decision_gate/${STEM}_train.jsonl" \
    "data/decision_gate/${STEM}_valid.jsonl" \
    "data/decision_gate/${STEM}_heldout.jsonl" \
    "outputs/${RUN_NAME}/metrics.json" \
    "outputs/${RUN_NAME}/loss_trace.csv" \
    "outputs/${RUN_NAME}/checkpoint" \
    "outputs/behavior_${RUN_NAME}/summary.json" \
    "outputs/behavior_${RUN_NAME}/base_decision_gate_behavior.json" \
    "outputs/behavior_${RUN_NAME}/${RUN_NAME}_decision_gate_behavior.json" \
    2>/dev/null || true
}

finish() {
  code=$?
  echo "[$(date '+%F %T')] EXIT_CODE=${code}"
  package_artifacts
  echo "[$(date '+%F %T')] Run finished; instance left running for inspection/download."
}
trap finish EXIT

echo "[$(date '+%F %T')] AutoDL Phase2J decision gate run started"
pwd
nvidia-smi || true
mkdir -p "${TMPDIR}"
export TMPDIR

if [[ -n "${VENV_DIR}" ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[$(date '+%F %T')] Creating venv at ${VENV_DIR}"
    python -m venv "${VENV_DIR}"
  fi
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi

python -V
python -m pip --version

python -m pip install --no-cache-dir -U pip setuptools wheel
if [[ "${INSTALL_TORCH_CU128}" == "1" ]]; then
  python -m pip install --no-cache-dir --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
fi
python -m pip install --no-cache-dir -r requirements-4090.txt

python - <<'PY'
import json
import os
from pathlib import Path

model_repo = os.environ["MODEL_REPO"]
model_dir = Path(os.environ["MODEL_DIR"])
model_dir.mkdir(parents=True, exist_ok=True)

def model_complete(path: Path) -> bool:
    required = ["config.json", "tokenizer_config.json"]
    if not all((path / item).exists() for item in required):
        return False
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return any(path.glob("*.safetensors"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        return False
    return all((path / shard).exists() and (path / shard).stat().st_size > 1024 * 1024 for shard in shards)

if model_complete(model_dir):
    print(f"Model already exists at {model_dir}")
    raise SystemExit(0)

print(f"Downloading {model_repo} to {model_dir}")
try:
    from modelscope import snapshot_download as ms_snapshot_download

    ms_snapshot_download(
        model_id=model_repo,
        local_dir=str(model_dir),
    )
except Exception as ms_exc:  # noqa: BLE001
    print(f"ModelScope download failed: {type(ms_exc).__name__}: {ms_exc}")
    print("Trying HuggingFace fallback...")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_repo,
        local_dir=str(model_dir),
        resume_download=True,
    )

if not model_complete(model_dir):
    raise RuntimeError(f"Model download incomplete at {model_dir}")
PY

echo "[$(date '+%F %T')] Dependencies and model ready"
BASE_MODEL="${MODEL_DIR}" \
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS}" \
STEPS="${STEPS}" \
LR="${LR}" \
LORA_R="${LORA_R}" \
LORA_ALPHA="${LORA_ALPHA}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
EVAL_PROBES="${EVAL_PROBES}" \
USE_4BIT="${USE_4BIT}" \
bash scripts/run_decision_gate_sft_4090.sh 2>&1 | tee run_decision_gate_4090.log
