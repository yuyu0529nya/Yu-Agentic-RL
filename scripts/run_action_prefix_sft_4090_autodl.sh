#!/usr/bin/env bash
set -euo pipefail

# AutoDL wrapper for the first rented-GPU Action-Prefix SFT run.
# It installs a stable training stack, downloads Qwen2.5 to the data disk,
# launches the 4090 QLoRA run, and leaves the instance alive for inspection.
# AUTO_SHUTDOWN is kept for compatibility, but shutdown is suppressed.

export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib:/root/miniconda3/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
SAFETY_SHUTDOWN_MINUTES="${SAFETY_SHUTDOWN_MINUTES:-0}"
ARTIFACT_TAR="${ARTIFACT_TAR:-/root/autodl-tmp/yuyu_action_prefix_artifacts_$(date '+%Y%m%d_%H%M%S').tar.gz}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/qwen25-7b-instruct}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-2048}"
STEPS="${STEPS:-120}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
export MODEL_REPO MODEL_DIR MAX_SAMPLE_TOKENS STEPS LR LORA_R LORA_ALPHA

if [[ "${SAFETY_SHUTDOWN_MINUTES}" != "0" ]]; then
  /usr/bin/shutdown -c >/dev/null 2>&1 || true
  echo "[$(date '+%F %T')] Safety shutdown suppressed by no-auto-shutdown policy."
fi

package_artifacts() {
  echo "[$(date '+%F %T')] Packaging artifacts to ${ARTIFACT_TAR}"
  tar -czf "${ARTIFACT_TAR}" \
    run_4090.log \
    reports/sft_action_prefix_v2_qwen25_7b_qlora_2048.md \
    reports/behavior_sft_action_prefix_v2_qwen25_7b_qlora_2048.md \
    outputs/sft_action_prefix_v2_qwen25_7b_qlora_2048/metrics.json \
    outputs/sft_action_prefix_v2_qwen25_7b_qlora_2048/loss_trace.csv \
    outputs/sft_action_prefix_v2_qwen25_7b_qlora_2048/checkpoint \
    outputs/behavior_sft_action_prefix_v2_qwen25_7b_qlora_2048/summary.json \
    outputs/behavior_sft_action_prefix_v2_qwen25_7b_qlora_2048/base_behavior.json \
    outputs/behavior_sft_action_prefix_v2_qwen25_7b_qlora_2048/sft_action_prefix_v2_qwen25_7b_qlora_2048_behavior.json \
    2>/dev/null || true
}

finish() {
  code=$?
  echo "[$(date '+%F %T')] EXIT_CODE=${code}"
  if [[ "${AUTO_SHUTDOWN}" == "1" ]]; then
    package_artifacts
    echo "[$(date '+%F %T')] AUTO_SHUTDOWN=1 requested, but shutdown is suppressed."
    echo "[$(date '+%F %T')] Shutdown suppressed by no-auto-shutdown policy."
  else
    echo "[$(date '+%F %T')] AUTO_SHUTDOWN=0; instance left running for inspection/download."
  fi
}
trap finish EXIT

echo "[$(date '+%F %T')] AutoDL run started"
pwd
nvidia-smi || true
python -V
python -m pip --version

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-4090.txt

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
bash scripts/run_action_prefix_sft_4090.sh
