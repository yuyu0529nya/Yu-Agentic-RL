#!/usr/bin/env bash
set -uo pipefail
# Build the GRPO env on the COMPANY 2x5090 box, ENTIRELY on /data (system disk is 96% full).
# Replicates the weste/round-3 cu130 stack (Blackwell sm_120 needs cu130 torch 2.11).
# ALL caches redirected to /data so nothing grows the system disk.
ROOT=$HOME/agentic-rl
export HF_HOME="$ROOT/hf-cache"
export PIP_CACHE_DIR="$ROOT/.pipcache"
export MODELSCOPE_CACHE="$ROOT/.mscache"
export TRITON_CACHE_DIR="$ROOT/.triton"
export TMPDIR="$ROOT/tmp"; mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$MODELSCOPE_CACHE" "$TRITON_CACHE_DIR"
cd "$ROOT"

echo "######## SETUP_ENV START $(date) ROOT=$ROOT ########"
echo "=== which python3 ==="; which python3; python3 --version

# venv on /data
if [ ! -x "$ROOT/venv/bin/python" ]; then
  python3 -m venv "$ROOT/venv" || { echo "venv create FAILED"; exit 1; }
fi
source "$ROOT/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel 2>&1 | tail -2

echo "=== [1/4] torch 2.11 cu130 (Blackwell 5090) ==="
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -4

echo "=== [2/4] vllm + training stack (torch already satisfied, won't downgrade) ==="
pip install vllm==0.23.0 2>&1 | tail -4
pip install transformers==4.57.6 peft==0.19.1 accelerate==1.14.0 datasets==5.0.0 \
  bitsandbytes modelscope==1.37.1 2>&1 | tail -4
pip install "setuptools<81" 2>&1 | tail -1   # vllm needs <81 (weste lesson)

echo "=== [3/4] sanity: torch sees 2x5090 + key imports ==="
python - <<'PY' 2>&1 | tail -8
import torch
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  gpu", i, torch.cuda.get_device_name(i), "cap", torch.cuda.get_device_capability(i))
import vllm, transformers, peft
print("vllm", vllm.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
PY

echo "=== [4/4] cache locations (confirm all on /data) ==="
echo "HF_HOME=$HF_HOME"; echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"; echo "MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
du -sh "$ROOT" 2>/dev/null
echo "######## SETUP_ENV DONE $(date) ########"
