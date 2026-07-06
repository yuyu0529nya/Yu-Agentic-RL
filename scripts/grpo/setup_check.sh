#!/bin/bash
# Round-3 company-box preflight. Run this FIRST — it diagnoses what's present so we know
# exactly what to install. Read-only (installs nothing). 5090 = Blackwell sm_120 → needs
# CUDA 12.8+ / torch >=2.7 built for it, and transformers must be 4.57.x (render-mask pin).
echo "=== HOST / GPUs ==="
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv 2>&1
echo "nvcc:"; nvcc --version 2>/dev/null | tail -1 || echo "  (no nvcc — fine if torch ships its own CUDA)"
echo
echo "=== PYTHON ==="
for p in python python3; do command -v $p && $p --version 2>&1; done
command -v conda >/dev/null && { echo "conda envs:"; conda env list; }
echo
echo "=== KEY DEPS (versions / MISSING) ==="
PYBIN="${PYBIN:-python}"
$PYBIN - <<'PY' 2>&1 | tail -25
for m in ['torch','transformers','vllm','peft','bitsandbytes','datasets','accelerate']:
    try:
        x=__import__(m); print(f"{m:14s} {getattr(x,'__version__','?')}")
    except Exception as e:
        print(f"{m:14s} MISSING ({type(e).__name__})")
try:
    import torch
    ok = torch.cuda.is_available()
    print("torch.cuda.is_available:", ok, "| device_count:", torch.cuda.device_count() if ok else 0)
    if ok:
        for i in range(torch.cuda.device_count()):
            print(f"  GPU{i}: cap {torch.cuda.get_device_capability(i)} {torch.cuda.get_device_name(i)}")
        print("  (5090 needs capability (12,0); if torch can't see it, the build is too old for Blackwell)")
    import transformers as t
    print("transformers pin:", t.__version__, "OK" if t.__version__.startswith("4.57") else "<-- MUST be 4.57.6")
except Exception as e:
    print("torch/transformers check failed:", repr(e)[:120])
PY
echo
echo "=== MODEL / DATA / DISK ==="
find "$HOME" /data /workspace -maxdepth 4 -iname '*Qwen2.5-7B*' -type d 2>/dev/null | head
echo "HotpotQA cache:"; find "$HOME" /data -maxdepth 6 -iname '*hotpot*' 2>/dev/null | head -3
echo "disk (home):"; df -h "$HOME" 2>/dev/null | tail -1
echo "PREFLIGHT_DONE"
