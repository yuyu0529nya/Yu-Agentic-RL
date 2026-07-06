param(
  [string]$Python = ".\.venv-train\Scripts\python.exe",
  [string]$Report = "reports\training_env_check.md"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not (Test-Path $Python)) {
  Write-Error "Python not found: $Python. Create it with: &'E:\anaconda navicator\python.exe' -m venv .venv-train"
}

$env:TRAINING_ENV_REPORT = $Report

$check = @'
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


REQUIRED = [
    "numpy",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "tokenizers",
    "jinja2",
]

OPTIONAL = [
    "bitsandbytes",
]


def version_of(module):
    return getattr(module, "__version__", "unknown")


def import_result(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        return {
            "name": name,
            "ok": True,
            "version": version_of(module),
            "file": getattr(module, "__file__", None),
            "error": None,
        }
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "version": None,
            "file": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def pip_check() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode == 0, proc.stdout.strip()


def main() -> int:
    required = [import_result(name) for name in REQUIRED]
    optional = [import_result(name) for name in OPTIONAL]

    torch_info = {
        "available": False,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "device_names": [],
        "error": None,
    }
    try:
        import torch

        torch_info.update(
            {
                "available": True,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "device_names": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as exc:
        torch_info["error"] = f"{type(exc).__name__}: {exc}"

    sft_trainer = {"ok": False, "error": None}
    try:
        from trl import SFTConfig, SFTTrainer  # noqa: F401

        sft_trainer["ok"] = True
    except Exception as exc:
        sft_trainer["error"] = f"{type(exc).__name__}: {exc}"

    pip_ok, pip_output = pip_check()
    arch = platform.architecture()[0]
    required_ok = all(item["ok"] for item in required)
    status_ok = required_ok and sft_trainer["ok"] and pip_ok and arch == "64bit"

    report = {
        "python": sys.executable,
        "version": sys.version.replace("\n", " "),
        "architecture": arch,
        "required": required,
        "optional": optional,
        "torch": torch_info,
        "sft_trainer": sft_trainer,
        "pip_check": {"ok": pip_ok, "output": pip_output},
        "status_ok": status_ok,
    }

    report_path = Path(os.environ.get("TRAINING_ENV_REPORT", "reports/training_env_check.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training Environment Check",
        "",
        f"- Python: `{report['python']}`",
        f"- Version: `{report['version']}`",
        f"- Architecture: `{arch}`",
        f"- Status: `{'OK' if status_ok else 'FAILED'}`",
        "",
        "## Required Packages",
        "",
        "| Package | Status | Version |",
        "| --- | --- | --- |",
    ]
    for item in required:
        lines.append(
            f"| `{item['name']}` | {'OK' if item['ok'] else 'FAILED'} | `{item['version'] or item['error']}` |"
        )

    lines.extend(
        [
            "",
            "## Torch / CUDA",
            "",
            f"- Torch available: `{torch_info['available']}`",
            f"- CUDA available: `{torch_info['cuda_available']}`",
            f"- Torch CUDA version: `{torch_info['cuda_version']}`",
            f"- Device count: `{torch_info['device_count']}`",
            f"- Devices: `{', '.join(torch_info['device_names']) or 'none'}`",
            "",
            "## TRL",
            "",
            f"- SFTTrainer import: `{'OK' if sft_trainer['ok'] else sft_trainer['error']}`",
            "",
            "## Optional Packages",
            "",
            "| Package | Status | Version / Error |",
            "| --- | --- | --- |",
        ]
    )
    for item in optional:
        lines.append(
            f"| `{item['name']}` | {'OK' if item['ok'] else 'MISSING'} | `{item['version'] or item['error']}` |"
        )

    lines.extend(
        [
            "",
            "## Pip Check",
            "",
            f"- Status: `{'OK' if pip_ok else 'FAILED'}`",
            "```text",
            pip_output or "No broken requirements found.",
            "```",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote: {report_path.resolve()}")
    return 0 if status_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@

$check | & $Python -
exit $LASTEXITCODE
