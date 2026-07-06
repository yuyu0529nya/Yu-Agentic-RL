# Training Environment Check

- Python: `E:\yuyu\.venv-train\Scripts\python.exe`
- Version: `3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 15:03:56) [MSC v.1929 64 bit (AMD64)]`
- Architecture: `64bit`
- Status: `OK`

## Required Packages

| Package | Status | Version |
| --- | --- | --- |
| `numpy` | OK | `2.4.4` |
| `torch` | OK | `2.5.1+cu121` |
| `transformers` | OK | `4.57.6` |
| `datasets` | OK | `5.0.0` |
| `accelerate` | OK | `1.14.0` |
| `peft` | OK | `0.19.1` |
| `trl` | OK | `1.6.0` |
| `tokenizers` | OK | `0.22.2` |
| `jinja2` | OK | `3.1.6` |

## Torch / CUDA

- Torch available: `True`
- CUDA available: `True`
- Torch CUDA version: `12.1`
- Device count: `1`
- Devices: `NVIDIA GeForce RTX 3060 Laptop GPU`

## TRL

- SFTTrainer import: `OK`

## Optional Packages

| Package | Status | Version / Error |
| --- | --- | --- |
| `bitsandbytes` | MISSING | `ModuleNotFoundError: No module named 'bitsandbytes'` |

## Pip Check

- Status: `OK`
```text
No broken requirements found.
```
