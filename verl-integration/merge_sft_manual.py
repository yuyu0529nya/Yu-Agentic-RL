"""Manual bf16 LoRA merge — pure torch, no peft/bitsandbytes (which hangs the
QLoRA adapter on a CPU-only box). Reads the LoRA A/B matrices directly and adds
delta = (B @ A) * (alpha/r) to each target weight. Low-rank, so cheap.

Usage:
    python merge_sft_manual.py --base <7b> --adapter <dir> --out <dir>
"""

import argparse
import json
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", help="cpu (no-GPU, but CPU ops segfault on this box) or cuda (fast, stable)")
    args = ap.parse_args()
    dev = args.device

    cfg = json.load(open(f"{args.adapter}/adapter_config.json"))
    scaling = cfg["lora_alpha"] / cfg["r"]
    print(f"[merge] r={cfg['r']} alpha={cfg['lora_alpha']} scaling={scaling}", flush=True)

    t0 = time.time()
    print(f"[merge] loading base bf16 (materialized in RAM, no mmap): {args.base}", flush=True)
    # low_cpu_mem_usage=False → weights are normal writable tensors, not mmap
    # (in-place += on an mmap'd weight segfaults on this box).
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, low_cpu_mem_usage=(dev != "cpu"),
    ).to(dev)
    lora = load_file(f"{args.adapter}/adapter_model.safetensors", device=dev)
    n_expected = len(lora) // 2
    print(f"[merge] base + lora loaded ({time.time() - t0:.0f}s). merging {n_expected} modules ...", flush=True)

    merged = 0
    for name, module in model.named_modules():
        a_key = f"base_model.model.{name}.lora_A.weight"
        b_key = f"base_model.model.{name}.lora_B.weight"
        if a_key in lora and b_key in lora:
            A = lora[a_key].to(torch.float32)   # (r, in)
            B = lora[b_key].to(torch.float32)   # (out, r)
            delta = (B @ A) * scaling           # (out, in)
            # non-in-place: build a fresh tensor, then assign (avoids writing mmap)
            new_w = (module.weight.data.to(torch.float32) + delta).to(torch.bfloat16)
            module.weight.data = new_w
            merged += 1
            if merged % 50 == 0:
                print(f"[merge]   {merged}/{n_expected} modules merged ({time.time() - t0:.0f}s)", flush=True)
    print(f"[merge] merged {merged} modules ({time.time() - t0:.0f}s). saving -> {args.out}", flush=True)
    assert merged == len(lora) // 2, f"merged {merged} != expected {len(lora)//2} (key-name mismatch!)"

    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"[merge] DONE in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
