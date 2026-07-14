"""Merge the distillation-SFT LoRA adapter into the Qwen2.5-7B base, on CPU.

Produces a plain SFT'd 7B that the veRL GRPO stage treats as a normal base
model (then trains a fresh LoRA on top). Runs GPU-free: loads bf16 on CPU,
peft merge_and_unload, saves. ~10-20 min for 7B on CPU.

Usage:
    python merge_sft_adapter.py \
        --base   /root/autodl-tmp/verl-work/models/qwen25-7b-instruct \
        --adapter /root/autodl-tmp/verl-work/sft_assets/sft_adapter \
        --out    /root/autodl-tmp/verl-work/models/qwen25-7b-sft-airline
"""

import argparse
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[merge] loading base bf16 on CPU: {args.base}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    print(f"[merge] base loaded ({time.time() - t0:.0f}s). applying SFT LoRA: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter, device_map="cpu")
    print("[merge] merge_and_unload ...", flush=True)
    model = model.merge_and_unload()
    print(f"[merge] merged ({time.time() - t0:.0f}s). saving -> {args.out}", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # tokenizer travels with the model
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"[merge] DONE in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
