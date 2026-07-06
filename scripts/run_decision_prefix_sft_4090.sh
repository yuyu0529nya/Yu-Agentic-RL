#!/usr/bin/env bash
set -euo pipefail

# Decision-Prefix SFT: train the next assistant action, either natural text
# ask/confirm or one executable tool call.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-2048}"
STEPS="${STEPS:-220}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SEED="${SEED:-7}"
MAX_TARGET_CONTENT_CHARS="${MAX_TARGET_CONTENT_CHARS:-1200}"

STEM="tau2_airline_decision_prefix_v1_${MAX_SAMPLE_TOKENS}"
RUN_NAME="sft_decision_prefix_v1_qwen25_7b_qlora_${MAX_SAMPLE_TOKENS}"

python scripts/build_decision_prefix_dataset.py \
  --tokenizer-model "${BASE_MODEL}" \
  --max-sample-tokens "${MAX_SAMPLE_TOKENS}" \
  --max-target-content-chars "${MAX_TARGET_CONTENT_CHARS}" \
  --output-stem "${STEM}" \
  --report "reports/decision_prefix_dataset_v1_${MAX_SAMPLE_TOKENS}.md" \
  --manifest "data/decision_prefix/tau2_airline_decision_prefix_manifest_v1_${MAX_SAMPLE_TOKENS}.json"

python scripts/train_sft_smoke.py \
  --train "data/decision_prefix/${STEM}_train.jsonl" \
  --valid "data/decision_prefix/${STEM}_valid.jsonl" \
  --tokenizer-model "${BASE_MODEL}" \
  --pretrained-model "${BASE_MODEL}" \
  --model-init pretrained \
  --no-local-files-only \
  --output-dir "outputs/${RUN_NAME}" \
  --report "reports/${RUN_NAME}.md" \
  --max-seq-len "${MAX_SAMPLE_TOKENS}" \
  --truncation-side right \
  --max-train-samples 100000 \
  --max-valid-samples 100000 \
  --batch-size 1 \
  --steps "${STEPS}" \
  --eval-batches 16 \
  --eval-every 20 \
  --log-every 5 \
  --rolling-window 10 \
  --lr "${LR}" \
  --warmup-steps 10 \
  --fp16 \
  --load-in-4bit \
  --bnb-4bit-quant-type nf4 \
  --bnb-4bit-compute-dtype bfloat16 \
  --bnb-4bit-use-double-quant \
  --lora \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --shuffle

python scripts/evaluate_sft_behavior.py \
  --base-model "${BASE_MODEL}" \
  --data "data/decision_prefix/${STEM}_heldout.jsonl" \
  --adapter "${RUN_NAME}=outputs/${RUN_NAME}/checkpoint" \
  --max-probes 32 \
  --include-later-turns \
  --max-seq-len "${MAX_SAMPLE_TOKENS}" \
  --max-new-tokens 160 \
  --stop-sequence "</tool_call>" \
  --output-dir "outputs/behavior_${RUN_NAME}" \
  --report "reports/behavior_${RUN_NAME}.md" \
  --fp16
