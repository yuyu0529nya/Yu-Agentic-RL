#!/usr/bin/env bash
set -euo pipefail

# Single-tool protocol SFT.
# Combines slot-grounded v3 data with executable <tool_call> supervision
# and exactly one target tool call per sample.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-1536}"
STEPS="${STEPS:-160}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

STEM="tau2_airline_single_tool_protocol_v4_${MAX_SAMPLE_TOKENS}"
RUN_NAME="sft_single_tool_protocol_v4_qwen25_7b_qlora_${MAX_SAMPLE_TOKENS}"

python scripts/build_single_tool_protocol_dataset.py \
  --tokenizer-model "${BASE_MODEL}" \
  --max-sample-tokens "${MAX_SAMPLE_TOKENS}" \
  --output-stem "${STEM}" \
  --report "reports/single_tool_protocol_dataset_v4_${MAX_SAMPLE_TOKENS}.md" \
  --manifest "data/tool_call_protocol/tau2_airline_single_tool_protocol_manifest_v4_${MAX_SAMPLE_TOKENS}.json"

python scripts/train_sft_smoke.py \
  --train "data/tool_call_protocol/${STEM}_train.jsonl" \
  --valid "data/tool_call_protocol/${STEM}_valid.jsonl" \
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
  --data "data/tool_call_protocol/${STEM}_heldout.jsonl" \
  --adapter "${RUN_NAME}=outputs/${RUN_NAME}/checkpoint" \
  --max-probes 32 \
  --include-later-turns \
  --max-seq-len "${MAX_SAMPLE_TOKENS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --stop-sequence "</tool_call>" \
  --output-dir "outputs/behavior_${RUN_NAME}" \
  --report "reports/behavior_${RUN_NAME}.md" \
  --fp16
