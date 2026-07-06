#!/usr/bin/env bash
set -euo pipefail

# Phase2H mixed dialogue/tool-policy SFT.
# This trains on assistant text turns, grounded single-tool turns, and
# sequentialized one-call targets converted from gold parallel tool-call turns.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-2048}"
STEPS="${STEPS:-260}"
LR="${LR:-1.5e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
EVAL_PROBES="${EVAL_PROBES:-64}"

STEM="tau2_airline_mixed_dialogue_tool_policy_v1_${MAX_SAMPLE_TOKENS}"
RUN_NAME="sft_mixed_dialogue_tool_policy_v1_qwen25_7b_qlora_${MAX_SAMPLE_TOKENS}"

python scripts/build_mixed_dialogue_tool_policy_dataset.py \
  --tokenizer-model "${BASE_MODEL}" \
  --max-sample-tokens "${MAX_SAMPLE_TOKENS}" \
  --output-stem "${STEM}" \
  --report "reports/mixed_dialogue_tool_policy_dataset_v1_${MAX_SAMPLE_TOKENS}.md" \
  --manifest "data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_manifest_v1_${MAX_SAMPLE_TOKENS}.json"

python scripts/train_sft_smoke.py \
  --train "data/mixed_policy/${STEM}_train.jsonl" \
  --valid "data/mixed_policy/${STEM}_valid.jsonl" \
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
  --eval-batches 24 \
  --eval-every 20 \
  --log-every 5 \
  --rolling-window 10 \
  --lr "${LR}" \
  --warmup-steps 12 \
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

python scripts/evaluate_mixed_policy_behavior.py \
  --base-model "${BASE_MODEL}" \
  --data "data/mixed_policy/${STEM}_heldout.jsonl" \
  --adapter "${RUN_NAME}=outputs/${RUN_NAME}/checkpoint" \
  --max-probes "${EVAL_PROBES}" \
  --max-seq-len "${MAX_SAMPLE_TOKENS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --stop-sequence "</tool_call>" \
  --output-dir "outputs/behavior_${RUN_NAME}" \
  --report "reports/behavior_${RUN_NAME}.md" \
  --fp16

