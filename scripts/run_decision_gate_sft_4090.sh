#!/usr/bin/env bash
set -euo pipefail

# Phase2J decision-only gate SFT.
# The adapter only learns whether the next assistant action should be
# `assistant_text` or `tool_call`; tool JSON generation stays in Phase2H.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-2048}"
STEPS="${STEPS:-180}"
LR="${LR:-1.0e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
EVAL_PROBES="${EVAL_PROBES:-128}"
USE_4BIT="${USE_4BIT:-0}"

RUN_VARIANT="fp16_lora"
TRAIN_EXTRA_ARGS=(--fp16)
if [[ "${USE_4BIT}" == "1" ]]; then
  RUN_VARIANT="qlora"
  TRAIN_EXTRA_ARGS+=(
    --load-in-4bit
    --bnb-4bit-quant-type nf4
    --bnb-4bit-compute-dtype bfloat16
    --bnb-4bit-use-double-quant
  )
fi

STEM="tau2_airline_decision_gate_v1_${MAX_SAMPLE_TOKENS}"
RUN_NAME="sft_decision_gate_v1_qwen25_7b_${RUN_VARIANT}_${MAX_SAMPLE_TOKENS}"

python scripts/build_decision_gate_dataset.py \
  --output-stem "${STEM}" \
  --report "reports/decision_gate_dataset_v1_${MAX_SAMPLE_TOKENS}.md" \
  --manifest "data/decision_gate/tau2_airline_decision_gate_manifest_v1_${MAX_SAMPLE_TOKENS}.json"

python scripts/train_sft_smoke.py \
  --train "data/decision_gate/${STEM}_train.jsonl" \
  --valid "data/decision_gate/${STEM}_valid.jsonl" \
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
  --warmup-steps 10 \
  "${TRAIN_EXTRA_ARGS[@]}" \
  --lora \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --shuffle

python scripts/evaluate_decision_gate_behavior.py \
  --base-model "${BASE_MODEL}" \
  --data "data/decision_gate/${STEM}_heldout.jsonl" \
  --adapter "${RUN_NAME}=outputs/${RUN_NAME}/checkpoint" \
  --max-probes "${EVAL_PROBES}" \
  --max-seq-len "${MAX_SAMPLE_TOKENS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "outputs/behavior_${RUN_NAME}" \
  --report "reports/behavior_${RUN_NAME}.md" \
  --fp16
