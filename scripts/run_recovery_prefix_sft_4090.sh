#!/usr/bin/env bash
set -euo pipefail

# Recovery-Prefix SFT: continue the behavior-policy route with online
# anti-guess corrections from failed tau2 traces.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_SAMPLE_TOKENS="${MAX_SAMPLE_TOKENS:-2048}"
STEPS="${STEPS:-650}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-16}"
DEFAULT_FAILURE_RESULTS="autodl_artifacts/phase2b_decision_prefix_sftonly_2task_4090_20260614_194545/results.json;autodl_artifacts/phase2b_recovery_gate_sftonly_2task_4090_20260614_201913/third_party__tau2-bench__data__simulations__airline_qwen25_7b_sft_phase2b_recovery_gate_sftonly_2task_4090_20260614_201913__results.json;autodl_artifacts/debug_task16_allguards_parser_40steps_4090_20260614_211045/root__autodl-tmp__yuyu__third_party__tau2-bench__data__simulations__airline_qwen25_7b_sft_debug_task16_allguards_parser_40steps_4090_20260614_211045__results.json;autodl_artifacts/debug_task16_allguards_unknown_40steps_160tok_4090_20260614_212838/root__autodl-tmp__yuyu__third_party__tau2-bench__data__simulations__airline_qwen25_7b_sft_debug_task16_allguards_unknown_40steps_160tok_4090_20260614_212838__results.json;autodl_artifacts/phase2b_recovery_v4_allguards_sftonly_2task_4090_20260614_2207/third_party__tau2-bench__data__simulations__airline_qwen25_7b_sft_phase2b_recovery_v4_allguards_sftonly_2task_4090_20260614_2207__results.json;autodl_artifacts/phase2b_recovery_v4_allguards_sftonly_4task_no16_4090_20260614_2219/third_party__tau2-bench__data__simulations__airline_qwen25_7b_sft_phase2b_recovery_v4_allguards_sftonly_4task_no16_4090_20260614_2219__results.json"
FAILURE_RESULTS="${FAILURE_RESULTS:-${DEFAULT_FAILURE_RESULTS}}"
RECOVERY_VERSION="${RECOVERY_VERSION:-v5}"

STEM="tau2_airline_recovery_prefix_${RECOVERY_VERSION}_${MAX_SAMPLE_TOKENS}"
RUN_NAME="sft_recovery_prefix_${RECOVERY_VERSION}_qwen25_7b_qlora_${MAX_SAMPLE_TOKENS}"
IFS=";" read -r -a FAILURE_RESULT_LIST <<< "${FAILURE_RESULTS}"
FAILURE_RESULT_ARGS=()
for failure_result in "${FAILURE_RESULT_LIST[@]}"; do
  if [[ -n "${failure_result}" ]]; then
    FAILURE_RESULT_ARGS+=(--failure-results "${failure_result}")
  fi
done

python scripts/build_recovery_prefix_dataset.py \
  "${FAILURE_RESULT_ARGS[@]}" \
  --oversample-factor "${OVERSAMPLE_FACTOR}" \
  --output-stem "${STEM}" \
  --report "reports/recovery_prefix_dataset_${RECOVERY_VERSION}_${MAX_SAMPLE_TOKENS}.md" \
  --manifest "data/recovery_prefix/tau2_airline_recovery_prefix_manifest_${RECOVERY_VERSION}_${MAX_SAMPLE_TOKENS}.json"

python scripts/train_sft_smoke.py \
  --train "data/recovery_prefix/${STEM}_train.jsonl" \
  --valid "data/recovery_prefix/${STEM}_valid.jsonl" \
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
  --data "data/recovery_prefix/${STEM}_heldout.jsonl" \
  --adapter "${RUN_NAME}=outputs/${RUN_NAME}/checkpoint" \
  --max-probes 32 \
  --include-later-turns \
  --max-seq-len "${MAX_SAMPLE_TOKENS}" \
  --max-new-tokens 160 \
  --stop-sequence "</tool_call>" \
  --output-dir "outputs/behavior_${RUN_NAME}" \
  --report "reports/behavior_${RUN_NAME}.md" \
  --fp16
