#!/usr/bin/env bash
# ==============================================================================
# veRL GRPO — tau2 airline × Qwen2.5-7B-Instruct — FLAGSHIP, dual-GPU
# Both 5090s run the 7B POLICY (FSDP FULL_SHARD across 2 GPUs + vLLM TP=2), so
# the two 7B weight copies (train + rollout) are sharded to ~half per card and
# fit. The user simulator runs on a HOSTED API (OpenRouter) instead of a local
# vLLM, freeing both GPUs for the policy.
#
# Why not single-GPU 7B: hybrid engine keeps a training copy AND a vLLM copy of
# the 7B on the same card (~28GB of weights) — exceeds one 32GB 5090 at init.
#
# Secret: put your OpenRouter key in $ROOT/.tau2_env (chmod 600), one line:
#     export OPENROUTER_API_KEY=sk-or-v1-...
# This file is git-ignored and sourced below; the key is never printed.
#
# Usage:  bash run_tau2_grpo_7b_2gpu.sh
# ==============================================================================
set -euo pipefail   # (no -x: avoid echoing the sourced key)

ROOT=/root/autodl-tmp/verl-work
VENV=/root/autodl-tmp/venv-verl
INTEG=$ROOT/tau2_integration

export TMPDIR=$ROOT/tmp
export HF_HOME=$ROOT/.hfhome
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_CACHE_ROOT=$ROOT/.vllmcache
export TRITON_CACHE_DIR=$ROOT/.triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/.inductor
export XDG_CACHE_HOME=$ROOT/.xdgcache
export RAY_TMPDIR=$ROOT/tmp
export VLLM_USE_V1=1
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# ---- TP=2 hang fix: RTX 5090 are consumer cards with NO working GPU-to-GPU
# P2P/NVLink; NCCL's default P2P path deadlocks at vLLM TP init. Force the
# shared-memory/host path so the two ranks actually handshake. ----
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ---- both GPUs for the policy ----
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1}
export PYTHONPATH="$INTEG:${PYTHONPATH:-}"

# ---- OpenRouter-hosted user simulator ----
export TAU2_DOMAIN=airline
export TAU2_USER_LLM=${TAU2_USER_LLM:-openrouter/qwen/qwen-2.5-72b-instruct}
export TAU2_USER_API_BASE=""        # empty → litellm uses openrouter/ routing
export TAU2_USER_TEMPERATURE=0.0
export TAU2_EVAL_TYPE=all
export TAU2_MAX_ERRORS=10
# Load the OpenRouter key (kept out of the script/logs).
if [ -f "$ROOT/.tau2_env" ]; then set +x; source "$ROOT/.tau2_env"; fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "[ABORT] OPENROUTER_API_KEY not set. Put it in $ROOT/.tau2_env"; exit 1
fi

# ---- precheck: both GPUs mostly free ----
for g in 0 1; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d " ")
  echo "[precheck] GPU $g used ${used} MiB"
  if [ "${used}" -gt 6000 ]; then echo "[ABORT] GPU $g busy (${used}MiB)."; exit 1; fi
done
# ---- precheck: OpenRouter reachable ----
if ! curl -sf https://openrouter.ai/api/v1/models -H "Authorization: Bearer $OPENROUTER_API_KEY" >/dev/null 2>&1; then
  echo "[WARN] OpenRouter /models probe failed (continuing; key may still work for chat)."
fi

source "$VENV/bin/activate"
cd "$ROOT/verl"

MODEL_PATH=${POLICY_MODEL:-$ROOT/models/qwen25-7b-instruct}
TRAIN_FILE=$ROOT/data/tau2_airline/train.parquet
TEST_FILE=$ROOT/data/tau2_airline/test.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=${TRAIN_BS:-16} \
    data.max_prompt_length=${MAX_PROMPT:-6144} \
    data.max_response_length=${MAX_RESP:-3072} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI:-8} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD:-True} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.agent.default_agent_loop=tau2_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$INTEG/agent_loop_config.yaml" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-10240} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL:-0.55} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${SYNC_BUCKET:-256} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.use_v1=False \
    trainer.critic_warmup=0 \
    trainer.logger=[console,tensorboard] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.project_name=verl_tau2 \
    trainer.experiment_name=tau2_airline_qwen25_7b_grpo_2gpu \
    trainer.default_local_dir=$ROOT/ckpts/tau2_airline_7b_2gpu \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=${EPOCHS:-3} \
    2>&1 | tee $ROOT/tau2_airline_7b_2gpu_grpo.log
