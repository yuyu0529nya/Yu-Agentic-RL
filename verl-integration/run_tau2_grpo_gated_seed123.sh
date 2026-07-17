#!/usr/bin/env bash
# ==============================================================================
# tau2-airline / veRL — gate ON, seed=123, 20-step matched A/B replication.
#
# This is intentionally a separate launcher.  It does not alter the ordinary
# GRPO launchers or silently turn their runs into gated experiments.
#
# Run through run_tau2_grpo_gated_seed123_managed.sh for artifact collection and
# optional safe automatic shutdown.  This file stays foreground-only so that
# the manager can observe the true trainer exit code.
# ==============================================================================
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/verl-work}
VENV=${VENV:-/root/autodl-tmp/venv-verl}
INTEG=${INTEG:-$ROOT/tau2_integration}

# This experiment is intentionally fixed.  A caller cannot accidentally turn
# this into a different seed while retaining the seed-123 result name.
SEED=123
EXPECTED_STEPS=20
RUN_ID=${RUN_ID:-tau2_airline_gated_seed123_manual}
RUN_DIR=${RUN_DIR:-$ROOT/runs/$RUN_ID}

POLICY_MODEL=${POLICY_MODEL:-$ROOT/models/qwen25-7b-sft-airline}
TRAIN_FILE=${TRAIN_FILE:-$ROOT/data/tau2_airline/train.parquet}
TEST_FILE=${TEST_FILE:-$ROOT/data/tau2_airline/test.parquet}

# The matched A/B was run on one RTX PRO 6000 96GB card.  Do not silently
# fall back to the older dual-5090 launcher: changing sharding/offload changes
# the experiment as well as the hardware.  GPU is selectable only to choose
# which single 96GB card to use on an equivalent host.
GPU=${GPU:-0}

# Conditions recorded in RESULTS_ab_gating.md for the seed-42 A/B.
LR=1e-4
TRAIN_BS=24
ROLLOUT_N=12
TEST_FREQ=5
PPO_MINI=${PPO_MINI:-8}
SAVE_FREQ=${SAVE_FREQ:-5}

# The report establishes mean@4 but did not preserve its sampling knobs.  Make
# the chosen values explicit in the run manifest rather than silently falling
# back to the old mean@1 greedy defaults.  Override only if the GPU-box command
# or raw A/B log is recovered.
VAL_N=4
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}

fail() { echo "[ABORT] $*" >&2; exit 1; }

[[ -x "$VENV/bin/python" ]] || fail "missing venv python: $VENV/bin/python"
[[ -d "$ROOT/verl" ]] || fail "missing veRL checkout: $ROOT/verl"
[[ -f "$INTEG/gated_runtime/sitecustomize.py" ]] || fail "missing gated runtime hook"
[[ -f "$INTEG/tau2_agent_loop.py" ]] || fail "missing tau2 integration"
[[ -f "$TRAIN_FILE" ]] || fail "missing train parquet: $TRAIN_FILE"
[[ -f "$TEST_FILE" ]] || fail "missing held-out parquet: $TEST_FILE"
[[ -d "$POLICY_MODEL" ]] || fail "missing SFT policy checkpoint: $POLICY_MODEL"

# Read the secret without xtrace.  The file is expected to be chmod 600 and
# contain export OPENROUTER_API_KEY=...; an already-exported key also works.
if [[ -f "$ROOT/.tau2_env" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/.tau2_env"
fi
[[ -n "${OPENROUTER_API_KEY:-}" ]] || fail "OPENROUTER_API_KEY is not set"

# A prior AutoDL instance mounted /root/autodl-tmp on the small overlay.  Refuse
# that layout by default; set ALLOW_ROOT_OVERLAY=1 only after an intentional
# capacity check.
root_fs=$(df -P "$ROOT" | awk 'NR == 2 {print $1}')
slash_fs=$(df -P / | awk 'NR == 2 {print $1}')
if [[ "$root_fs" == "$slash_fs" && "${ALLOW_ROOT_OVERLAY:-0}" != "1" ]]; then
  fail "$ROOT shares the root filesystem ($root_fs); verify a persistent data disk or set ALLOW_ROOT_OVERLAY=1 intentionally"
fi
min_free_gb=${MIN_FREE_GB:-30}
available_kib=$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')
(( available_kib >= min_free_gb * 1024 * 1024 )) || fail "less than ${min_free_gb} GiB free under $ROOT"

gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
echo "[precheck] GPU $GPU total=${gpu_total} MiB used=${used} MiB"
[[ "$gpu_total" -ge "${MIN_GPU_MEMORY_MIB:-90000}" ]] || \
  fail "GPU $GPU has ${gpu_total} MiB; the matched A/B requires a 96GB-class single GPU"
[[ "$used" -le "${PRECHECK_MAX:-6000}" ]] || fail "GPU $GPU is busy (${used} MiB)"

if ! curl -sf https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" >/dev/null 2>&1; then
  echo "[WARN] OpenRouter model probe failed; continuing because a chat request may still succeed."
fi

mkdir -p "$RUN_DIR" "$ROOT/tmp" "$ROOT/.hfhome" "$ROOT/.vllmcache" \
  "$ROOT/.triton" "$ROOT/.inductor" "$ROOT/.xdgcache"

export TMPDIR=$ROOT/tmp
export HF_HOME=$ROOT/.hfhome
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=$ROOT/.mscache
export VLLM_CACHE_ROOT=$ROOT/.vllmcache
export TRITON_CACHE_DIR=$ROOT/.triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/.inductor
export XDG_CACHE_HOME=$ROOT/.xdgcache
export RAY_TMPDIR=$ROOT/tmp
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$INTEG/gated_runtime:$INTEG:${PYTHONPATH:-}"
export USE_DYNAMIC_SAMPLING=1
export HYDRA_FULL_ERROR=1

export TAU2_DOMAIN=airline
export TAU2_USER_LLM=${TAU2_USER_LLM:-openrouter/meta-llama/llama-3.3-70b-instruct}
export TAU2_USER_API_BASE=""
export TAU2_USER_TEMPERATURE=${TAU2_USER_TEMPERATURE:-0.0}
export TAU2_EVAL_TYPE=all
export TAU2_MAX_ERRORS=10
export TAU2_USER_ALLOW_FALLBACKS=${TAU2_USER_ALLOW_FALLBACKS:-true}

"$VENV/bin/python" - "$TEST_FILE" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
n = len(pd.read_parquet(path))
assert n == 20, f"gate-on seed123 requires exactly 20 held-out tasks, found {n}: {path}"
print(f"[precheck] held-out task count={n}")
PY

# This confirms that sitecustomize ran in this interpreter, registered the
# custom estimator, and patched the same legacy trainer seam used by main_ppo.
"$VENV/bin/python" - <<'PY'
from verl.trainer.ppo import ray_trainer

assert getattr(ray_trainer, "_tau2_gated_patch", False), "gated sitecustomize patch is absent"
print("[precheck] GATED_BOOTSTRAP_READY verified in driver interpreter")
PY

cd "$ROOT/verl"

exec "$VENV/bin/python" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo_gated \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size=$TRAIN_BS \
  data.seed=$SEED \
  data.max_prompt_length=${MAX_PROMPT:-6144} \
  data.max_response_length=${MAX_RESP:-3072} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$POLICY_MODEL" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.data_loader_seed=$SEED \
  actor_rollout_ref.actor.fsdp_config.seed=$SEED \
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-False} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD:-False} \
  actor_rollout_ref.ref.fsdp_config.seed=$SEED \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed=$SEED \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
  actor_rollout_ref.rollout.agent.default_agent_loop=tau2_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$INTEG/agent_loop_config.yaml" \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS:-2} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-10240} \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL:-0.30} \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${SYNC_BUCKET:-2048} \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
  actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
  actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
  algorithm.use_kl_in_reward=False \
  trainer.use_v1=False \
  trainer.critic_warmup=0 \
  trainer.logger='[console,tensorboard]' \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.project_name=verl_tau2 \
  trainer.experiment_name="$RUN_ID" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.total_epochs=$EXPECTED_STEPS
