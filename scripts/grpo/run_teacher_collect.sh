#!/usr/bin/env bash
set -euo pipefail
# Teacher-trajectory collection for distillation SFT — NO GPU NEEDED.
# deepseek-chat plays the AGENT (via the OpenAI-compatible endpoint) against a deepseek
# user-simulator, through the SAME tau2 machinery as training/eval (same system prompt,
# same tool schemas), so successful trajectories are directly renderable by
# grpo_update.py --rft (render-twice-diff on the Qwen template).
#
# Run on the GPU box CPU during the day (Mac may lack DNS to api.deepseek.com):
#   smoke : TRAIN_TASK_IDS=2,3 TRIALS=2 RUN=teacher_smoke bash scripts/grpo/run_teacher_collect.sh
#   full  : bash scripts/grpo/run_teacher_collect.sh
WORKDIR="${WORKDIR:-$HOME/agentic-rl}"; PYBIN="${PYBIN:-$WORKDIR/venv/bin/python}"
cd "$WORKDIR"
export PATH="${EXTRA_PATH:-$WORKDIR/venv/bin}:$HOME/.local/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
# tau2 editable-install .pth gets a macOS 'hidden' flag re-applied by iCloud Desktop sync,
# and Python 3.12.4+ silently SKIPS hidden .pth files -> ModuleNotFoundError. PYTHONPATH is
# immune to file flags (harmless no-op where the editable install works, e.g. GPU box).
export PYTHONPATH="$WORKDIR/third_party/tau2-bench/src${PYTHONPATH:+:$PYTHONPATH}"
export TAU2_CMD="${TAU2_CMD:-tau2}"   # shared /data env: tau2 CLI in venv (NOT uv)
set -a; for f in .env .tau2_secret_env; do [ -f "$f" ] && . "./$f"; done; set +a

# TEACHER_BACKEND: openrouter (default/primary — $10 credit, also unlocks v4-pro/r1 teachers)
# | deepseek (fallback — official api.deepseek.com). Both are the user's OWN keys; the litellm
# here is only the open-source client library inside tau2 — the external litellm proxy is never used.
TEACHER_BACKEND="${TEACHER_BACKEND:-openrouter}"
if [ "$TEACHER_BACKEND" = "openrouter" ]; then
  : "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY missing — put it in $WORKDIR/.env}"
  TEACHER_MODEL="${TEACHER_MODEL:-deepseek/deepseek-v4-flash}"
  DS_BASE="${DS_BASE:-https://openrouter.ai/api/v1}"
  TEACHER_API_KEY="$OPENROUTER_API_KEY"
  USER_LLM="${USER_LLM:-openrouter/$TEACHER_MODEL}"     # litellm native openrouter provider
else
  : "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY missing — put it in $WORKDIR/.env}"
  TEACHER_MODEL="${TEACHER_MODEL:-deepseek-chat}"
  DS_BASE="${DS_BASE:-https://api.deepseek.com/v1}"
  TEACHER_API_KEY="$DEEPSEEK_API_KEY"
  USER_LLM="${USER_LLM:-deepseek/$TEACHER_MODEL}"
fi

RUN="${RUN:-teacher_ds_$(date +%m%d)}"
DOMAIN="${DOMAIN:-airline}"
# 30 RL-train tasks only (airline_rl_split.json) — NEVER the 20 held-out eval tasks.
TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-2,3,4,7,8,9,12,13,14,17,18,19,22,23,24,27,28,29,32,33,34,37,38,39,42,43,44,47,48,49}"
TRIALS="${TRIALS:-6}"            # attempts per task; more trials -> better coverage of hard tasks
TEMP="${TEMP:-0.7}"              # teacher sampling temp (diversity across trials)
USER_TEMP="${USER_TEMP:-0.7}"    # usersim temp: >0 diversifies TRAINING dialogues (eval protocol pins 0 elsewhere)
MAX_STEPS="${MAX_STEPS:-40}"; AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1024}"
MAXC="${MAXC:-6}"                # API concurrency (no local serving bottleneck)
SEED="${SEED:-1234}"
CAP="${CAP:-4}"                  # max kept successes per task in the distill set
OUT="outputs/$RUN"; mkdir -p "$OUT"
exec > >(tee -a "$OUT/master.log") 2>&1

echo "######## TEACHER COLLECT START $(date) run=$RUN backend=$TEACHER_BACKEND teacher=$TEACHER_MODEL trials=$TRIALS ########"
# NOTE: --stop "" disables the </tool_call> stop trick (Qwen-specific); deepseek uses
# native function calling so tau2 gets clean nested tool_calls in the stored messages.
$PYBIN scripts/grpo/collect_rollouts.py \
  --domain "$DOMAIN" \
  --served-model "$TEACHER_MODEL" \
  --api-base "$DS_BASE" --api-key "$TEACHER_API_KEY" \
  --user-llm "$USER_LLM" --user-temperature "$USER_TEMP" \
  --task-ids "$TRAIN_TASK_IDS" --num-trials "$TRIALS" \
  --temperature "$TEMP" --max-tokens "$AGENT_MAX_TOKENS" --stop "" \
  --max-steps "$MAX_STEPS" --max-concurrency "$MAXC" \
  --seed "$SEED" --timeout-seconds 21600 \
  --save-to "${RUN}_sims" --out "$OUT/teacher_rollouts.jsonl"

echo "-------- build success-only distill dataset (cap $CAP/task) --------"
$PYBIN scripts/grpo/build_rft_dataset.py --rollouts "$OUT/teacher_rollouts.jsonl" \
  --out "$OUT/distill_dataset.jsonl" --cap-per-task "$CAP"

echo "ALLDONE_TEACHER RUN=$RUN dataset=$OUT/distill_dataset.jsonl $(date)"
