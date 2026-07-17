#!/usr/bin/env bash
# ==============================================================================
# Safe manager for run_tau2_grpo_gated_seed123.sh.
#
# It owns the exact child process/session, records an auditable run directory,
# archives small launch artifacts, and only powers off after a verified complete
# run.  It never kills jobs by a global process-name match.
#
# Usage:
#   cd /root/autodl-tmp/verl-work/tau2_integration
#   AUTO_SHUTDOWN=1 nohup bash run_tau2_grpo_gated_seed123_managed.sh \
#     > /root/autodl-tmp/verl-work/gated_seed123_manager.log 2>&1 &
#
# AUTO_SHUTDOWN defaults to 0.  Set it to 1 only when this AutoDL instance has
# no unrelated work that should survive a successful run.
# ==============================================================================
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/verl-work}
INTEG=${INTEG:-$ROOT/tau2_integration}
LAUNCHER=${LAUNCHER:-$INTEG/run_tau2_grpo_gated_seed123.sh}
EXPECTED_STEP=20
MAX_RUNTIME=${MAX_RUNTIME:-18000}
TERM_GRACE=${TERM_GRACE:-120}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}

fail() { echo "[ABORT] $*" >&2; exit 1; }
[[ -x /usr/bin/setsid || -x /bin/setsid || -n "$(command -v setsid 2>/dev/null || true)" ]] || fail "setsid is required"
[[ -f "$LAUNCHER" ]] || fail "missing launcher: $LAUNCHER"
[[ "$AUTO_SHUTDOWN" == "0" || "$AUTO_SHUTDOWN" == "1" ]] || fail "AUTO_SHUTDOWN must be 0 or 1"

RUN_ID=${RUN_ID:-tau2_airline_gated_seed123_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$ROOT/runs/$RUN_ID}
LOG="$RUN_DIR/train.log"
SUCCESS="$RUN_DIR/TRAINING_SUCCEEDED"
FAILED="$RUN_DIR/TRAINING_FAILED"
CAP="$RUN_DIR/RUNTIME_CAP"
mkdir -p "$RUN_DIR"

git_rev=$(git -C "$INTEG" rev-parse HEAD 2>/dev/null || true)
{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'expected_step=%s\n' "$EXPECTED_STEP"
  printf 'max_runtime_seconds=%s\n' "$MAX_RUNTIME"
  printf 'auto_shutdown=%s\n' "$AUTO_SHUTDOWN"
  printf 'git_revision=%s\n' "$git_rev"
  printf 'launcher=%s\n' "$LAUNCHER"
  printf 'launcher_sha256=%s\n' "$(sha256sum "$LAUNCHER" | awk '{print $1}')"
  printf 'sitecustomize_sha256=%s\n' "$(sha256sum "$INTEG/gated_runtime/sitecustomize.py" | awk '{print $1}')"
  printf 'tau2_agent_loop_sha256=%s\n' "$(sha256sum "$INTEG/tau2_agent_loop.py" | awk '{print $1}')"
} > "$RUN_DIR/manifest.env"
cp "$LAUNCHER" "$RUN_DIR/launcher.snapshot.sh"
cp "$INTEG/gated_runtime/sitecustomize.py" "$RUN_DIR/sitecustomize.snapshot.py"
cp "$INTEG/tau2_agent_loop.py" "$RUN_DIR/tau2_agent_loop.snapshot.py"
cp "$INTEG/grpo_gated/grpo_gated.py" "$RUN_DIR/grpo_gated.snapshot.py"
cp "$INTEG/grpo_gated/dynamic_sampling.py" "$RUN_DIR/dynamic_sampling.snapshot.py"

echo "[manager] run=$RUN_ID"
echo "[manager] log=$LOG"
echo "[manager] auto_shutdown=$AUTO_SHUTDOWN, runtime_cap=${MAX_RUNTIME}s"

# The launcher execs main_ppo in the foreground.  setsid gives this one run a
# private session so a timeout can stop only its descendants.
setsid env RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" bash "$LAUNCHER" >"$LOG" 2>&1 &
TRAIN_PID=$!
sleep 1
TRAIN_SID=$(ps -o sid= -p "$TRAIN_PID" | tr -d ' ')
[[ "$TRAIN_SID" == "$TRAIN_PID" ]] || fail "setsid did not create a private session for PID $TRAIN_PID"
printf '%s\n' "$TRAIN_PID" > "$RUN_DIR/train.pid"
printf '%s\n' "$TRAIN_SID" > "$RUN_DIR/train.sid"

session_pids() {
  ps -eo pid=,sid= | awk -v sid="$TRAIN_SID" '$2 == sid {print $1}'
}

stop_run_session() {
  local signal=$1
  mapfile -t pids < <(session_pids)
  if ((${#pids[@]})); then
    echo "[manager] sending $signal to this run's session: ${pids[*]}" >> "$LOG"
    kill "-$signal" "${pids[@]}" 2>/dev/null || true
  fi
}

(
  sleep "$MAX_RUNTIME"
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    date -Is > "$CAP"
    echo "[manager] runtime cap reached; terminating only run session $TRAIN_SID" >> "$LOG"
    stop_run_session TERM
    sleep "$TERM_GRACE"
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      stop_run_session KILL
    fi
  fi
) &
CAP_TIMER=$!

set +e
wait "$TRAIN_PID"
TRAIN_RC=$?
set -e
kill "$CAP_TIMER" 2>/dev/null || true
wait "$CAP_TIMER" 2>/dev/null || true

completed_step=false
bootstrap_seen=false
sampling_seen=false
grep -Eq "training/global_step:[[:space:]]*${EXPECTED_STEP}([^0-9]|$)" "$LOG" && completed_step=true
grep -Fq "GATED_BOOTSTRAP_READY" "$LOG" && bootstrap_seen=true
grep -Fq "GATED_DYNAMIC_SAMPLING" "$LOG" && sampling_seen=true

if [[ ! -e "$CAP" && "$TRAIN_RC" -eq 0 && "$completed_step" == true && \
      "$bootstrap_seen" == true && "$sampling_seen" == true ]]; then
  {
    printf 'completed_at=%s\n' "$(date -Is)"
    printf 'train_pid=%s\ntrain_sid=%s\nexit_code=%s\nexpected_step=%s\n' \
      "$TRAIN_PID" "$TRAIN_SID" "$TRAIN_RC" "$EXPECTED_STEP"
  } > "$SUCCESS.tmp"
  mv "$SUCCESS.tmp" "$SUCCESS"
  outcome=success
else
  {
    printf 'ended_at=%s\n' "$(date -Is)"
    printf 'train_pid=%s\ntrain_sid=%s\nexit_code=%s\n' "$TRAIN_PID" "$TRAIN_SID" "$TRAIN_RC"
    printf 'completed_step=%s\nbootstrap_seen=%s\nsampling_seen=%s\nruntime_cap=%s\n' \
      "$completed_step" "$bootstrap_seen" "$sampling_seen" "$([[ -e "$CAP" ]] && echo true || echo false)"
  } > "$FAILED"
  outcome=failure
fi

{
  printf '# %s\n\n' "$RUN_ID"
  printf 'outcome: %s\n' "$outcome"
  printf 'exit_code: %s\n' "$TRAIN_RC"
  printf 'completion_marker: %s\n\n' "$([[ -e "$SUCCESS" ]] && echo yes || echo no)"
  printf '## Gated runtime markers\n\n'
  grep -F '[gated-bootstrap]' "$LOG" || true
  printf '\n## Validation metrics\n\n'
  grep -Eo 'val-core/tau2/airline/reward/mean@4:[^ -]+' "$LOG" || true
  printf '\n## Tail\n\n```text\n'
  tail -n 80 "$LOG" || true
  printf '\n```\n'
} > "$RUN_DIR/SUMMARY.md"

if [[ -e "$SUCCESS" ]]; then
  tar -C "$RUN_DIR" -czf "$RUN_DIR/launch_artifacts.tgz" \
    manifest.env launcher.snapshot.sh sitecustomize.snapshot.py tau2_agent_loop.snapshot.py \
    grpo_gated.snapshot.py dynamic_sampling.snapshot.py train.log SUMMARY.md TRAINING_SUCCEEDED \
    2>/dev/null || touch "$RUN_DIR/ARCHIVE_FAILED"
fi

foreign_work=false
while IFS= read -r pid; do
  [[ -z "$pid" ]] && continue
  sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  if [[ -n "$sid" && "$sid" != "$TRAIN_SID" ]]; then
    echo "foreign_gpu_pid=$pid sid=$sid" >> "$RUN_DIR/foreign_work.txt"
    foreign_work=true
  fi
done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)

while IFS= read -r pid; do
  [[ -z "$pid" ]] && continue
  sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  if [[ -n "$sid" && "$sid" != "$TRAIN_SID" ]]; then
    echo "foreign_main_ppo_pid=$pid sid=$sid" >> "$RUN_DIR/foreign_work.txt"
    foreign_work=true
  fi
done < <(pgrep -f 'verl.trainer.main_ppo' 2>/dev/null || true)

if [[ "$AUTO_SHUTDOWN" == "1" && -e "$SUCCESS" && "$foreign_work" == false ]]; then
  echo "[manager] verified complete run; powering off host to stop billing." | tee -a "$LOG"
  /usr/sbin/shutdown -h now 2>/dev/null || shutdown -h now 2>/dev/null || poweroff 2>/dev/null
elif [[ "$AUTO_SHUTDOWN" == "1" && "$foreign_work" == true ]]; then
  date -Is > "$RUN_DIR/SHUTDOWN_SKIPPED_FOREIGN_WORK"
  echo "[manager] foreign GPU/PPO work found; leaving host on." | tee -a "$LOG"
elif [[ "$AUTO_SHUTDOWN" == "1" && ! -e "$SUCCESS" ]]; then
  echo "[manager] run was not verified complete; leaving host on for diagnosis." | tee -a "$LOG"
fi

if [[ -e "$SUCCESS" ]]; then
  echo "[manager] success: $RUN_DIR"
  exit 0
fi

echo "[manager] incomplete/failed run retained at: $RUN_DIR" >&2
if [[ "$TRAIN_RC" -eq 0 ]]; then
  # A zero process exit without the expected final step or gated-runtime
  # markers is not a scientifically valid completion.
  exit 1
fi
exit "$TRAIN_RC"
