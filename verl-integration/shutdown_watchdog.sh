#!/usr/bin/env bash
# ==============================================================================
# Overnight watchdog: shut the machine down once the tau2 training finishes,
# so the GPU stops billing while the user sleeps.
#
# Logic: poll for the main_ppo training process. Once it is gone (whether it
# completed all epochs or died), wait a short grace period, then power off.
# Either way there is nothing left to bill for. The /root/autodl-tmp data disk
# (logs, checkpoints, curves) persists across shutdown on AutoDL.
#
# Usage: nohup bash shutdown_watchdog.sh > /root/autodl-tmp/verl-work/watchdog.log 2>&1 &
# ==============================================================================
LOG=/root/autodl-tmp/verl-work/tau2_airline_3b_grpo.log
MAX_RUNTIME=${MAX_RUNTIME:-18000}   # cost cap: shut down after 5h even if still training
START=$(date +%s)
echo "[watchdog] started $(date); finish-or-${MAX_RUNTIME}s cap, then power off."

# Give training a moment to be fully up before we start watching.
sleep 120

# Shut down ONLY on (a) clean completion, or (b) the runtime cap. If training
# dies with an error, do NOT power off -- leave the box up so the operator can
# fix and restart (a premature shutdown on a fixable crash wastes the night).
while true; do
  now=$(date +%s)
  if [ $((now - START)) -ge "$MAX_RUNTIME" ]; then
    echo "[watchdog] runtime cap reached ($MAX_RUNTIME s); stopping + powering off."
    for p in $(pgrep -f "verl.trainer.main_ppo"); do kill -9 "$p" 2>/dev/null || true; done
    sleep 15
    break
  fi
  if ! pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; then
    sleep 30   # let logs flush
    if pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; then continue; fi
    if tail -30 "$LOG" 2>/dev/null | grep -qiE "Error executing job|RayTaskError|OutOfMemory|Traceback"; then
      echo "[watchdog] training CRASHED at $(date) -- NOT powering off (waiting for operator)."
      # Idle-wait until the runtime cap; operator may restart training meanwhile.
      sleep 300
      continue
    fi
    echo "[watchdog] training completed cleanly at $(date)."
    break
  fi
  sleep 60
done

# Extract the learning curve into an easy-to-find summary on the (persistent)
# data disk before powering off, so the user sees results on restart.
SUMMARY=/root/autodl-tmp/verl-work/RESULTS_tau2_3b.txt
{
  echo "tau2 airline x Qwen2.5-3B GRPO (veRL) — run summary  $(date)"
  echo "=== val reward curve (val-core/tau2/airline/reward/mean@1) ==="
  grep -oE "step:[0-9]+ - val-core/tau2/airline/reward/mean@1:[^ ]+ - val-aux/num_turns/mean:[^ ]+" "$LOG" 2>/dev/null
  echo "=== train reward per step (critic/rewards/mean) ==="
  grep -oE "training/global_step:[0-9]+|critic/rewards/mean:[^ ]+" "$LOG" 2>/dev/null | paste - -
  echo "=== last 30 log lines ==="
  tail -30 "$LOG" 2>/dev/null
} > "$SUMMARY" 2>&1
echo "[watchdog] wrote $SUMMARY"

echo "[watchdog] powering off to stop GPU billing at $(date)."
# AutoDL: shutting down the instance stops GPU billing; data disk persists.
/usr/sbin/shutdown -h now 2>/dev/null || shutdown -h now 2>/dev/null || poweroff 2>/dev/null
