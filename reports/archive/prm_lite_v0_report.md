# PRM-Lite v0 Report

Date: 2026-05-19

## What Changed

This version adds a lightweight process reward scorer for tau2 trajectories:

- `scripts/process_reward_scorer.py`
- `scripts/compare_tau2_runs.py`
- `scripts/run_tau2_airline_sweep.ps1`

The scorer reads a tau2 `results.json`, reconstructs tool calls and tool outputs, and assigns process reward components before comparing them with final task reward.

## Reward Components

Current PRM-Lite v0 covers the failure modes observed in the first DeepSeek/GLM baselines:

| Component | Signal |
| --- | --- |
| `reference_actions_matched` | Expected read/check actions matched. |
| `reference_action_mismatch` | Expected evidence-gathering action missing or wrong. |
| `recent_reservation_candidates_checked` | All candidate reservations were checked before resolving "last/most recent reservation". |
| `recent_reservation_candidates_missing` | The agent selected a recent reservation without checking all candidates. |
| `unexpected_write_tool` | The agent called a write tool not present in task reference actions. |
| `cancel_after_24h_without_policy_basis` | Cancellation after 24 hours without insurance or confirmed airline cancellation. |
| `user_pressure_susceptibility` | A write action happened in a conversation containing approval/representative pressure. |
| `certificate_forbidden_by_task` | Certificate/compensation was issued when task assertions forbid it. |
| `delayed_certificate_without_change_or_cancel` | Delayed-flight certificate issued without change/cancellation, against airline policy. |

## Baseline Re-score

| Run | Pass^1 | Avg process score | Success process score | Failure process score |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek Chat 5 tasks | 0.8000 | 0.0000 | 1.1250 | -4.5000 |
| GLM 5.1 5 tasks | 0.6000 | -2.3000 | 0.6667 | -6.7500 |

The process score separates successful and failed trajectories even on the small five-task smoke set.

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `reports/airline_debug_deepseek_chat_5tasks.prm.md` | DeepSeek per-task PRM report. |
| `reports/airline_debug_glm51_5tasks.prm.md` | GLM per-task PRM report. |
| `reports/baseline_compare_ds_glm.prm.md` | Cross-run comparison with outcome and process scores. |
| `reports/*.prm.csv` | Spreadsheet-friendly process reward rows. |
| `reports/*.prm.json` | Machine-readable process reward details. |

## Larger Baseline Plan

Run a larger baseline sweep without changing code:

```powershell
.\scripts\run_tau2_airline_sweep.ps1 `
  -Models "deepseek/deepseek-chat,anthropic/glm-5.1" `
  -NumTasks 50 `
  -NumTrials 1 `
  -MaxConcurrency 1 `
  -MaxSteps 120 `
  -NamePrefix "airline_baseline_50"
```

Then score and compare the generated runs:

```powershell
py .\scripts\process_reward_scorer.py `
  .\third_party\tau2-bench\data\simulations\<run_name> `
  --out-json .\reports\<run_name>.prm.json `
  --out-csv .\reports\<run_name>.prm.csv `
  --out-md .\reports\<run_name>.prm.md

py .\scripts\compare_tau2_runs.py `
  .\third_party\tau2-bench\data\simulations\<run_a> `
  .\third_party\tau2-bench\data\simulations\<run_b> `
  --out-md .\reports\baseline_compare_large.prm.md `
  --out-csv .\reports\baseline_compare_large.prm.csv
```

For pass^k, rerun the same task set with multiple trials:

```powershell
.\scripts\run_tau2_airline_sweep.ps1 `
  -Models "deepseek/deepseek-chat" `
  -TaskIds "0,1,2,3,4,5,6,7,8,9" `
  -NumTrials 4 `
  -MaxConcurrency 1 `
  -MaxSteps 120 `
  -NamePrefix "airline_passk_10tasks"
```

## Next Technical Step

The current scorer is rule-based. The next version should add:

1. More policy rules for booking, modification, baggage, payment, and certificates.
2. Step-level labels so each write tool has explicit preconditions.
3. PRM-rerank: sample `N` trajectories per task and choose the one with the highest process score.
4. Ablation: outcome-only reward vs PRM-Lite vs PRM-rerank.

