# PRM Ablation Report: airline_failed21_n4_timeout300_v2_merged

## Summary

| Ablation | Pass@N | Selected | Drop vs full | Gap to oracle | Selection acc. | Description |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `full` | 0.7619 | 16/21 | +0.0000 | 0.0000 | 1.0000 | All PRM-Lite components. |
| `full_eval_tiebreak` | 0.7619 | 16/21 | +0.0000 | 0.0000 | 1.0000 | All components plus reward-info DB/communication tags for tie-break only. |
| `zero_score` | 0.3333 | 7/21 | +0.4286 | 0.4286 | 0.4375 | Remove all explicit components; deterministic tie-break reduces to first trial. |
| `scoreless_eval_tiebreak` | 0.7619 | 16/21 | +0.0000 | 0.0000 | 1.0000 | Remove all explicit components but keep reward-info DB/communication tie-break tags. |
| `no_reference_actions` | 0.6190 | 13/21 | +0.1429 | 0.1429 | 0.8125 | Remove direct tau2 reference action match/mismatch components. |
| `no_privileged_action_rules` | 0.5714 | 12/21 | +0.1905 | 0.1905 | 0.7500 | Remove reference actions plus expected-write and deferral components. |
| `no_premature_write` | 0.5238 | 11/21 | +0.2381 | 0.2381 | 0.6875 | Remove components tagged as premature write. |
| `no_payment_planning` | 0.7619 | 16/21 | +0.0000 | 0.0000 | 1.0000 | Remove payment-planning and calculation components. |
| `no_tool_affordance` | 0.7619 | 16/21 | +0.0000 | 0.0000 | 1.0000 | Remove tool affordance and premature-deferral components. |
| `no_temporal_policy` | 0.6667 | 14/21 | +0.0952 | 0.0952 | 0.8750 | Remove temporal policy components. |
| `no_compensation_policy` | 0.7619 | 16/21 | +0.0000 | 0.0000 | 1.0000 | Remove compensation policy components. |
| `no_evidence_object` | 0.7143 | 15/21 | +0.0476 | 0.0476 | 0.9375 | Remove incomplete-evidence and object-selection components. |
| `negative_only` | 0.7619 | 16/21 | +0.0000 | 0.0000 | 1.0000 | Keep only negative penalty components. |
| `positive_only` | 0.4286 | 9/21 | +0.3333 | 0.3333 | 0.5625 | Keep only positive support components. |
| `reference_only` | 0.5238 | 11/21 | +0.2381 | 0.2381 | 0.6875 | Keep only reference action match/mismatch components. |
| `write_safety_only` | 0.5714 | 12/21 | +0.1905 | 0.1905 | 0.7500 | Keep only premature-write components. |
| `payment_planning_only` | 0.3333 | 7/21 | +0.4286 | 0.4286 | 0.4375 | Keep only payment-planning and calculation components. |
| `tool_affordance_only` | 0.3333 | 7/21 | +0.4286 | 0.4286 | 0.4375 | Keep only tool-affordance and premature-deferral components. |
| `temporal_policy_only` | 0.4286 | 9/21 | +0.3333 | 0.3333 | 0.5625 | Keep only temporal policy components. |
| `evidence_object_only` | 0.5714 | 12/21 | +0.1905 | 0.1905 | 0.7500 | Keep only incomplete-evidence and object-selection components. |

## Leave-One-Family-Out

| Ablation | Selected Tasks Changed vs Full | New Misses on Solvable Tasks |
| --- | --- | --- |
| `no_reference_actions` | 7:t0->t2, 15:t2->t0, 18:t2->t0, 33:t1->t0 | 15, 18, 33 |
| `no_privileged_action_rules` | 7:t0->t2, 12:t3->t0, 15:t2->t0, 18:t2->t0, 33:t1->t0 | 12, 15, 18, 33 |
| `no_premature_write` | 2:t2->t1, 12:t3->t0, 27:t2->t0, 34:t1->t0, 42:t0->t1 | 2, 12, 27, 34, 42 |
| `no_payment_planning` | - | - |
| `no_tool_affordance` | - | - |
| `no_temporal_policy` | 34:t1->t0, 42:t0->t1 | 34, 42 |
| `no_compensation_policy` | - | - |
| `no_evidence_object` | 7:t0->t2, 12:t3->t1, 33:t1->t0, 44:t3->t0 | 33 |

## Single-Family-Only

| Ablation | Pass@N | Selected Tasks |
| --- | ---: | --- |
| `reference_only` | 0.5238 | 15:t2=1.0000, 16:t0=1.0000, 18:t2=1.0000, 20:t0=1.0000, 23:t1=1.0000, 25:t0=1.0000, 32:t0=1.0000, 33:t1=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
| `write_safety_only` | 0.5714 | 1:t1=1.0000, 2:t2=1.0000, 12:t1=1.0000, 16:t0=1.0000, 20:t0=1.0000, 25:t0=1.0000, 27:t2=1.0000, 32:t0=1.0000, 34:t1=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
| `payment_planning_only` | 0.3333 | 16:t0=1.0000, 20:t0=1.0000, 25:t0=1.0000, 32:t0=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
| `tool_affordance_only` | 0.3333 | 16:t0=1.0000, 20:t0=1.0000, 25:t0=1.0000, 32:t0=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
| `temporal_policy_only` | 0.4286 | 1:t1=1.0000, 16:t0=1.0000, 20:t0=1.0000, 25:t0=1.0000, 32:t0=1.0000, 34:t1=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
| `evidence_object_only` | 0.5714 | 2:t2=1.0000, 15:t2=1.0000, 16:t0=1.0000, 18:t2=1.0000, 20:t0=1.0000, 23:t1=1.0000, 25:t0=1.0000, 32:t0=1.0000, 33:t1=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
| `positive_only` | 0.4286 | 15:t2=1.0000, 16:t0=1.0000, 18:t2=1.0000, 20:t0=1.0000, 23:t1=1.0000, 25:t0=1.0000, 32:t0=1.0000, 37:t0=1.0000, 38:t0=1.0000 |
| `negative_only` | 0.7619 | 1:t1=1.0000, 2:t2=1.0000, 12:t3=1.0000, 15:t2=1.0000, 16:t0=1.0000, 18:t2=1.0000, 20:t0=1.0000, 23:t1=1.0000, 25:t0=1.0000, 27:t2=1.0000, 32:t0=1.0000, 33:t1=1.0000, 34:t1=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
| `scoreless_eval_tiebreak` | 0.7619 | 1:t1=1.0000, 2:t2=1.0000, 12:t1=1.0000, 15:t2=1.0000, 16:t0=1.0000, 18:t2=1.0000, 20:t0=1.0000, 23:t1=1.0000, 25:t0=1.0000, 27:t2=1.0000, 32:t0=1.0000, 33:t1=1.0000, 34:t1=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
| `zero_score` | 0.3333 | 16:t0=1.0000, 20:t0=1.0000, 25:t0=1.0000, 32:t0=1.0000, 37:t0=1.0000, 38:t0=1.0000, 42:t0=1.0000 |
