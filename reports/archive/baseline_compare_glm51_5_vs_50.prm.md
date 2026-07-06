# Tau2 Baseline Run Comparison

| run | agent_llm | tasks | sims | avg_reward | pass^1 | db_match | db_mismatch | avg_process_score | risk_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airline_debug_glm51_5tasks | anthropic/glm-5.1 | 5 | 5 | 0.6000 | 0.6000 | 3 | 2 | -2.3000 | action_mismatch:1;communication_db_gap:2;compensation_policy_error:1;incomplete_evidence:1;object_selection_error:1;premature_write:2;temporal_policy_error:1;user_pressure_susceptibility:1 |
| airline_baseline_50_anthropic_glm_5_1_50tasks_1trials | anthropic/glm-5.1 | 50 | 50 | 0.5800 | 0.5800 | 29 | 21 | -0.8200 | action_mismatch:17;communication_db_gap:18;compensation_policy_error:3;incomplete_evidence:17;object_selection_error:1;premature_write:9;temporal_policy_error:3;user_pressure_susceptibility:1 |
