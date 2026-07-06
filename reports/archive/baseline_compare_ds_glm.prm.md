# Tau2 Baseline Run Comparison

| run | agent_llm | tasks | sims | avg_reward | pass^1 | db_match | db_mismatch | avg_process_score | risk_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airline_debug_deepseek_chat_5tasks | deepseek/deepseek-chat | 5 | 5 | 0.8000 | 0.8000 | 4 | 1 | 0.0000 | communication_db_gap:1;premature_write:1;temporal_policy_error:1;user_pressure_susceptibility:1 |
| airline_debug_glm51_5tasks | anthropic/glm-5.1 | 5 | 5 | 0.6000 | 0.6000 | 3 | 2 | -2.3000 | action_mismatch:1;communication_db_gap:2;compensation_policy_error:1;incomplete_evidence:1;object_selection_error:1;premature_write:2;temporal_policy_error:1;user_pressure_susceptibility:1 |
