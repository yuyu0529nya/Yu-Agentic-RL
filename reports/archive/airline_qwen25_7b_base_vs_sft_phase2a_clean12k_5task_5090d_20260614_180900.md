# Tau2 Baseline Run Comparison

| run | agent_llm | tasks | sims | avg_reward | pass^1 | db_match | db_mismatch | avg_process_score | risk_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airline_qwen25_7b_base_phase2a_clean12k_5task_5090d_20260614_180900 | openai/qwen25-7b-base | 5 | 5 | 0.2500 | 0.2000 | 1 | 2 | - | - |
| airline_qwen25_7b_sft_phase2a_clean12k_5task_5090d_20260614_180900 | openai/qwen25-7b-action-prefix-sft | 5 | 5 | 0.2000 | 0.2000 | 1 | 4 | - | - |
