# Tau2 Baseline Run Comparison

| run | agent_llm | tasks | sims | avg_reward | pass^1 | db_match | db_mismatch | avg_process_score | risk_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airline_qwen25_7b_base_phase2a_clean12k_5090d_20260614_175700 | openai/qwen25-7b-base | 2 | 2 | 0.0000 | 0.0000 | 0 | 2 | - | - |
| airline_qwen25_7b_sft_phase2a_clean12k_5090d_20260614_175700 | openai/qwen25-7b-action-prefix-sft | 2 | 2 | 0.5000 | 0.5000 | 1 | 1 | - | - |
