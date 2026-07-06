# Tau2 Baseline Run Comparison

| run | agent_llm | tasks | sims | avg_reward | pass^1 | db_match | db_mismatch | avg_process_score | risk_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airline_qwen25_7b_base_phase2g_full_tau2_eager64_20260616_112623 | openai/qwen25-7b-base | 5 | 5 | 0.2000 | 0.2000 | 1 | 4 | - | - |
| airline_qwen25_7b_slot_grounded_v3_phase2g_full_tau2_eager64_20260616_112623 | openai/qwen25-7b-slot-grounded-v3 | 5 | 5 | 0.2000 | 0.2000 | 1 | 4 | - | - |
| airline_qwen25_7b_single_tool_v4_phase2g_v4_eager32_20260616_115520 | openai/qwen25-7b-single-tool-v4 | 5 | 5 | 0.2000 | 0.2000 | 1 | 4 | - | - |
