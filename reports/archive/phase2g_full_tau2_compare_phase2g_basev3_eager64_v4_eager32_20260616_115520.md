# Phase2G Full Tau2 Rollout Comparison

## Configuration

- tasks: `2,16,18,25,44`
- trials: `1`
- max steps: `80`
- base/v3 agent max tokens: `64`
- v4 agent max tokens: `32` (reduced to avoid 12288 context boundary)
- max model len: `12288`
- user simulator: `anthropic/glm-5.1`

## Results

| run | agent_llm | tasks | sims | success_count | pass^1 | avg_reward | db_match | db_mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| airline_qwen25_7b_base_phase2g_full_tau2_eager64_20260616_112623 | openai/qwen25-7b-base | 5 | 5 | 1 | 0.2000 | 0.2000 | 1 | 4 |
| airline_qwen25_7b_slot_grounded_v3_phase2g_full_tau2_eager64_20260616_112623 | openai/qwen25-7b-slot-grounded-v3 | 5 | 5 | 1 | 0.2000 | 0.2000 | 1 | 4 |
| airline_qwen25_7b_single_tool_v4_phase2g_v4_eager32_20260616_115520 | openai/qwen25-7b-single-tool-v4 | 5 | 5 | 1 | 0.2000 | 0.2000 | 1 | 4 |

## Source Files

- comparison json: `reports/airline_qwen25_7b_base_vs_v3_vs_v4_phase2g_basev3_eager64_v4_eager32_20260616_115520.json`
