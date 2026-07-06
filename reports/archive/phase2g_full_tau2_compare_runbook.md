# Phase2G Full Tau2 Rollout Compare Runbook

## Goal

Validate whether the Phase2F Single-Tool Protocol v4 offline behavior gain
transfers to real end-to-end tau2 airline task success.

Comparison:

```text
Base Qwen2.5-7B-Instruct
vs Slot-Grounded Action-Prefix v3 QLoRA
vs Single-Tool Protocol v4 QLoRA
```

## Default Run

On AutoDL, from the project root:

```bash
cd /root/autodl-tmp/yuyu

RUN_TAG=phase2g_full_tau2_$(date '+%Y%m%d_%H%M%S') \
TASK_IDS=2,16,18,25,44 \
NUM_TRIALS=1 \
MAX_STEPS=80 \
MAX_MODEL_LEN=12288 \
AGENT_MAX_TOKENS=128 \
AGENT_STOP_SEQUENCE='</tool_call>' \
AGENT_INCLUDE_STOP_STR=1 \
AUTO_SHUTDOWN=0 \
bash scripts/run_phase2g_full_tau2_compare_autodl.sh
```

Default policy: the instance is left running. Only set `AUTO_SHUTDOWN=1` if the
user explicitly asks to shut down after this run.

## Expected Inputs

The script expects:

- base model: `/root/autodl-tmp/models/qwen25-7b-instruct`
- tau2-bench checkout: `third_party/tau2-bench`
- v3 artifact tarball:
  `autodl_artifacts/slot_grounded_v3_1536_4090_20260615_201005/yuyu_slot_grounded_action_prefix_artifacts_20260615_201013.tar.gz`
- v4 artifact tarball:
  `autodl_artifacts/phase2f_single_tool_protocol_v4_4090_20260615_224720/yuyu_single_tool_protocol_v4_4090_20260615_224720.tar.gz`

If v3/v4 adapters are not already extracted, the script extracts them into:

```text
outputs/sft_action_prefix_slot_grounded_v3_qwen25_7b_qlora_1536/checkpoint
outputs/sft_single_tool_protocol_v4_qwen25_7b_qlora_1536/checkpoint
```

## Why These Defaults

- `TASK_IDS=2,16,18,25,44`: continuity set from prior Phase2A/2D/2E diagnostics.
- `MAX_MODEL_LEN=12288`: avoids the earlier 8192-token context overflow on
  task 16 while staying safer than 16K on a 24GB 4090D.
- `AGENT_MAX_TOKENS=128`: keeps tool-call turns short and reduces context growth.
- `AGENT_STOP_SEQUENCE='</tool_call>'` and `AGENT_INCLUDE_STOP_STR=1`: preserve
  the executable tool-call boundary learned by Phase2F v4.

## Outputs

Each run writes tau2 trajectories under:

```text
third_party/tau2-bench/data/simulations/
```

Summaries and comparison reports are written under:

```text
reports/
```

Important expected files:

```text
reports/airline_qwen25_7b_base_<RUN_TAG>.summary.json
reports/airline_qwen25_7b_slot_grounded_v3_<RUN_TAG>.summary.json
reports/airline_qwen25_7b_single_tool_v4_<RUN_TAG>.summary.json
reports/airline_qwen25_7b_base_vs_v3_vs_v4_<RUN_TAG>.md
reports/airline_qwen25_7b_base_vs_v3_vs_v4_<RUN_TAG>.json
reports/phase2g_full_tau2_compare_<RUN_TAG>.md
```

vLLM logs are written under:

```text
outputs/vllm_logs/
```

## Success Criteria

Primary criterion:

```text
v4 pass^1 > base pass^1 and v4 pass^1 >= v3 pass^1
```

Interpretation:

- If v4 improves `pass^1`, the tool protocol fix transferred from offline
  behavior proxy to real tau2 rollout.
- If v4 does not improve `pass^1`, inspect failed trajectories before training
  more. Likely next issues are parser integration, natural assistant turns, or
  remaining airline policy decisions.

## Fast Checks Before Running

```bash
cd /root/autodl-tmp/yuyu
nvidia-smi
test -d third_party/tau2-bench && echo TAU2_OK
test -d /root/autodl-tmp/models/qwen25-7b-instruct && echo MODEL_OK
bash -n scripts/run_phase2g_full_tau2_compare_autodl.sh
DRY_RUN=1 bash scripts/run_phase2g_full_tau2_compare_autodl.sh
```
