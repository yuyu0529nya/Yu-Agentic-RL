# Phase 2A: Qwen2.5-7B Base vs Action-Prefix SFT tau2 Eval

## Goal

Run an end-to-end tau2 airline `pass^1` comparison:

```text
Qwen2.5-7B-Instruct base
vs
Qwen2.5-7B-Instruct + Action-Prefix SFT LoRA
```

This answers the next real project question: whether the behavior gain from
Action-Prefix SFT improves complete tool-agent task success, not only isolated
next-tool-call probes.

## New Runner

- Script: `scripts/run_tau2_base_vs_sft_vllm_autodl.sh`
- Serving: local vLLM OpenAI-compatible server
- Agent LLMs:
  - `openai/qwen25-7b-base`
  - `openai/qwen25-7b-action-prefix-sft`
- User simulator default: `anthropic/glm-5.1`
- Default tasks: `2,16,18,25,44`
- Default trials: `1`
- Default max steps: `80`

## Shutdown Policy

Default is no automatic shutdown:

```bash
AUTO_SHUTDOWN=0
```

Only set `AUTO_SHUTDOWN=1` when the user explicitly asks for a run-finished
shutdown.

Current operating rule:

- Do not proactively shut down AutoDL instances after remote runs.
- Only add shutdown commands when the user explicitly requests shutdown for
  that specific run.
- If GPU supply is tight, prefer leaving the instance available and reporting
  run status/artifact locations promptly.

## Expected Outputs

The script writes:

- `reports/airline_qwen25_7b_base_<tag>.summary.json`
- `reports/airline_qwen25_7b_sft_<tag>.summary.json`
- `reports/airline_qwen25_7b_base_vs_sft_<tag>.md`
- `reports/airline_qwen25_7b_base_vs_sft_<tag>.json`
- `outputs/vllm_logs/*.log`

tau2 trajectories are written under:

```text
third_party/tau2-bench/data/simulations/
```

## First Remote Command

On AutoDL, from `/root/autodl-tmp/yuyu`:

```bash
INSTALL_VLLM=1 TASK_IDS=2,16 NUM_TRIALS=1 MAX_STEPS=60 TIMEOUT_SECONDS=900 \
bash scripts/run_tau2_base_vs_sft_vllm_autodl.sh
```

If the smoke works, expand to:

```bash
TASK_IDS=2,16,18,25,44 NUM_TRIALS=1 MAX_STEPS=80 TIMEOUT_SECONDS=1800 \
bash scripts/run_tau2_base_vs_sft_vllm_autodl.sh
```

## 2026-06-14 Remote Smoke Attempt

Attempt tag:

```text
phase2a_smoke_20260614_154827
```

Observed progress:

- Uploaded the lightweight tau2-bench checkout and confirmed the remote repo layout.
- Installed vLLM on the 4090D instance.
- Started the Qwen2.5-7B base vLLM OpenAI-compatible server.
- vLLM reached `/v1/models` successfully and occupied about 16GB GPU memory.

Failure point:

- tau2 did not start because `find_uv()` captured `pip install uv` logs into the
  command variable, so `timeout` tried to execute the install log instead of the
  `uv` binary.

Fix:

- `scripts/run_tau2_base_vs_sft_vllm_autodl.sh` now sends uv install logs to
  stderr, re-detects the installed `uv` executable, checks both common AutoDL
  locations, and falls back to a local shim using `python -m uv`.

This attempt validates the serving side. The next attempt should reuse the
installed vLLM/uv environment on the same instance and proceed directly to tau2
evaluation after uploading the fixed runner.

## 2026-06-14 Remote Smoke Attempt 2

Attempt tag:

```text
phase2a_smoke2_20260614_162354
```

Observed progress:

- Uploaded the fixed runner.
- Reused the installed vLLM environment.
- Started both the base vLLM server and the SFT LoRA vLLM server.
- tau2 executed for both base and SFT and wrote summaries/comparison files.

Failure point:

- The run was not a valid model-quality result. All tasks ended as
  infrastructure errors because LiteLLM looked for `ANTHROPIC_API_KEY` while
  the remote env file used `ANTHROPIC_AUTH_TOKEN`.

Fix:

- The runner now maps `ANTHROPIC_AUTH_TOKEN` to `ANTHROPIC_API_KEY` and
  `ANTHROPIC_BASE_URL` to `ANTHROPIC_API_BASE` before launching tau2.

This attempt validates the full eval harness path up to external user-simulator
authentication. The next attempt should produce real task trajectories.

## 2026-06-14 Remote Smoke Attempt 3

Attempt tag:

```text
phase2a_smoke3_20260614_164046
```

Observed progress:

- GLM/Anthropic-compatible authentication was fixed.
- Base and SFT LoRA vLLM servers both launched.
- tau2 produced real trajectories rather than immediate auth failures.

Observed summaries:

```text
base: pass^1 = 0.5000, success_count = 1 / 2
SFT:  pass^1 = 0.0000, success_count = 0 / 2
```

Interpretation caveat:

- This is not a valid model-quality comparison yet.
- Base had one valid success on task 2 and one context-window infrastructure
  error on task 16.
- SFT had two context-window infrastructure errors. The failures occurred at
  about 7681 input tokens with 512 output tokens reserved, exceeding the 8192
  vLLM context limit by one or more tokens.

Fix:

- The runner now exposes `AGENT_MAX_TOKENS`.
- Next remote smoke should use:

```bash
AGENT_MAX_TOKENS=128 TASK_IDS=2,16 NUM_TRIALS=1 MAX_STEPS=60 TIMEOUT_SECONDS=1200 \
bash scripts/run_tau2_base_vs_sft_vllm_autodl.sh
```

This keeps the agent endpoint within the 8K vLLM serving budget without
confusing context overflow with model behavior.

## 2026-06-14 Remote Smoke Attempt 4

Attempt tag:

```text
phase2a_smoke4_20260614_165135
```

Configuration:

```text
AGENT_MAX_TOKENS=128
MAX_MODEL_LEN=8192
TASK_IDS=2,16
```

Observed summaries:

```text
base: pass^1 = 0.0000, success_count = 0 / 2
SFT:  pass^1 = 0.5000, success_count = 1 / 2
```

Details:

- Base task 2 finished normally but failed DB check.
- Base task 16 still hit a context-window infrastructure error:
  8065 input tokens + 128 output tokens = 8193, just over the 8192 limit.
- SFT task 2 finished normally and succeeded.
- SFT task 16 finished normally but failed DB check.

Interpretation caveat:

- This is the first smoke where the SFT run produced two normal task endings.
- The SFT direction is encouraging on this tiny sample, but the comparison is
  still not fully clean because the base run had one context-window infra error.

Artifact location:

```text
autodl_artifacts/phase2a_smoke4_20260614_165135/
```

Next run should use:

```bash
MAX_MODEL_LEN=9216 AGENT_MAX_TOKENS=128 TASK_IDS=2,16 NUM_TRIALS=1 MAX_STEPS=60 TIMEOUT_SECONDS=1200 \
bash scripts/run_tau2_base_vs_sft_vllm_autodl.sh
```

This should remove the remaining 8192-token boundary failure while staying close
to the 4090D memory limit.

## 2026-06-14 Remote Smoke Attempt 5

Attempt tag:

```text
phase2a_smoke5_20260614_170033
```

Configuration:

```text
AGENT_MAX_TOKENS=128
MAX_MODEL_LEN=9216
TASK_IDS=2,16
```

Observed summaries:

```text
base: pass^1 = 0.0000, success_count = 0 / 2
SFT:  pass^1 = 0.5000, success_count = 1 / 2
```

Details:

- Base still failed both tasks as context-window infrastructure errors.
- The exact boundary was 9089 input tokens + 128 output tokens = 9217, just
  over the 9216 limit.
- SFT finished both tasks normally: task 2 succeeded, task 16 failed DB check.
- The SFT run no longer had context-window infra failures at this setting.

Interpretation caveat:

- This is encouraging for the Action-Prefix SFT direction, because the SFT
  agent produced one real end-to-end success while the base run produced none.
- It is still not a clean base-vs-SFT quality comparison, because the base side
  is being dominated by prompt-length infrastructure failures.
- Small context increases are likely to keep chasing an off-by-one boundary.
  The next clean comparison should either choose shorter smoke tasks, raise
  context substantially, or add an evaluation-side history truncation strategy.

Artifact locations:

```text
autodl_artifacts/phase2a_smoke5_20260614_170033/
reports/airline_qwen25_7b_base_vs_sft_phase2a_smoke5_20260614_170033.md
```

## 2026-06-14 AutoDL Clone Check

New cloned instance status:

- GPU: RTX 5090 D, 32GB VRAM.
- GPU was idle at check time.
- `/root/autodl-tmp/yuyu` exists.
- Qwen2.5-7B-Instruct base model exists under `/root/autodl-tmp/models/`.
- Action-Prefix SFT LoRA checkpoint exists under
  `/root/autodl-tmp/yuyu/outputs/sft_action_prefix_v2_qwen25_7b_qlora_2048/checkpoint`.
- Prior Phase 2A smoke reports and artifacts are present.
- No vLLM/tau2/shutdown process was running at check time.

Environment note:

- `uv` and `tau2` are available.
- vLLM initially failed to import on this 5090D clone because torch 2.11 raised
  `AssertionError: duplicate template name` during an optional Inductor fallback
  patch.
- A reversible site-package patch was applied on the remote instance:
  `vllm/env_override.py` now skips that optional patch if torch raises during
  import. The original file was backed up beside it with a `.bak_yuyu_20260614`
  suffix.
- Additional 5090D compatibility settings were required for vLLM serving:
  `--enforce-eager`, `--no-enable-flashinfer-autotune`,
  `TORCHDYNAMO_DISABLE=1`, and `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Additional reversible site-package patches were applied to make optional
  AITer/MLA fusion and inductor compile-cache imports fail closed under this
  image. These are compatibility patches for eval serving, not model changes.
- After these patches/settings, vLLM 12K serving succeeded on the 5090D clone.

## 2026-06-14 Clean 12K 5090D Smoke

Attempt tag:

```text
phase2a_clean12k_5090d_20260614_175700
```

Configuration:

```text
MAX_MODEL_LEN=12288
GPU_MEMORY_UTILIZATION=0.82
AGENT_MAX_TOKENS=128
TASK_IDS=2,16
VLLM_ENFORCE_EAGER=1
VLLM_ENABLE_FLASHINFER_AUTOTUNE=0
VLLM_TORCHDYNAMO_DISABLE=1
VLLM_USE_FLASHINFER_SAMPLER=0
```

Observed summaries:

```text
base: pass^1 = 0.0000, success_count = 0 / 2
SFT:  pass^1 = 0.5000, success_count = 1 / 2
```

Details:

- This is the first clean Qwen2.5-7B base-vs-SFT tau2 comparison for this
  project: both base and SFT finished with normal `user_stop` terminations.
- No context-window infrastructure errors occurred.
- Base failed both tasks with DB mismatch.
- SFT succeeded on task 2 and failed task 16 with DB mismatch.
- This gives the first end-to-end pass^1 signal that Action-Prefix SFT improves
  task success on the small smoke set: `0.0000 -> 0.5000`.

Interpretation caveat:

- N=2 is still only a smoke result, not a final claim.
- The next result worth writing up should scale the same clean 12K/5090D
  configuration to a larger task slice, ideally 5-10 tasks first.

Artifact locations:

```text
autodl_artifacts/phase2a_clean12k_5090d_20260614_175700/
reports/airline_qwen25_7b_base_vs_sft_phase2a_clean12k_5090d_20260614_175700.md
```

## Decision Rule

If SFT pass^1 improves over base, scale data and context.

If next-tool-call metrics improve but pass^1 does not, switch to one of:

- constrained tool decoding,
- stricter tool-call-only target formatting,
- PRM rerank over multiple sampled trajectories.

## 2026-06-14 5-Task 12K 5090D Slice

Attempt tag:

```text
phase2a_clean12k_5task_5090d_20260614_180900
```

Configuration:

```text
MAX_MODEL_LEN=12288
GPU_MEMORY_UTILIZATION=0.82
AGENT_MAX_TOKENS=128
TASK_IDS=2,16,18,25,44
VLLM_ENFORCE_EAGER=1
VLLM_ENABLE_FLASHINFER_AUTOTUNE=0
VLLM_TORCHDYNAMO_DISABLE=1
VLLM_USE_FLASHINFER_SAMPLER=0
AUTO_SHUTDOWN=0
```

Observed summaries:

```text
base: pass^1 = 0.2000, success_count = 1 / 5, avg_reward = 0.2500
SFT:  pass^1 = 0.2000, success_count = 1 / 5, avg_reward = 0.2000
```

Details:

- Base succeeded only on task 2.
- SFT also succeeded only on task 2.
- SFT finished all 5 tasks normally, but did not improve pass^1 over base on
  this larger slice.
- Base task 44 hit a context-window infrastructure error:
  12161 input tokens + 128 reserved output tokens exceeded the 12288 limit by
  one token.
- The pass^1 conclusion is still meaningful as a warning signal, but this run
  should be labeled `context-caveated`, not a fully clean 5-task comparison.

Interpretation:

- The earlier 2-task clean smoke result looked encouraging, but this 5-task
  slice does not support claiming that Action-Prefix SFT v2 improves full tau2
  task success.
- The next algorithmic step should not be blind scale-up of the same SFT data.
  We need inspect the failed trajectories and move toward a sharper target:
  tool-call-only formatting, constrained tool decoding, or PRM-guided reranking.
- For fully clean 5-task comparison including task 44, rerun with a larger
  serving context such as 16K, or reduce reserved `AGENT_MAX_TOKENS`.

Artifact locations:

```text
autodl_artifacts/phase2a_clean12k_5task_5090d_20260614_180900/
reports/airline_qwen25_7b_base_vs_sft_phase2a_clean12k_5task_5090d_20260614_180900.md
```

Follow-up failure analysis:

```text
reports/phase2_action_prefix_sft_real_eval_failure_notes.md
```
