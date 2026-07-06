# Mac Migration Notes

This project can be moved from Windows to macOS, but the Windows `.venv` should not be reused.
Rebuild the tau2-bench environment on the Mac with `uv`.

## 1. Unzip

```bash
mkdir -p ~/work
unzip ~/Downloads/yuyu_project_mac_private.zip -d ~/work
cd ~/work/yuyu
```

## 2. Install Basic Tools

If Homebrew is available:

```bash
brew install python@3.12 uv git
```

If `uv` is not available through Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal after installing `uv` if the command is not found.

## 3. Rebuild tau2-bench Environment

```bash
cd ~/work/yuyu/third_party/tau2-bench
uv sync
uv run tau2 check-data
```

The private migration zip keeps `third_party/tau2-bench/.env`, so API configuration should already be present.
Do not publish this zip or commit the `.env` file.

## 4. Verify Existing Reports

```bash
cd ~/work/yuyu
python3 scripts/summarize_tau2_results.py \
  third_party/tau2-bench/data/simulations/airline_prm_rerank_glm51_hard5_n4

python3 scripts/prm_rerank_tau2.py \
  third_party/tau2-bench/data/simulations/airline_prm_rerank_glm51_hard5_n4 \
  --score-mode heuristic_only
```

Expected hard5 N=4 rerank headline:

- first-trial pass: `0.4000`
- oracle pass@N: `0.8000`
- PRM-rerank pass@N: `0.8000`

## 5. Run a Small GLM 5.1 Test

From the project root:

```bash
bash scripts/run_tau2_airline_baseline.sh \
  --agent-llm anthropic/glm-5.1 \
  --user-llm anthropic/glm-5.1 \
  --task-ids 0,1 \
  --num-trials 1 \
  --max-concurrency 1 \
  --max-steps 80 \
  --save-to mac_smoke_glm51_2tasks
```

LiteLLM may still print cost-mapping/provider warnings for `glm-5.1`; those warnings affect cost display, not necessarily the simulation itself.
