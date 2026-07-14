# tau2-bench airline → veRL GRPO (custom multi-turn agent loop)

On-policy GRPO training of **Qwen2.5-7B-Instruct** on **τ²-bench airline** using
**veRL**, via a custom agent loop that puts a real user simulator and a stateful
tool environment inside veRL's token-level rollout.

This is not a "register a few tools" integration. veRL's stock `ToolAgentLoop`
**terminates** as soon as the model stops emitting tool calls — it has no concept
of a user-simulator turn, and this veRL build ships no `interactions/` / `recipe/`
abstraction for one. So the core deliverable is a custom `@register("tau2_agent")`
loop that reproduces τ²'s agent ↔ user ↔ env turn-taking *at the token level*,
with correct response masking and a faithful reward.

## Architecture

```
                        ┌─────────────────────────── Tau2AgentLoop (veRL) ──────────────────────────┐
 policy vLLM  ──gen──▶  │  GENERATING ──has tool_call?──▶ PROCESSING_TOOLS ─┐                          │
 (GPU1, LoRA)           │      ▲                              │(exec on private env)                  │
                        │      │                              ▼                                        │
                        │      └──────────── mask=0 tokens ◀── tool response                           │
                        │      │                                                                       │
                        │      └─ no tool_call ─▶ PROCESSING_USER ─▶ user_sim.generate ─▶ mask=0 turn  │
                        │                              │(STOP? → TERMINATED, reward)                    │
                        └──────────────────────────────┼───────────────────────────────────────────────┘
                                                        ▼
 usersim vLLM (GPU0, :18001)  ◀── litellm openai/usersim ──┘        reward = tau2 evaluator(messages, task)
```

* **Policy tokens are trained (mask=1); tool responses and user turns are context (mask=0).**
* Each trajectory owns a **deep-copied `FlightDB`** — concurrent rollouts never share mutable state.
* **Reward is a pure function of (messages, task)**: τ²'s evaluator rebuilds a fresh env, replays the
  trajectory's tool calls, and compares DB hashes + string-matches `communicate_info`. No LLM at reward
  time (all 50 airline base tasks use `reward_basis = {DB, COMMUNICATE}`).

## Files

| file | role |
|---|---|
| `tau2_agent_loop.py`  | **Adapter B** — the `@register("tau2_agent")` loop. Reuses veRL's tested tokenization/masking; changes only (1) tools→tau2, (2) no-tool-call→user turn, (3) end→tau2 reward. |
| `tau2_bridge.py`      | **Adapters A+D** — per-trajectory env store, tool dispatch (`env.get_response`), structured-message builders (matched tool-call ids), reward via `evaluate_simulation`. All tau2 logic; zero veRL deps. |
| `rollout_context.py`  | **Adapter A** — `ContextVar`-based per-trajectory isolation (`rollout_scope` / `current_rollout`). |
| `data_prep_airline.py`| **Adapter E** — `tasks.json` → veRL parquet (`extra_info.task_id` is load-bearing; routing via `default_agent_loop`). |
| `serve_usersim_7b.sh` | **Adapter C** — user-simulator vLLM server (OpenAI-compatible, `:18001`, GPU0). |
| `agent_loop_config.yaml` | Registers `tau2_agent` → `_target_` for veRL/hydra. |
| `run_tau2_grpo_7b.sh` | Training launch (GPU1, 7B LoRA GRPO, multi-turn agent loop). Built on the working GSM8K config. |
| `test_tau2_loop_offline.py` | **CPU-only** validation of everything except GPU token-gen. |

## What is validated GPU-free (passing)

`python test_tau2_loop_offline.py` on the no-GPU box:

* **Bridge plumbing** — tools dispatch to the private env; `tool_call.id` matches the following
  `ToolMessage.id` (τ²'s evaluator requires this); user STOP is detected.
* **Reward fidelity** — replaying a task's **gold** actions + `communicate_info` → **reward 1.0**
  (`{DB:1.0, COMMUNICATE:1.0}`); an **unfinished** trajectory → **reward 0.0** (τ²'s premature-termination
  guard). This proves the reward is wired faithfully, matching the hand-written pipeline's `reward_info.reward`.
* **Concurrency isolation** — 8 trajectories driven concurrently keep 8 **distinct** DBs; the
  `ContextVar` scope resolves to the right trajectory in each async task.
* **Loop import + registration** — `tau2_agent_loop` imports against real veRL and registers as `tau2_agent`.

What still needs GPU: policy token generation (vLLM), the real usersim server, and veRL's runtime
`tokenization_sanity_check` (turn-by-turn vs whole-conversation token equality).

## Run (needs GPU mode)

```bash
# 0) build data (GPU-free, already done)
python data_prep_airline.py                       # → data/tau2_airline/{train,test}.parquet

# 1) usersim server on GPU0
bash serve_usersim_7b.sh

# 2) sanity: offline test still green
python test_tau2_loop_offline.py

# 3) GRPO on GPU1 (policy LoRA + colocated rollout; usersim on GPU0)
bash run_tau2_grpo_7b.sh
```

## Design decisions worth calling out (interview-facing)

* **Why a custom loop and not offline data?** veRL 0.9's trainer consumes the agent loop's token-level
  `response_mask`; feeding externally-collected OpenAI-format trajectories means re-tokenizing + rebuilding
  masks offline and losing on-policy freshness. For faithful on-policy τ² GRPO, the custom loop is the
  straight road.
* **Per-request env lifecycle.** veRL's `ToolAgentLoop._call_tool` does `create()`+`release()` *per call*,
  so BaseTool per-trajectory state does **not** persist across calls on that path. We therefore keep the
  τ² env in the trajectory-local `agent_data.extra_fields` (+ ContextVar), created once per `run()` and GC'd
  with it — no global dict, no leak.
* **Incremental tokenization.** Non-assistant turns (tool + user) are rendered in isolation with
  `remove_system_prompt=True` and prefixed with `turn_separator`, exactly as veRL's tested tool path does,
  so turn-by-turn tokens match a full re-tokenization.
```
