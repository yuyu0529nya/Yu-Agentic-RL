"""Collect GRPO rollouts by driving the EXISTING tau2 rollout machinery.

We do NOT reimplement the agent loop / tool execution / reward — we reuse the
battle-tested `tau2 run` path (same as our eval harness), just with
temperature>0 and N trials per task to get a diverse GROUP per task, then parse
the resulting simulations into a flat rollouts JSONL for the GRPO updater.

Assumes a vLLM server is ALREADY serving the current policy at OPENAI_API_BASE
(the driver script starts/stops it). The user simulator is GLM-5.1 via API
(ANTHROPIC_* env), so no local 72B is needed.

Output JSONL: one line per rollout:
  {"task_id","trial","reward","messages":[...openai-format...]}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def build_agent_args(api_base: str, api_key: str, temperature: float, max_tokens: int,
                     stop: str, include_stop: bool, parallel_tool_calls: bool) -> str:
    args: dict = {
        "api_base": api_base,
        "api_key": api_key,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if stop:
        args["stop"] = [stop]
        if include_stop:
            args["extra_body"] = {"include_stop_str_in_output": True}
    args["parallel_tool_calls"] = bool(parallel_tool_calls)
    return json.dumps(args)


def run_tau2(tau2_root: Path, tau2_cmd: str, domain: str, served_model: str, user_llm: str,
             agent_args: str, task_ids: list[str], num_trials: int, max_steps: int,
             timeout_s: int, seed: int, save_to: str, max_concurrency: int,
             max_retries: int, retry_delay: float, auto_resume: bool = True,
             user_llm_args: str = "") -> None:
    cmd = [
        *tau2_cmd.split(), "run",
        "--domain", domain,
        "--agent", "llm_agent",
        "--user", "user_simulator",
        "--agent-llm", f"openai/{served_model}",
        "--user-llm", user_llm,
        *(["--user-llm-args", user_llm_args] if user_llm_args else []),
        "--agent-llm-args", agent_args,
        "--num-trials", str(num_trials),
        "--max-concurrency", str(max_concurrency),
        "--max-steps", str(max_steps),
        "--max-retries", str(max_retries),
        "--retry-delay", str(retry_delay),
        "--seed", str(seed),
        "--save-to", save_to,
    ]
    if auto_resume:
        cmd.append("--auto-resume")
    cmd += ["--task-ids", *task_ids]
    print("[collect] running:", " ".join(cmd[:14]), "... tasks=", len(task_ids), "trials=", num_trials)
    # tau2 run wraps each sim; let it stream. Hard timeout guards a stuck run.
    subprocess.run(cmd, cwd=str(tau2_root), check=True, timeout=timeout_s)


def load_sims(sim_dir: Path) -> list[dict]:
    rj = sim_dir / "results.json"
    if rj.exists():
        data = json.loads(rj.read_text(encoding="utf-8"))
        sims = data.get("simulations") or []
        if sims:
            return sims
    # fallback: per-sim files
    sub = sim_dir / "simulations"
    if sub.is_dir():
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(sub.glob("*.json"))]
    raise SystemExit(f"no simulations found under {sim_dir}")


def reward_of(sim: dict) -> float:
    return float((sim.get("reward_info") or {}).get("reward") or 0.0)


def count_broken_tool_calls(sims: list[dict]) -> int:
    """assistant turns that emitted '<tool_call>' with NO closing tag and NO parsed
    tool_calls == a truncated/broken call (the max_tokens=256 defect). 0 in clean rollouts."""
    n = 0
    for sim in sims:
        for m in (sim.get("messages") or []):
            if m.get("role") != "assistant" or m.get("tool_calls"):
                continue
            c = m.get("content") or ""
            if "<tool_call>" in c and "</tool_call>" not in c:
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect GRPO rollouts via tau2 run (temp>0, N trials).")
    ap.add_argument("--domain", default="retail")
    ap.add_argument("--served-model", default="qwen25-7b-policy")
    ap.add_argument("--user-llm", default=os.environ.get("USER_LLM", "anthropic/glm-5.1"))
    ap.add_argument("--user-temperature", type=float, default=None,
                    help="pin the USER-SIMULATOR LLM temperature (e.g. 0.0) to cut its stochasticity — the dominant eval noise source. Also reads env USER_TEMP.")
    ap.add_argument("--user-api-base", default=None,
                    help="route user-sim to THIS endpoint for an INDEPENDENT usersim (not self-play)")
    ap.add_argument("--task-ids", required=True, help="comma-separated task ids")
    ap.add_argument("--num-trials", type=int, default=6, help="group size N per task")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--stop", default="</tool_call>")
    ap.add_argument("--include-stop", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--parallel-tool-calls", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-concurrency", type=int, default=int(os.environ.get("MAX_CONCURRENCY", "2")))
    ap.add_argument("--max-retries", type=int, default=5, help="tau2 retries (GLM overloaded_error 1305 needs backoff)")
    ap.add_argument("--retry-delay", type=float, default=6.0)
    ap.add_argument("--timeout-seconds", type=int, default=5400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    ap.add_argument("--tau2-root", default=str(REPO_ROOT / "third_party" / "tau2-bench"))
    ap.add_argument("--tau2-cmd", default=os.environ.get("TAU2_CMD", "tau2"),
                    help="how to invoke tau2 CLI (default installed 'tau2'; use 'uv run tau2' if uv-based)")
    ap.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True,
                    help="resume tau2 sims sharing --save-to (default True). Use --no-auto-resume for a FRESH eval (avoids reusing a prior checkpoint's cached sims).")
    ap.add_argument("--assert-no-broken-calls", action="store_true",
                    help="exit nonzero if any collected rollout contains a truncated/broken tool call (training hygiene gate).")
    ap.add_argument("--save-to", required=True, help="tau2 save_to name for this rollout batch")
    ap.add_argument("--out", required=True, help="output rollouts jsonl")
    args = ap.parse_args()

    tau2_root = Path(args.tau2_root)
    sim_dir = tau2_root / "data" / "simulations" / args.save_to
    if not args.auto_resume and sim_dir.exists():
        # tau2 INTERACTIVELY prompts "results.json exists, resume? (y/n)" when a save-to dir
        # is present -> EOFError under nohup. Delete it so the run is genuinely fresh.
        shutil.rmtree(sim_dir, ignore_errors=True)
        print(f"[collect] fresh run (--no-auto-resume): removed prior {sim_dir}")
    task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    agent_args = build_agent_args(args.api_base, args.api_key, args.temperature,
                                  args.max_tokens, args.stop, args.include_stop,
                                  args.parallel_tool_calls)

    ut = args.user_temperature
    if ut is None and os.environ.get("USER_TEMP"):
        ut = float(os.environ["USER_TEMP"])
    _ua: dict = {}
    if ut is not None:
        _ua["temperature"] = ut
    if args.user_api_base:
        _ua["api_base"] = args.user_api_base
        _ua["api_key"] = "dummy"
    user_llm_args = json.dumps(_ua) if _ua else ""
    if user_llm_args:
        print(f"[collect] user-llm-args: {user_llm_args}")

    run_tau2(tau2_root, args.tau2_cmd, args.domain, args.served_model, args.user_llm,
             agent_args, task_ids, args.num_trials, args.max_steps,
             args.timeout_seconds, args.seed, args.save_to,
             args.max_concurrency, args.max_retries, args.retry_delay, args.auto_resume,
             user_llm_args)

    sim_dir = tau2_root / "data" / "simulations" / args.save_to
    sims = load_sims(sim_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    rewards: list[float] = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, sim in enumerate(sims):
            msgs = sim.get("messages") or []
            if not msgs:
                continue
            r = reward_of(sim)
            rewards.append(r)
            f.write(json.dumps({
                "task_id": str(sim.get("task_id")),
                "trial": sim.get("trial", i),
                "reward": r,
                "messages": msgs,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    succ = sum(1 for r in rewards if r >= 1.0 - 1e-6)
    print(f"[collect] wrote {n_written} rollouts -> {out_path}")
    print(f"[collect] reward: success {succ}/{len(rewards)} mean={sum(rewards)/max(len(rewards),1):.3f}")
    # group reward std signal (GRPO needs intra-group variance to learn)
    by_task: dict[str, list[float]] = {}
    for sim in sims:
        by_task.setdefault(str(sim.get("task_id")), []).append(reward_of(sim))
    dead = sum(1 for v in by_task.values() if len(set(v)) <= 1)
    print(f"[collect] tasks with ZERO intra-group variance (no GRPO signal): {dead}/{len(by_task)}")
    broken = count_broken_tool_calls(sims)
    print(f"[collect] broken/truncated tool-calls (no </tool_call>, no parsed call): {broken}")
    if args.assert_no_broken_calls and broken > 0:
        print(f"[collect] FAIL: {broken} broken tool calls present -> raise --max-tokens. aborting (--assert-no-broken-calls).")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
