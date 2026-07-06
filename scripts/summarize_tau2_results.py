from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    if path.is_dir():
        meta_path = path / "results.json"
        sims_dir = path / "simulations"
    else:
        meta_path = path
        sims_dir = path.parent / "simulations"

    if not meta_path.exists():
        raise SystemExit(f"results.json not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if sims_dir.is_dir():
        sims = []
        for sim_file in sorted(sims_dir.glob("*.json")):
            with sim_file.open("r", encoding="utf-8") as f:
                sims.append(json.load(f))
        data["simulations"] = sims

    return data


def is_success(reward: float | None) -> bool:
    return reward is not None and (1.0 - 1e-6) <= reward <= (1.0 + 1e-6)


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    if num_trials < k:
        raise ValueError(f"num_trials={num_trials} is less than k={k}")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def reward_of(sim: dict[str, Any]) -> float | None:
    reward_info = sim.get("reward_info") or {}
    return reward_info.get("reward")


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    sims = data.get("simulations") or []
    info = data.get("info") or {}
    env_info = info.get("environment_info") or {}
    agent_info = info.get("agent_info") or {}
    user_info = info.get("user_info") or {}

    rewards = [reward_of(sim) for sim in sims if reward_of(sim) is not None]
    by_task: dict[str, list[bool]] = defaultdict(list)
    termination = Counter()
    db = Counter()
    communicate = Counter()

    for sim in sims:
        reward = reward_of(sim)
        task_id = str(sim.get("task_id"))
        by_task[task_id].append(is_success(reward))
        termination[str(sim.get("termination_reason"))] += 1

        reward_info = sim.get("reward_info") or {}
        db_check = reward_info.get("db_check")
        if db_check is None:
            db["not_checked"] += 1
        elif db_check.get("db_match"):
            db["match"] += 1
        else:
            db["mismatch"] += 1

        breakdown = reward_info.get("reward_breakdown") or {}
        if "COMMUNICATE" in breakdown:
            communicate[str(breakdown["COMMUNICATE"])] += 1
        else:
            communicate["not_checked"] += 1

    max_k = min((len(v) for v in by_task.values()), default=0)
    pass_k = {}
    for k in range(1, max_k + 1):
        values = []
        for successes in by_task.values():
            values.append(pass_hat_k(len(successes), sum(successes), k))
        pass_k[f"pass^{k}"] = sum(values) / len(values) if values else 0.0

    return {
        "domain": env_info.get("domain_name"),
        "agent": agent_info.get("implementation"),
        "agent_llm": agent_info.get("llm"),
        "user": user_info.get("implementation"),
        "user_llm": user_info.get("llm"),
        "num_tasks": len(by_task),
        "num_simulations": len(sims),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "success_count": sum(1 for r in rewards if is_success(r)),
        "pass_k": pass_k,
        "termination": dict(termination),
        "db": dict(db),
        "communicate": dict(communicate),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("Tau2 result summary")
    print("===================")
    print(f"domain: {summary['domain']}")
    print(f"agent: {summary['agent']} ({summary['agent_llm']})")
    print(f"user: {summary['user']} ({summary['user_llm']})")
    print(f"tasks: {summary['num_tasks']}")
    print(f"simulations: {summary['num_simulations']}")
    print(f"avg_reward: {summary['avg_reward']:.4f}")
    print(f"success_count: {summary['success_count']}")

    if summary["pass_k"]:
        print("\npass^k:")
        for key, value in summary["pass_k"].items():
            print(f"  {key}: {value:.4f}")

    print("\ntermination:")
    for key, value in summary["termination"].items():
        print(f"  {key}: {value}")

    print("\nDB check:")
    for key, value in summary["db"].items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tau2 results.")
    parser.add_argument("path", type=Path, help="Path to a results directory or results.json")
    parser.add_argument("--out", type=Path, help="Optional JSON summary output path")
    args = parser.parse_args()

    summary = summarize(load_results(args.path))
    print_summary(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
