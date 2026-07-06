from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_tau2_results import load_results, reward_of


def compact(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = value.replace("\r\n", "\n").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def pick_simulation(
    simulations: list[dict[str, Any]],
    task_id: str | None,
    trial: int | None,
    index: int,
) -> dict[str, Any]:
    if task_id is None:
        if index < 0 or index >= len(simulations):
            raise IndexError(f"Simulation index {index} out of range")
        return simulations[index]

    matches = [sim for sim in simulations if str(sim.get("task_id")) == task_id]
    if trial is not None:
        matches = [sim for sim in matches if sim.get("trial") == trial]
    if not matches:
        raise ValueError(f"No simulation found for task_id={task_id!r}, trial={trial}")
    return matches[0]


def print_tool_calls(message: dict[str, Any], limit: int) -> None:
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        name = call.get("name", "unknown_tool")
        arguments = compact(call.get("arguments", {}), limit)
        print(f"    tool_call: {name} {arguments}")


def print_tool_message(message: dict[str, Any], limit: int) -> None:
    if "tool_messages" in message:
        for item in message.get("tool_messages") or []:
            print_tool_message(item, limit)
        return

    name = message.get("name") or message.get("tool_name") or "tool"
    content = compact(message.get("content") or message.get("result"), limit)
    print(f"[tool:{name}] {content}")


def print_message(message: dict[str, Any], limit: int) -> None:
    role = message.get("role", "unknown")
    if role == "tool" or "tool_messages" in message:
        print_tool_message(message, limit)
        return

    content = compact(message.get("content"), limit)
    print(f"[{role}] {content}")
    print_tool_calls(message, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one tau2 trajectory.")
    parser.add_argument("path", type=Path, help="Run directory or results.json path.")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--trial", type=int, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=700)
    args = parser.parse_args()

    data = load_results(args.path)
    simulations = data.get("simulations", [])
    if not simulations:
        raise ValueError("No simulations found in results")

    sim = pick_simulation(simulations, args.task_id, args.trial, args.index)
    print("tau2 trajectory")
    print("===============")
    print(f"id:                 {sim.get('id')}")
    print(f"task_id:            {sim.get('task_id')}")
    print(f"trial:              {sim.get('trial')}")
    print(f"reward:             {reward_of(sim)}")
    print(f"termination_reason: {sim.get('termination_reason')}")
    print("")

    reward_info = sim.get("reward_info") or {}
    if reward_info:
        print("reward_info")
        print(json.dumps(reward_info, indent=2, ensure_ascii=False))
        print("")

    messages = sim.get("messages") or []
    print(f"messages ({len(messages)})")
    for idx, message in enumerate(messages):
        print("")
        print(f"--- #{idx} ---")
        print_message(message, args.limit)


if __name__ == "__main__":
    main()

