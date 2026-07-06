from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def short(text: str | None, limit: int = 140) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tau2 domain tasks.")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument(
        "--tau2-root",
        type=Path,
        default=project_root() / "third_party" / "tau2-bench",
    )
    args = parser.parse_args()

    domain_dir = args.tau2_root / "data" / "tau2" / "domains" / args.domain
    tasks_path = domain_dir / "tasks.json"
    splits_path = domain_dir / "split_tasks.json"

    if not tasks_path.exists():
        raise SystemExit(f"tasks.json not found: {tasks_path}")

    tasks = load_json(tasks_path)
    splits = load_json(splits_path) if splits_path.exists() else {}

    reward_basis_counts: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    communicate_counts: Counter[int] = Counter()
    nl_assertion_counts: Counter[int] = Counter()
    tool_counts: Counter[str] = Counter()

    for task in tasks:
        criteria = task.get("evaluation_criteria") or {}
        reward_basis = tuple(criteria.get("reward_basis") or [])
        reward_basis_counts["+".join(reward_basis) or "<none>"] += 1

        actions = criteria.get("actions") or []
        action_counts[len(actions)] += 1
        for action in actions:
            tool_counts[action.get("name", "<unknown>")] += 1

        communicate_counts[len(criteria.get("communicate_info") or [])] += 1
        nl_assertion_counts[len(criteria.get("nl_assertions") or [])] += 1

    print(f"Domain: {args.domain}")
    print(f"Tasks: {len(tasks)}")
    if splits:
        print("Splits:")
        for name, ids in splits.items():
            print(f"  {name}: {len(ids)}")

    print("\nReward basis:")
    for key, count in reward_basis_counts.most_common():
        print(f"  {key}: {count}")

    print("\nReference action count per task:")
    for count, n_tasks in sorted(action_counts.items()):
        print(f"  {count} actions: {n_tasks} tasks")

    print("\nCommunicate-info count per task:")
    for count, n_tasks in sorted(communicate_counts.items()):
        print(f"  {count} items: {n_tasks} tasks")

    print("\nTop tools in reference actions:")
    for tool, count in tool_counts.most_common(12):
        print(f"  {tool}: {count}")

    print("\nExample tasks:")
    for task in tasks[: args.examples]:
        criteria = task.get("evaluation_criteria") or {}
        instructions = ((task.get("user_scenario") or {}).get("instructions") or {})
        purpose = (task.get("description") or {}).get("purpose")
        actions = criteria.get("actions") or []
        action_names = [a.get("name", "<unknown>") for a in actions[:6]]
        print(f"\n  Task {task.get('id')}")
        print(f"    purpose: {short(purpose)}")
        print(f"    reason: {short(instructions.get('reason_for_call'))}")
        print(f"    reward_basis: {criteria.get('reward_basis')}")
        print(f"    reference_actions: {len(actions)} {action_names}")


if __name__ == "__main__":
    main()
