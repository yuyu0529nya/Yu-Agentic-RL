from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from process_reward_scorer import (
    default_tau2_root,
    load_tasks,
    parse_policy_current_time,
    score_simulation,
    summarize_scores,
)
from summarize_tau2_results import load_results, summarize


def load_prm_summary(path: Path, domain: str, tau2_root: Path) -> dict[str, Any]:
    data = load_results(path)
    domain_dir = tau2_root / "data" / "tau2" / "domains" / domain
    tasks = load_tasks(domain_dir / "tasks.json")
    current_time = parse_policy_current_time(domain_dir / "policy.md")
    scores = [
        score_simulation(sim, tasks.get(str(sim.get("task_id"))), current_time)
        for sim in data.get("simulations") or []
    ]
    return summarize_scores(scores)


def run_name(path: Path) -> str:
    return path.name if path.is_dir() else path.parent.name


def collect_row(path: Path, domain: str, tau2_root: Path, with_prm: bool) -> dict[str, Any]:
    outcome = summarize(load_results(path))
    pass_k = outcome.get("pass_k") or {}
    db = outcome.get("db") or {}
    communicate = outcome.get("communicate") or {}

    row: dict[str, Any] = {
        "run": run_name(path),
        "agent_llm": outcome.get("agent_llm"),
        "user_llm": outcome.get("user_llm"),
        "tasks": outcome.get("num_tasks"),
        "sims": outcome.get("num_simulations"),
        "avg_reward": outcome.get("avg_reward"),
        "success_count": outcome.get("success_count"),
        "pass^1": pass_k.get("pass^1"),
        "pass^2": pass_k.get("pass^2"),
        "pass^3": pass_k.get("pass^3"),
        "pass^4": pass_k.get("pass^4"),
        "db_match": db.get("match", 0),
        "db_mismatch": db.get("mismatch", 0),
        "communicate_1.0": communicate.get("1.0", 0),
    }

    if with_prm:
        prm = load_prm_summary(path, domain, tau2_root)
        row.update(
            {
                "avg_process_score": prm.get("avg_process_score"),
                "avg_process_score_success": prm.get("avg_process_score_success"),
                "avg_process_score_failure": prm.get("avg_process_score_failure"),
                "risk_tags": ";".join(
                    f"{tag}:{count}"
                    for tag, count in sorted((prm.get("risk_tags") or {}).items())
                ),
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run",
        "agent_llm",
        "tasks",
        "sims",
        "avg_reward",
        "pass^1",
        "db_match",
        "db_mismatch",
        "avg_process_score",
        "risk_tags",
    ]
    lines = [
        "# Tau2 Baseline Run Comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [format_cell(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|")


def print_rows(rows: list[dict[str, Any]]) -> None:
    print("Tau2 baseline comparison")
    print("========================")
    for row in rows:
        print(
            f"{row['run']}: reward={format_cell(row.get('avg_reward'))} "
            f"pass^1={format_cell(row.get('pass^1'))} "
            f"db={row.get('db_match')}/{row.get('db_match', 0) + row.get('db_mismatch', 0)} "
            f"prm={format_cell(row.get('avg_process_score'))}"
        )

    tags = Counter()
    for row in rows:
        for item in str(row.get("risk_tags") or "").split(";"):
            if not item:
                continue
            tag, _, count = item.partition(":")
            try:
                tags[tag] += int(count)
            except ValueError:
                pass
    if tags:
        print("\ncombined risk tags:")
        for tag, count in tags.most_common():
            print(f"  {tag}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tau2 run directories.")
    parser.add_argument("paths", type=Path, nargs="+", help="Run dirs or results.json files.")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--tau2-root", type=Path, default=default_tau2_root())
    parser.add_argument("--no-prm", action="store_true", help="Skip PRM-Lite scoring.")
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    rows = [
        collect_row(path, args.domain, args.tau2_root, with_prm=not args.no_prm)
        for path in args.paths
    ]
    print_rows(rows)

    if args.out_csv:
        write_csv(args.out_csv, rows)
        print(f"\nwrote: {args.out_csv}")
    if args.out_md:
        write_md(args.out_md, rows)
        print(f"wrote: {args.out_md}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote: {args.out_json}")


if __name__ == "__main__":
    main()
