"""Leakage-aware tau2 summary: pass^1 overall + by seen/unseen task split.

M1 deliverable. Reuses the parsing logic of summarize_tau2_results.py and adds a
seen/unseen breakdown so a credible generalization number can be reported:
- seen tasks were used in SFT train/valid -> high seen pass^1 may be memorization.
- unseen tasks were never trained on -> unseen pass^1 is the honest generalization metric.

Usage:
    python scripts/summarize_tau2_seen_unseen.py \
        third_party/tau2-bench/data/simulations/<run> \
        --split scripts/m1_eval_task_split.json \
        --out-json reports/<run>.seen_unseen.json \
        --out-md reports/<run>.seen_unseen.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_tau2_results import is_success, load_results, pass_hat_k, reward_of  # noqa: E402


def split_stats(by_task: dict[str, list[bool]], task_ids: list[str]) -> dict[str, Any]:
    """pass^1 averaged over the given task ids that are present in the run."""
    present = [t for t in task_ids if t in by_task]
    if not present:
        return {"tasks": 0, "trials": 0, "successes": 0, "pass^1": None}
    pass1_vals = []
    trials = 0
    successes = 0
    for t in present:
        outcomes = by_task[t]
        trials += len(outcomes)
        successes += sum(outcomes)
        pass1_vals.append(pass_hat_k(len(outcomes), sum(outcomes), 1))
    return {
        "tasks": len(present),
        "trials": trials,
        "successes": successes,
        "pass^1": sum(pass1_vals) / len(pass1_vals),
    }


def summarize_seen_unseen(data: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    sims = data.get("simulations") or []
    by_task: dict[str, list[bool]] = defaultdict(list)
    for sim in sims:
        by_task[str(sim.get("task_id"))].append(is_success(reward_of(sim)))

    seen_ids = [str(x) for x in split.get("seen_task_ids", [])]
    unseen_ids = [str(x) for x in split.get("unseen_task_ids", [])]
    known = set(seen_ids) | set(unseen_ids)
    other_ids = [t for t in by_task if t not in known]

    return {
        "num_tasks": len(by_task),
        "num_simulations": len(sims),
        "overall": split_stats(by_task, list(by_task.keys())),
        "seen": split_stats(by_task, seen_ids),
        "unseen": split_stats(by_task, unseen_ids),
        "unlabeled": split_stats(by_task, other_ids),
        "evaluated_seen_task_ids": sorted((t for t in by_task if t in set(seen_ids)), key=lambda x: int(x) if x.isdigit() else 1 << 30),
        "evaluated_unseen_task_ids": sorted((t for t in by_task if t in set(unseen_ids)), key=lambda x: int(x) if x.isdigit() else 1 << 30),
    }


def _fmt(stats: dict[str, Any]) -> str:
    p = stats["pass^1"]
    p = "-" if p is None else f"{p:.4f}"
    return f"{p} | {stats['successes']}/{stats['trials']} | {stats['tasks']}"


def to_md(summary: dict[str, Any], run_name: str) -> str:
    lines = [
        f"# Seen/Unseen Tau2 Summary: {run_name}",
        "",
        f"- simulations: {summary['num_simulations']}",
        f"- tasks evaluated: {summary['num_tasks']}",
        "",
        "| Split | pass^1 | successes/trials | tasks |",
        "| --- | ---: | ---: | ---: |",
        f"| overall | {_fmt(summary['overall'])} |",
        f"| seen (SFT train/valid) | {_fmt(summary['seen'])} |",
        f"| **unseen (generalization)** | {_fmt(summary['unseen'])} |",
    ]
    if summary["unlabeled"]["tasks"]:
        lines.append(f"| unlabeled | {_fmt(summary['unlabeled'])} |")
    lines += [
        "",
        "> unseen pass^1 is the honest, leakage-free generalization metric.",
        "> A large seen-minus-unseen gap indicates memorization of SFT tasks.",
        "",
        f"- evaluated seen ids: `{summary['evaluated_seen_task_ids']}`",
        f"- evaluated unseen ids: `{summary['evaluated_unseen_task_ids']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Seen/unseen tau2 pass^1 summary.")
    ap.add_argument("path", type=Path, help="tau2 simulations dir or results.json")
    ap.add_argument("--split", type=Path, default=Path(__file__).with_name("m1_eval_task_split.json"))
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-md", type=Path)
    args = ap.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    summary = summarize_seen_unseen(load_results(args.path), split)
    run_name = args.path.name.rstrip("/") or str(args.path)

    print(to_md(summary, run_name))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote: {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_md(summary, run_name), encoding="utf-8")
        print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
