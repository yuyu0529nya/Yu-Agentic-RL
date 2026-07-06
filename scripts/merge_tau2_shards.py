from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from summarize_tau2_results import is_success, load_results, reward_of


TASK_TRIAL_PATTERNS = [
    re.compile(r"task[_-]?(?P<task_id>\d+)[_-]trial[_-]?(?P<trial>\d+)", re.I),
    re.compile(r"t(?P<task_id>\d+)[_-]tr(?:ial)?[_-]?(?P<trial>\d+)", re.I),
]


def parse_task_ids(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [item.strip() for item in re.split(r"[,\s]+", raw) if item.strip()]


def parse_task_trial(name: str) -> tuple[str | None, int | None]:
    for pattern in TASK_TRIAL_PATTERNS:
        match = pattern.search(name)
        if match:
            return match.group("task_id"), int(match.group("trial"))
    return None, None


def result_path_for_out(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    return path / "results.json"


def find_shard_results(shards_root: Path, prefix: str) -> list[Path]:
    results: list[Path] = []
    for result_path in sorted(shards_root.rglob("results.json")):
        if prefix and not result_path.parent.name.startswith(prefix):
            continue
        results.append(result_path)
    return results


def load_shard(result_path: Path) -> dict[str, Any]:
    return load_results(result_path)


def merge_shards(
    result_paths: list[Path],
    *,
    rewrite_trial_from_name: bool,
    expected_num_trials: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not result_paths:
        raise SystemExit("No shard results found.")

    base_info: dict[str, Any] | None = None
    tasks_by_id: dict[str, dict[str, Any]] = {}
    simulations: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen_task_trial: set[tuple[str, int | None]] = set()
    shard_records: list[dict[str, Any]] = []

    for result_path in result_paths:
        shard_name = result_path.parent.name
        shard_task_id, shard_trial = parse_task_trial(shard_name)
        data = load_shard(result_path)

        if base_info is None:
            base_info = copy.deepcopy(data.get("info") or {})

        for task in data.get("tasks") or []:
            task_id = str(task.get("id"))
            tasks_by_id.setdefault(task_id, copy.deepcopy(task))

        shard_sims = data.get("simulations") or []
        shard_records.append(
            {
                "name": shard_name,
                "path": str(result_path),
                "parsed_task_id": shard_task_id,
                "parsed_trial": shard_trial,
                "simulation_count": len(shard_sims),
            }
        )

        for sim in shard_sims:
            merged_sim = copy.deepcopy(sim)
            task_id = str(merged_sim.get("task_id"))
            trial = merged_sim.get("trial")

            if rewrite_trial_from_name and shard_trial is not None:
                merged_sim["trial"] = shard_trial
                trial = shard_trial

            if shard_task_id is not None and task_id != shard_task_id:
                print(
                    "warning: shard name task_id does not match simulation: "
                    f"{shard_name} has task {shard_task_id}, simulation has task {task_id}"
                )

            key = (task_id, int(trial) if trial is not None else None)
            if key in seen_task_trial:
                duplicates.append(
                    {
                        "task_id": task_id,
                        "trial": trial,
                        "simulation_id": merged_sim.get("id"),
                        "source": str(result_path),
                    }
                )
                continue

            seen_task_trial.add(key)
            simulations.append(merged_sim)

    info = base_info or {}
    if expected_num_trials is not None:
        info["num_trials"] = expected_num_trials

    simulations.sort(
        key=lambda sim: (
            int(sim.get("task_id")),
            int(sim.get("trial") if sim.get("trial") is not None else 0),
            str(sim.get("id")),
        )
    )
    tasks = [tasks_by_id[task_id] for task_id in sorted(tasks_by_id, key=int)]

    merged = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "info": info,
        "tasks": tasks,
        "simulations": simulations,
        "simulation_index": None,
    }

    samples_by_task = Counter(str(sim.get("task_id")) for sim in simulations)
    success_by_task: dict[str, int] = Counter()
    for sim in simulations:
        if is_success(reward_of(sim)):
            success_by_task[str(sim.get("task_id"))] += 1

    manifest = {
        "shard_count": len(result_paths),
        "simulation_count": len(simulations),
        "task_count": len(tasks),
        "samples_by_task": dict(sorted(samples_by_task.items(), key=lambda item: int(item[0]))),
        "successes_by_task": dict(
            sorted(success_by_task.items(), key=lambda item: int(item[0]))
        ),
        "duplicates_skipped": duplicates,
        "shards": shard_records,
    }
    return merged, manifest


def add_missing_manifest(
    manifest: dict[str, Any],
    task_ids: list[str],
    num_trials: int | None,
) -> None:
    if not task_ids or num_trials is None:
        return

    completed = {
        (str(sim_record["parsed_task_id"]), int(sim_record["parsed_trial"]))
        for sim_record in manifest["shards"]
        if sim_record["parsed_task_id"] is not None
        and sim_record["parsed_trial"] is not None
        and sim_record["simulation_count"] > 0
    }
    missing = []
    for task_id in task_ids:
        for trial in range(num_trials):
            if (str(task_id), trial) not in completed:
                missing.append({"task_id": str(task_id), "trial": trial})
    manifest["expected_task_ids"] = task_ids
    manifest["expected_num_trials"] = num_trials
    manifest["missing_shards"] = missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge timeout-safe tau2 shard runs.")
    parser.add_argument(
        "--shards-root",
        type=Path,
        required=True,
        help="Root directory containing shard result directories.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only merge shard directories whose names start with this prefix.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output results directory or results.json path.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Expected number of trials for the merged run metadata.",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Expected task ids, comma- or whitespace-separated, for missing-shard reporting.",
    )
    parser.add_argument(
        "--rewrite-trial-from-name",
        action="store_true",
        help="Rewrite simulation.trial from shard directory names like task_18_trial_1.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional manifest JSON path. Defaults to <out-dir>/merge_manifest.json.",
    )
    args = parser.parse_args()

    result_paths = find_shard_results(args.shards_root, args.prefix)
    out_results = result_path_for_out(args.out)
    result_paths = [path for path in result_paths if path.resolve() != out_results.resolve()]

    merged, manifest = merge_shards(
        result_paths,
        rewrite_trial_from_name=args.rewrite_trial_from_name,
        expected_num_trials=args.num_trials,
    )
    task_ids = parse_task_ids(args.task_ids)
    add_missing_manifest(manifest, task_ids, args.num_trials)

    out_results.parent.mkdir(parents=True, exist_ok=True)
    with out_results.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    manifest_out = args.manifest_out or (out_results.parent / "merge_manifest.json")
    with manifest_out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Merged tau2 shards")
    print("==================")
    print(f"shards: {manifest['shard_count']}")
    print(f"simulations: {manifest['simulation_count']}")
    print(f"tasks: {manifest['task_count']}")
    if "missing_shards" in manifest:
        print(f"missing_shards: {len(manifest['missing_shards'])}")
    if manifest["duplicates_skipped"]:
        print(f"duplicates_skipped: {len(manifest['duplicates_skipped'])}")
    print(f"wrote: {out_results}")
    print(f"wrote: {manifest_out}")


if __name__ == "__main__":
    main()
