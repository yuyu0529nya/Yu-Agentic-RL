from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from process_reward_scorer import (
    ScoreResult,
    classify_tags,
    default_tau2_root,
    format_optional,
    load_tasks,
    parse_policy_current_time,
    result_to_dict,
    score_simulation,
)
from summarize_tau2_results import is_success, load_results, reward_of


REFERENCE_ACTION_COMPONENTS = {
    "reference_actions_matched",
    "reference_action_mismatch",
}

PRIVILEGED_ACTION_COMPONENTS = REFERENCE_ACTION_COMPONENTS | {
    "unexpected_write_tool",
    "premature_deferral",
    "expected_write_plan_missing",
}


@dataclass
class RerankRow:
    task_id: str
    sample_count: int
    success_count: int
    first_trial: int | None
    first_reward: float | None
    first_process_score: float
    oracle_success: bool
    selected_trial: int | None
    selected_reward: float | None
    selected_process_score: float
    selected_normalized_process_score: float
    selected_tags: list[str]
    best_success_trial: int | None
    best_success_process_score: float | None
    process_scores: list[float]
    rewards: list[float | None]


def trial_sort_key(item: tuple[dict[str, Any], ScoreResult]) -> tuple[int, str]:
    sim, score = item
    trial = score.trial
    if trial is None:
        trial = sim.get("trial")
    trial_key = int(trial) if trial is not None else 0
    return trial_key, str(score.simulation_id)


def component_risk_tags(components) -> list[str]:
    tags = Counter()
    for component in components:
        if component.value < 0:
            for tag in component.tags:
                tags[tag] += 1
    return sorted(tags)


def select_by_process(candidates: list[tuple[dict[str, Any], ScoreResult]]) -> tuple[dict[str, Any], ScoreResult]:
    return max(
        candidates,
        key=lambda item: (
            item[1].process_score,
            item[1].normalized_process_score,
            -len(item[1].risk_tags),
            -(item[1].trial if item[1].trial is not None else 10**9),
        ),
    )


def build_rows(
    sims: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    current_time,
    score_mode: str,
    use_eval_tiebreak: bool,
) -> tuple[list[RerankRow], list[ScoreResult]]:
    scored: list[tuple[dict[str, Any], ScoreResult]] = []
    for sim in sims:
        task_id = str(sim.get("task_id"))
        score = score_simulation(sim, tasks.get(task_id), current_time)
        scored.append((sim, apply_score_mode(sim, score, score_mode, use_eval_tiebreak)))

    by_task: dict[str, list[tuple[dict[str, Any], ScoreResult]]] = defaultdict(list)
    for item in scored:
        by_task[str(item[0].get("task_id"))].append(item)

    rows: list[RerankRow] = []
    for task_id, candidates in by_task.items():
        ordered = sorted(candidates, key=trial_sort_key)
        first_sim, first_score = ordered[0]
        selected_sim, selected_score = select_by_process(ordered)
        successes = [
            (sim, score)
            for sim, score in ordered
            if is_success(score.reward if score.reward is not None else reward_of(sim))
        ]
        best_success_trial = None
        best_success_process_score = None
        if successes:
            _, best_success_score = max(
                successes,
                key=lambda item: (
                    item[1].process_score,
                    item[1].normalized_process_score,
                    -(item[1].trial if item[1].trial is not None else 10**9),
                ),
            )
            best_success_trial = best_success_score.trial
            best_success_process_score = best_success_score.process_score

        rows.append(
            RerankRow(
                task_id=task_id,
                sample_count=len(ordered),
                success_count=len(successes),
                first_trial=first_score.trial,
                first_reward=first_score.reward,
                first_process_score=first_score.process_score,
                oracle_success=bool(successes),
                selected_trial=selected_score.trial,
                selected_reward=selected_score.reward,
                selected_process_score=selected_score.process_score,
                selected_normalized_process_score=selected_score.normalized_process_score,
                selected_tags=selected_score.risk_tags,
                best_success_trial=best_success_trial,
                best_success_process_score=best_success_process_score,
                process_scores=[score.process_score for _, score in ordered],
                rewards=[score.reward for _, score in ordered],
            )
        )

    return sorted(rows, key=lambda row: int(row.task_id)), [score for _, score in scored]


def apply_score_mode(
    sim: dict[str, Any],
    score: ScoreResult,
    score_mode: str,
    use_eval_tiebreak: bool,
) -> ScoreResult:
    if score_mode == "full":
        components = score.components
        process_score = score.process_score
        normalized = score.normalized_process_score
    elif score_mode == "no_reference_actions":
        excluded = REFERENCE_ACTION_COMPONENTS
        components = [
            component for component in score.components if component.name not in excluded
        ]
        process_score = sum(component.value for component in components)
        max_abs = sum(abs(component.value) for component in components) or 1.0
        normalized = process_score / max_abs
    elif score_mode == "heuristic_only":
        excluded = PRIVILEGED_ACTION_COMPONENTS
        components = [
            component for component in score.components if component.name not in excluded
        ]
        process_score = sum(component.value for component in components)
        max_abs = sum(abs(component.value) for component in components) or 1.0
        normalized = process_score / max_abs
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    risk_tags = (
        classify_tags(sim, components, score.db_match)
        if use_eval_tiebreak
        else component_risk_tags(components)
    )

    return ScoreResult(
        simulation_id=score.simulation_id,
        task_id=score.task_id,
        trial=score.trial,
        reward=score.reward,
        db_match=score.db_match,
        communicate_score=score.communicate_score,
        process_score=process_score,
        normalized_process_score=normalized,
        tool_calls=score.tool_calls,
        write_calls=score.write_calls,
        risk_tags=risk_tags,
        components=components,
    )


def summarize_rows(rows: list[RerankRow]) -> dict[str, Any]:
    task_count = len(rows)
    sim_count = sum(row.sample_count for row in rows)
    first_successes = sum(1 for row in rows if is_success(row.first_reward))
    selected_successes = sum(1 for row in rows if is_success(row.selected_reward))
    oracle_successes = sum(1 for row in rows if row.oracle_success)
    sample_successes = sum(row.success_count for row in rows)
    solvable = [row for row in rows if row.oracle_success]
    selected_solved = [row for row in solvable if is_success(row.selected_reward)]

    return {
        "num_tasks": task_count,
        "num_simulations": sim_count,
        "min_samples_per_task": min((row.sample_count for row in rows), default=0),
        "max_samples_per_task": max((row.sample_count for row in rows), default=0),
        "first_trial_pass": first_successes / task_count if task_count else 0.0,
        "sample_success_rate": sample_successes / sim_count if sim_count else 0.0,
        "oracle_pass_at_n": oracle_successes / task_count if task_count else 0.0,
        "prm_rerank_pass_at_n": selected_successes / task_count if task_count else 0.0,
        "prm_gain_vs_first": (
            (selected_successes - first_successes) / task_count if task_count else 0.0
        ),
        "prm_gap_to_oracle": (
            (oracle_successes - selected_successes) / task_count if task_count else 0.0
        ),
        "selection_accuracy_on_solvable": (
            len(selected_solved) / len(solvable) if solvable else None
        ),
        "first_success_count": first_successes,
        "selected_success_count": selected_successes,
        "oracle_success_count": oracle_successes,
        "sample_success_count": sample_successes,
        "solvable_task_count": len(solvable),
    }


def row_to_dict(row: RerankRow) -> dict[str, Any]:
    return {
        "task_id": row.task_id,
        "sample_count": row.sample_count,
        "success_count": row.success_count,
        "first_trial": row.first_trial,
        "first_reward": row.first_reward,
        "first_process_score": row.first_process_score,
        "oracle_success": row.oracle_success,
        "selected_trial": row.selected_trial,
        "selected_reward": row.selected_reward,
        "selected_process_score": row.selected_process_score,
        "selected_normalized_process_score": row.selected_normalized_process_score,
        "selected_tags": row.selected_tags,
        "best_success_trial": row.best_success_trial,
        "best_success_process_score": row.best_success_process_score,
        "process_scores": row.process_scores,
        "rewards": row.rewards,
    }


def write_csv(path: Path, rows: list[RerankRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "sample_count",
                "success_count",
                "first_trial",
                "first_reward",
                "first_process_score",
                "oracle_success",
                "selected_trial",
                "selected_reward",
                "selected_process_score",
                "selected_normalized_process_score",
                "selected_tags",
                "best_success_trial",
                "best_success_process_score",
                "process_scores",
                "rewards",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task_id": row.task_id,
                    "sample_count": row.sample_count,
                    "success_count": row.success_count,
                    "first_trial": row.first_trial,
                    "first_reward": row.first_reward,
                    "first_process_score": f"{row.first_process_score:.4f}",
                    "oracle_success": row.oracle_success,
                    "selected_trial": row.selected_trial,
                    "selected_reward": row.selected_reward,
                    "selected_process_score": f"{row.selected_process_score:.4f}",
                    "selected_normalized_process_score": (
                        f"{row.selected_normalized_process_score:.4f}"
                    ),
                    "selected_tags": "|".join(row.selected_tags),
                    "best_success_trial": row.best_success_trial,
                    "best_success_process_score": format_optional(row.best_success_process_score),
                    "process_scores": "|".join(f"{score:.1f}" for score in row.process_scores),
                    "rewards": "|".join(format_optional(reward) for reward in row.rewards),
                }
            )


def write_markdown(
    path: Path,
    run_name: str,
    score_mode: str,
    tie_break_mode: str,
    summary: dict[str, Any],
    rows: list[RerankRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_label = (
        str(summary["min_samples_per_task"])
        if summary["min_samples_per_task"] == summary["max_samples_per_task"]
        else f"{summary['min_samples_per_task']}-{summary['max_samples_per_task']}"
    )
    lines = [
        f"# PRM-rerank Report: {run_name}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Score mode | `{score_mode}` |",
        f"| Tie-break mode | `{tie_break_mode}` |",
        f"| Tasks | {summary['num_tasks']} |",
        f"| Simulations | {summary['num_simulations']} |",
        f"| Samples per task | {n_label} |",
        f"| First-trial pass | {summary['first_trial_pass']:.4f} |",
        f"| Raw sample success rate | {summary['sample_success_rate']:.4f} |",
        f"| Oracle pass@N | {summary['oracle_pass_at_n']:.4f} |",
        f"| PRM-rerank pass@N | {summary['prm_rerank_pass_at_n']:.4f} |",
        f"| PRM gain vs first trial | {summary['prm_gain_vs_first']:+.4f} |",
        f"| PRM gap to oracle | {summary['prm_gap_to_oracle']:.4f} |",
        "| Selection accuracy on solvable tasks | "
        f"{format_optional(summary['selection_accuracy_on_solvable'])} |",
        "",
        "## Per-task Rerank",
        "",
        "| Task | Samples | Successes | First | Oracle | PRM Selected | Best Success | Scores | Tags |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        first = f"t{row.first_trial}: {format_optional(row.first_reward)} / {row.first_process_score:.1f}"
        selected = (
            f"t{row.selected_trial}: {format_optional(row.selected_reward)} / "
            f"{row.selected_process_score:.1f}"
        )
        best_success = (
            "-"
            if row.best_success_trial is None
            else f"t{row.best_success_trial}: {row.best_success_process_score:.1f}"
        )
        scores = "<br>".join(
            f"r={format_optional(reward)}, p={score:.1f}"
            for reward, score in zip(row.rewards, row.process_scores)
        )
        tags = ", ".join(f"`{tag}`" for tag in row.selected_tags) or "-"
        lines.append(
            f"| {row.task_id} | {row.sample_count} | {row.success_count} | {first} | "
            f"{row.oracle_success} | {selected} | {best_success} | {scores} | {tags} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(
    run_name: str,
    summary: dict[str, Any],
    rows: list[RerankRow],
    tie_break_mode: str,
) -> None:
    print("PRM-rerank summary")
    print("==================")
    print(f"run: {run_name}")
    print(f"tie_break_mode: {tie_break_mode}")
    print(f"tasks: {summary['num_tasks']}")
    print(f"simulations: {summary['num_simulations']}")
    print(
        "samples/task: "
        f"{summary['min_samples_per_task']}-{summary['max_samples_per_task']}"
    )
    print(f"first_trial_pass: {summary['first_trial_pass']:.4f}")
    print(f"sample_success_rate: {summary['sample_success_rate']:.4f}")
    print(f"oracle_pass@N: {summary['oracle_pass_at_n']:.4f}")
    print(f"prm_rerank_pass@N: {summary['prm_rerank_pass_at_n']:.4f}")
    print(f"gain_vs_first: {summary['prm_gain_vs_first']:+.4f}")
    print(f"gap_to_oracle: {summary['prm_gap_to_oracle']:.4f}")
    print(
        "selection_accuracy_on_solvable: "
        f"{format_optional(summary['selection_accuracy_on_solvable'])}"
    )
    print("\nper task:")
    for row in rows:
        print(
            f"  task {row.task_id}: successes={row.success_count}/{row.sample_count}, "
            f"first={format_optional(row.first_reward)}@{row.first_process_score:.1f}, "
            f"selected=t{row.selected_trial} "
            f"{format_optional(row.selected_reward)}@{row.selected_process_score:.1f}, "
            f"oracle={row.oracle_success}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank tau2 multi-sample trajectories with PRM-Lite process scores."
    )
    parser.add_argument("path", type=Path, help="Path to tau2 run dir or results.json.")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--tau2-root", type=Path, default=default_tau2_root())
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument(
        "--score-mode",
        choices=["full", "no_reference_actions", "heuristic_only"],
        default="full",
        help=(
            "full uses all PRM-Lite components; no_reference_actions removes direct "
            "tau2 action-check components; heuristic_only also removes components that "
            "depend on expected action names."
        ),
    )
    parser.add_argument(
        "--include-scores",
        action="store_true",
        help="Include full per-simulation PRM component details in JSON output.",
    )
    parser.add_argument(
        "--eval-tiebreak",
        action="store_true",
        help=(
            "Diagnostic only: allow post-hoc evaluator tags such as db/communication "
            "mismatch to break ties. The default online-safe mode uses only PRM "
            "component tags."
        ),
    )
    args = parser.parse_args()

    data = load_results(args.path)
    sims = data.get("simulations") or []
    if not sims:
        raise SystemExit(f"No simulations found in {args.path}")

    domain_dir = args.tau2_root / "data" / "tau2" / "domains" / args.domain
    tasks = load_tasks(domain_dir / "tasks.json")
    current_time = parse_policy_current_time(domain_dir / "policy.md")
    rows, scores = build_rows(
        sims, tasks, current_time, args.score_mode, args.eval_tiebreak
    )
    summary = summarize_rows(rows)
    run_name = args.path.name if args.path.is_dir() else args.path.parent.name
    tie_break_mode = "eval_tiebreak" if args.eval_tiebreak else "online_safe"

    payload: dict[str, Any] = {
        "run": run_name,
        "score_mode": args.score_mode,
        "tie_break_mode": tie_break_mode,
        "summary": summary,
        "tasks": [row_to_dict(row) for row in rows],
    }
    if args.include_scores:
        payload["simulations"] = [result_to_dict(score) for score in scores]

    print_summary(run_name, summary, rows, tie_break_mode)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nwrote: {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, rows)
        print(f"wrote: {args.out_csv}")
    if args.out_md:
        write_markdown(args.out_md, run_name, args.score_mode, tie_break_mode, summary, rows)
        print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
