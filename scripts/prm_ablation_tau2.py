from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REFERENCE_ACTION_NAMES = {
    "reference_actions_matched",
    "reference_action_mismatch",
}

PRIVILEGED_ACTION_NAMES = REFERENCE_ACTION_NAMES | {
    "unexpected_write_tool",
    "premature_deferral",
    "expected_write_plan_missing",
}


@dataclass(frozen=True)
class ComponentFilter:
    names: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    positive: bool = False
    negative: bool = False

    def matches(self, component: dict[str, Any]) -> bool:
        name = str(component.get("name") or "")
        tags = set(component.get("tags") or [])
        value = float(component.get("value") or 0.0)
        if name in self.names:
            return True
        if self.tags & tags:
            return True
        if self.positive and value > 0:
            return True
        if self.negative and value < 0:
            return True
        return False


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    exclude: ComponentFilter = field(default_factory=ComponentFilter)
    include: ComponentFilter | None = None
    zero_score: bool = False
    use_eval_tiebreak_tags: bool = False


ABLATIONS: list[AblationSpec] = [
    AblationSpec("full", "All PRM-Lite components."),
    AblationSpec(
        "full_eval_tiebreak",
        "All components plus reward-info DB/communication tags for tie-break only.",
        use_eval_tiebreak_tags=True,
    ),
    AblationSpec(
        "zero_score",
        "Remove all explicit components; deterministic tie-break reduces to first trial.",
        zero_score=True,
    ),
    AblationSpec(
        "scoreless_eval_tiebreak",
        "Remove all explicit components but keep reward-info DB/communication tie-break tags.",
        zero_score=True,
        use_eval_tiebreak_tags=True,
    ),
    AblationSpec(
        "no_reference_actions",
        "Remove direct tau2 reference action match/mismatch components.",
        exclude=ComponentFilter(names=REFERENCE_ACTION_NAMES),
    ),
    AblationSpec(
        "no_privileged_action_rules",
        "Remove reference actions plus expected-write and deferral components.",
        exclude=ComponentFilter(names=PRIVILEGED_ACTION_NAMES),
    ),
    AblationSpec(
        "no_premature_write",
        "Remove components tagged as premature write.",
        exclude=ComponentFilter(tags={"premature_write"}),
    ),
    AblationSpec(
        "no_payment_planning",
        "Remove payment-planning and calculation components.",
        exclude=ComponentFilter(tags={"payment_planning_error", "calculation_error"}),
    ),
    AblationSpec(
        "no_tool_affordance",
        "Remove tool affordance and premature-deferral components.",
        exclude=ComponentFilter(tags={"tool_affordance_error", "premature_deferral"}),
    ),
    AblationSpec(
        "no_temporal_policy",
        "Remove temporal policy components.",
        exclude=ComponentFilter(tags={"temporal_policy_error", "policy_precedence_error"}),
    ),
    AblationSpec(
        "no_compensation_policy",
        "Remove compensation policy components.",
        exclude=ComponentFilter(tags={"compensation_policy_error"}),
    ),
    AblationSpec(
        "no_evidence_object",
        "Remove incomplete-evidence and object-selection components.",
        exclude=ComponentFilter(tags={"incomplete_evidence", "object_selection_error"}),
    ),
    AblationSpec(
        "no_slot_grounding",
        "Remove slot-grounding and argument hallucination components.",
        exclude=ComponentFilter(tags={"slot_grounding_error", "argument_hallucination"}),
    ),
    AblationSpec(
        "negative_only",
        "Keep only negative penalty components.",
        include=ComponentFilter(negative=True),
    ),
    AblationSpec(
        "positive_only",
        "Keep only positive support components.",
        include=ComponentFilter(positive=True),
    ),
    AblationSpec(
        "reference_only",
        "Keep only reference action match/mismatch components.",
        include=ComponentFilter(names=REFERENCE_ACTION_NAMES),
    ),
    AblationSpec(
        "write_safety_only",
        "Keep only premature-write components.",
        include=ComponentFilter(tags={"premature_write"}),
    ),
    AblationSpec(
        "payment_planning_only",
        "Keep only payment-planning and calculation components.",
        include=ComponentFilter(tags={"payment_planning_error", "calculation_error"}),
    ),
    AblationSpec(
        "tool_affordance_only",
        "Keep only tool-affordance and premature-deferral components.",
        include=ComponentFilter(tags={"tool_affordance_error", "premature_deferral"}),
    ),
    AblationSpec(
        "temporal_policy_only",
        "Keep only temporal policy components.",
        include=ComponentFilter(tags={"temporal_policy_error", "policy_precedence_error"}),
    ),
    AblationSpec(
        "evidence_object_only",
        "Keep only incomplete-evidence and object-selection components.",
        include=ComponentFilter(tags={"incomplete_evidence", "object_selection_error"}),
    ),
    AblationSpec(
        "slot_grounding_only",
        "Keep only slot-grounding and argument hallucination components.",
        include=ComponentFilter(tags={"slot_grounding_error", "argument_hallucination"}),
    ),
]


def is_success(reward: Any) -> bool:
    try:
        return float(reward) >= 1.0
    except (TypeError, ValueError):
        return False


def filtered_components(sim: dict[str, Any], spec: AblationSpec) -> list[dict[str, Any]]:
    components = list(sim.get("components") or [])
    if spec.zero_score:
        return []
    if spec.include is not None:
        components = [component for component in components if spec.include.matches(component)]
    if spec.exclude.names or spec.exclude.tags or spec.exclude.positive or spec.exclude.negative:
        components = [component for component in components if not spec.exclude.matches(component)]
    return components


def score_components(
    sim: dict[str, Any], components: list[dict[str, Any]], use_eval_tiebreak_tags: bool
) -> tuple[float, float, list[str]]:
    score = sum(float(component.get("value") or 0.0) for component in components)
    max_abs = sum(abs(float(component.get("value") or 0.0)) for component in components)
    normalized = score / max_abs if max_abs else 0.0
    tags = {
        tag
        for component in components
        if float(component.get("value") or 0.0) < 0
        for tag in component.get("tags") or []
    }

    if use_eval_tiebreak_tags:
        if sim.get("db_match") is False and sim.get("communicate_score") == 1.0:
            tags.add("communication_db_gap")
        if sim.get("db_match") is False and not tags:
            tags.add("unclassified_db_failure")
    return score, normalized, sorted(tags)


def trial_key(sim: dict[str, Any]) -> tuple[int, str]:
    trial = sim.get("trial")
    return int(trial) if trial is not None else 0, str(sim.get("simulation_id") or "")


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            item["ablation_score"],
            item["ablation_normalized_score"],
            -len(item["ablation_tags"]),
            -(item.get("trial") if item.get("trial") is not None else 10**9),
        ),
    )


def build_rows(simulations: list[dict[str, Any]], spec: AblationSpec) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for sim in simulations:
        components = filtered_components(sim, spec)
        score, normalized, tags = score_components(
            sim, components, spec.use_eval_tiebreak_tags
        )
        item = dict(sim)
        item["ablation_name"] = spec.name
        item["ablation_score"] = score
        item["ablation_normalized_score"] = normalized
        item["ablation_tags"] = tags
        item["ablation_component_count"] = len(components)
        scored.append(item)

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        by_task[str(item.get("task_id"))].append(item)

    rows: list[dict[str, Any]] = []
    for task_id, candidates in by_task.items():
        ordered = sorted(candidates, key=trial_key)
        first = ordered[0]
        selected = select_candidate(ordered)
        successes = [item for item in ordered if is_success(item.get("reward"))]
        best_success = None
        if successes:
            best_success = max(
                successes,
                key=lambda item: (
                    item["ablation_score"],
                    item["ablation_normalized_score"],
                    -(item.get("trial") if item.get("trial") is not None else 10**9),
                ),
            )
        rows.append(
            {
                "ablation": spec.name,
                "task_id": task_id,
                "sample_count": len(ordered),
                "success_count": len(successes),
                "first_trial": first.get("trial"),
                "first_reward": first.get("reward"),
                "first_score": first["ablation_score"],
                "oracle_success": bool(successes),
                "selected_trial": selected.get("trial"),
                "selected_reward": selected.get("reward"),
                "selected_score": selected["ablation_score"],
                "selected_normalized_score": selected["ablation_normalized_score"],
                "selected_tags": selected["ablation_tags"],
                "best_success_trial": best_success.get("trial") if best_success else None,
                "best_success_score": best_success["ablation_score"] if best_success else None,
                "scores": [item["ablation_score"] for item in ordered],
                "rewards": [item.get("reward") for item in ordered],
                "component_counts": [item["ablation_component_count"] for item in ordered],
            }
        )
    return sorted(rows, key=lambda row: int(row["task_id"]))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_count = len(rows)
    sim_count = sum(int(row["sample_count"]) for row in rows)
    first_successes = sum(1 for row in rows if is_success(row["first_reward"]))
    selected_successes = sum(1 for row in rows if is_success(row["selected_reward"]))
    oracle_successes = sum(1 for row in rows if row["oracle_success"])
    sample_successes = sum(int(row["success_count"]) for row in rows)
    solvable = [row for row in rows if row["oracle_success"]]
    selected_solved = [row for row in solvable if is_success(row["selected_reward"])]
    return {
        "num_tasks": task_count,
        "num_simulations": sim_count,
        "first_trial_pass": first_successes / task_count if task_count else 0.0,
        "sample_success_rate": sample_successes / sim_count if sim_count else 0.0,
        "oracle_pass_at_n": oracle_successes / task_count if task_count else 0.0,
        "rerank_pass_at_n": selected_successes / task_count if task_count else 0.0,
        "gain_vs_first": (selected_successes - first_successes) / task_count if task_count else 0.0,
        "gap_to_oracle": (oracle_successes - selected_successes) / task_count if task_count else 0.0,
        "selection_accuracy_on_solvable": (
            len(selected_solved) / len(solvable) if solvable else None
        ),
        "first_success_count": first_successes,
        "selected_success_count": selected_successes,
        "oracle_success_count": oracle_successes,
        "sample_success_count": sample_successes,
        "solvable_task_count": len(solvable),
    }


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ablation",
                "task_id",
                "sample_count",
                "success_count",
                "first_trial",
                "first_reward",
                "first_score",
                "oracle_success",
                "selected_trial",
                "selected_reward",
                "selected_score",
                "selected_normalized_score",
                "selected_tags",
                "best_success_trial",
                "best_success_score",
                "scores",
                "rewards",
                "component_counts",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{
                        key: row.get(key)
                        for key in writer.fieldnames
                        if key not in {"selected_tags", "scores", "rewards", "component_counts"}
                    },
                    "selected_tags": "|".join(row.get("selected_tags") or []),
                    "scores": "|".join(format_float(score) for score in row.get("scores") or []),
                    "rewards": "|".join(format_float(reward) for reward in row.get("rewards") or []),
                    "component_counts": "|".join(
                        str(count) for count in row.get("component_counts") or []
                    ),
                }
            )


def write_markdown(
    path: Path,
    run_name: str,
    summaries: dict[str, dict[str, Any]],
    all_rows: dict[str, list[dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full = summaries["full"]
    lines = [
        f"# PRM Ablation Report: {run_name}",
        "",
        "## Summary",
        "",
        "| Ablation | Pass@N | Selected | Drop vs full | Gap to oracle | Selection acc. | Description |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for spec in ABLATIONS:
        summary = summaries[spec.name]
        drop = full["rerank_pass_at_n"] - summary["rerank_pass_at_n"]
        lines.append(
            f"| `{spec.name}` | {summary['rerank_pass_at_n']:.4f} | "
            f"{summary['selected_success_count']}/{summary['num_tasks']} | "
            f"{drop:+.4f} | {summary['gap_to_oracle']:.4f} | "
            f"{format_float(summary['selection_accuracy_on_solvable'])} | "
            f"{spec.description} |"
        )

    lines += [
        "",
        "## Leave-One-Family-Out",
        "",
        "| Ablation | Selected Tasks Changed vs Full | New Misses on Solvable Tasks |",
        "| --- | --- | --- |",
    ]
    full_selected = {row["task_id"]: row for row in all_rows["full"]}
    for spec in ABLATIONS:
        if spec.name in {
            "full",
            "full_eval_tiebreak",
            "zero_score",
            "scoreless_eval_tiebreak",
            "negative_only",
            "positive_only",
            "reference_only",
            "write_safety_only",
            "payment_planning_only",
            "tool_affordance_only",
            "temporal_policy_only",
            "evidence_object_only",
        }:
            continue
        changed: list[str] = []
        misses: list[str] = []
        for row in all_rows[spec.name]:
            base = full_selected[row["task_id"]]
            if row["selected_trial"] != base["selected_trial"]:
                changed.append(
                    f"{row['task_id']}:t{base['selected_trial']}->t{row['selected_trial']}"
                )
            if row["oracle_success"] and is_success(base["selected_reward"]) and not is_success(row["selected_reward"]):
                misses.append(str(row["task_id"]))
        lines.append(
            f"| `{spec.name}` | {', '.join(changed) or '-'} | {', '.join(misses) or '-'} |"
        )

    lines += [
        "",
        "## Single-Family-Only",
        "",
        "| Ablation | Pass@N | Selected Tasks |",
        "| --- | ---: | --- |",
    ]
    for name in [
        "reference_only",
        "write_safety_only",
        "payment_planning_only",
        "tool_affordance_only",
        "temporal_policy_only",
        "evidence_object_only",
        "positive_only",
        "negative_only",
        "scoreless_eval_tiebreak",
        "zero_score",
    ]:
        rows = all_rows[name]
        selected = [
            f"{row['task_id']}:t{row['selected_trial']}={format_float(row['selected_reward'])}"
            for row in rows
            if is_success(row["selected_reward"])
        ]
        lines.append(
            f"| `{name}` | {summaries[name]['rerank_pass_at_n']:.4f} | "
            f"{', '.join(selected) or '-'} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PRM-Lite component ablations on a process_reward_scorer JSON file."
    )
    parser.add_argument("path", type=Path, help="Path to *.prm_v2.json.")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    simulations = data.get("simulations") or []
    if not simulations:
        raise SystemExit(f"No simulations found in {args.path}")

    run_name = data.get("run") or args.path.stem
    all_rows: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []
    for spec in ABLATIONS:
        rows = build_rows(simulations, spec)
        summary = summarize(rows)
        summaries[spec.name] = {"description": spec.description, **summary}
        all_rows[spec.name] = rows
        flat_rows.extend(rows)

    payload = {
        "run": run_name,
        "source": str(args.path),
        "summaries": summaries,
        "tasks": all_rows,
    }

    print("PRM ablation summary")
    print("====================")
    full_pass = summaries["full"]["rerank_pass_at_n"]
    for spec in ABLATIONS:
        summary = summaries[spec.name]
        drop = full_pass - summary["rerank_pass_at_n"]
        print(
            f"{spec.name:28s} pass={summary['rerank_pass_at_n']:.4f} "
            f"selected={summary['selected_success_count']}/{summary['num_tasks']} "
            f"drop={drop:+.4f}"
        )

    if args.out_json:
        write_json(args.out_json, payload)
        print(f"\nwrote: {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, flat_rows)
        print(f"wrote: {args.out_csv}")
    if args.out_md:
        write_markdown(args.out_md, run_name, summaries, all_rows)
        print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
