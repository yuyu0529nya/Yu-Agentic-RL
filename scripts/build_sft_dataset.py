from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from summarize_tau2_results import is_success, load_results, reward_of


DEFAULT_RUN = (
    "third_party/tau2-bench/data/simulations/"
    "airline_failed21_n4_timeout300_v2_merged"
)
DEFAULT_PRM = "reports/airline_failed21_n4_timeout300_v2_merged.prm_v3.json"


STABLE_FAIL_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "7": {
        "failure_families": [
            "communication_target_miss",
            "mid_conversation_intent_forgetting",
            "policy_workaround_execution",
        ],
        "correction_goals": [
            "Execute the upgrade-and-cancel sequence for XEHM4B and 59XX6W.",
            "Maintain a checklist of unresolved user intents after each user turn.",
            "After database writes, explicitly answer the inserted total-cost request.",
            "Ensure the final assistant response contains the required value 1628.",
        ],
        "expected_focus": "correct actions plus final communication of 1628",
    },
    "14": {
        "failure_families": [
            "payment_planning_error",
            "calculation_error",
            "basic_economy_update_attempt",
            "cancel_rebook_plan_not_closed",
        ],
        "correction_goals": [
            "Compute gift-card and certificate totals before any write.",
            "Apply the max-one-certificate constraint before payment arithmetic.",
            "Detect that basic economy requires cancel-and-book rather than update.",
            "Use one certificate, both gift cards, and Mastercard charge 1786.",
            "Communicate the final Mastercard charge before booking.",
        ],
        "expected_focus": "cancel-and-book with one certificate and Mastercard 1786",
    },
    "21": {
        "failure_families": [
            "temporal_route_optimization_error",
            "object_selection_error",
            "payment_method_selection_error",
            "baggage_policy_error",
        ],
        "correction_goals": [
            "Search return options from DEN to IAH on May 27.",
            "Compute full return duration including stopover time.",
            "Select HAT290/HAT175 as the fastest valid return.",
            "Use the smallest-balance gift card gift_card_6276644.",
            "Set nonfree_baggages=0 when membership/policy makes the bags free.",
        ],
        "expected_focus": "route-duration optimization plus exact write arguments",
    },
    "29": {
        "failure_families": [
            "policy_gate_missed",
            "cancel_rebook_plan_not_closed",
            "unexpected_write_tool",
            "payment_argument_error",
        ],
        "correction_goals": [
            "Recognize destination changes cannot be handled as a normal update.",
            "Cancel VA5SGQ before booking the replacement itinerary.",
            "Book HAT169/HAT033 as a new economy reservation.",
            "Preserve one checked bag, no insurance, and credit_card_8003957.",
            "Use the expected payment amount 282.",
        ],
        "expected_focus": "destination-change cancel-and-book policy gate",
    },
    "44": {
        "failure_families": [
            "tool_affordance_error",
            "incomplete_evidence",
            "multi_reservation_planning_error",
            "premature_write",
        ],
        "correction_goals": [
            "Use schedule-returning search tools rather than get_flight_status for duration.",
            "Build a table covering every future reservation and every leg.",
            "Classify cancel/upgrade eligibility before any write.",
            "Communicate total upgrade cost before write calls.",
            "Upgrade only the expected reservations and avoid invalid cancellations.",
        ],
        "expected_focus": "schedule evidence table before multi-reservation writes",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_official_splits(path: Path) -> dict[str, str]:
    data = load_json(path)
    result: dict[str, str] = {}
    for split_name in ("train", "test"):
        for task_id in data.get(split_name) or []:
            result[str(task_id)] = split_name
    return result


def simulation_key(sim: dict[str, Any]) -> str:
    return str(sim.get("id") or sim.get("simulation_id"))


def prm_by_simulation(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    return {str(item.get("simulation_id")): item for item in data.get("simulations") or []}


def build_task_map(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(task.get("id")): task for task in results.get("tasks") or []}


def task_summary(task: dict[str, Any] | None) -> dict[str, Any]:
    if not task:
        return {}
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions") or {}
    criteria = task.get("evaluation_criteria") or {}
    return {
        "description": task.get("description"),
        "user_instructions": instructions,
        "evaluation_criteria": criteria,
    }


def normalize_tool_calls(tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    calls = []
    for call in tool_calls or []:
        name = call.get("name") or (call.get("function") or {}).get("name")
        arguments = call.get("arguments")
        if arguments is None:
            arguments = (call.get("function") or {}).get("arguments") or {}
        calls.append(
            {
                "id": str(call.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(name or ""),
                    "arguments": arguments,
                },
            }
        )
    return calls


def normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[bool]]:
    call_names: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    loss_mask: list[bool] = []

    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        item: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
        if role == "assistant":
            calls = normalize_tool_calls(message.get("tool_calls"))
            if calls:
                item["tool_calls"] = calls
                for call in calls:
                    call_names[call["id"]] = call["function"]["name"]
        elif role == "tool":
            call_id = str(message.get("id") or message.get("tool_call_id") or "")
            item["tool_call_id"] = call_id
            item["name"] = str(message.get("name") or call_names.get(call_id) or "tool")

        normalized.append(item)
        loss_mask.append(role == "assistant")

    return normalized, loss_mask


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = 0
    for message in messages:
        chars += len(json.dumps(message, ensure_ascii=False))
    return max(1, round(chars / 4))


def tool_call_count(messages: list[dict[str, Any]]) -> int:
    return sum(len(message.get("tool_calls") or []) for message in messages)


def write_call_count(messages: list[dict[str, Any]]) -> int:
    write_tools = {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
    count = 0
    for message in messages:
        for call in message.get("tool_calls") or []:
            if call.get("function", {}).get("name") in write_tools:
                count += 1
    return count


def sample_base_metadata(
    sim: dict[str, Any],
    task: dict[str, Any] | None,
    prm: dict[str, Any],
    official_split: str,
) -> dict[str, Any]:
    task_id = str(sim.get("task_id"))
    reward = reward_of(sim)
    return {
        "simulation_id": simulation_key(sim),
        "task_id": task_id,
        "trial": sim.get("trial"),
        "official_split": official_split,
        "trainable": official_split == "train",
        "reward": reward,
        "process_score": prm.get("process_score"),
        "normalized_process_score": prm.get("normalized_process_score"),
        "risk_tags": prm.get("risk_tags") or [],
        "components": [
            {
                "name": component.get("name"),
                "value": component.get("value"),
                "tags": component.get("tags") or [],
            }
            for component in prm.get("components") or []
        ],
        "task": task_summary(task),
    }


def build_sft_samples(
    simulations: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    prm_scores: dict[str, dict[str, Any]],
    official_splits: dict[str, str],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sim in simulations:
        reward = reward_of(sim)
        if not is_success(reward):
            continue

        task_id = str(sim.get("task_id"))
        messages, loss_mask = normalize_messages(sim.get("messages") or [])
        prm = prm_scores.get(simulation_key(sim), {})
        official_split = official_splits.get(task_id, "unknown")
        sample = {
            "id": f"sft_success_task{task_id}_trial{sim.get('trial')}",
            "sample_type": "success_imitation",
            "format_version": "tau2_airline_sft_v1",
            "domain": "airline",
            "messages": messages,
            "loss_mask": loss_mask,
            "loss_policy": {
                "assistant_content": True,
                "assistant_tool_calls": True,
                "user": False,
                "tool": False,
            },
            "estimated_tokens": estimate_tokens(messages),
            "tool_calls": tool_call_count(messages),
            "write_calls": write_call_count(messages),
        }
        sample.update(
            sample_base_metadata(
                sim, tasks.get(task_id), prm, official_split
            )
        )
        samples.append(sample)
    return samples


def score_for_sort(sim: dict[str, Any], prm_scores: dict[str, dict[str, Any]]) -> float:
    prm = prm_scores.get(simulation_key(sim), {})
    value = prm.get("process_score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compact_pair_side(
    sim: dict[str, Any],
    task: dict[str, Any] | None,
    prm: dict[str, Any],
    official_split: str,
) -> dict[str, Any]:
    messages, loss_mask = normalize_messages(sim.get("messages") or [])
    side = {
        "simulation_id": simulation_key(sim),
        "task_id": str(sim.get("task_id")),
        "trial": sim.get("trial"),
        "official_split": official_split,
        "reward": reward_of(sim),
        "process_score": prm.get("process_score"),
        "risk_tags": prm.get("risk_tags") or [],
        "messages": messages,
        "loss_mask": loss_mask,
        "estimated_tokens": estimate_tokens(messages),
        "tool_calls": tool_call_count(messages),
        "write_calls": write_call_count(messages),
        "task": task_summary(task),
    }
    return side


def build_preference_pairs(
    simulations: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    prm_scores: dict[str, dict[str, Any]],
    official_splits: dict[str, str],
    max_pairs_per_task: int,
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sim in simulations:
        by_task[str(sim.get("task_id"))].append(sim)

    pairs: list[dict[str, Any]] = []
    for task_id, task_sims in sorted(by_task.items(), key=lambda item: int(item[0])):
        successes = [sim for sim in task_sims if is_success(reward_of(sim))]
        failures = [sim for sim in task_sims if not is_success(reward_of(sim))]
        if not successes or not failures:
            continue

        successes = sorted(
            successes,
            key=lambda sim: (-score_for_sort(sim, prm_scores), int(sim.get("trial") or 0)),
        )
        failures = sorted(
            failures,
            key=lambda sim: (score_for_sort(sim, prm_scores), int(sim.get("trial") or 0)),
        )
        chosen = successes[0]
        official_split = official_splits.get(task_id, "unknown")

        for rejected in failures[:max_pairs_per_task]:
            chosen_prm = prm_scores.get(simulation_key(chosen), {})
            rejected_prm = prm_scores.get(simulation_key(rejected), {})
            pair = {
                "id": (
                    f"pref_task{task_id}_chosen{chosen.get('trial')}"
                    f"_rejected{rejected.get('trial')}"
                ),
                "sample_type": "preference_pair",
                "format_version": "tau2_airline_preference_v1",
                "domain": "airline",
                "task_id": task_id,
                "official_split": official_split,
                "trainable": official_split == "train",
                "pair_policy": "best_success_vs_failure",
                "chosen": compact_pair_side(
                    chosen, tasks.get(task_id), chosen_prm, official_split
                ),
                "rejected": compact_pair_side(
                    rejected, tasks.get(task_id), rejected_prm, official_split
                ),
                "delta_reward": (reward_of(chosen) or 0.0) - (reward_of(rejected) or 0.0),
                "delta_process_score": (
                    score_for_sort(chosen, prm_scores)
                    - score_for_sort(rejected, prm_scores)
                ),
            }
            pairs.append(pair)
    return pairs


def build_correction_blueprints(
    tasks: dict[str, dict[str, Any]],
    official_splits: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for task_id, blueprint in sorted(
        STABLE_FAIL_BLUEPRINTS.items(), key=lambda item: int(item[0])
    ):
        official_split = official_splits.get(task_id, "unknown")
        rows.append(
            {
                "id": f"correction_blueprint_task{task_id}",
                "sample_type": "targeted_correction_blueprint",
                "format_version": "tau2_airline_correction_blueprint_v1",
                "domain": "airline",
                "task_id": task_id,
                "official_split": official_split,
                "trainable": official_split == "train",
                "synthetic_status": "blueprint_only_not_imitation",
                "source_report": "reports/stable_fail_audit_v1.md",
                "task": task_summary(tasks.get(task_id)),
                **blueprint,
            }
        )
    return rows


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[idx]


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("official_split")) for row in rows))


def trainable_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("trainable"))


def risk_tag_counts_from_pairs(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for pair in pairs:
        for tag in pair.get("rejected", {}).get("risk_tags") or []:
            counts[str(tag)] += 1
    return dict(sorted(counts.items()))


def task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("task_id")) for row in rows)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


def write_dataset_card(
    path: Path,
    run_path: Path,
    prm_path: Path,
    sft_samples: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    blueprints: list[dict[str, Any]],
    sft_path: Path,
    pair_path: Path,
    blueprint_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    token_values = [int(row.get("estimated_tokens") or 0) for row in sft_samples]
    pair_token_values = [
        int(pair["chosen"].get("estimated_tokens") or 0)
        + int(pair["rejected"].get("estimated_tokens") or 0)
        for pair in pairs
    ]
    train_tasks = sorted(
        {row["task_id"] for row in sft_samples if row.get("trainable")},
        key=int,
    )
    heldout_tasks = sorted(
        {row["task_id"] for row in sft_samples if not row.get("trainable")},
        key=int,
    )
    pair_tasks = sorted({pair["task_id"] for pair in pairs}, key=int)
    blueprint_train = [row["task_id"] for row in blueprints if row.get("trainable")]
    blueprint_heldout = [row["task_id"] for row in blueprints if not row.get("trainable")]

    lines = [
        "# SFT Dataset v1 Card",
        "",
        "## Source",
        "",
        f"- Tau2 run: `{display_path(run_path)}`",
        f"- PRM scores: `{display_path(prm_path)}`",
        "- Domain: `airline`",
        "- Official split source: `third_party/tau2-bench/data/tau2/domains/airline/split_tasks.json`",
        "",
        "## Outputs",
        "",
        f"- SFT imitation: `{display_path(sft_path)}`",
        f"- Preference pairs: `{display_path(pair_path)}`",
        f"- Correction blueprints: `{display_path(blueprint_path)}`",
        "",
        "## Summary",
        "",
        "| Dataset | Rows | Trainable | Split counts |",
        "| --- | ---: | ---: | --- |",
        (
            f"| Success imitation | {len(sft_samples)} | {trainable_count(sft_samples)} | "
            f"`{json.dumps(split_counts(sft_samples), sort_keys=True)}` |"
        ),
        (
            f"| Preference pairs | {len(pairs)} | {trainable_count(pairs)} | "
            f"`{json.dumps(split_counts(pairs), sort_keys=True)}` |"
        ),
        (
            f"| Correction blueprints | {len(blueprints)} | {trainable_count(blueprints)} | "
            f"`{json.dumps(split_counts(blueprints), sort_keys=True)}` |"
        ),
        "",
        "## Length Statistics",
        "",
        "Token counts are approximate character-based estimates for Phase 1A. "
        "Phase 1B should replace this with Qwen tokenizer rendering.",
        "",
        "| Dataset | Mean | P50 | P90 | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| SFT imitation | {mean(token_values):.1f} | "
            f"{percentile(token_values, 0.5)} | {percentile(token_values, 0.9)} | "
            f"{max(token_values) if token_values else 0} |"
        ),
        (
            f"| Preference chosen+rejected | {mean(pair_token_values):.1f} | "
            f"{percentile(pair_token_values, 0.5)} | {percentile(pair_token_values, 0.9)} | "
            f"{max(pair_token_values) if pair_token_values else 0} |"
        )
        if pair_token_values
        else "| Preference chosen+rejected | 0.0 | 0 | 0 | 0 |",
        "",
        "## Task Coverage",
        "",
        f"- Trainable SFT tasks: `{', '.join(train_tasks) or '-'}`",
        f"- Heldout SFT tasks: `{', '.join(heldout_tasks) or '-'}`",
        f"- Preference-pair tasks: `{', '.join(pair_tasks) or '-'}`",
        f"- Trainable correction blueprint tasks: `{', '.join(blueprint_train) or '-'}`",
        f"- Heldout correction blueprint tasks: `{', '.join(blueprint_heldout) or '-'}`",
        "",
        "### Success Imitation Rows Per Task",
        "",
        "| Task | Rows |",
        "| ---: | ---: |",
    ]
    for task_id, count in task_counts(sft_samples).items():
        lines.append(f"| {task_id} | {count} |")

    lines += [
        "",
        "## Failure-Signal Coverage From Rejected Preference Sides",
        "",
        "| Risk tag | Count |",
        "| --- | ---: |",
    ]
    tag_counts = risk_tag_counts_from_pairs(pairs)
    if tag_counts:
        for tag, count in tag_counts.items():
            lines.append(f"| `{tag}` | {count} |")
    else:
        lines.append("| - | 0 |")

    lines += [
        "",
        "## Data Policy",
        "",
        "- `success_imitation` rows are real reward=1 trajectories from the N=4 run.",
        "- `preference_pair` rows compare the best successful trajectory against failed trajectories for the same task.",
        "- `targeted_correction_blueprint` rows are not real demonstrations; they are planning specs for future synthetic or human-audited correction data.",
        "- Official test tasks are marked `trainable=false` to keep heldout evaluation clean.",
        "- Tool responses are retained as context. The intended SFT loss is only on assistant messages, including assistant tool calls and final answers.",
        "",
        "## Next Checks",
        "",
        "1. Render these rows with the target Qwen tokenizer/chat template.",
        "2. Verify assistant-only loss masks at token level.",
        "3. Filter `trainable=true` for actual SFT training.",
        "4. Keep official-test rows for heldout diagnostics, not training.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Build Phase 1A SFT and preference datasets from tau2 trajectories."
    )
    parser.add_argument("--run", type=Path, default=root / DEFAULT_RUN)
    parser.add_argument("--prm", type=Path, default=root / DEFAULT_PRM)
    parser.add_argument(
        "--split-tasks",
        type=Path,
        default=root / "third_party/tau2-bench/data/tau2/domains/airline/split_tasks.json",
    )
    parser.add_argument(
        "--out-sft",
        type=Path,
        default=root / "data/sft/tau2_airline_sft_v1.jsonl",
    )
    parser.add_argument(
        "--out-pairs",
        type=Path,
        default=root / "data/preferences/tau2_airline_pairs_v1.jsonl",
    )
    parser.add_argument(
        "--out-blueprints",
        type=Path,
        default=root / "data/sft/tau2_airline_correction_blueprints_v1.jsonl",
    )
    parser.add_argument(
        "--out-card",
        type=Path,
        default=root / "reports/sft_dataset_v1_card.md",
    )
    parser.add_argument("--max-pairs-per-task", type=int, default=8)
    args = parser.parse_args()

    results = load_results(args.run)
    simulations = results.get("simulations") or []
    if not simulations:
        raise SystemExit(f"No simulations found in {args.run}")

    tasks = build_task_map(results)
    official_splits = load_official_splits(args.split_tasks)
    prm_scores = prm_by_simulation(args.prm)

    sft_samples = build_sft_samples(simulations, tasks, prm_scores, official_splits)
    pairs = build_preference_pairs(
        simulations,
        tasks,
        prm_scores,
        official_splits,
        max_pairs_per_task=args.max_pairs_per_task,
    )
    blueprints = build_correction_blueprints(tasks, official_splits)

    write_jsonl(args.out_sft, sft_samples)
    write_jsonl(args.out_pairs, pairs)
    write_jsonl(args.out_blueprints, blueprints)
    write_dataset_card(
        args.out_card,
        args.run,
        args.prm,
        sft_samples,
        pairs,
        blueprints,
        args.out_sft,
        args.out_pairs,
        args.out_blueprints,
    )

    print("SFT dataset build complete")
    print("==========================")
    print(f"success_imitation: {len(sft_samples)} rows ({trainable_count(sft_samples)} trainable)")
    print(f"preference_pairs:  {len(pairs)} rows ({trainable_count(pairs)} trainable)")
    print(f"blueprints:        {len(blueprints)} rows ({trainable_count(blueprints)} trainable)")
    print(f"wrote: {args.out_sft}")
    print(f"wrote: {args.out_pairs}")
    print(f"wrote: {args.out_blueprints}")
    print(f"wrote: {args.out_card}")


if __name__ == "__main__":
    main()
