from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_action_prefix_dataset import (  # noqa: E402
    DEFAULT_HELDOUT,
    DEFAULT_TOKENIZER_MODEL,
    DEFAULT_TRAIN,
    DEFAULT_VALID,
    build_split,
    display_path,
    load_jsonl,
    summarize,
    validate_samples,
    write_json,
    write_jsonl,
)
from build_action_prefix_dataset_v2 import (  # noqa: E402
    add_final_token_stats,
    trim_split,
    trim_summary,
)
from slot_grounding_validator import (  # noqa: E402
    SlotState,
    add_slots_from_text,
    add_slots_from_tool_result,
    call_name_and_args,
    extract_slots_from_text,
    parse_json_maybe,
    validate_call,
)
from train_sft_smoke import load_tokenizer  # noqa: E402


DEFAULT_OUT_DIR = "data/action_prefix"
DEFAULT_REPORT = "reports/action_prefix_slot_grounded_dataset_v3_1536.md"
DEFAULT_MANIFEST = "data/action_prefix/tau2_airline_action_prefix_slot_grounded_manifest_v3_1536.json"
DEFAULT_OUTPUT_STEM = "tau2_airline_action_prefix_slot_grounded_v3_1536"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def issue_to_dict(issue: Any) -> dict[str, Any]:
    return {
        "component_name": issue.component_name,
        "slot_type": issue.slot_type,
        "value": issue.value,
        "tool_name": issue.tool_name,
        "message_index": issue.message_index,
        "call_id": issue.call_id,
        "severity": issue.severity,
        "detail": issue.detail,
        "tags": issue.tags,
    }


def validate_target_grounding(row: dict[str, Any]) -> dict[str, Any]:
    """Validate only the target assistant tool-call turn against online prefix evidence."""
    messages = row.get("messages") or []
    target_index = len(messages) - 1
    target = messages[-1] if messages else {}
    state = SlotState()
    call_by_id: dict[str, tuple[str, dict[str, Any]]] = {}

    for message_index, message in enumerate(messages[:-1]):
        role = message.get("role")
        if role == "user":
            add_slots_from_text(state, str(message.get("content") or ""), "user", message_index)
            continue

        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name, arguments, call_id = call_name_and_args(call)
                if call_id:
                    call_by_id[call_id] = (name, arguments)
            continue

        if role == "tool":
            call_id = str(message.get("id") or message.get("tool_call_id") or "")
            name, arguments = call_by_id.get(call_id, ("", {}))
            content = message.get("content")
            content_text = str(content or "")
            if message.get("error") or content_text.lower().startswith("error:"):
                for slot_type, value in extract_slots_from_text(content_text):
                    state.remember_not_found(slot_type, value)
                continue
            add_slots_from_tool_result(
                state,
                name,
                arguments,
                parse_json_maybe(content),
                message_index,
            )

    issues = []
    assistant_text = str(target.get("content") or "")
    for call in target.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name, arguments, call_id = call_name_and_args(call)
        issues.extend(validate_call(state, name, arguments, call_id, target_index, assistant_text))

    return {
        "is_grounded": len(issues) == 0,
        "issue_count": len(issues),
        "issues": [issue_to_dict(issue) for issue in issues],
        "known_slot_counts_before_target": state.counts(),
    }


def annotate_grounding(row: dict[str, Any]) -> dict[str, Any]:
    result = validate_target_grounding(row)
    row.setdefault("metadata", {})["target_slot_grounding"] = result
    row["format_version"] = "tau2_airline_action_prefix_slot_grounded_v3"
    row["id"] = str(row.get("id")).replace("action_prefix_v2_", "action_prefix_slot_grounded_v3_", 1)
    return row


def split_clean_rejected(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in samples:
        annotated = annotate_grounding(row)
        grounding = annotated.get("metadata", {}).get("target_slot_grounding", {})
        if grounding.get("is_grounded"):
            clean.append(annotated)
        else:
            rejected.append(annotated)
    return clean, rejected


def grounding_summary(clean: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counts = Counter()
    task_rejections = Counter()
    tool_rejections = Counter()
    for row in rejected:
        metadata = row.get("metadata") or {}
        task_rejections[str(metadata.get("task_id") or "")] += 1
        for tool in metadata.get("target_tool_names") or []:
            tool_rejections[str(tool)] += 1
        grounding = metadata.get("target_slot_grounding") or {}
        for issue in grounding.get("issues") or []:
            issue_counts[str(issue.get("component_name") or "")] += 1

    total = len(clean) + len(rejected)
    return {
        "candidates": total,
        "kept": len(clean),
        "rejected": len(rejected),
        "keep_rate": (len(clean) / total) if total else 0.0,
        "issue_counts": dict(sorted(issue_counts.items())),
        "task_rejections": dict(sorted(task_rejections.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[0]))),
        "tool_rejections": dict(sorted(tool_rejections.items())),
    }


def validation_error_summary(errors: list[str]) -> dict[str, Any]:
    by_kind = Counter()
    for error in errors:
        kind = error.split(":", 1)[-1]
        by_kind[kind] += 1
    return {"count": len(errors), "by_kind": dict(sorted(by_kind.items()))}


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Slot-Grounded Action-Prefix Dataset v3",
        "",
        "## Goal",
        "",
        "Build action-prefix SFT samples where the target tool-call arguments are grounded in the online prefix. This directly targets the Phase2D failure mode: hallucinated `user_id` and `reservation_id` arguments.",
        "",
        "## Outputs",
        "",
    ]
    for split, out_path in payload["outputs"].items():
        lines.append(f"- {split}: `{out_path}`")
    for split, out_path in payload["rejected_outputs"].items():
        lines.append(f"- {split} rejected: `{out_path}`")

    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Tokenizer model: `{payload['tokenizer_model']}`",
            f"- Max sample tokens: `{payload['max_sample_tokens']}`",
            "- Base route: action-prefix v2 trimming plus target-only slot-grounding validation",
            "",
            "## Summary",
            "",
            "| Split | Candidates | Kept | Rejected | Keep rate | Mean tokens | Max tokens | Mean target tokens | Trimmed rows |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in payload["summaries"]:
        summary = payload["summaries"][split]
        grounding = payload["grounding_summaries"][split]
        trim = payload["trim_summaries"][split]
        lines.append(
            f"| {split} | {grounding['candidates']} | {grounding['kept']} | {grounding['rejected']} | "
            f"{grounding['keep_rate']:.3f} | {summary['mean_tokens']:.1f} | {summary['max_tokens']} | "
            f"{summary['mean_target_tokens']:.1f} | {trim['trimmed_rows']} |"
        )

    lines.extend(["", "## Rejection Issues", ""])
    for split, grounding in payload["grounding_summaries"].items():
        lines.append(f"### {split}")
        if grounding["issue_counts"]:
            lines.extend(["", "| Issue | Count |", "| --- | ---: |"])
            for issue, count in grounding["issue_counts"].items():
                lines.append(f"| `{issue}` | {count} |")
        else:
            lines.extend(["", "No rejected grounding issues."])
        lines.append("")

    lines.extend(["## Kept Task Coverage", ""])
    lines.extend(["| Split | Tasks | Tools |", "| --- | --- | --- |"])
    for split, summary in payload["summaries"].items():
        tasks = ", ".join(f"{task}:{count}" for task, count in summary["tasks"].items())
        tools = ", ".join(f"{tool}:{count}" for tool, count in summary["tools"].items())
        lines.append(f"| {split} | `{tasks}` | `{tools}` |")

    lines.extend(["", "## Validation", ""])
    validation = payload["validation_errors"]
    if validation["count"]:
        lines.append(f"Validation errors: `{validation['count']}`")
        for kind, count in validation["by_kind"].items():
            lines.append(f"- `{kind}`: {count}")
    else:
        lines.append("No structural/token validation errors on kept samples.")

    lines.extend(
        [
            "",
            "## Training Use",
            "",
            "- Train on the kept train split first.",
            "- Use the rejected files as hard negatives or future correction targets, not as SFT positives.",
            "- After training, rerun the Phase2D targeted tasks and check whether slot-grounding issues drop before expecting pass@4 to rise.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build slot-grounded action-prefix v3 SFT samples.")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--max-sample-tokens", type=int, default=1536)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(str(repo_path(args.tokenizer_model)), local_files_only=True)

    input_paths = {
        "train": repo_path(args.train),
        "valid": repo_path(args.valid),
        "heldout": repo_path(args.heldout),
    }
    out_dir = repo_path(args.out_dir)
    output_paths = {
        split: out_dir / f"{args.output_stem}_{split}.jsonl"
        for split in input_paths
    }
    rejected_paths = {
        split: out_dir / f"{args.output_stem}_{split}_rejected.jsonl"
        for split in input_paths
    }

    splits: dict[str, list[dict[str, Any]]] = {}
    rejected_splits: dict[str, list[dict[str, Any]]] = {}
    grounding_summaries: dict[str, dict[str, Any]] = {}
    validation_errors: list[str] = []

    for split, input_path in input_paths.items():
        source_rows = load_jsonl(input_path)
        candidates = build_split(source_rows, split)
        candidates = trim_split(candidates, tokenizer, args.max_sample_tokens)
        validation_errors.extend(add_final_token_stats(candidates, tokenizer, args.max_sample_tokens))
        clean, rejected = split_clean_rejected(candidates)
        validation_errors.extend(validate_samples(clean))
        splits[split] = clean
        rejected_splits[split] = rejected
        grounding_summaries[split] = grounding_summary(clean, rejected)
        write_jsonl(output_paths[split], clean)
        write_jsonl(rejected_paths[split], rejected)

    summaries = {split: summarize(samples) for split, samples in splits.items()}
    trim_summaries = {split: trim_summary(samples) for split, samples in splits.items()}
    manifest = {
        "format_version": "tau2_airline_action_prefix_slot_grounded_manifest_v3",
        "inputs": {split: display_path(path) for split, path in input_paths.items()},
        "outputs": {split: display_path(path) for split, path in output_paths.items()},
        "rejected_outputs": {split: display_path(path) for split, path in rejected_paths.items()},
        "report": display_path(repo_path(args.report)),
        "tokenizer_model": args.tokenizer_model,
        "max_sample_tokens": args.max_sample_tokens,
        "summaries": summaries,
        "trim_summaries": trim_summaries,
        "grounding_summaries": grounding_summaries,
        "validation_errors": validation_error_summary(validation_errors),
    }
    write_json(repo_path(args.manifest), manifest)
    write_report(repo_path(args.report), manifest)

    print("Slot-grounded action-prefix v3 dataset complete")
    print("===============================================")
    for split, grounding in grounding_summaries.items():
        summary = summaries[split]
        print(
            f"{split}: candidates={grounding['candidates']} kept={grounding['kept']} "
            f"rejected={grounding['rejected']} keep_rate={grounding['keep_rate']:.3f} "
            f"max_tokens={summary['max_tokens']}"
        )
        if grounding["issue_counts"]:
            print("  issues: " + ", ".join(f"{k}={v}" for k, v in grounding["issue_counts"].items()))
    print(f"validation_errors: {len(validation_errors)}")
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
