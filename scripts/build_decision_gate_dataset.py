from __future__ import annotations

import argparse
import copy
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

from build_mixed_dialogue_tool_policy_dataset import (  # noqa: E402
    display_path,
    load_jsonl,
    write_json,
    write_jsonl,
)
from check_sft_render_mask import render_with_spans, validate_structure  # noqa: E402


DEFAULT_TRAIN = "data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_train.jsonl"
DEFAULT_VALID = "data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_valid.jsonl"
DEFAULT_HELDOUT = "data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_v1_2048_heldout.jsonl"
DEFAULT_OUT_DIR = "data/decision_gate"
DEFAULT_OUTPUT_STEM = "tau2_airline_decision_gate_v1_2048"
DEFAULT_REPORT = "reports/decision_gate_dataset_v1_2048.md"
DEFAULT_MANIFEST = "data/decision_gate/tau2_airline_decision_gate_manifest_v1_2048.json"

VALID_LABELS = {"assistant_text", "tool_call"}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def decision_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    metadata = row.get("metadata") or {}
    source_id = str(metadata.get("source_id") or row.get("id") or "")
    turn_index = str(metadata.get("turn_index") if metadata.get("turn_index") is not None else "")
    call_index = str(metadata.get("source_call_index") if metadata.get("source_call_index") is not None else "")
    action = str(metadata.get("target_action") or "")
    prefix_count = str(metadata.get("prefix_message_count") if metadata.get("prefix_message_count") is not None else "")
    return source_id, turn_index, call_index, action, prefix_count


def is_protocol_only(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return bool(metadata.get("protocol_only") or row.get("sample_type") == "mixed_policy_protocol_tool")


def dedupe_decision_points(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []

    for row in rows:
        key = decision_key(row)
        if key not in kept_by_key:
            kept_by_key[key] = row
            continue

        current = kept_by_key[key]
        if is_protocol_only(current) and not is_protocol_only(row):
            dropped.append(current)
            kept_by_key[key] = row
        else:
            dropped.append(row)

    return list(kept_by_key.values()), dropped


def make_gate_sample(row: dict[str, Any], split: str) -> dict[str, Any]:
    messages = copy.deepcopy(row.get("messages") or [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"{row.get('id')} does not end with assistant target")

    metadata = copy.deepcopy(row.get("metadata") or {})
    label = str(metadata.get("target_action") or "")
    if label not in VALID_LABELS:
        raise ValueError(f"{row.get('id')} has invalid target_action={label!r}")

    prefix = messages[:-1]
    target = {
        "role": "assistant",
        "content": label,
    }

    gate_metadata = {
        "source_id": row.get("id"),
        "source_format_version": row.get("format_version"),
        "source_sample_type": row.get("sample_type"),
        "source_decision_key": list(decision_key(row)),
        "source_protocol_only": is_protocol_only(row),
        "domain": metadata.get("domain"),
        "task_id": str(metadata.get("task_id") or ""),
        "trial": metadata.get("trial"),
        "simulation_id": metadata.get("simulation_id"),
        "official_split": metadata.get("official_split"),
        "split": split,
        "turn_index": metadata.get("turn_index"),
        "prefix_message_count": len(prefix),
        "gate_label": label,
        "source_target_action": label,
        "source_target_tool_names": metadata.get("target_tool_names") or [],
        "source_target_tool_call_count": metadata.get("target_tool_call_count") or 0,
        "source_target_content_chars": metadata.get("target_content_chars") or 0,
    }

    return {
        "id": f"decision_gate_{split}_{row.get('id')}",
        "format_version": "tau2_airline_decision_gate_v1",
        "sample_type": "decision_gate_label",
        "split": split,
        "messages": prefix + [target],
        "loss_mask": [False] * len(prefix) + [True],
        "loss_policy": {
            "assistant_content": True,
            "assistant_tool_calls": False,
            "assistant_tool_call_wrappers": False,
            "user": False,
            "tool": False,
        },
        "metadata": gate_metadata,
    }


def validate_samples(samples: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for row in samples:
        row_id = str(row.get("id"))
        if row_id in seen:
            errors.append(f"duplicate_id:{row_id}")
        seen.add(row_id)

        structure_errors, _warnings = validate_structure(row)
        errors.extend(f"{row_id}:{error}" for error in structure_errors)

        messages = row.get("messages") or []
        target = messages[-1] if messages else {}
        metadata = row.get("metadata") or {}
        label = metadata.get("gate_label")
        if label not in VALID_LABELS:
            errors.append(f"{row_id}:invalid_gate_label:{label}")
        if target.get("content") != label:
            errors.append(f"{row_id}:target_content_not_gate_label")
        if target.get("tool_calls"):
            errors.append(f"{row_id}:gate_target_has_tool_calls")
        if row.get("loss_mask") != [False] * (len(messages) - 1) + [True]:
            errors.append(f"{row_id}:invalid_loss_mask")

        rendered = render_with_spans(row)
        if not rendered.target_spans:
            errors.append(f"{row_id}:no_target_spans")
        marked = rendered.marked_text
        if label and label not in marked:
            errors.append(f"{row_id}:label_missing_from_render")
    return errors


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actions = Counter((row.get("metadata") or {}).get("gate_label") for row in rows)
    tasks = Counter(str((row.get("metadata") or {}).get("task_id") or "") for row in rows)
    turns = [int((row.get("metadata") or {}).get("prefix_message_count") or 0) for row in rows]
    return {
        "rows": len(rows),
        "actions": dict(sorted(actions.items())),
        "tasks": dict(sorted(tasks.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 999)),
        "mean_prefix_messages": mean(turns) if turns else 0.0,
        "max_prefix_messages": max(turns) if turns else 0,
    }


def build_split(rows: list[dict[str, Any]], split: str, include_initial_greeting: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    skipped: Counter[str] = Counter()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        if str(metadata.get("target_action") or "") not in VALID_LABELS:
            skipped["invalid_target_action"] += 1
            continue
        if not include_initial_greeting and int(metadata.get("turn_index") or 0) == 0:
            skipped["initial_greeting"] += 1
            continue
        filtered.append(row)

    deduped, dropped = dedupe_decision_points(filtered)
    samples = [make_gate_sample(row, split) for row in deduped]
    return samples, dropped, skipped


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Decision Gate Dataset v1",
        "",
        "## Goal",
        "",
        "Train a decision-only gate that predicts exactly one label for the next assistant action: `assistant_text` or `tool_call`.",
        "",
        "This is Phase2J's first artifact. It separates action decision from tool-call generation.",
        "",
        "## Outputs",
        "",
    ]
    for split, out_path in payload["outputs"].items():
        lines.append(f"- {split}: `{out_path}`")
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Input source: mixed-policy v1 rows",
            f"- Include initial greeting: `{payload['include_initial_greeting']}`",
            "- Duplicate decision points are removed by `(source_id, turn_index, call_index, action, prefix_count)`.",
            "- If a protocol-only and non-protocol row describe the same decision, the non-protocol row is kept.",
            "",
            "## Summary",
            "",
            "| Split | Rows | Actions | Tasks | Mean prefix msgs | Max prefix msgs | Dropped duplicates | Skipped |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for split in ("train", "valid", "heldout"):
        summary = payload["summaries"][split]
        actions = ", ".join(f"{key}:{value}" for key, value in summary["actions"].items())
        tasks = ", ".join(f"{key}:{value}" for key, value in summary["tasks"].items())
        skipped = ", ".join(f"{key}:{value}" for key, value in payload["skipped"][split].items()) or "-"
        lines.append(
            f"| {split} | {summary['rows']} | `{actions}` | `{tasks}` | "
            f"{summary['mean_prefix_messages']:.1f} | {summary['max_prefix_messages']} | "
            f"{payload['dedupe_dropped'][split]} | `{skipped}` |"
        )

    lines.extend(["", "## Validation", ""])
    if payload["validation_errors"]:
        lines.append(f"Validation errors: `{len(payload['validation_errors'])}`")
        for error in payload["validation_errors"][:50]:
            lines.append(f"- `{error}`")
    else:
        lines.append("No structural validation errors.")

    lines.extend(
        [
            "",
            "## Training Use",
            "",
            "- Train with normal assistant-content SFT; the target is only the short label.",
            "- At inference, classify generated text into `assistant_text` or `tool_call`.",
            "- Then route `tool_call` cases to a stronger tool generator such as Phase2H.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2J decision-only gate dataset.")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--include-initial-greeting", action="store_true")
    args = parser.parse_args()

    split_inputs = {
        "train": repo_path(args.train),
        "valid": repo_path(args.valid),
        "heldout": repo_path(args.heldout),
    }
    out_dir = repo_path(args.out_dir)
    outputs = {
        split: out_dir / f"{args.output_stem}_{split}.jsonl"
        for split in split_inputs
    }

    all_errors: list[str] = []
    summaries: dict[str, Any] = {}
    skipped: dict[str, dict[str, int]] = {}
    dedupe_dropped: dict[str, int] = {}
    built: dict[str, list[dict[str, Any]]] = {}

    for split, path in split_inputs.items():
        rows = load_jsonl(path)
        samples, dropped, split_skipped = build_split(rows, split, args.include_initial_greeting)
        errors = validate_samples(samples)
        all_errors.extend(errors)
        built[split] = samples
        summaries[split] = summarize(samples)
        skipped[split] = dict(sorted(split_skipped.items()))
        dedupe_dropped[split] = len(dropped)

    for split, rows in built.items():
        write_jsonl(outputs[split], rows)

    payload = {
        "format_version": "tau2_airline_decision_gate_manifest_v1",
        "inputs": {split: display_path(path) for split, path in split_inputs.items()},
        "outputs": {split: display_path(path) for split, path in outputs.items()},
        "include_initial_greeting": args.include_initial_greeting,
        "summaries": summaries,
        "skipped": skipped,
        "dedupe_dropped": dedupe_dropped,
        "validation_errors": all_errors,
    }
    write_json(repo_path(args.manifest), payload)
    write_report(repo_path(args.report), payload)

    print("Decision gate dataset complete")
    for split in ("train", "valid", "heldout"):
        print(f"{split}: rows={summaries[split]['rows']} actions={summaries[split]['actions']} dropped={dedupe_dropped[split]} skipped={skipped[split]}")
    print(f"validation_errors: {len(all_errors)}")
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")


if __name__ == "__main__":
    main()
