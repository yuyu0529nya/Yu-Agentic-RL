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

from build_action_prefix_dataset import (  # noqa: E402
    display_path,
    load_jsonl,
    tool_call_payload,
    validate_samples,
    write_json,
    write_jsonl,
)
from build_tool_call_protocol_dataset import normalize_tool_call  # noqa: E402
from train_sft_smoke import encode_row, load_tokenizer  # noqa: E402
from build_action_prefix_dataset_v2 import token_count  # noqa: E402


DEFAULT_TRAIN = "data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_train.jsonl"
DEFAULT_VALID = "data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_valid.jsonl"
DEFAULT_HELDOUT = "data/action_prefix/tau2_airline_action_prefix_slot_grounded_v3_1536_heldout.jsonl"
DEFAULT_OUT_DIR = "data/tool_call_protocol"
DEFAULT_REPORT = "reports/single_tool_protocol_dataset_v4_1536.md"
DEFAULT_MANIFEST = "data/tool_call_protocol/tau2_airline_single_tool_protocol_manifest_v4_1536.json"
DEFAULT_OUTPUT_STEM = "tau2_airline_single_tool_protocol_v4_1536"
DEFAULT_TOKENIZER_MODEL = "models/Qwen2.5-0.5B-Instruct"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def convert_row(row: dict[str, Any], split: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    messages = row.get("messages") or []
    if not messages:
        rejected = copy.deepcopy(row)
        rejected.setdefault("metadata", {})["single_tool_protocol_reject_reason"] = "empty_messages"
        return None, rejected

    target = messages[-1]
    target_calls = target.get("tool_calls") or []
    rejected = copy.deepcopy(row)
    rejected_metadata = rejected.setdefault("metadata", {})
    grounding = (row.get("metadata") or {}).get("target_slot_grounding") or {}
    if grounding and not grounding.get("is_grounded", False):
        rejected_metadata["single_tool_protocol_reject_reason"] = "target_not_grounded"
        return None, rejected
    if target.get("role") != "assistant":
        rejected_metadata["single_tool_protocol_reject_reason"] = "target_not_assistant"
        return None, rejected
    if len(target_calls) != 1:
        rejected_metadata["single_tool_protocol_reject_reason"] = f"target_tool_call_count_{len(target_calls)}"
        return None, rejected

    normalized_call = normalize_tool_call(target_calls[0])
    payload = tool_call_payload(normalized_call)
    converted = copy.deepcopy(row)
    converted_messages = converted["messages"]
    converted_messages[-1] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [normalized_call],
    }
    converted["id"] = str(row.get("id")).replace(
        "action_prefix_slot_grounded_v3_",
        "single_tool_protocol_v4_",
        1,
    )
    converted["format_version"] = "tau2_airline_single_tool_protocol_v4"
    converted["sample_type"] = "single_tool_protocol"
    converted["split"] = split
    converted["loss_mask"] = [False] * (len(converted_messages) - 1) + [True]
    converted["loss_policy"] = {
        "assistant_content": False,
        "assistant_tool_calls": True,
        "assistant_tool_call_wrappers": True,
        "user": False,
        "tool": False,
    }
    metadata = converted.setdefault("metadata", {})
    metadata["source_id"] = row.get("id")
    metadata["source_format_version"] = row.get("format_version")
    metadata["target_content_removed"] = True
    metadata["single_tool_protocol"] = True
    metadata["protocol_wrapper_loss"] = True
    metadata["target_tool_call_count"] = 1
    metadata["target_tool_names"] = [payload["name"]]
    metadata["target_tool_calls"] = [payload]
    return converted, None


def add_token_stats(samples: list[dict[str, Any]], tokenizer: Any, max_sample_tokens: int) -> list[str]:
    errors: list[str] = []
    for row in samples:
        final_token_count = token_count(row, tokenizer)
        try:
            encoded = encode_row(row, tokenizer, max_seq_len=max_sample_tokens)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('id')}: {type(exc).__name__}: {exc}")
            continue
        row["metadata"]["qwen_tokens"] = final_token_count
        row["metadata"]["qwen_target_tokens"] = encoded.target_token_count
        row["metadata"]["truncated_at_stats_max_seq_len"] = final_token_count > max_sample_tokens
        if final_token_count > max_sample_tokens:
            errors.append(f"{row.get('id')}:over_budget:{final_token_count}>{max_sample_tokens}")
        if encoded.target_token_count <= 0:
            errors.append(f"{row.get('id')}:target_tokens_zero")
    return errors


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    tools = Counter()
    token_counts: list[int] = []
    target_token_counts: list[int] = []
    prefix_messages: list[int] = []
    for row in samples:
        metadata = row.get("metadata") or {}
        tasks[str(metadata.get("task_id") or "")] += 1
        prefix_messages.append(int(metadata.get("prefix_message_count") or len(row.get("messages") or []) - 1))
        for tool in metadata.get("target_tool_names") or []:
            tools[str(tool)] += 1
        if metadata.get("qwen_tokens") is not None:
            token_counts.append(int(metadata["qwen_tokens"]))
        if metadata.get("qwen_target_tokens") is not None:
            target_token_counts.append(int(metadata["qwen_target_tokens"]))
    return {
        "rows": len(samples),
        "tasks": dict(sorted(tasks.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[0]))),
        "tools": dict(sorted(tools.items())),
        "mean_prefix_messages": mean(prefix_messages) if prefix_messages else 0,
        "max_prefix_messages": max(prefix_messages, default=0),
        "mean_tokens": mean(token_counts) if token_counts else 0,
        "p90_tokens": percentile(token_counts, 0.9),
        "max_tokens": max(token_counts, default=0),
        "mean_target_tokens": mean(target_token_counts) if target_token_counts else 0,
        "max_target_tokens": max(target_token_counts, default=0),
    }


def reject_summary(rejected: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter((row.get("metadata") or {}).get("single_tool_protocol_reject_reason") for row in rejected)
    tasks = Counter(str((row.get("metadata") or {}).get("task_id") or "") for row in rejected)
    return {
        "rows": len(rejected),
        "reasons": dict(sorted(reasons.items())),
        "tasks": dict(sorted(tasks.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[0]))),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Single-Tool Protocol Dataset v4",
        "",
        "## Goal",
        "",
        "Combine Phase2E slot-grounded supervision with the earlier executable tool-call protocol. Each target is exactly one grounded tool call with empty assistant text, and loss covers the `<tool_call>` wrapper plus JSON payload.",
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
            "- Source: slot-grounded action-prefix v3 kept samples",
            "- Target assistant content: removed",
            "- Target tool calls: exactly one",
            "- Loss: `assistant_tool_call_wrappers=True`, `assistant_tool_calls=True`, `assistant_content=False`",
            "",
            "## Summary",
            "",
            "| Split | Rows | Rejected | Tasks | Tools | Mean tokens | P90 tokens | Max tokens | Mean target tokens |",
            "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, summary in payload["summaries"].items():
        rejected = payload["rejected_summaries"][split]
        tasks = ", ".join(f"{task}:{count}" for task, count in summary["tasks"].items())
        tools = ", ".join(f"{tool}:{count}" for tool, count in summary["tools"].items())
        lines.append(
            f"| {split} | {summary['rows']} | {rejected['rows']} | `{tasks}` | `{tools}` | "
            f"{summary['mean_tokens']:.1f} | {summary['p90_tokens']} | {summary['max_tokens']} | "
            f"{summary['mean_target_tokens']:.1f} |"
        )

    lines.extend(["", "## Rejections", ""])
    for split, rejected in payload["rejected_summaries"].items():
        lines.append(f"### {split}")
        if rejected["reasons"]:
            lines.extend(["", "| Reason | Count |", "| --- | ---: |"])
            for reason, count in rejected["reasons"].items():
                lines.append(f"| `{reason}` | {count} |")
        else:
            lines.extend(["", "No rejected rows."])
        lines.append("")

    lines.extend(["## Validation", ""])
    if payload["validation_errors"]:
        lines.extend(["Structural/token validation errors:"] + [f"- `{error}`" for error in payload["validation_errors"]])
    else:
        lines.append("No structural/token validation errors.")

    lines.extend(
        [
            "",
            "## Intended Next Run",
            "",
            "- Train Qwen2.5-7B QLoRA on this dataset.",
            "- Evaluate with `--stop-sequence </tool_call>` and small `--max-new-tokens` such as 64 or 96.",
            "- Target: keep tool-name accuracy high while raising single-call rate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build slot-grounded single-tool protocol SFT data.")
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
    validation_errors: list[str] = []
    for split, input_path in input_paths.items():
        rows = load_jsonl(input_path)
        converted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for row in rows:
            sample, reject = convert_row(row, split)
            if sample is not None:
                converted.append(sample)
            if reject is not None:
                rejected.append(reject)
        validation_errors.extend(validate_samples(converted))
        validation_errors.extend(add_token_stats(converted, tokenizer, args.max_sample_tokens))
        splits[split] = converted
        rejected_splits[split] = rejected
        write_jsonl(output_paths[split], converted)
        write_jsonl(rejected_paths[split], rejected)

    summaries = {split: summarize(samples) for split, samples in splits.items()}
    rejected_summaries = {split: reject_summary(samples) for split, samples in rejected_splits.items()}
    manifest = {
        "format_version": "tau2_airline_single_tool_protocol_manifest_v4",
        "inputs": {split: display_path(path) for split, path in input_paths.items()},
        "outputs": {split: display_path(path) for split, path in output_paths.items()},
        "rejected_outputs": {split: display_path(path) for split, path in rejected_paths.items()},
        "report": display_path(repo_path(args.report)),
        "tokenizer_model": args.tokenizer_model,
        "max_sample_tokens": args.max_sample_tokens,
        "summaries": summaries,
        "rejected_summaries": rejected_summaries,
        "validation_errors": validation_errors,
    }
    write_json(repo_path(args.manifest), manifest)
    write_report(repo_path(args.report), manifest)

    print("Single-tool protocol v4 dataset complete")
    print("========================================")
    for split, summary in summaries.items():
        rejected = rejected_summaries[split]
        print(
            f"{split}: rows={summary['rows']} rejected={rejected['rows']} "
            f"max_tokens={summary['max_tokens']} mean_target_tokens={summary['mean_target_tokens']:.1f}"
        )
        if rejected["reasons"]:
            print("  rejected: " + ", ".join(f"{k}={v}" for k, v in rejected["reasons"].items()))
    print(f"validation_errors: {len(validation_errors)}")
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
