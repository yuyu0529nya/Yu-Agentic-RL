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
    DEFAULT_HELDOUT,
    DEFAULT_TOKENIZER_MODEL,
    DEFAULT_TRAIN,
    DEFAULT_VALID,
    display_path,
    load_jsonl,
    tool_call_payload,
    validate_samples,
    write_json,
    write_jsonl,
)
from build_action_prefix_dataset_v2 import token_count  # noqa: E402
from train_sft_smoke import encode_row, load_tokenizer  # noqa: E402


DEFAULT_OUT_DIR = "data/tool_call_protocol"
DEFAULT_REPORT = "reports/tool_call_protocol_dataset_v1.md"
DEFAULT_MANIFEST = "data/tool_call_protocol/tau2_airline_tool_call_protocol_manifest_v1.json"
DEFAULT_OUTPUT_STEM = "tau2_airline_tool_call_protocol_v1"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(call)
    function = item.setdefault("function", {})
    if not function.get("name"):
        function["name"] = item.get("name") or ""
    if "arguments" not in function:
        function["arguments"] = item.get("arguments") or {}
    if isinstance(function.get("arguments"), str):
        try:
            function["arguments"] = json.loads(function["arguments"])
        except json.JSONDecodeError:
            pass
    item["type"] = item.get("type") or "function"
    item["id"] = item.get("id") or f"call_protocol_{abs(hash(json.dumps(tool_call_payload(item), sort_keys=True, default=str)))}"
    return item


def with_messages(row: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(row)
    updated["messages"] = messages
    updated["loss_mask"] = [False] * (len(messages) - 1) + [True]
    updated["metadata"]["prefix_message_count"] = len(messages) - 1
    return updated


def drop_dangling_leading_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = list(messages)
    while kept and kept[0].get("role") == "tool":
        kept = kept[1:]
    return kept


def trim_prefix_to_budget(row: dict[str, Any], tokenizer: Any, max_sample_tokens: int) -> dict[str, Any]:
    original = copy.deepcopy(row)
    messages = original.get("messages") or []
    if len(messages) <= 1:
        return original

    prefix = list(messages[:-1])
    target = messages[-1]
    original_token_count = token_count(original, tokenizer)

    candidate = original
    kept_prefix = list(prefix)
    while token_count(candidate, tokenizer) > max_sample_tokens and kept_prefix:
        kept_prefix = drop_dangling_leading_tool_messages(kept_prefix[1:])
        candidate = with_messages(original, kept_prefix + [target])

    final_token_count = token_count(candidate, tokenizer)
    if final_token_count > max_sample_tokens:
        candidate = with_messages(original, [target])
        final_token_count = token_count(candidate, tokenizer)

    metadata = candidate.setdefault("metadata", {})
    metadata["context_trim"] = {
        "strategy": "message_suffix_target_preserving",
        "max_sample_tokens": max_sample_tokens,
        "original_qwen_tokens": original_token_count,
        "final_qwen_tokens": final_token_count,
        "original_prefix_message_count": len(prefix),
        "kept_prefix_message_count": len(candidate.get("messages") or []) - 1,
        "dropped_prefix_message_count": len(prefix) - (len(candidate.get("messages") or []) - 1),
    }
    metadata["qwen_tokens"] = final_token_count
    return candidate


def make_protocol_sample(
    source_row: dict[str, Any],
    split: str,
    turn_index: int,
    call_index: int,
    source_message: dict[str, Any],
) -> dict[str, Any] | None:
    source_calls = source_message.get("tool_calls") or []
    if len(source_calls) != 1:
        return None

    messages = source_row.get("messages") or []
    metadata = source_row.get("metadata") or {}
    call = normalize_tool_call(source_calls[call_index])
    payload = tool_call_payload(call)
    target = {
        "role": "assistant",
        "content": "",
        "tool_calls": [call],
    }
    sample_id = f"tool_call_protocol_{split}_{source_row.get('id')}_turn{turn_index}_call{call_index}"
    return {
        "id": sample_id,
        "format_version": "tau2_airline_tool_call_protocol_v1",
        "sample_type": "tool_call_protocol",
        "split": split,
        "messages": copy.deepcopy(messages[:turn_index]) + [target],
        "loss_mask": [False] * turn_index + [True],
        "loss_policy": {
            "assistant_content": False,
            "assistant_tool_calls": True,
            "assistant_tool_call_wrappers": True,
            "user": False,
            "tool": False,
        },
        "metadata": {
            "source_id": source_row.get("id"),
            "source_format_version": source_row.get("format_version"),
            "source_sample_type": source_row.get("sample_type"),
            "domain": metadata.get("domain"),
            "task_id": str(metadata.get("task_id") or ""),
            "trial": metadata.get("trial"),
            "simulation_id": metadata.get("simulation_id"),
            "official_split": metadata.get("official_split"),
            "source_reward": metadata.get("reward"),
            "source_process_score": metadata.get("process_score"),
            "source_risk_tags": metadata.get("risk_tags") or [],
            "turn_index": turn_index,
            "call_index": call_index,
            "prefix_message_count": turn_index,
            "target_tool_call_count": 1,
            "target_tool_names": [payload["name"]],
            "target_tool_calls": [payload],
            "source_assistant_content_chars": len(str(source_message.get("content") or "")),
            "target_content_removed": True,
            "protocol_wrapper_loss": True,
        },
    }


def build_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        for turn_index, message in enumerate(row.get("messages") or []):
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            source_calls = message.get("tool_calls") or []
            for call_index, _call in enumerate(source_calls):
                sample = make_protocol_sample(row, split, turn_index, call_index, message)
                if sample is not None:
                    samples.append(sample)
    return samples


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
    token_counts = []
    target_token_counts = []
    dropped_prefix = []
    source_content_chars = []
    for row in samples:
        metadata = row.get("metadata") or {}
        tasks[metadata.get("task_id")] += 1
        for tool in metadata.get("target_tool_names") or []:
            tools[tool] += 1
        if metadata.get("qwen_tokens") is not None:
            token_counts.append(int(metadata["qwen_tokens"]))
        if metadata.get("qwen_target_tokens") is not None:
            target_token_counts.append(int(metadata["qwen_target_tokens"]))
        trim = metadata.get("context_trim") or {}
        dropped_prefix.append(int(trim.get("dropped_prefix_message_count") or 0))
        source_content_chars.append(int(metadata.get("source_assistant_content_chars") or 0))
    return {
        "rows": len(samples),
        "tasks": dict(sorted(tasks.items(), key=lambda item: (int(item[0]) if str(item[0]).isdigit() else 10**9, str(item[0])))),
        "tools": dict(sorted(tools.items())),
        "mean_tokens": mean(token_counts) if token_counts else 0,
        "p90_tokens": percentile(token_counts, 0.9),
        "max_tokens": max(token_counts, default=0),
        "mean_target_tokens": mean(target_token_counts) if target_token_counts else 0,
        "max_target_tokens": max(target_token_counts, default=0),
        "trimmed_rows": sum(1 for value in dropped_prefix if value > 0),
        "mean_dropped_prefix_messages": mean(dropped_prefix) if dropped_prefix else 0,
        "max_dropped_prefix_messages": max(dropped_prefix, default=0),
        "mean_removed_content_chars": mean(source_content_chars) if source_content_chars else 0,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tool-Call Protocol Dataset v1",
        "",
        "## Goal",
        "",
        "Build executable tool-call supervision samples for Qwen/tau2. Each target assistant turn contains no natural language and exactly one tool call. The loss covers the `<tool_call>` protocol wrapper and JSON payload.",
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
            f"- Tokenizer model: `{payload['tokenizer_model']}`",
            f"- Max sample tokens: `{payload['max_sample_tokens']}`",
            "- Target assistant content: removed",
            "- Target tool calls: exactly one",
            "- Loss: `assistant_tool_call_wrappers=True`, `assistant_tool_calls=True`, `assistant_content=False`",
            "",
            "## Summary",
            "",
            "| Split | Rows | Tasks | Tools | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed rows | Mean dropped prefix msgs | Mean removed content chars |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, summary in payload["summaries"].items():
        tasks = ", ".join(f"{task}:{count}" for task, count in summary["tasks"].items())
        tools = ", ".join(f"{tool}:{count}" for tool, count in summary["tools"].items())
        lines.append(
            f"| {split} | {summary['rows']} | `{tasks}` | `{tools}` | "
            f"{summary['mean_tokens']:.1f} | {summary['p90_tokens']} | {summary['max_tokens']} | "
            f"{summary['mean_target_tokens']:.1f} | {summary['trimmed_rows']} | "
            f"{summary['mean_dropped_prefix_messages']:.1f} | {summary['mean_removed_content_chars']:.1f} |"
        )

    lines.extend(["", "## Validation", ""])
    if payload["validation_errors"]:
        lines.extend(["Structural/token validation errors:"] + [f"- `{error}`" for error in payload["validation_errors"]])
    else:
        lines.append("No structural/token validation errors.")

    lines.extend(
        [
            "",
            "## Why This Replaces Action-Prefix v2",
            "",
            "Action-Prefix v2 improved offline next-tool-call probes but the 5-task tau2 run showed protocol drift: the model emitted JSON as normal assistant text, so no tools were executed. This dataset explicitly trains the executable tool-call protocol.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build executable tool-call protocol SFT samples.")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--max-sample-tokens", type=int, default=2048)
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

    splits: dict[str, list[dict[str, Any]]] = {}
    validation_errors: list[str] = []
    for split, input_path in input_paths.items():
        source_rows = load_jsonl(input_path)
        samples = build_split(source_rows, split)
        samples = [trim_prefix_to_budget(row, tokenizer, args.max_sample_tokens) for row in samples]
        validation_errors.extend(validate_samples(samples))
        validation_errors.extend(add_token_stats(samples, tokenizer, args.max_sample_tokens))
        splits[split] = samples
        write_jsonl(output_paths[split], samples)

    summaries = {split: summarize(samples) for split, samples in splits.items()}
    manifest = {
        "format_version": "tau2_airline_tool_call_protocol_manifest_v1",
        "inputs": {split: display_path(path) for split, path in input_paths.items()},
        "outputs": {split: display_path(path) for split, path in output_paths.items()},
        "report": display_path(repo_path(args.report)),
        "tokenizer_model": args.tokenizer_model,
        "max_sample_tokens": args.max_sample_tokens,
        "summaries": summaries,
        "validation_errors": validation_errors,
    }
    write_json(repo_path(args.manifest), manifest)
    write_report(repo_path(args.report), manifest)

    print("Tool-call protocol dataset complete")
    print("===================================")
    for split, summary in summaries.items():
        print(
            f"{split}: rows={summary['rows']} max_tokens={summary['max_tokens']} "
            f"mean_target_tokens={summary['mean_target_tokens']:.1f}"
        )
    print(f"validation_errors: {len(validation_errors)}")
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
