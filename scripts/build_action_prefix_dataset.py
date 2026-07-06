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

from check_sft_render_mask import validate_structure  # noqa: E402
from train_sft_smoke import encode_row, load_tokenizer  # noqa: E402


DEFAULT_TRAIN = "data/train/tau2_airline_sft_train.jsonl"
DEFAULT_VALID = "data/train/tau2_airline_sft_valid.jsonl"
DEFAULT_HELDOUT = "data/train/tau2_airline_sft_heldout.jsonl"
DEFAULT_OUT_DIR = "data/action_prefix"
DEFAULT_REPORT = "reports/action_prefix_dataset_v1.md"
DEFAULT_MANIFEST = "data/action_prefix/tau2_airline_action_prefix_manifest_v1.json"
DEFAULT_TOKENIZER_MODEL = "models/Qwen2.5-0.5B-Instruct"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tool_call_payload(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    args = function.get("arguments")
    if args is None:
        args = call.get("arguments") or {}
    return {
        "name": function.get("name") or call.get("name") or "",
        "arguments": args,
    }


def make_action_prefix_sample(
    source_row: dict[str, Any],
    split: str,
    turn_index: int,
    message: dict[str, Any],
) -> dict[str, Any]:
    messages = source_row.get("messages") or []
    metadata = source_row.get("metadata") or {}
    prefix = messages[:turn_index]
    target = dict(message)
    target_calls = [tool_call_payload(call) for call in target.get("tool_calls") or []]
    target_tool_names = [call["name"] for call in target_calls]
    sample_id = f"action_prefix_{split}_{source_row.get('id')}_turn{turn_index}"
    return {
        "id": sample_id,
        "format_version": "tau2_airline_action_prefix_v1",
        "sample_type": "action_prefix_tool_call",
        "split": split,
        "messages": prefix + [target],
        "loss_mask": [False] * len(prefix) + [True],
        "loss_policy": {
            "assistant_content": True,
            "assistant_tool_calls": True,
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
            "prefix_message_count": len(prefix),
            "target_tool_call_count": len(target_calls),
            "target_tool_names": target_tool_names,
            "target_tool_calls": target_calls,
        },
    }


def build_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        for turn_index, message in enumerate(row.get("messages") or []):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                samples.append(make_action_prefix_sample(row, split, turn_index, message))
    return samples


def validate_samples(samples: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for row in samples:
        row_id = str(row.get("id"))
        if row_id in seen:
            errors.append(f"duplicate_id:{row_id}")
        seen.add(row_id)

        structure_errors, _warnings = validate_structure(row)
        for error in structure_errors:
            errors.append(f"{row_id}:{error}")

        messages = row.get("messages") or []
        loss_mask = row.get("loss_mask") or []
        if not messages or messages[-1].get("role") != "assistant":
            errors.append(f"{row_id}:last_message_not_assistant")
        if not messages or not messages[-1].get("tool_calls"):
            errors.append(f"{row_id}:target_missing_tool_calls")
        if loss_mask != [False] * (len(messages) - 1) + [True]:
            errors.append(f"{row_id}:loss_mask_not_prefix_false_target_true")
    return errors


def add_token_stats(samples: list[dict[str, Any]], tokenizer: Any | None, max_seq_len: int) -> None:
    if tokenizer is None:
        return
    for row in samples:
        encoded = encode_row(row, tokenizer, max_seq_len=max_seq_len)
        row["metadata"]["qwen_tokens"] = encoded.token_count
        row["metadata"]["qwen_target_tokens"] = encoded.target_token_count
        row["metadata"]["truncated_at_stats_max_seq_len"] = encoded.truncated


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[idx]


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    tools = Counter()
    prefix_lengths = []
    tool_call_counts = []
    token_counts = []
    target_token_counts = []
    truncated = 0
    for row in samples:
        metadata = row.get("metadata") or {}
        tasks[metadata.get("task_id")] += 1
        prefix_lengths.append(int(metadata.get("prefix_message_count") or 0))
        tool_call_counts.append(int(metadata.get("target_tool_call_count") or 0))
        for tool in metadata.get("target_tool_names") or []:
            tools[tool] += 1
        if metadata.get("qwen_tokens") is not None:
            token_counts.append(int(metadata["qwen_tokens"]))
            target_token_counts.append(int(metadata.get("qwen_target_tokens") or 0))
            truncated += int(bool(metadata.get("truncated_at_stats_max_seq_len")))

    return {
        "rows": len(samples),
        "tasks": dict(sorted(tasks.items(), key=lambda item: (int(item[0]) if str(item[0]).isdigit() else 10**9, str(item[0])))),
        "tools": dict(sorted(tools.items())),
        "mean_prefix_messages": mean(prefix_lengths) if prefix_lengths else 0,
        "max_prefix_messages": max(prefix_lengths, default=0),
        "mean_tool_calls_per_target": mean(tool_call_counts) if tool_call_counts else 0,
        "max_tool_calls_per_target": max(tool_call_counts, default=0),
        "mean_tokens": mean(token_counts) if token_counts else None,
        "p50_tokens": percentile(token_counts, 0.5) if token_counts else None,
        "p90_tokens": percentile(token_counts, 0.9) if token_counts else None,
        "max_tokens": max(token_counts, default=None),
        "mean_target_tokens": mean(target_token_counts) if target_token_counts else None,
        "max_target_tokens": max(target_token_counts, default=None),
        "truncated": truncated,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Action-Prefix Dataset v1",
        "",
        "## Goal",
        "",
        "Convert successful tau2 airline trajectories into dense next-tool-call supervision samples. Each sample contains a conversation prefix plus exactly one target assistant tool-call turn.",
        "",
        "## Outputs",
        "",
    ]
    for split, out_path in payload["outputs"].items():
        lines.append(f"- {split}: `{out_path}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Split | Rows | Tasks | Tools | Mean prefix msgs | Max prefix msgs | Mean tokens | P90 tokens | Max tokens | Mean target tokens |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, summary in payload["summaries"].items():
        tasks = ", ".join(f"{task}:{count}" for task, count in summary["tasks"].items())
        tools = ", ".join(f"{tool}:{count}" for tool, count in summary["tools"].items())
        lines.append(
            f"| {split} | {summary['rows']} | `{tasks}` | `{tools}` | "
            f"{summary['mean_prefix_messages']:.1f} | {summary['max_prefix_messages']} | "
            f"{summary['mean_tokens']:.1f} | {summary['p90_tokens']} | {summary['max_tokens']} | "
            f"{summary['mean_target_tokens']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
        ]
    )
    if payload["validation_errors"]:
        lines.extend(["Structural validation errors:"] + [f"- `{error}`" for error in payload["validation_errors"]])
    else:
        lines.append("No structural validation errors.")

    lines.extend(
        [
            "",
            "## Training Use",
            "",
            "- Use the train split for action-prefix SFT.",
            "- Keep valid and heldout for behavior evaluation.",
            "- The current local 3060 can run 2K-context smoke experiments; larger context or 7B/8B models should wait until this action-prefix route shows a behavior gain.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build action-prefix next-tool-call SFT samples from successful trajectories.")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--max-seq-len-for-stats", type=int, default=16384)
    parser.add_argument("--skip-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = {
        "train": repo_path(args.train),
        "valid": repo_path(args.valid),
        "heldout": repo_path(args.heldout),
    }
    out_dir = repo_path(args.out_dir)
    output_paths = {
        "train": out_dir / "tau2_airline_action_prefix_train.jsonl",
        "valid": out_dir / "tau2_airline_action_prefix_valid.jsonl",
        "heldout": out_dir / "tau2_airline_action_prefix_heldout.jsonl",
    }

    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = load_tokenizer(str(repo_path(args.tokenizer_model)), local_files_only=True)

    splits: dict[str, list[dict[str, Any]]] = {}
    validation_errors: list[str] = []
    for split, input_path in input_paths.items():
        rows = load_jsonl(input_path)
        samples = build_split(rows, split)
        add_token_stats(samples, tokenizer, max_seq_len=args.max_seq_len_for_stats)
        split_errors = validate_samples(samples)
        validation_errors.extend(split_errors)
        splits[split] = samples
        write_jsonl(output_paths[split], samples)

    summaries = {split: summarize(samples) for split, samples in splits.items()}
    manifest = {
        "format_version": "tau2_airline_action_prefix_manifest_v1",
        "inputs": {split: display_path(path) for split, path in input_paths.items()},
        "outputs": {split: display_path(path) for split, path in output_paths.items()},
        "report": display_path(repo_path(args.report)),
        "tokenizer_model": args.tokenizer_model if not args.skip_tokenizer else None,
        "max_seq_len_for_stats": args.max_seq_len_for_stats,
        "summaries": summaries,
        "validation_errors": validation_errors,
    }
    write_json(repo_path(args.manifest), manifest)
    write_report(repo_path(args.report), manifest)

    print("Action-prefix dataset complete")
    print("==============================")
    for split, summary in summaries.items():
        print(
            f"{split}: rows={summary['rows']} tasks={len(summary['tasks'])} "
            f"tools={len(summary['tools'])} max_tokens={summary['max_tokens']}"
        )
    print(f"validation_errors: {len(validation_errors)}")
    for split, path in output_paths.items():
        print(f"wrote {split}: {path}")
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
