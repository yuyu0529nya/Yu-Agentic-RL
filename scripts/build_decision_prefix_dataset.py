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
    write_json,
    write_jsonl,
)
from build_action_prefix_dataset_v2 import (  # noqa: E402
    drop_dangling_leading_tool_messages,
    token_count,
)
from build_tool_call_protocol_dataset import normalize_tool_call  # noqa: E402
from check_sft_render_mask import validate_structure  # noqa: E402
from train_sft_smoke import encode_row, load_tokenizer  # noqa: E402


DEFAULT_OUT_DIR = "data/decision_prefix"
DEFAULT_REPORT = "reports/decision_prefix_dataset_v1.md"
DEFAULT_MANIFEST = "data/decision_prefix/tau2_airline_decision_prefix_manifest_v1.json"
DEFAULT_OUTPUT_STEM = "tau2_airline_decision_prefix_v1"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def with_messages(row: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(row)
    updated["messages"] = messages
    updated["loss_mask"] = [False] * (len(messages) - 1) + [True]
    updated.setdefault("metadata", {})["prefix_message_count"] = len(messages) - 1
    return updated


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


def base_metadata(source_row: dict[str, Any], split: str, turn_index: int) -> dict[str, Any]:
    metadata = source_row.get("metadata") or {}
    return {
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
        "split": split,
        "turn_index": turn_index,
    }


def make_text_sample(
    source_row: dict[str, Any],
    split: str,
    turn_index: int,
    message: dict[str, Any],
) -> dict[str, Any]:
    messages = source_row.get("messages") or []
    prefix = copy.deepcopy(messages[:turn_index])
    target = {
        "role": "assistant",
        "content": str(message.get("content") or ""),
        "tool_calls": None,
    }
    metadata = base_metadata(source_row, split, turn_index)
    metadata.update(
        {
            "target_action": "assistant_text",
            "prefix_message_count": len(prefix),
            "target_content_chars": len(target["content"]),
            "target_tool_call_count": 0,
            "target_tool_names": [],
        }
    )
    return {
        "id": f"decision_prefix_{split}_{source_row.get('id')}_turn{turn_index}_text",
        "format_version": "tau2_airline_decision_prefix_v1",
        "sample_type": "decision_prefix_text",
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
        "metadata": metadata,
    }


def make_tool_sample(
    source_row: dict[str, Any],
    split: str,
    turn_index: int,
    call_index: int,
    message: dict[str, Any],
) -> dict[str, Any]:
    messages = source_row.get("messages") or []
    prefix = copy.deepcopy(messages[:turn_index])
    call = normalize_tool_call((message.get("tool_calls") or [])[call_index])
    payload = tool_call_payload(call)
    target = {
        "role": "assistant",
        "content": "",
        "tool_calls": [call],
    }
    metadata = base_metadata(source_row, split, turn_index)
    metadata.update(
        {
            "target_action": "tool_call",
            "prefix_message_count": len(prefix),
            "call_index": call_index,
            "source_tool_call_count": len(message.get("tool_calls") or []),
            "source_assistant_content_chars": len(str(message.get("content") or "")),
            "target_content_removed": True,
            "target_tool_call_count": 1,
            "target_tool_names": [payload["name"]],
            "target_tool_calls": [payload],
            "protocol_wrapper_loss": True,
        }
    )
    return {
        "id": f"decision_prefix_{split}_{source_row.get('id')}_turn{turn_index}_call{call_index}",
        "format_version": "tau2_airline_decision_prefix_v1",
        "sample_type": "decision_prefix_tool_call",
        "split": split,
        "messages": prefix + [target],
        "loss_mask": [False] * len(prefix) + [True],
        "loss_policy": {
            "assistant_content": False,
            "assistant_tool_calls": True,
            "assistant_tool_call_wrappers": True,
            "user": False,
            "tool": False,
        },
        "metadata": metadata,
    }


def build_split(
    rows: list[dict[str, Any]],
    split: str,
    *,
    include_greeting: bool,
    max_target_content_chars: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    samples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        messages = row.get("messages") or []
        for turn_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            if turn_index == 0 and not include_greeting:
                skipped["initial_greeting"] += 1
                continue

            tool_calls = message.get("tool_calls") or []
            content = str(message.get("content") or "").strip()
            if tool_calls:
                for call_index, _call in enumerate(tool_calls):
                    samples.append(make_tool_sample(row, split, turn_index, call_index, message))
            elif content:
                if len(content) > max_target_content_chars:
                    skipped["text_target_too_long"] += 1
                    continue
                samples.append(make_text_sample(row, split, turn_index, message))
            else:
                skipped["empty_assistant"] += 1
    return samples, skipped


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
        target = messages[-1] if messages else {}
        action = (row.get("metadata") or {}).get("target_action")
        if not messages or target.get("role") != "assistant":
            errors.append(f"{row_id}:last_message_not_assistant")
        if loss_mask != [False] * (len(messages) - 1) + [True]:
            errors.append(f"{row_id}:loss_mask_not_prefix_false_target_true")
        if action == "tool_call" and len(target.get("tool_calls") or []) != 1:
            errors.append(f"{row_id}:tool_target_not_single_call")
        if action == "assistant_text":
            if target.get("tool_calls"):
                errors.append(f"{row_id}:text_target_has_tool_calls")
            if not str(target.get("content") or "").strip():
                errors.append(f"{row_id}:text_target_empty")
    return errors


def add_token_stats(samples: list[dict[str, Any]], tokenizer: Any, max_sample_tokens: int) -> list[str]:
    errors: list[str] = []
    for row in samples:
        final_token_count = token_count(row, tokenizer)
        try:
            encoded = encode_row(row, tokenizer, max_seq_len=max_sample_tokens)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('id')}: {type(exc).__name__}: {exc}")
            continue
        metadata = row.setdefault("metadata", {})
        metadata["qwen_tokens"] = final_token_count
        metadata["qwen_target_tokens"] = encoded.target_token_count
        metadata["truncated_at_stats_max_seq_len"] = final_token_count > max_sample_tokens
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
    actions = Counter()
    tools = Counter()
    token_counts: list[int] = []
    target_token_counts: list[int] = []
    content_chars: list[int] = []
    dropped_prefix: list[int] = []
    for row in samples:
        metadata = row.get("metadata") or {}
        tasks[metadata.get("task_id")] += 1
        actions[metadata.get("target_action")] += 1
        for tool in metadata.get("target_tool_names") or []:
            tools[tool] += 1
        if metadata.get("qwen_tokens") is not None:
            token_counts.append(int(metadata["qwen_tokens"]))
        if metadata.get("qwen_target_tokens") is not None:
            target_token_counts.append(int(metadata["qwen_target_tokens"]))
        if metadata.get("target_content_chars") is not None:
            content_chars.append(int(metadata["target_content_chars"]))
        trim = metadata.get("context_trim") or {}
        dropped_prefix.append(int(trim.get("dropped_prefix_message_count") or 0))
    return {
        "rows": len(samples),
        "tasks": dict(sorted(tasks.items(), key=lambda item: (int(item[0]) if str(item[0]).isdigit() else 10**9, str(item[0])))),
        "actions": dict(sorted(actions.items())),
        "tools": dict(sorted(tools.items())),
        "mean_tokens": mean(token_counts) if token_counts else 0,
        "p90_tokens": percentile(token_counts, 0.9),
        "max_tokens": max(token_counts, default=0),
        "mean_target_tokens": mean(target_token_counts) if target_token_counts else 0,
        "max_target_tokens": max(target_token_counts, default=0),
        "mean_text_target_chars": mean(content_chars) if content_chars else 0,
        "max_text_target_chars": max(content_chars, default=0),
        "trimmed_rows": sum(1 for value in dropped_prefix if value > 0),
        "mean_dropped_prefix_messages": mean(dropped_prefix) if dropped_prefix else 0,
        "max_dropped_prefix_messages": max(dropped_prefix, default=0),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Decision-Prefix Dataset v1",
        "",
        "## Goal",
        "",
        "Train the model to choose the next assistant action: either ask/confirm in natural language or emit exactly one executable tool call. This addresses the protocol SFT failure mode where the model can format tools but keeps calling tools when it should first collect evidence.",
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
            f"- Include initial greeting: `{payload['include_greeting']}`",
            f"- Max target content chars: `{payload['max_target_content_chars']}`",
            "- Tool targets: content removed, one tool call per sample, wrapper loss enabled",
            "- Text targets: content loss enabled, tool-call loss disabled",
            "",
            "## Summary",
            "",
            "| Split | Rows | Actions | Tasks | Tools | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed rows | Mean dropped prefix msgs |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, summary in payload["summaries"].items():
        actions = ", ".join(f"{key}:{value}" for key, value in summary["actions"].items())
        tasks = ", ".join(f"{key}:{value}" for key, value in summary["tasks"].items())
        tools = ", ".join(f"{key}:{value}" for key, value in summary["tools"].items())
        lines.append(
            f"| {split} | {summary['rows']} | `{actions}` | `{tasks}` | `{tools}` | "
            f"{summary['mean_tokens']:.1f} | {summary['p90_tokens']} | {summary['max_tokens']} | "
            f"{summary['mean_target_tokens']:.1f} | {summary['trimmed_rows']} | "
            f"{summary['mean_dropped_prefix_messages']:.1f} |"
        )

    lines.extend(["", "## Skipped", ""])
    for split, skipped in payload["skipped"].items():
        pretty = ", ".join(f"{key}:{value}" for key, value in skipped.items()) or "none"
        lines.append(f"- {split}: `{pretty}`")

    lines.extend(["", "## Validation", ""])
    if payload["validation_errors"]:
        lines.extend(["Structural/token validation errors:"] + [f"- `{error}`" for error in payload["validation_errors"]])
    else:
        lines.append("No structural/token validation errors.")

    lines.extend(
        [
            "",
            "## Training Use",
            "",
            "- Train this after the tool-call protocol adapter as the first behavior-policy adapter.",
            "- Evaluate with offline behavior probes for both `assistant_text` and `tool_call`, then run tau2 SFT-only smoke with `AGENT_STOP_SEQUENCE=</tool_call>`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build decision-prefix SFT samples with text-vs-tool next-action supervision.")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--max-sample-tokens", type=int, default=2048)
    parser.add_argument("--max-target-content-chars", type=int, default=1200)
    parser.add_argument("--include-greeting", action=argparse.BooleanOptionalAction, default=False)
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
    skipped_by_split: dict[str, dict[str, int]] = {}
    validation_errors: list[str] = []
    for split, input_path in input_paths.items():
        source_rows = load_jsonl(input_path)
        samples, skipped = build_split(
            source_rows,
            split,
            include_greeting=args.include_greeting,
            max_target_content_chars=args.max_target_content_chars,
        )
        samples = [trim_prefix_to_budget(row, tokenizer, args.max_sample_tokens) for row in samples]
        validation_errors.extend(validate_samples(samples))
        validation_errors.extend(add_token_stats(samples, tokenizer, args.max_sample_tokens))
        splits[split] = samples
        skipped_by_split[split] = dict(sorted(skipped.items()))
        write_jsonl(output_paths[split], samples)

    summaries = {split: summarize(samples) for split, samples in splits.items()}
    manifest = {
        "format_version": "tau2_airline_decision_prefix_manifest_v1",
        "inputs": {split: display_path(path) for split, path in input_paths.items()},
        "outputs": {split: display_path(path) for split, path in output_paths.items()},
        "report": display_path(repo_path(args.report)),
        "tokenizer_model": args.tokenizer_model,
        "max_sample_tokens": args.max_sample_tokens,
        "max_target_content_chars": args.max_target_content_chars,
        "include_greeting": args.include_greeting,
        "summaries": summaries,
        "skipped": skipped_by_split,
        "validation_errors": validation_errors,
    }
    write_json(repo_path(args.manifest), manifest)
    write_report(repo_path(args.report), manifest)

    print("Decision-prefix dataset complete")
    print("================================")
    for split, summary in summaries.items():
        print(
            f"{split}: rows={summary['rows']} actions={summary['actions']} "
            f"max_tokens={summary['max_tokens']} validation_errors={len(validation_errors)}"
        )
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
