from __future__ import annotations

import argparse
import copy
import json
import sys
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
from check_sft_render_mask import render_with_spans  # noqa: E402
from train_sft_smoke import encode_row, load_tokenizer  # noqa: E402


DEFAULT_OUT_DIR = "data/action_prefix"
DEFAULT_REPORT = "reports/action_prefix_dataset_v2.md"
DEFAULT_MANIFEST = "data/action_prefix/tau2_airline_action_prefix_manifest_v2.json"
DEFAULT_OUTPUT_STEM = "tau2_airline_action_prefix_v2"


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def token_count(row: dict[str, Any], tokenizer: Any) -> int:
    render = render_with_spans(row)
    encoded = tokenizer(render.text, add_special_tokens=False)
    return len(encoded["input_ids"])


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


def trim_prefix_to_budget(
    row: dict[str, Any],
    tokenizer: Any,
    max_sample_tokens: int,
) -> dict[str, Any]:
    original = copy.deepcopy(row)
    original_messages = original.get("messages") or []
    if len(original_messages) < 1:
        return original

    prefix = list(original_messages[:-1])
    target = original_messages[-1]
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
    candidate["format_version"] = "tau2_airline_action_prefix_v2"
    candidate["id"] = str(candidate.get("id")).replace("action_prefix_", "action_prefix_v2_", 1)
    return candidate


def add_final_token_stats(samples: list[dict[str, Any]], tokenizer: Any, max_sample_tokens: int) -> list[str]:
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


def trim_split(samples: list[dict[str, Any]], tokenizer: Any, max_sample_tokens: int) -> list[dict[str, Any]]:
    return [trim_prefix_to_budget(row, tokenizer, max_sample_tokens) for row in samples]


def trim_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    dropped = []
    original_tokens = []
    final_tokens = []
    for row in samples:
        trim = (row.get("metadata") or {}).get("context_trim") or {}
        dropped.append(int(trim.get("dropped_prefix_message_count") or 0))
        original_tokens.append(int(trim.get("original_qwen_tokens") or 0))
        final_tokens.append(int(trim.get("final_qwen_tokens") or 0))
    return {
        "trimmed_rows": sum(1 for value in dropped if value > 0),
        "mean_dropped_prefix_messages": mean(dropped) if dropped else 0,
        "max_dropped_prefix_messages": max(dropped, default=0),
        "mean_original_tokens": mean(original_tokens) if original_tokens else 0,
        "mean_final_tokens": mean(final_tokens) if final_tokens else 0,
        "max_final_tokens": max(final_tokens, default=0),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Action-Prefix Dataset v2",
        "",
        "## Goal",
        "",
        "Create one next-tool-call SFT sample per assistant tool-call turn, with message-level context trimming so each sample fits the local training token budget while preserving the target tool call.",
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
            "- Trim strategy: `message_suffix_target_preserving`",
            "",
            "## Summary",
            "",
            "| Split | Rows | Tasks | Tools | Mean tokens | Max tokens | Mean target tokens | Trimmed rows | Mean dropped prefix msgs | Max dropped prefix msgs |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, summary in payload["summaries"].items():
        trim = payload["trim_summaries"][split]
        tasks = ", ".join(f"{task}:{count}" for task, count in summary["tasks"].items())
        tools = ", ".join(f"{tool}:{count}" for tool, count in summary["tools"].items())
        lines.append(
            f"| {split} | {summary['rows']} | `{tasks}` | `{tools}` | "
            f"{summary['mean_tokens']:.1f} | {summary['max_tokens']} | "
            f"{summary['mean_target_tokens']:.1f} | {trim['trimmed_rows']} | "
            f"{trim['mean_dropped_prefix_messages']:.1f} | {trim['max_dropped_prefix_messages']} |"
        )

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
            "- Train with `--max-seq-len` equal to the v2 token budget and `--truncation-side right`; no target should be lost because samples are pre-trimmed.",
            "- This is the preferred local 3060 route before renting GPU: prove next-tool-call behavior improves on v2, then scale model/context.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build budgeted Action-Prefix v2 next-tool-call SFT samples.")
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

    splits: dict[str, list[dict[str, Any]]] = {}
    validation_errors: list[str] = []
    for split, input_path in input_paths.items():
        source_rows = load_jsonl(input_path)
        samples = build_split(source_rows, split)
        samples = trim_split(samples, tokenizer, args.max_sample_tokens)
        validation_errors.extend(validate_samples(samples))
        validation_errors.extend(add_final_token_stats(samples, tokenizer, args.max_sample_tokens))
        splits[split] = samples
        write_jsonl(output_paths[split], samples)

    summaries = {split: summarize(samples) for split, samples in splits.items()}
    trim_summaries = {split: trim_summary(samples) for split, samples in splits.items()}
    manifest = {
        "format_version": "tau2_airline_action_prefix_manifest_v2",
        "inputs": {split: display_path(path) for split, path in input_paths.items()},
        "outputs": {split: display_path(path) for split, path in output_paths.items()},
        "report": display_path(repo_path(args.report)),
        "tokenizer_model": args.tokenizer_model,
        "max_sample_tokens": args.max_sample_tokens,
        "summaries": summaries,
        "trim_summaries": trim_summaries,
        "validation_errors": validation_errors,
    }
    write_json(repo_path(args.manifest), manifest)
    write_report(repo_path(args.report), manifest)

    print("Action-prefix v2 dataset complete")
    print("=================================")
    for split, summary in summaries.items():
        trim = trim_summaries[split]
        print(
            f"{split}: rows={summary['rows']} max_tokens={summary['max_tokens']} "
            f"trimmed_rows={trim['trimmed_rows']} validation_errors={len(validation_errors)}"
        )
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
