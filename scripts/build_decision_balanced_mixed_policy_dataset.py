from __future__ import annotations

import argparse
import copy
import hashlib
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
    build_split,
    display_path,
    load_jsonl,
    load_tokenizer,
    repo_path,
    summarize,
    summarize_rejections,
    validate_and_finalize,
    write_json,
    write_jsonl,
)


DEFAULT_TRAIN = "data/train/tau2_airline_sft_train.jsonl"
DEFAULT_VALID = "data/train/tau2_airline_sft_valid.jsonl"
DEFAULT_HELDOUT = "data/train/tau2_airline_sft_heldout.jsonl"
DEFAULT_OUT_DIR = "data/mixed_policy"
DEFAULT_REPORT = "reports/mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048.md"
DEFAULT_MANIFEST = "data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_manifest_v1_2048.json"
DEFAULT_OUTPUT_STEM = "tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_v1_2048"
DEFAULT_TOKENIZER_MODEL = "models/Qwen2.5-0.5B-Instruct"


def model_source_arg(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or value.startswith(".") or (REPO_ROOT / value).exists():
        return str(repo_path(value))
    return value


def stable_score(row_id: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def is_text_row(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return metadata.get("target_action") == "assistant_text"


def is_protocol_tool_row(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    return metadata.get("target_action") == "tool_call" and bool(metadata.get("protocol_only"))


def annotate_balance(row: dict[str, Any], role: str, strategy: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(row)
    metadata = item.setdefault("metadata", {})
    metadata["phase2i_balance_role"] = role
    metadata["phase2i_balance_strategy"] = strategy["name"]
    metadata["phase2i_text_repeat"] = strategy["text_repeat"]
    metadata["phase2i_protocol_keep_ratio"] = strategy["protocol_keep_ratio"]
    metadata["phase2i_seed"] = strategy["seed"]
    metadata["phase2i_original_id"] = row.get("id")
    return item


def duplicate_text_row(row: dict[str, Any], repeat_index: int, strategy: dict[str, Any]) -> dict[str, Any]:
    item = annotate_balance(row, "assistant_text_repeat", strategy)
    item["id"] = f"{row.get('id')}__phase2i_text_repeat{repeat_index}"
    item["metadata"]["phase2i_repeat_index"] = repeat_index
    return item


def selected_protocol_ids(rows: list[dict[str, Any]], keep_ratio: float, seed: int) -> set[str]:
    protocol_rows = [row for row in rows if is_protocol_tool_row(row)]
    if keep_ratio <= 0:
        return set()
    if keep_ratio >= 1:
        return {str(row.get("id")) for row in protocol_rows}
    keep_count = int(len(protocol_rows) * keep_ratio + 0.5)
    keep_count = max(0, min(len(protocol_rows), keep_count))
    ranked = sorted(
        protocol_rows,
        key=lambda row: (stable_score(str(row.get("id")), seed), str(row.get("id"))),
    )
    return {str(row.get("id")) for row in ranked[:keep_count]}


def assert_unique_ids(rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("id")) for row in rows)
    duplicates = sorted(row_id for row_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("Duplicate balanced ids: " + ", ".join(duplicates[:10]))


def balance_train_samples(
    rows: list[dict[str, Any]],
    text_repeat: int,
    protocol_keep_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if text_repeat < 1:
        raise ValueError("--text-repeat must be >= 1")
    if not 0 <= protocol_keep_ratio <= 1:
        raise ValueError("--protocol-keep-ratio must be between 0 and 1")

    strategy = {
        "name": "phase2i_decision_balanced_text_repeat_protocol_downsample",
        "text_repeat": text_repeat,
        "protocol_keep_ratio": protocol_keep_ratio,
        "seed": seed,
    }
    keep_protocol = selected_protocol_ids(rows, protocol_keep_ratio, seed)
    balanced: list[dict[str, Any]] = []
    balance_dropped: list[dict[str, Any]] = []

    for row in rows:
        row_id = str(row.get("id"))
        if is_protocol_tool_row(row):
            if row_id not in keep_protocol:
                dropped = annotate_balance(row, "protocol_tool_downsampled", strategy)
                dropped.setdefault("metadata", {})["mixed_policy_reject_reason"] = "phase2i_protocol_downsampled"
                balance_dropped.append(dropped)
                continue
            balanced.append(annotate_balance(row, "protocol_tool_kept", strategy))
            continue

        if is_text_row(row):
            balanced.append(annotate_balance(row, "assistant_text_original", strategy))
            for repeat_index in range(1, text_repeat):
                balanced.append(duplicate_text_row(row, repeat_index, strategy))
            continue

        balanced.append(annotate_balance(row, "tool_original", strategy))

    assert_unique_ids(balanced)
    stats = {
        "text_repeat": text_repeat,
        "protocol_keep_ratio": protocol_keep_ratio,
        "seed": seed,
        "input_rows": len(rows),
        "output_rows": len(balanced),
        "balance_dropped_rows": len(balance_dropped),
        "input_target_actions": dict(sorted(Counter((row.get("metadata") or {}).get("target_action") for row in rows).items())),
        "output_target_actions": dict(sorted(Counter((row.get("metadata") or {}).get("target_action") for row in balanced).items())),
        "input_sample_types": dict(sorted(Counter(row.get("sample_type") for row in rows).items())),
        "output_sample_types": dict(sorted(Counter(row.get("sample_type") for row in balanced).items())),
        "balance_roles": dict(sorted(Counter((row.get("metadata") or {}).get("phase2i_balance_role") for row in balanced).items())),
    }
    return balanced, balance_dropped, stats


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def decision_mix(summary: dict[str, Any]) -> dict[str, Any]:
    actions = summary.get("target_actions") or {}
    text = int(actions.get("assistant_text") or 0)
    tools = int(actions.get("tool_call") or 0)
    total = text + tools
    return {
        "assistant_text": text,
        "tool_call": tools,
        "total": total,
        "text_share": ratio(text, total),
        "tool_share": ratio(tools, total),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase2I Decision-Balanced Mixed Policy Dataset v1",
        "",
        "## Goal",
        "",
        "Keep the Phase2H tool-call protocol gains while reducing over-calling on assistant text turns. The training split repeats assistant text targets and downsamples protocol-only tool targets; valid and heldout splits keep the original Phase2H distribution for unbiased behavior checks.",
        "",
        "## Outputs",
        "",
    ]
    for split, out_path in payload["outputs"].items():
        lines.append(f"- {split}: `{out_path}`")
    for split, out_path in payload["rejected_outputs"].items():
        lines.append(f"- {split} rejected: `{out_path}`")
    for split, out_path in payload["balance_dropped_outputs"].items():
        if out_path:
            lines.append(f"- {split} balance dropped: `{out_path}`")

    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- Tokenizer model: `{payload['tokenizer_model']}`",
            f"- Tokenizer stats enabled: `{payload['tokenizer_stats_enabled']}`",
            f"- Max sample tokens: `{payload['max_sample_tokens']}`",
            f"- Include protocol variants: `{payload['include_protocol_variants']}`",
            f"- Train text repeat: `{payload['text_repeat']}`",
            f"- Train protocol keep ratio: `{payload['protocol_keep_ratio']}`",
            f"- Balance seed: `{payload['balance_seed']}`",
            "- Tool targets are still required to be grounded in the online prefix.",
            "- Only `train` is decision-balanced; `valid` and `heldout` are not repeated or downsampled.",
            "",
            "## Decision Mix",
            "",
            "| Split | Stage | Rows | Text | Tool | Text share | Tool share | Sample types |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for split in ("train", "valid", "heldout"):
        stages = [("raw clean", payload["raw_summaries"][split])]
        if split == "train":
            stages.append(("balanced", payload["summaries"][split]))
        for stage, summary in stages:
            mix = decision_mix(summary)
            sample_types = ", ".join(f"{name}:{count}" for name, count in summary["sample_types"].items())
            lines.append(
                f"| {split} | {stage} | {summary['rows']} | {mix['assistant_text']} | {mix['tool_call']} | "
                f"{mix['text_share']:.3f} | {mix['tool_share']:.3f} | `{sample_types}` |"
            )

    train_stats = payload["balance_stats"]["train"]
    lines.extend(
        [
            "",
            "## Train Balance Stats",
            "",
            f"- Input rows: `{train_stats['input_rows']}`",
            f"- Output rows: `{train_stats['output_rows']}`",
            f"- Protocol rows downsampled: `{train_stats['balance_dropped_rows']}`",
            f"- Balance roles: `{train_stats['balance_roles']}`",
            "",
            "## Token Summary",
            "",
            "| Split | Rows | Rejected | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, summary in payload["summaries"].items():
        rejected = payload["rejected_summaries"][split]
        lines.append(
            f"| {split} | {summary['rows']} | {rejected['rows']} | {summary['mean_tokens']:.1f} | "
            f"{summary['p90_tokens']} | {summary['max_tokens']} | {summary['mean_target_tokens']:.1f} | "
            f"{summary['trimmed_rows']} |"
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
        lines.extend(["Structural/token validation errors:"] + [f"- `{error}`" for error in payload["validation_errors"][:50]])
        if len(payload["validation_errors"]) > 50:
            lines.append(f"- ... {len(payload['validation_errors']) - 50} more")
    else:
        lines.append("No structural/token validation errors on kept samples.")

    lines.extend(
        [
            "",
            "## Training Use",
            "",
            "- Train Qwen2.5-7B QLoRA from the base model on the balanced train split.",
            "- Run mixed-policy behavior eval before any full tau2 pass.",
            "- Desired behavior movement: keep tool-name/argument exact accuracy close to Phase2H while raising text no-tool rate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase2I decision-balanced mixed policy SFT samples.")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--max-sample-tokens", type=int, default=2048)
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--include-protocol-variants", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-repeat", type=int, default=2)
    parser.add_argument("--protocol-keep-ratio", type=float, default=0.5)
    parser.add_argument("--balance-seed", type=int, default=11)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = {
        "train": repo_path(args.train),
        "valid": repo_path(args.valid),
        "heldout": repo_path(args.heldout),
    }
    out_dir = repo_path(args.out_dir)
    output_paths = {split: out_dir / f"{args.output_stem}_{split}.jsonl" for split in input_paths}
    rejected_paths = {split: out_dir / f"{args.output_stem}_{split}_rejected.jsonl" for split in input_paths}
    balance_dropped_paths = {
        "train": out_dir / f"{args.output_stem}_train_balance_dropped.jsonl",
        "valid": None,
        "heldout": None,
    }

    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = load_tokenizer(model_source_arg(args.tokenizer_model), local_files_only=not args.no_local_files_only)

    raw_splits: dict[str, list[dict[str, Any]]] = {}
    splits: dict[str, list[dict[str, Any]]] = {}
    rejected_splits: dict[str, list[dict[str, Any]]] = {}
    balance_dropped_splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "heldout": []}
    balance_stats: dict[str, dict[str, Any]] = {}
    validation_errors: list[str] = []

    for split, input_path in input_paths.items():
        rows = load_jsonl(input_path)
        raw_samples, raw_rejected = build_split(rows, split, include_protocol_variants=args.include_protocol_variants)
        clean, rejected, errors = validate_and_finalize(raw_samples, raw_rejected, tokenizer, args.max_sample_tokens)
        validation_errors.extend(errors)
        raw_splits[split] = clean
        rejected_splits[split] = rejected

        if split == "train":
            balanced, balance_dropped, stats = balance_train_samples(
                clean,
                text_repeat=args.text_repeat,
                protocol_keep_ratio=args.protocol_keep_ratio,
                seed=args.balance_seed,
            )
            splits[split] = balanced
            balance_dropped_splits[split] = balance_dropped
            balance_stats[split] = stats
        else:
            splits[split] = clean
            balance_stats[split] = {
                "input_rows": len(clean),
                "output_rows": len(clean),
                "balance_dropped_rows": 0,
                "not_balanced": True,
            }

        write_jsonl(output_paths[split], splits[split])
        write_jsonl(rejected_paths[split], rejected_splits[split])

    write_jsonl(balance_dropped_paths["train"], balance_dropped_splits["train"])

    raw_summaries = {split: summarize(samples) for split, samples in raw_splits.items()}
    summaries = {split: summarize(samples) for split, samples in splits.items()}
    rejected_summaries = {split: summarize_rejections(samples) for split, samples in rejected_splits.items()}
    balance_dropped_summaries = {
        split: summarize_rejections(samples)
        for split, samples in balance_dropped_splits.items()
    }

    manifest = {
        "format_version": "tau2_airline_mixed_dialogue_tool_policy_phase2i_decision_balanced_manifest_v1",
        "inputs": {split: display_path(path) for split, path in input_paths.items()},
        "outputs": {split: display_path(path) for split, path in output_paths.items()},
        "rejected_outputs": {split: display_path(path) for split, path in rejected_paths.items()},
        "balance_dropped_outputs": {
            split: display_path(path) if path is not None else None
            for split, path in balance_dropped_paths.items()
        },
        "report": display_path(repo_path(args.report)),
        "tokenizer_model": args.tokenizer_model,
        "tokenizer_stats_enabled": tokenizer is not None,
        "max_sample_tokens": args.max_sample_tokens,
        "include_protocol_variants": args.include_protocol_variants,
        "text_repeat": args.text_repeat,
        "protocol_keep_ratio": args.protocol_keep_ratio,
        "balance_seed": args.balance_seed,
        "raw_summaries": raw_summaries,
        "summaries": summaries,
        "rejected_summaries": rejected_summaries,
        "balance_dropped_summaries": balance_dropped_summaries,
        "balance_stats": balance_stats,
        "validation_errors": validation_errors,
    }
    write_json(repo_path(args.manifest), manifest)
    write_report(repo_path(args.report), manifest)

    print("Phase2I decision-balanced mixed policy dataset complete")
    print("=====================================================")
    for split, summary in summaries.items():
        raw = raw_summaries[split]
        rejected = rejected_summaries[split]
        print(
            f"{split}: raw_rows={raw['rows']} rows={summary['rows']} rejected={rejected['rows']} "
            f"actions={summary['target_actions']} max_tokens={summary['max_tokens']}"
        )
        if split == "train":
            print(f"  balance_dropped={len(balance_dropped_splits['train'])} roles={balance_stats['train']['balance_roles']}")
        if rejected["reasons"]:
            print("  rejected: " + ", ".join(f"{k}={v}" for k, v in rejected["reasons"].items()))
    print(f"validation_errors: {len(validation_errors)}")
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
