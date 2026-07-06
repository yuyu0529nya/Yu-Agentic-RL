from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DEFAULT_MODEL = "models/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = "data/decision_gate/tau2_airline_decision_gate_v1_2048_heldout.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/behavior_decision_gate_v1"
DEFAULT_REPORT = "reports/behavior_decision_gate_v1.md"
VALID_LABELS = ("assistant_text", "tool_call")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_adapter_specs(values: list[str]) -> list[tuple[str, Path]]:
    adapters: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Adapter spec must be name=path, got: {value}")
        name, path = value.split("=", 1)
        path_obj = repo_path(path.strip())
        if not path_obj.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {path_obj}")
        adapters.append((name.strip(), path_obj))
    return adapters


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


def compact_generated(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def extract_gate_body(text: str) -> str:
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def classify_gate_label(text: str) -> str:
    body = extract_gate_body(text)
    normalized = re.sub(r"\s+", "", body.lower())
    for label in VALID_LABELS:
        if normalized.startswith(label):
            return label

    positions = {
        label: normalized.find(label)
        for label in VALID_LABELS
        if normalized.find(label) >= 0
    }
    if positions:
        return min(positions.items(), key=lambda item: item[1])[0]
    if "<tool_call>" in body or '"name"' in body:
        return "tool_call"
    return "unknown"


def build_probes(rows: list[dict[str, Any]], max_probes: int) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages") or []
        metadata = row.get("metadata") or {}
        if not messages or messages[-1].get("role") != "assistant":
            continue
        label = str(metadata.get("gate_label") or messages[-1].get("content") or "")
        if label not in VALID_LABELS:
            continue
        probes.append(
            {
                "probe_id": str(row.get("id")),
                "sample_id": str(metadata.get("source_id") or row.get("id")),
                "task_id": str(metadata.get("task_id") or ""),
                "target_label": label,
                "prefix_messages": messages[:-1],
                "prefix_message_count": metadata.get("prefix_message_count"),
                "source_sample_type": metadata.get("source_sample_type"),
            }
        )
        if len(probes) >= max_probes:
            break
    return probes


def evaluate_model(
    model_name: str,
    base_model_path: Path,
    adapter_path: Path | None,
    tokenizer: Any,
    probes: list[dict[str, Any]],
    device: Any,
    max_seq_len: int,
    max_new_tokens: int,
    stop_sequences: list[str],
    fp16: bool,
) -> dict[str, Any]:
    import torch
    from evaluate_mixed_policy_behavior import generate_one, load_model, render_prefix

    started = time.time()
    model = load_model(base_model_path, adapter_path, fp16=fp16)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, Any]] = []
    for idx, probe in enumerate(probes, start=1):
        prompt = render_prefix(probe["prefix_messages"])
        generation = generate_one(
            model,
            tokenizer,
            prompt,
            device=device,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
        )
        predicted_label = classify_gate_label(generation["generated_text"])
        match = predicted_label == probe["target_label"]
        row = {
            "probe_id": probe["probe_id"],
            "sample_id": probe["sample_id"],
            "task_id": probe["task_id"],
            "source_sample_type": probe["source_sample_type"],
            "target_label": probe["target_label"],
            "predicted_label": predicted_label,
            "match": match,
            "prompt_tokens": generation["prompt_tokens"],
            "new_tokens": generation["new_tokens"],
            "generated_text": generation["generated_text"],
            "generated_body": extract_gate_body(generation["generated_text"]),
        }
        rows.append(row)
        print(
            f"{model_name} {idx}/{len(probes)} "
            f"target={probe['target_label']} pred={predicted_label} "
            f"match={match} probe={probe['probe_id']}"
        )

    max_memory_mb = None
    if device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": model_name,
        "adapter_path": display_path(adapter_path),
        "elapsed_seconds": time.time() - started,
        "max_memory_mb": max_memory_mb,
        "rows": rows,
        "summary": summarize(rows),
    }


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_rows = [row for row in rows if row["target_label"] == "assistant_text"]
    tool_rows = [row for row in rows if row["target_label"] == "tool_call"]
    confusion = Counter((row["target_label"], row["predicted_label"]) for row in rows)
    return {
        "probes": len(rows),
        "assistant_text_probes": len(text_rows),
        "tool_call_probes": len(tool_rows),
        "accuracy": ratio(sum(1 for row in rows if row["match"]), len(rows)),
        "assistant_text_recall": ratio(sum(1 for row in text_rows if row["match"]), len(text_rows)),
        "tool_call_recall": ratio(sum(1 for row in tool_rows if row["match"]), len(tool_rows)),
        "unknown_rate": ratio(sum(1 for row in rows if row["predicted_label"] == "unknown"), len(rows)),
        "predicted_labels": dict(sorted(Counter(row["predicted_label"] for row in rows).items())),
        "confusion": {f"{ref}->{pred}": count for (ref, pred), count in sorted(confusion.items())},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Decision Gate Behavior Evaluation",
        "",
        "## Goal",
        "",
        "Evaluate a decision-only adapter that predicts the next assistant action label: `assistant_text` or `tool_call`.",
        "",
        "## Setup",
        "",
        f"- Base model: `{payload['base_model']}`",
        f"- Data: `{payload['data']}`",
        f"- Probes: `{len(payload['probes'])}`",
        f"- Max sequence length: `{payload['max_seq_len']}`",
        f"- Max new tokens: `{payload['max_new_tokens']}`",
        f"- Device: `{payload['device']}`",
        "",
        "## Summary",
        "",
        "| Model | Accuracy | Text recall | Tool recall | Unknown | Predicted labels |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in payload["results"]:
        summary = result["summary"]
        lines.append(
            f"| `{result['name']}` | {summary['accuracy']:.3f} | "
            f"{summary['assistant_text_recall']:.3f} | {summary['tool_call_recall']:.3f} | "
            f"{summary['unknown_rate']:.3f} | `{summary['predicted_labels']}` |"
        )

    lines.extend(["", "## Confusion", ""])
    for result in payload["results"]:
        lines.append(f"- `{result['name']}`: `{result['summary']['confusion']}`")

    lines.extend(["", "## Per-Probe Results", ""])
    for result in payload["results"]:
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "| Probe | Task | Target | Pred | Match | Generated preview |",
                "| --- | ---: | --- | --- | ---: | --- |",
            ]
        )
        for row in result["rows"]:
            lines.append(
                f"| `{row['probe_id']}` | {row['task_id']} | `{row['target_label']}` | "
                f"`{row['predicted_label']}` | {'Y' if row['match'] else 'N'} | "
                f"`{compact_generated(row['generated_body'])}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- If text recall is low, the agent still over-calls tools.",
            "- If tool recall is low, the gate blocks useful tool calls and the downstream tool policy cannot help.",
            "- The target is a high-recall gate on both classes before combining it with Phase2H tool generation.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decision-only gate behavior.")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--max-probes", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--stop-sequence", action="append", default=["<|im_end|>", "\n"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from train_sft_smoke import load_tokenizer

    base_model = repo_path(args.base_model)
    data_path = repo_path(args.data)
    output_dir = repo_path(args.output_dir)
    report_path = repo_path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = parse_adapter_specs(args.adapter)
    rows = load_jsonl(data_path)
    probes = build_probes(rows, max_probes=args.max_probes)
    if not probes:
        raise RuntimeError("No decision gate probes found.")

    tokenizer = load_tokenizer(str(base_model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model_specs: list[tuple[str, Path | None]] = [("base", None)]
    model_specs.extend(adapters)

    results = []
    for name, adapter_path in model_specs:
        result = evaluate_model(
            name,
            base_model,
            adapter_path,
            tokenizer,
            probes,
            device=device,
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            stop_sequences=args.stop_sequence,
            fp16=args.fp16,
        )
        results.append(result)
        write_json(output_dir / f"{name}_decision_gate_behavior.json", result)

    payload = {
        "base_model": display_path(base_model),
        "data": display_path(data_path),
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "stop_sequences": args.stop_sequence,
        "device": str(device),
        "probes": probes,
        "results": results,
    }
    write_json(output_dir / "summary.json", payload)
    write_report(report_path, payload)
    print(f"wrote: {report_path}")
    print(f"wrote: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
