from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_sft_smoke import encode_row, load_jsonl, load_tokenizer  # noqa: E402


DEFAULT_MODEL = "models/Qwen2.5-0.5B-Instruct"
DEFAULT_TRAIN = "data/train/tau2_airline_sft_train.jsonl"
DEFAULT_VALID = "data/train/tau2_airline_sft_valid.jsonl"
DEFAULT_HELDOUT = "data/train/tau2_airline_sft_heldout.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/sft_adapter_eval_phase1g"
DEFAULT_REPORT = "reports/sft_adapter_eval_phase1g.md"


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


def parse_adapter_specs(values: list[str]) -> list[tuple[str, Path]]:
    adapters: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Adapter spec must be name=path, got: {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        path_obj = repo_path(path.strip())
        if not name:
            raise ValueError(f"Adapter name is empty in spec: {value}")
        if not path_obj.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {path_obj}")
        adapters.append((name, path_obj))
    return adapters


def parse_splits(value: str) -> list[str]:
    allowed = {"train", "valid", "heldout"}
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        raise ValueError("--splits cannot be empty")
    unknown = sorted(set(splits) - allowed)
    if unknown:
        raise ValueError(f"Unknown split(s): {', '.join(unknown)}")
    return splits


def maybe_limit_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return rows[:max_rows]


def load_model(base_model_path: Path, adapter_path: Path | None, fp16: bool) -> torch.nn.Module:
    dtype = torch.float16 if fp16 else None
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def effective_target_count(labels: torch.Tensor) -> int:
    if labels.shape[-1] <= 1:
        return 0
    return int((labels[..., 1:] != -100).sum().item())


def eval_row(model: torch.nn.Module, row: dict[str, Any], tokenizer: Any, device: torch.device, max_seq_len: int) -> dict[str, Any]:
    encoded = encode_row(row, tokenizer, max_seq_len=max_seq_len)
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long, device=device)
    labels = torch.tensor([encoded.labels], dtype=torch.long, device=device)
    target_tokens = effective_target_count(labels)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = float(outputs.loss.detach().cpu())

    metadata = row.get("metadata") or {}
    return {
        "id": encoded.sample_id,
        "task_id": encoded.task_id,
        "official_split": metadata.get("official_split"),
        "token_count": encoded.token_count,
        "target_tokens": target_tokens,
        "truncated": encoded.truncated,
        "loss": loss,
        "nll_sum": loss * target_tokens,
        "perplexity": math.exp(min(loss, 20.0)),
        "warnings": encoded.warnings,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "weighted_loss": float("nan"),
            "mean_loss": float("nan"),
            "perplexity": float("nan"),
            "target_tokens": 0,
            "truncated": 0,
            "tasks": [],
        }
    target_tokens = sum(int(row["target_tokens"]) for row in rows)
    nll_sum = sum(float(row["nll_sum"]) for row in rows)
    weighted_loss = nll_sum / target_tokens if target_tokens else float("nan")
    return {
        "rows": len(rows),
        "weighted_loss": weighted_loss,
        "mean_loss": mean(float(row["loss"]) for row in rows),
        "perplexity": math.exp(min(weighted_loss, 20.0)),
        "target_tokens": target_tokens,
        "truncated": sum(1 for row in rows if row["truncated"]),
        "tasks": sorted({str(row["task_id"]) for row in rows}, key=lambda value: (int(value) if value.isdigit() else 10**9, value)),
    }


def evaluate_model(
    model_name: str,
    base_model_path: Path,
    adapter_path: Path | None,
    tokenizer: Any,
    splits: dict[str, list[dict[str, Any]]],
    device: torch.device,
    max_seq_len: int,
    fp16: bool,
) -> dict[str, Any]:
    started = time.time()
    model = load_model(base_model_path, adapter_path, fp16=fp16)
    model.to(device)
    model.eval()

    split_results: dict[str, Any] = {}
    max_memory_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for split_name, split_rows in splits.items():
        row_results = []
        for idx, row in enumerate(split_rows, start=1):
            result = eval_row(model, row, tokenizer, device, max_seq_len=max_seq_len)
            row_results.append(result)
            print(
                f"{model_name} {split_name} {idx}/{len(split_rows)} "
                f"loss={result['loss']:.4f} id={result['id']}"
            )
        split_results[split_name] = {
            "summary": summarize_rows(row_results),
            "rows": row_results,
        }

    if device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": model_name,
        "adapter_path": display_path(adapter_path) if adapter_path is not None else None,
        "elapsed_seconds": time.time() - started,
        "max_memory_mb": max_memory_mb,
        "splits": split_results,
    }


def build_deltas(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not results:
        return {}
    base = results[0]
    base_by_split = {
        split_name: split_payload["summary"]
        for split_name, split_payload in base["splits"].items()
    }
    deltas: dict[str, dict[str, Any]] = {}
    for result in results[1:]:
        model_delta: dict[str, Any] = {}
        for split_name, split_payload in result["splits"].items():
            current = split_payload["summary"]
            base_summary = base_by_split[split_name]
            base_loss = float(base_summary["weighted_loss"])
            current_loss = float(current["weighted_loss"])
            model_delta[split_name] = {
                "weighted_loss_delta": current_loss - base_loss,
                "relative_change_pct": ((current_loss - base_loss) / base_loss * 100.0) if base_loss else float("nan"),
            }
        deltas[result["name"]] = model_delta
    return deltas


def row_delta_table(base_result: dict[str, Any], adapter_result: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    base_rows = {row["id"]: row for row in base_result["splits"][split_name]["rows"]}
    adapter_rows = {row["id"]: row for row in adapter_result["splits"][split_name]["rows"]}
    rows = []
    for row_id, adapter_row in adapter_rows.items():
        base_row = base_rows[row_id]
        rows.append(
            {
                "id": row_id,
                "task_id": adapter_row["task_id"],
                "base_loss": base_row["loss"],
                "adapter_loss": adapter_row["loss"],
                "delta": adapter_row["loss"] - base_row["loss"],
                "target_tokens": adapter_row["target_tokens"],
            }
        )
    return sorted(rows, key=lambda item: item["delta"])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results = payload["results"]
    deltas = payload["deltas"]

    lines = [
        "# Phase 1G SFT Adapter Offline Evaluation",
        "",
        "## Goal",
        "",
        "Compare the base Qwen model against SFT LoRA adapters using assistant-only masked NLL on train, valid, and heldout splits.",
        "",
        "## Key Findings",
        "",
    ]
    if len(results) > 1:
        for adapter_name, split_deltas in deltas.items():
            parts = []
            improves_all = True
            for split_name in payload["splits"]:
                delta = split_deltas.get(split_name, {}).get("weighted_loss_delta")
                if delta is None:
                    continue
                improves_all = improves_all and delta < 0
                parts.append(f"{split_name} {delta:+.4f}")
            verdict = "improves selected splits" if improves_all and parts else "regresses at least one selected split"
            lines.append(f"- `{adapter_name}`: {', '.join(parts)} -> **{verdict}**.")
        ranking_split = "heldout" if "heldout" in payload["splits"] else payload["splits"][0]
        best_split = min(
            (
                (adapter_name, split_deltas[ranking_split]["weighted_loss_delta"])
                for adapter_name, split_deltas in deltas.items()
                if ranking_split in split_deltas
            ),
            key=lambda item: item[1],
            default=(None, None),
        )
        if best_split[0] is not None:
            lines.append(f"- Selected adapter for follow-up: `{best_split[0]}` based on `{ranking_split}` masked NLL.")
    else:
        lines.append("- Only the base model was evaluated; add adapters for delta analysis.")

    lines.extend(
        [
        "",
        "## Setup",
        "",
        f"- Base model: `{payload['base_model']}`",
        f"- Max sequence length: `{payload['max_seq_len']}`",
        f"- Device: `{payload['device']}`",
        f"- FP16: `{payload['fp16']}`",
        f"- Eval splits: `{', '.join(payload['splits'])}`",
        f"- Max rows per split: `{payload['max_rows_per_split']}`",
        "",
        "## Split Metrics",
        "",
        "| Model | Split | Rows | Weighted loss | Mean loss | PPL | Target tokens | Truncated |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        for split_name, split_payload in result["splits"].items():
            summary = split_payload["summary"]
            lines.append(
                f"| `{result['name']}` | {split_name} | {summary['rows']} | "
                f"{summary['weighted_loss']:.4f} | {summary['mean_loss']:.4f} | "
                f"{summary['perplexity']:.2f} | {summary['target_tokens']} | {summary['truncated']} |"
            )

    lines.extend(
        [
            "",
            "## Delta Vs Base",
            "",
            "| Adapter | Split | Loss delta | Relative change |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for adapter_name, split_deltas in deltas.items():
        for split_name, delta in split_deltas.items():
            lines.append(
                f"| `{adapter_name}` | {split_name} | "
                f"{delta['weighted_loss_delta']:+.4f} | {delta['relative_change_pct']:+.1f}% |"
            )

    if len(results) > 1:
        base_result = results[0]
        for adapter_result in results[1:]:
            lines.extend(
                [
                    "",
                    f"## Row Deltas: {adapter_result['name']}",
                    "",
                ]
            )
            for split_name in payload["splits"]:
                rows = row_delta_table(base_result, adapter_result, split_name)
                lines.extend(
                    [
                        f"### {split_name}",
                        "",
                        "| Sample | Task | Base | Adapter | Delta | Target tokens |",
                        "| --- | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for row in rows[:5]:
                    lines.append(
                        f"| `{row['id']}` | {row['task_id']} | {row['base_loss']:.4f} | "
                        f"{row['adapter_loss']:.4f} | {row['delta']:+.4f} | {row['target_tokens']} |"
                    )
                if len(rows) > 5:
                    lines.append("| ... worst regressions below ... |  |  |  |  |  |")
                    for row in rows[-5:]:
                        lines.append(
                            f"| `{row['id']}` | {row['task_id']} | {row['base_loss']:.4f} | "
                            f"{row['adapter_loss']:.4f} | {row['delta']:+.4f} | {row['target_tokens']} |"
                        )
                lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Lower masked NLL means the model assigns higher probability to the reference assistant/tool-call tokens.",
            "- Train improvements without valid/heldout improvements indicate memorization or distribution mismatch.",
            "- This is an offline teacher-forced metric, not an end-to-end tau2 pass rate.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base and LoRA SFT adapters with assistant-only masked NLL.")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--adapter", action="append", default=[], help="Adapter spec in name=path format. Can be repeated.")
    parser.add_argument("--splits", default="train,valid,heldout", help="Comma-separated subset of train,valid,heldout.")
    parser.add_argument("--max-rows-per-split", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_model = repo_path(args.base_model)
    if not base_model.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_model}")

    adapters = parse_adapter_specs(args.adapter)
    output_dir = repo_path(args.output_dir)
    report_path = repo_path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_splits = parse_splits(args.splits)
    all_split_paths = {
        "train": repo_path(args.train),
        "valid": repo_path(args.valid),
        "heldout": repo_path(args.heldout),
    }
    splits = {
        split_name: maybe_limit_rows(load_jsonl(all_split_paths[split_name]), args.max_rows_per_split)
        for split_name in requested_splits
    }
    tokenizer = load_tokenizer(str(base_model), local_files_only=True)
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
            splits,
            device=device,
            max_seq_len=args.max_seq_len,
            fp16=args.fp16,
        )
        results.append(result)
        write_json(output_dir / f"{name}_eval.json", result)

    payload = {
        "base_model": display_path(base_model),
        "max_seq_len": args.max_seq_len,
        "device": str(device),
        "fp16": args.fp16,
        "max_rows_per_split": args.max_rows_per_split,
        "splits": list(splits),
        "results": results,
        "deltas": build_deltas(results),
    }
    write_json(output_dir / "summary.json", payload)
    write_report(report_path, payload)
    print(f"wrote: {report_path}")
    print(f"wrote: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
