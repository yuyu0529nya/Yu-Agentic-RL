from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT = "data/sft/tau2_airline_sft_v1.jsonl"
DEFAULT_RENDER_REPORT = "reports/sft_render_mask_v1.json"
DEFAULT_OUT_DIR = "data/train"
DEFAULT_REPORT = "reports/sft_training_export_v1.md"
DEFAULT_MANIFEST = "data/train/tau2_airline_sft_manifest_v1.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


def task_sort_key(task_id: str) -> tuple[int, str]:
    try:
        return int(task_id), task_id
    except ValueError:
        return 10**9, task_id


def row_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    task_id = str(row.get("task_id"))
    trial = row.get("trial")
    try:
        trial_key = int(trial)
    except (TypeError, ValueError):
        trial_key = 10**9
    return task_sort_key(task_id)[0], trial_key, str(row.get("id"))


def load_render_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(row.get("id")): row for row in payload.get("rows") or []}


def parse_task_ids(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def row_tokens(row: dict[str, Any], render_rows: dict[str, dict[str, Any]]) -> int:
    render = render_rows.get(str(row.get("id"))) or {}
    value = render.get("tokens") or row.get("estimated_tokens") or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def select_validation_tasks(
    trainable_rows: list[dict[str, Any]],
    render_rows: dict[str, dict[str, Any]],
    valid_ratio: float,
    min_valid_rows: int,
    requested_valid_tasks: set[str] | None,
) -> set[str]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trainable_rows:
        by_task[str(row.get("task_id"))].append(row)

    if requested_valid_tasks is not None:
        missing = requested_valid_tasks - set(by_task)
        if missing:
            raise ValueError(
                "Requested validation task ids are not trainable: "
                + ", ".join(sorted(missing, key=task_sort_key))
            )
        return requested_valid_tasks

    target_rows = max(min_valid_rows, round(len(trainable_rows) * valid_ratio))
    task_infos = []
    for task_id, rows in by_task.items():
        max_tokens = max(row_tokens(row, render_rows) for row in rows)
        task_infos.append((max_tokens, len(rows), task_id))

    selected: set[str] = set()
    selected_rows = 0
    for _max_tokens, count, task_id in sorted(
        task_infos, key=lambda item: (-item[0], task_sort_key(item[2]))
    ):
        selected.add(task_id)
        selected_rows += count
        if selected_rows >= target_rows:
            break
    return selected


def export_row(row: dict[str, Any], split: str, render_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    render = render_rows.get(str(row.get("id"))) or {}
    messages = row.get("messages") or []
    loss_mask = row.get("loss_mask") or []
    return {
        "id": row.get("id"),
        "format_version": "tau2_airline_sft_training_v1",
        "split": split,
        "messages": messages,
        "loss_mask": loss_mask,
        "loss_policy": row.get("loss_policy")
        or {
            "assistant_content": True,
            "assistant_tool_calls": True,
            "user": False,
            "tool": False,
        },
        "metadata": {
            "source_format_version": row.get("format_version"),
            "sample_type": row.get("sample_type"),
            "domain": row.get("domain"),
            "task_id": str(row.get("task_id")),
            "trial": row.get("trial"),
            "simulation_id": row.get("simulation_id"),
            "official_split": row.get("official_split"),
            "reward": row.get("reward"),
            "process_score": row.get("process_score"),
            "normalized_process_score": row.get("normalized_process_score"),
            "risk_tags": row.get("risk_tags") or [],
            "tool_calls": row.get("tool_calls"),
            "write_calls": row.get("write_calls"),
            "estimated_tokens": row.get("estimated_tokens"),
            "qwen2_5_tokens": render.get("tokens"),
            "qwen2_5_target_tokens": render.get("target_tokens"),
            "qwen2_5_target_token_ratio": render.get("target_token_ratio"),
            "render_check_errors": render.get("errors") or [],
            "render_check_warnings": render.get("warnings") or [],
        },
    }


def split_rows(
    rows: list[dict[str, Any]],
    render_rows: dict[str, dict[str, Any]],
    valid_ratio: float,
    min_valid_rows: int,
    requested_valid_tasks: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    trainable_rows = [row for row in rows if row.get("trainable")]
    heldout_rows = [row for row in rows if not row.get("trainable")]
    valid_tasks = select_validation_tasks(
        trainable_rows,
        render_rows,
        valid_ratio=valid_ratio,
        min_valid_rows=min_valid_rows,
        requested_valid_tasks=requested_valid_tasks,
    )

    train_rows = [
        row for row in trainable_rows if str(row.get("task_id")) not in valid_tasks
    ]
    valid_rows = [
        row for row in trainable_rows if str(row.get("task_id")) in valid_tasks
    ]
    return (
        sorted(train_rows, key=row_sort_key),
        sorted(valid_rows, key=row_sort_key),
        sorted(heldout_rows, key=row_sort_key),
        valid_tasks,
    )


def validate_export(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    train_tasks = {row["metadata"]["task_id"] for row in train_rows}
    valid_tasks = {row["metadata"]["task_id"] for row in valid_rows}
    heldout_tasks = {row["metadata"]["task_id"] for row in heldout_rows}

    if train_tasks & valid_tasks:
        errors.append("train_valid_task_overlap:" + ",".join(sorted(train_tasks & valid_tasks)))
    if (train_tasks | valid_tasks) & heldout_tasks:
        errors.append(
            "official_train_test_overlap:"
            + ",".join(sorted((train_tasks | valid_tasks) & heldout_tasks))
        )

    for split_name, split_rows_ in (
        ("train", train_rows),
        ("valid", valid_rows),
        ("heldout", heldout_rows),
    ):
        for row in split_rows_:
            if len(row.get("messages") or []) != len(row.get("loss_mask") or []):
                errors.append(f"{split_name}:{row.get('id')}:loss_mask_length_mismatch")
            if not row.get("messages"):
                errors.append(f"{split_name}:{row.get('id')}:empty_messages")
            for message, target in zip(row.get("messages") or [], row.get("loss_mask") or []):
                role = message.get("role")
                if role in {"user", "tool"} and target:
                    errors.append(f"{split_name}:{row.get('id')}:non_assistant_target:{role}")

    return errors


def counts_by_task(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["metadata"]["task_id"] for row in rows)
    return dict(sorted(counts.items(), key=lambda item: task_sort_key(item[0])))


def tokens_for(rows: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = row.get("metadata", {}).get("qwen2_5_tokens")
        if value is None:
            value = row.get("metadata", {}).get("estimated_tokens")
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            pass
    return values


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_values = tokens_for(rows)
    target_ratios = [
        row.get("metadata", {}).get("qwen2_5_target_token_ratio")
        for row in rows
        if row.get("metadata", {}).get("qwen2_5_target_token_ratio") is not None
    ]
    return {
        "rows": len(rows),
        "tasks": counts_by_task(rows),
        "mean_tokens": mean(token_values) if token_values else 0.0,
        "p50_tokens": percentile(token_values, 0.5),
        "p90_tokens": percentile(token_values, 0.9),
        "max_tokens": max(token_values) if token_values else 0,
        "mean_target_token_ratio": mean(target_ratios) if target_ratios else None,
        "tool_calls": sum(int(row.get("metadata", {}).get("tool_calls") or 0) for row in rows),
        "write_calls": sum(int(row.get("metadata", {}).get("write_calls") or 0) for row in rows),
    }


def recommend_max_seq_len(max_tokens: int) -> int:
    for candidate in (4096, 8192, 16384, 32768):
        if max_tokens <= candidate:
            return candidate
    return 32768


def write_report(
    path: Path,
    input_path: Path,
    render_report: Path | None,
    outputs: dict[str, Path],
    summaries: dict[str, dict[str, Any]],
    valid_tasks: set[str],
    validation_errors: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_tokens = max(summary["max_tokens"] for summary in summaries.values())
    recommended = recommend_max_seq_len(max_tokens)
    lines = [
        "# SFT Training Export v1",
        "",
        "## Source",
        "",
        f"- Input SFT data: `{display_path(input_path)}`",
        f"- Render/mask report: `{display_path(render_report) if render_report else '-'}`",
        "- Export format: `tau2_airline_sft_training_v1`",
        "- Training rows include `messages`, `loss_mask`, `loss_policy`, and metadata only.",
        "- Full task/evaluation criteria are intentionally not exported into training rows.",
        "",
        "## Outputs",
        "",
    ]
    for name, output_path in outputs.items():
        lines.append(f"- {name}: `{display_path(output_path)}`")

    lines += [
        "",
        "## Split Summary",
        "",
        "| Split | Rows | Tasks | Mean tokens | P90 | Max | Target ratio | Tool calls | Write calls |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "valid", "heldout"):
        summary = summaries[split_name]
        tasks = ", ".join(f"{task}:{count}" for task, count in summary["tasks"].items())
        target_ratio = summary["mean_target_token_ratio"]
        target_text = "-" if target_ratio is None else f"{target_ratio:.4f}"
        lines.append(
            f"| {split_name} | {summary['rows']} | `{tasks or '-'}` | "
            f"{summary['mean_tokens']:.1f} | {summary['p90_tokens']} | "
            f"{summary['max_tokens']} | {target_text} | "
            f"{summary['tool_calls']} | {summary['write_calls']} |"
        )

    lines += [
        "",
        "## Validation Split",
        "",
        f"- Validation task ids: `{', '.join(sorted(valid_tasks, key=task_sort_key))}`",
        "- Validation is grouped by task id, so no task appears in both train and valid.",
        "- Official test tasks are exported only to heldout.",
        "",
        "## Sequence Length Recommendation",
        "",
        f"- Max Qwen2.5 token count across exported splits: `{max_tokens}`",
        f"- Recommended first SFT `max_seq_len`: `{recommended}`",
        "- A 4K context would truncate several trajectories. Use 8K only for filtered short experiments; use 16K for the full first SFT run.",
        "",
        "## Validation",
        "",
    ]
    if validation_errors:
        lines += ["| Error |", "| --- |"]
        for error in validation_errors:
            lines.append(f"| `{error}` |")
    else:
        lines.append("No structural export errors.")

    lines += [
        "",
        "## Training Notes",
        "",
        "- Use only `train.jsonl` for fitting.",
        "- Use `valid.jsonl` for loss/format monitoring.",
        "- Use `heldout.jsonl` for offline inspection or post-training evaluation, not fitting.",
        "- Loss should be applied only where `loss_mask` is true: assistant content and assistant tool calls.",
        "- The current Windows global Python is 32-bit and has a broken NumPy import; create a fresh 64-bit training environment before running SFT.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Export Phase 1C train/valid/heldout SFT data."
    )
    parser.add_argument("--input", type=Path, default=root / DEFAULT_INPUT)
    parser.add_argument("--render-report", type=Path, default=root / DEFAULT_RENDER_REPORT)
    parser.add_argument("--out-dir", type=Path, default=root / DEFAULT_OUT_DIR)
    parser.add_argument("--out-report", type=Path, default=root / DEFAULT_REPORT)
    parser.add_argument("--out-manifest", type=Path, default=root / DEFAULT_MANIFEST)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--min-valid-rows", type=int, default=4)
    parser.add_argument(
        "--valid-task-ids",
        default=None,
        help="Comma-separated trainable task ids to use for validation.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    render_rows = load_render_rows(args.render_report)
    valid_task_ids = parse_task_ids(args.valid_task_ids)

    raw_train, raw_valid, raw_heldout, valid_tasks = split_rows(
        rows,
        render_rows,
        valid_ratio=args.valid_ratio,
        min_valid_rows=args.min_valid_rows,
        requested_valid_tasks=valid_task_ids,
    )

    train_rows = [export_row(row, "train", render_rows) for row in raw_train]
    valid_rows = [export_row(row, "valid", render_rows) for row in raw_valid]
    heldout_rows = [export_row(row, "heldout", render_rows) for row in raw_heldout]
    validation_errors = validate_export(train_rows, valid_rows, heldout_rows)

    outputs = {
        "train": args.out_dir / "tau2_airline_sft_train.jsonl",
        "valid": args.out_dir / "tau2_airline_sft_valid.jsonl",
        "heldout": args.out_dir / "tau2_airline_sft_heldout.jsonl",
    }
    write_jsonl(outputs["train"], train_rows)
    write_jsonl(outputs["valid"], valid_rows)
    write_jsonl(outputs["heldout"], heldout_rows)

    summaries = {
        "train": split_summary(train_rows),
        "valid": split_summary(valid_rows),
        "heldout": split_summary(heldout_rows),
    }
    manifest = {
        "format_version": "tau2_airline_sft_training_export_v1",
        "input": display_path(args.input),
        "render_report": display_path(args.render_report),
        "outputs": {name: display_path(path) for name, path in outputs.items()},
        "report": display_path(args.out_report),
        "valid_tasks": sorted(valid_tasks, key=task_sort_key),
        "summaries": summaries,
        "validation_errors": validation_errors,
        "recommended_max_seq_len": recommend_max_seq_len(
            max(summary["max_tokens"] for summary in summaries.values())
        ),
    }
    write_json(args.out_manifest, manifest)
    write_report(
        args.out_report,
        args.input,
        args.render_report,
        outputs,
        summaries,
        valid_tasks,
        validation_errors,
    )

    print("SFT training export complete")
    print("============================")
    for split_name, summary in summaries.items():
        print(
            f"{split_name}: rows={summary['rows']} "
            f"tasks={len(summary['tasks'])} max_tokens={summary['max_tokens']}"
        )
    print(f"valid_tasks: {', '.join(sorted(valid_tasks, key=task_sort_key))}")
    print(f"recommended_max_seq_len: {manifest['recommended_max_seq_len']}")
    print(f"validation_errors: {len(validation_errors)}")
    for output_path in outputs.values():
        print(f"wrote: {output_path}")
    print(f"wrote: {args.out_manifest}")
    print(f"wrote: {args.out_report}")


if __name__ == "__main__":
    main()
