from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_SFT = "data/sft/tau2_airline_sft_v1.jsonl"
DEFAULT_REPORT_JSON = "reports/sft_render_mask_v1.json"
DEFAULT_REPORT_MD = "reports/sft_render_mask_v1.md"


@dataclass
class RenderResult:
    text: str
    target_spans: list[tuple[int, int]]
    marked_text: str


@dataclass
class DirectTokenizerBundle:
    tokenizer: Any
    chat_template: str | None
    special_tokens: dict[str, Any]

    def encode_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer.encode(text)
        return list(encoded.ids), list(encoded.offsets)

    def render_chat_template(self, messages: list[dict[str, Any]]) -> str:
        if not self.chat_template:
            raise ValueError("tokenizer_config.json does not contain chat_template")
        from jinja2 import Environment

        def raise_exception(message: str) -> None:
            raise ValueError(message)

        env = Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda _fmt: ""
        template = env.from_string(self.chat_template)
        return template.render(
            messages=to_qwen_messages(messages),
            tools=None,
            add_generation_prompt=False,
            **self.special_tokens,
        )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def compact_counter(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def message_roles(messages: list[dict[str, Any]]) -> dict[str, int]:
    return compact_counter(Counter(str(message.get("role")) for message in messages))


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tool_call_payload(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "name": function.get("name") or call.get("name") or "",
        "arguments": function.get("arguments") or call.get("arguments") or {},
    }


def append_target_part(
    parts: list[str],
    spans: list[tuple[int, int]],
    cursor: int,
    text: str,
    target: bool,
) -> int:
    if target and text:
        spans.append((cursor, cursor + len(text)))
    parts.append(text)
    return cursor + len(text)


def render_message(
    message: dict[str, Any],
    target: bool,
    assistant_content_loss: bool,
    assistant_tool_call_loss: bool,
    assistant_tool_call_wrapper_loss: bool = False,
) -> tuple[str, list[tuple[int, int]]]:
    role = str(message.get("role") or "unknown")
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0

    header = f"<|im_start|>{role}"
    if role == "tool":
        name = str(message.get("name") or "tool")
        tool_call_id = str(message.get("tool_call_id") or "")
        header += f" name={name} tool_call_id={tool_call_id}"
    header += "\n"
    cursor = append_target_part(parts, spans, cursor, header, False)

    content = str(message.get("content") or "")
    if role == "assistant":
        cursor = append_target_part(
            parts,
            spans,
            cursor,
            content,
            target and assistant_content_loss and bool(content),
        )
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            if content or parts[-1] != header:
                cursor = append_target_part(parts, spans, cursor, "\n", False)
            target_wrapper = (
                target
                and assistant_tool_call_loss
                and assistant_tool_call_wrapper_loss
            )
            cursor = append_target_part(parts, spans, cursor, "<tool_call>\n", target_wrapper)
            payload = json_text(tool_call_payload(call))
            cursor = append_target_part(
                parts,
                spans,
                cursor,
                payload,
                target and assistant_tool_call_loss and bool(payload),
            )
            cursor = append_target_part(parts, spans, cursor, "\n</tool_call>", target_wrapper)
    else:
        cursor = append_target_part(parts, spans, cursor, content, False)

    cursor = append_target_part(parts, spans, cursor, "\n<|im_end|>\n", False)
    return "".join(parts), spans


def mark_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for idx, (start, end) in enumerate(sorted(spans), start=1):
        pieces.append(text[cursor:start])
        pieces.append(f"@@TARGET_{idx}_START@@")
        pieces.append(text[start:end])
        pieces.append(f"@@TARGET_{idx}_END@@")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def strip_markers(text: str, count: int) -> str:
    for idx in range(1, count + 1):
        text = text.replace(f"@@TARGET_{idx}_START@@", "")
        text = text.replace(f"@@TARGET_{idx}_END@@", "")
    return text


def render_with_spans(row: dict[str, Any]) -> RenderResult:
    messages = row.get("messages") or []
    loss_mask = row.get("loss_mask") or []
    policy = row.get("loss_policy") or {}
    assistant_content_loss = bool(policy.get("assistant_content", True))
    assistant_tool_call_loss = bool(policy.get("assistant_tool_calls", True))
    assistant_tool_call_wrapper_loss = bool(policy.get("assistant_tool_call_wrappers", False))

    parts: list[str] = []
    absolute_spans: list[tuple[int, int]] = []
    cursor = 0
    for message, target in zip(messages, loss_mask):
        segment, spans = render_message(
            message,
            bool(target),
            assistant_content_loss,
            assistant_tool_call_loss,
            assistant_tool_call_wrapper_loss,
        )
        for start, end in spans:
            absolute_spans.append((cursor + start, cursor + end))
        parts.append(segment)
        cursor += len(segment)

    text = "".join(parts)
    marked = mark_spans(text, absolute_spans)
    if strip_markers(marked, len(absolute_spans)) != text:
        raise AssertionError("Render-Twice-Diff marker round-trip failed.")
    return RenderResult(text=text, target_spans=absolute_spans, marked_text=marked)


def validate_structure(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    messages = row.get("messages") or []
    loss_mask = row.get("loss_mask") or []

    if not messages:
        errors.append("empty_messages")
    if len(messages) != len(loss_mask):
        errors.append("messages_loss_mask_length_mismatch")

    seen_tool_calls: set[str] = set()
    seen_tool_responses: set[str] = set()
    valid_roles = {"system", "user", "assistant", "tool"}

    for idx, message in enumerate(messages):
        role = message.get("role")
        target = bool(loss_mask[idx]) if idx < len(loss_mask) else False
        if role not in valid_roles:
            errors.append(f"invalid_role:{idx}:{role}")
        if role in {"user", "tool"} and target:
            errors.append(f"non_assistant_target:{idx}:{role}")
        if role == "assistant" and not target:
            warnings.append(f"assistant_not_targeted:{idx}")

        if role == "assistant":
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            if target and not content and not tool_calls:
                warnings.append(f"empty_target_assistant:{idx}")
            for call in tool_calls:
                call_id = str(call.get("id") or "")
                function = call.get("function") or {}
                if not call_id:
                    errors.append(f"tool_call_missing_id:{idx}")
                if not function.get("name"):
                    errors.append(f"tool_call_missing_name:{idx}:{call_id}")
                if call_id:
                    seen_tool_calls.add(call_id)

        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            if not call_id:
                errors.append(f"tool_message_missing_call_id:{idx}")
            elif call_id not in seen_tool_calls:
                errors.append(f"tool_message_without_prior_call:{idx}:{call_id}")
            else:
                seen_tool_responses.add(call_id)
            if not message.get("name"):
                warnings.append(f"tool_message_missing_name:{idx}:{call_id}")

    missing_responses = sorted(seen_tool_calls - seen_tool_responses)
    if missing_responses:
        warnings.append("tool_calls_without_response:" + ",".join(missing_responses[:5]))

    if messages and messages[0].get("role") == "assistant":
        warnings.append("conversation_starts_with_assistant")

    return errors, warnings


def try_load_tokenizer(model: str, local_files_only: bool):
    transformers_status = ""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ModuleNotFoundError:
        transformers_status = "missing_transformers"
    except Exception as exc:  # noqa: BLE001
        transformers_status = f"transformers_import_failed:{type(exc).__name__}:{exc}"
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=local_files_only,
            )
            return tokenizer, "transformers_loaded"
        except Exception as exc:  # noqa: BLE001
            transformers_status = f"transformers_load_failed:{type(exc).__name__}:{exc}"

    direct_tokenizer, direct_status = try_load_direct_tokenizer(model, local_files_only)
    if direct_tokenizer is not None:
        return direct_tokenizer, f"direct_tokenizers_loaded_after_{transformers_status}"
    return None, f"{transformers_status}; direct_tokenizers_failed:{direct_status}"


def try_load_direct_tokenizer(model: str, local_files_only: bool):
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except Exception as exc:  # noqa: BLE001
        return None, f"load_failed:{type(exc).__name__}:{exc}"

    try:
        tokenizer_path = hf_hub_download(
            repo_id=model,
            filename="tokenizer.json",
            local_files_only=local_files_only,
        )
        config_path = hf_hub_download(
            repo_id=model,
            filename="tokenizer_config.json",
            local_files_only=local_files_only,
        )
        tokenizer = Tokenizer.from_file(tokenizer_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        special_tokens = {
            key: value
            for key, value in config.items()
            if key.endswith("_token") and isinstance(value, str)
        }
        return (
            DirectTokenizerBundle(
                tokenizer=tokenizer,
                chat_template=config.get("chat_template"),
                special_tokens=special_tokens,
            ),
            "direct_tokenizers_loaded",
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}:{exc}"


def token_stats(tokenizer: Any, text: str, spans: list[tuple[int, int]]) -> dict[str, Any]:
    if hasattr(tokenizer, "encode_offsets"):
        input_ids, offsets = tokenizer.encode_offsets(text)
    else:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        input_ids = encoded.get("input_ids") or []
        if offsets is None:
            return {"tokenizer_offsets": False}

    target_tokens = 0
    for start, end in offsets:
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            target_tokens += 1

    return {
        "tokenizer_offsets": True,
        "tokens": len(input_ids),
        "target_tokens": target_tokens,
        "target_token_ratio": (
            target_tokens / len(input_ids)
            if input_ids
            else 0.0
        ),
    }


def to_qwen_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qwen_messages: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "assistant" and item.get("tool_calls"):
            calls = []
            for call in item.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments") or {}
                calls.append(
                    {
                        "id": call.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": function.get("name") or "",
                            "arguments": json_text(arguments),
                        },
                    }
                )
            item["tool_calls"] = calls
        qwen_messages.append(item)
    return qwen_messages


def official_template_check(tokenizer: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    if tokenizer is None:
        return {"official_template_status": "skipped_no_tokenizer"}
    if hasattr(tokenizer, "render_chat_template"):
        try:
            rendered = tokenizer.render_chat_template(messages)
            return {
                "official_template_status": "ok",
                "official_render_chars": len(rendered),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "official_template_status": "failed",
                "official_template_error": f"{type(exc).__name__}: {exc}",
            }
    if not hasattr(tokenizer, "apply_chat_template"):
        return {"official_template_status": "skipped_no_apply_chat_template"}
    try:
        rendered = tokenizer.apply_chat_template(
            to_qwen_messages(messages),
            tokenize=False,
            add_generation_prompt=False,
        )
        return {
            "official_template_status": "ok",
            "official_render_chars": len(rendered),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "official_template_status": "failed",
            "official_template_error": f"{type(exc).__name__}: {exc}",
        }


def check_row(row: dict[str, Any], tokenizer: Any | None) -> dict[str, Any]:
    errors, warnings = validate_structure(row)
    render = render_with_spans(row)
    target_chars = sum(end - start for start, end in render.target_spans)

    result: dict[str, Any] = {
        "id": row.get("id"),
        "task_id": row.get("task_id"),
        "trial": row.get("trial"),
        "trainable": row.get("trainable"),
        "official_split": row.get("official_split"),
        "message_count": len(row.get("messages") or []),
        "roles": message_roles(row.get("messages") or []),
        "assistant_target_messages": sum(1 for value in row.get("loss_mask") or [] if value),
        "render_chars": len(render.text),
        "target_spans": len(render.target_spans),
        "target_chars": target_chars,
        "target_char_ratio": target_chars / len(render.text) if render.text else 0.0,
        "errors": errors,
        "warnings": warnings,
    }
    if tokenizer is not None:
        result.update(token_stats(tokenizer, render.text, render.target_spans))
        result.update(official_template_check(tokenizer, row.get("messages") or []))
    else:
        result.update({"tokenizer_offsets": False, "official_template_status": "skipped"})
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_status = str(summary["tokenizer_status"]).replace("\n", " ")

    lines = [
        "# SFT Render/Mask Check v1",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Input rows | {summary['rows']} |",
        f"| Checked rows | {summary['checked_rows']} |",
        f"| Rows with errors | {summary['rows_with_errors']} |",
        f"| Rows with warnings | {summary['rows_with_warnings']} |",
        f"| Tokenizer status | `{tokenizer_status}` |",
        f"| Official template ok | {summary['official_template_ok']} |",
        f"| Official template failed | {summary['official_template_failed']} |",
        "",
        "## Lengths",
        "",
        "| Render | Mean | P50 | P90 | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| chars | {summary['render_chars_mean']:.1f} | "
            f"{summary['render_chars_p50']} | {summary['render_chars_p90']} | "
            f"{summary['render_chars_max']} |"
        ),
    ]
    if summary.get("tokenizer_tokens_max") is not None:
        lines.append(
            f"| tokens | {summary['tokenizer_tokens_mean']:.1f} | "
            f"{summary['tokenizer_tokens_p50']} | {summary['tokenizer_tokens_p90']} | "
            f"{summary['tokenizer_tokens_max']} |"
        )

    lines += [
        "",
        "## Loss Target Coverage",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean target char ratio | {summary['target_char_ratio_mean']:.4f} |",
        f"| Min target char ratio | {summary['target_char_ratio_min']:.4f} |",
        f"| Max target char ratio | {summary['target_char_ratio_max']:.4f} |",
    ]
    if summary.get("target_token_ratio_mean") is not None:
        lines.append(
            f"| Mean target token ratio | {summary['target_token_ratio_mean']:.4f} |"
        )

    lines += [
        "",
        "## Error And Warning Types",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    issues = summary.get("issue_counts") or {}
    if issues:
        for issue, count in sorted(issues.items()):
            lines.append(f"| `{issue}` | {count} |")
    else:
        lines.append("| - | 0 |")

    lines += [
        "",
        "## Per-row Check",
        "",
        "| Sample | Task | Split | Msgs | Target spans | Chars | Errors | Warnings |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        errors = ", ".join(f"`{item}`" for item in row["errors"]) or "-"
        warnings = ", ".join(f"`{item}`" for item in row["warnings"][:3]) or "-"
        if len(row["warnings"]) > 3:
            warnings += f", +{len(row['warnings']) - 3}"
        lines.append(
            f"| `{row['id']}` | {row['task_id']} | {row['official_split']} | "
            f"{row['message_count']} | {row['target_spans']} | {row['render_chars']} | "
            f"{errors} | {warnings} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- This v1 check always runs a project-native ChatML-style renderer.",
        "- It first tries `transformers`; if that is unavailable or broken, it falls back to `huggingface_hub` + `tokenizers` + Jinja for Qwen tokenizer/template checks.",
        "- Token-level label spans are computed on the project-native rendered text; official Qwen `chat_template` rendering is checked for compatibility.",
        "- The `conversation_starts_with_assistant` warning is expected for tau2 because the agent opens the conversation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, Any]], checked: list[dict[str, Any]], tokenizer_status: str) -> dict[str, Any]:
    render_chars = [int(row["render_chars"]) for row in checked]
    target_ratios = [float(row["target_char_ratio"]) for row in checked]
    token_counts = [
        int(row["tokens"])
        for row in checked
        if row.get("tokenizer_offsets") and row.get("tokens") is not None
    ]
    token_target_ratios = [
        float(row["target_token_ratio"])
        for row in checked
        if row.get("tokenizer_offsets") and row.get("target_token_ratio") is not None
    ]
    issue_counts = Counter()
    for row in checked:
        issue_counts.update(row.get("errors") or [])
        issue_counts.update(row.get("warnings") or [])

    return {
        "rows": len(rows),
        "checked_rows": len(checked),
        "rows_with_errors": sum(1 for row in checked if row.get("errors")),
        "rows_with_warnings": sum(1 for row in checked if row.get("warnings")),
        "tokenizer_status": tokenizer_status,
        "official_template_ok": sum(
            1 for row in checked if row.get("official_template_status") == "ok"
        ),
        "official_template_failed": sum(
            1 for row in checked if row.get("official_template_status") == "failed"
        ),
        "render_chars_mean": mean(render_chars) if render_chars else 0.0,
        "render_chars_p50": percentile(render_chars, 0.5),
        "render_chars_p90": percentile(render_chars, 0.9),
        "render_chars_max": max(render_chars) if render_chars else 0,
        "target_char_ratio_mean": mean(target_ratios) if target_ratios else 0.0,
        "target_char_ratio_min": min(target_ratios) if target_ratios else 0.0,
        "target_char_ratio_max": max(target_ratios) if target_ratios else 0.0,
        "tokenizer_tokens_mean": mean(token_counts) if token_counts else None,
        "tokenizer_tokens_p50": percentile(token_counts, 0.5) if token_counts else None,
        "tokenizer_tokens_p90": percentile(token_counts, 0.9) if token_counts else None,
        "tokenizer_tokens_max": max(token_counts) if token_counts else None,
        "target_token_ratio_mean": (
            mean(token_target_ratios) if token_target_ratios else None
        ),
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Check SFT message rendering and assistant-only loss mask spans."
    )
    parser.add_argument("--input", type=Path, default=root / DEFAULT_SFT)
    parser.add_argument("--out-json", type=Path, default=root / DEFAULT_REPORT_JSON)
    parser.add_argument("--out-md", type=Path, default=root / DEFAULT_REPORT_MD)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--trainable-only", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download tokenizer files when transformers is available.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Only run structural and project-native render checks.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    selected = [row for row in rows if row.get("trainable") or not args.trainable_only]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    tokenizer = None
    tokenizer_status = "skipped"
    if not args.skip_tokenizer:
        tokenizer, tokenizer_status = try_load_tokenizer(args.model, args.local_files_only)

    checked = [check_row(row, tokenizer) for row in selected]
    payload = {
        "input": str(args.input),
        "model": args.model,
        "summary": summarize(rows, checked, tokenizer_status),
        "rows": checked,
    }
    write_json(args.out_json, payload)
    write_markdown(args.out_md, payload)

    summary = payload["summary"]
    print("SFT render/mask check")
    print("=====================")
    print(f"input_rows: {summary['rows']}")
    print(f"checked_rows: {summary['checked_rows']}")
    print(f"rows_with_errors: {summary['rows_with_errors']}")
    print(f"rows_with_warnings: {summary['rows_with_warnings']}")
    print(f"tokenizer_status: {summary['tokenizer_status']}")
    print(f"official_template_ok: {summary['official_template_ok']}")
    print(f"official_template_failed: {summary['official_template_failed']}")
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
