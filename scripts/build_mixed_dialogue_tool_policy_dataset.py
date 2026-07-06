from __future__ import annotations

import argparse
import copy
import hashlib
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

from check_sft_render_mask import render_with_spans, validate_structure  # noqa: E402
from slot_grounding_validator import (  # noqa: E402
    SlotState,
    add_slots_from_text,
    add_slots_from_tool_result,
    call_name_and_args,
    extract_slots_from_text,
    parse_json_maybe,
    validate_call,
)


DEFAULT_TRAIN = "data/train/tau2_airline_sft_train.jsonl"
DEFAULT_VALID = "data/train/tau2_airline_sft_valid.jsonl"
DEFAULT_HELDOUT = "data/train/tau2_airline_sft_heldout.jsonl"
DEFAULT_OUT_DIR = "data/mixed_policy"
DEFAULT_REPORT = "reports/mixed_dialogue_tool_policy_dataset_v1_2048.md"
DEFAULT_MANIFEST = "data/mixed_policy/tau2_airline_mixed_dialogue_tool_policy_manifest_v1_2048.json"
DEFAULT_OUTPUT_STEM = "tau2_airline_mixed_dialogue_tool_policy_v1_2048"
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


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


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
    if not item.get("id"):
        stable = hashlib.sha1(
            json_text({"name": function.get("name"), "arguments": function.get("arguments")}).encode("utf-8")
        ).hexdigest()[:16]
        item["id"] = f"call_mixed_{stable}"
    return item


def tool_payload(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    args = function.get("arguments")
    if args is None:
        args = call.get("arguments") or {}
    return {
        "name": function.get("name") or call.get("name") or "",
        "arguments": args,
    }


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    for message in messages:
        item = copy.deepcopy(message)
        role = item.get("role")
        if role == "assistant":
            calls = [normalize_tool_call(call) for call in item.get("tool_calls") or []]
            for call in calls:
                call_id = str(call.get("id") or "")
                if call_id:
                    call_names[call_id] = str((call.get("function") or {}).get("name") or "")
            if calls:
                item["tool_calls"] = calls
            elif "tool_calls" in item:
                item.pop("tool_calls", None)
            if item.get("content") is None:
                item["content"] = ""
        elif role == "tool":
            call_id = str(item.get("tool_call_id") or item.get("id") or "")
            if call_id:
                item["tool_call_id"] = call_id
            if not item.get("name") and call_id in call_names:
                item["name"] = call_names[call_id]
            if item.get("content") is None:
                item["content"] = ""
        normalized.append(item)
    return normalized


def issue_to_dict(issue: Any) -> dict[str, Any]:
    return {
        "component_name": issue.component_name,
        "slot_type": issue.slot_type,
        "value": issue.value,
        "tool_name": issue.tool_name,
        "message_index": issue.message_index,
        "call_id": issue.call_id,
        "severity": issue.severity,
        "detail": issue.detail,
        "tags": issue.tags,
    }


def validate_target_grounding(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    target_index = len(messages) - 1
    target = messages[-1] if messages else {}
    state = SlotState()
    call_by_id: dict[str, tuple[str, dict[str, Any]]] = {}

    for message_index, message in enumerate(messages[:-1]):
        role = message.get("role")
        if role == "user":
            add_slots_from_text(state, str(message.get("content") or ""), "user", message_index)
            continue
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name, arguments, call_id = call_name_and_args(call)
                if call_id:
                    call_by_id[call_id] = (name, arguments)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("id") or "")
            name, arguments = call_by_id.get(call_id, ("", {}))
            content = message.get("content")
            content_text = str(content or "")
            if message.get("error") or content_text.lower().startswith("error:"):
                for slot_type, value in extract_slots_from_text(content_text):
                    state.remember_not_found(slot_type, value)
                continue
            add_slots_from_tool_result(state, name, arguments, parse_json_maybe(content), message_index)

    issues = []
    assistant_text = str(target.get("content") or "")
    for call in target.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name, arguments, call_id = call_name_and_args(call)
        issues.extend(validate_call(state, name, arguments, call_id, target_index, assistant_text))

    return {
        "is_grounded": len(issues) == 0,
        "issue_count": len(issues),
        "issues": [issue_to_dict(issue) for issue in issues],
        "known_slot_counts_before_target": state.counts(),
    }


def load_tokenizer(model: str, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    start_a, end_a = span_a
    start_b, end_b = span_b
    return start_a < end_b and end_a > start_b


def token_stats(row: dict[str, Any], tokenizer: Any) -> tuple[int, int]:
    render = render_with_spans(row)
    encoded = tokenizer(
        render.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    target_tokens = 0
    for start, end in offsets:
        if start == end:
            continue
        if any(overlap((start, end), span) for span in render.target_spans):
            target_tokens += 1
    return len(input_ids), target_tokens


def with_messages(row: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    updated = copy.deepcopy(row)
    updated["messages"] = messages
    updated["loss_mask"] = [False] * (len(messages) - 1) + [True]
    updated.setdefault("metadata", {})["prefix_message_count"] = len(messages) - 1
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
    original_tokens, _target_tokens = token_stats(original, tokenizer)

    candidate = original
    kept_prefix = list(prefix)
    while token_stats(candidate, tokenizer)[0] > max_sample_tokens and kept_prefix:
        kept_prefix = drop_dangling_leading_tool_messages(kept_prefix[1:])
        candidate = with_messages(original, kept_prefix + [target])

    final_tokens, _ = token_stats(candidate, tokenizer)
    if final_tokens > max_sample_tokens:
        candidate = with_messages(original, [target])
        final_tokens, _ = token_stats(candidate, tokenizer)

    metadata = candidate.setdefault("metadata", {})
    metadata["context_trim"] = {
        "strategy": "message_suffix_target_preserving",
        "max_sample_tokens": max_sample_tokens,
        "original_qwen_tokens": original_tokens,
        "final_qwen_tokens": final_tokens,
        "original_prefix_message_count": len(prefix),
        "kept_prefix_message_count": len(candidate.get("messages") or []) - 1,
        "dropped_prefix_message_count": len(prefix) - (len(candidate.get("messages") or []) - 1),
    }
    return candidate


def source_target_indices(row: dict[str, Any]) -> list[int]:
    messages = row.get("messages") or []
    loss_mask = row.get("loss_mask") or []
    if len(loss_mask) != len(messages):
        return [idx for idx, message in enumerate(messages) if message.get("role") == "assistant"]
    return [
        idx
        for idx, (message, target) in enumerate(zip(messages, loss_mask))
        if target and message.get("role") == "assistant"
    ]


def base_metadata(row: dict[str, Any], split: str, turn_index: int) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "source_id": row.get("id"),
        "source_format_version": row.get("format_version"),
        "source_sample_type": row.get("sample_type") or metadata.get("sample_type"),
        "domain": metadata.get("domain", "airline"),
        "task_id": str(metadata.get("task_id") or ""),
        "trial": metadata.get("trial"),
        "simulation_id": metadata.get("simulation_id"),
        "official_split": metadata.get("official_split") or split,
        "source_reward": metadata.get("reward"),
        "source_process_score": metadata.get("process_score"),
        "source_risk_tags": metadata.get("risk_tags") or [],
        "turn_index": turn_index,
        "split": split,
    }


def make_sample(
    sample_id: str,
    split: str,
    sample_type: str,
    prefix: list[dict[str, Any]],
    target: dict[str, Any],
    loss_policy: dict[str, bool],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    messages = copy.deepcopy(prefix) + [copy.deepcopy(target)]
    return {
        "id": sample_id,
        "format_version": "tau2_airline_mixed_dialogue_tool_policy_v1",
        "sample_type": sample_type,
        "split": split,
        "messages": messages,
        "loss_mask": [False] * len(prefix) + [True],
        "loss_policy": {
            "assistant_content": bool(loss_policy.get("assistant_content", False)),
            "assistant_tool_calls": bool(loss_policy.get("assistant_tool_calls", False)),
            "assistant_tool_call_wrappers": bool(loss_policy.get("assistant_tool_call_wrappers", False)),
            "user": False,
            "tool": False,
        },
        "metadata": metadata,
    }


def text_sample(row: dict[str, Any], split: str, turn_index: int, message: dict[str, Any]) -> dict[str, Any] | None:
    content = str(message.get("content") or "").strip()
    if not content:
        return None
    metadata = base_metadata(row, split, turn_index)
    metadata.update(
        {
            "target_action": "assistant_text",
            "target_content_chars": len(content),
            "prefix_message_count": turn_index,
            "source_target_tool_call_count": 0,
        }
    )
    target = {"role": "assistant", "content": message.get("content") or ""}
    return make_sample(
        f"mixed_policy_{split}_{row.get('id')}_turn{turn_index}_text",
        split,
        "mixed_policy_text",
        row["messages"][:turn_index],
        target,
        {"assistant_content": True},
        metadata,
    )


def tool_sample(
    row: dict[str, Any],
    split: str,
    turn_index: int,
    prefix: list[dict[str, Any]],
    call: dict[str, Any],
    content: str,
    sample_type: str,
    sample_suffix: str,
    source_call_count: int,
    call_index: int,
    protocol_only: bool,
) -> dict[str, Any]:
    normalized_call = normalize_tool_call(call)
    payload = tool_payload(normalized_call)
    target_content = "" if protocol_only else content
    metadata = base_metadata(row, split, turn_index)
    metadata.update(
        {
            "target_action": "tool_call",
            "target_tool_call_count": 1,
            "target_tool_names": [payload["name"]],
            "target_tool_calls": [payload],
            "prefix_message_count": len(prefix),
            "source_target_tool_call_count": source_call_count,
            "source_call_index": call_index,
            "source_had_assistant_content": bool(content),
            "target_content_chars": len(target_content),
            "protocol_only": protocol_only,
            "protocol_wrapper_loss": True,
        }
    )
    target = {
        "role": "assistant",
        "content": target_content,
        "tool_calls": [normalized_call],
    }
    return make_sample(
        f"mixed_policy_{split}_{row.get('id')}_turn{turn_index}_{sample_suffix}",
        split,
        sample_type,
        prefix,
        target,
        {
            "assistant_content": bool(target_content),
            "assistant_tool_calls": True,
            "assistant_tool_call_wrappers": True,
        },
        metadata,
    )


def following_tool_responses(messages: list[dict[str, Any]], turn_index: int) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    for message in messages[turn_index + 1 :]:
        if message.get("role") != "tool":
            break
        call_id = str(message.get("tool_call_id") or message.get("id") or "")
        if call_id:
            responses[call_id] = copy.deepcopy(message)
    return responses


def build_multi_tool_sequence(
    row: dict[str, Any],
    split: str,
    turn_index: int,
    message: dict[str, Any],
    rejects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls = [normalize_tool_call(call) for call in message.get("tool_calls") or []]
    responses = following_tool_responses(row["messages"], turn_index)
    prefix = copy.deepcopy(row["messages"][:turn_index])
    samples: list[dict[str, Any]] = []
    source_content = str(message.get("content") or "")

    for call_index, call in enumerate(calls):
        call_id = str(call.get("id") or "")
        target_content = source_content if call_index == 0 else ""
        sample = tool_sample(
            row=row,
            split=split,
            turn_index=turn_index,
            prefix=copy.deepcopy(prefix),
            call=call,
            content=target_content,
            sample_type="mixed_policy_sequential_tool",
            sample_suffix=f"seq_call{call_index}",
            source_call_count=len(calls),
            call_index=call_index,
            protocol_only=False,
        )
        sample["metadata"]["sequentialized_from_parallel_tool_calls"] = True
        samples.append(sample)

        response = responses.get(call_id)
        if response is None:
            rejected = copy.deepcopy(sample)
            rejected.setdefault("metadata", {})["mixed_policy_reject_reason"] = "missing_tool_response_for_sequence"
            rejects.append(rejected)
            break
        prefix.append(
            {
                "role": "assistant",
                "content": target_content,
                "tool_calls": [call],
            }
        )
        prefix.append(response)
    return samples


def build_split(rows: list[dict[str, Any]], split: str, include_protocol_variants: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for source_row in rows:
        row = copy.deepcopy(source_row)
        row["messages"] = normalize_messages(row.get("messages") or [])
        for turn_index in source_target_indices(row):
            message = row["messages"][turn_index]
            calls = message.get("tool_calls") or []
            content = str(message.get("content") or "")
            if not calls:
                sample = text_sample(row, split, turn_index, message)
                if sample is not None:
                    samples.append(sample)
                continue
            if len(calls) == 1:
                samples.append(
                    tool_sample(
                        row=row,
                        split=split,
                        turn_index=turn_index,
                        prefix=copy.deepcopy(row["messages"][:turn_index]),
                        call=calls[0],
                        content=content,
                        sample_type="mixed_policy_single_tool",
                        sample_suffix="single_tool",
                        source_call_count=1,
                        call_index=0,
                        protocol_only=False,
                    )
                )
                if include_protocol_variants:
                    samples.append(
                        tool_sample(
                            row=row,
                            split=split,
                            turn_index=turn_index,
                            prefix=copy.deepcopy(row["messages"][:turn_index]),
                            call=calls[0],
                            content=content,
                            sample_type="mixed_policy_protocol_tool",
                            sample_suffix="single_tool_protocol",
                            source_call_count=1,
                            call_index=0,
                            protocol_only=True,
                        )
                    )
                continue
            samples.extend(build_multi_tool_sequence(row, split, turn_index, message, rejected))
    return samples, rejected


def reject(row: dict[str, Any], reason: str, rejected: list[dict[str, Any]]) -> None:
    item = copy.deepcopy(row)
    item.setdefault("metadata", {})["mixed_policy_reject_reason"] = reason
    rejected.append(item)


def validate_and_finalize(
    samples: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    tokenizer: Any | None,
    max_sample_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    clean: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()

    for row in samples:
        row_id = str(row.get("id"))
        if row_id in seen:
            reject(row, "duplicate_id", rejected)
            errors.append(f"{row_id}:duplicate_id")
            continue
        seen.add(row_id)

        if tokenizer is not None:
            row = trim_prefix_to_budget(row, tokenizer, max_sample_tokens)

        structure_errors, structure_warnings = validate_structure(row)
        if structure_errors:
            reject(row, "structure_error", rejected)
            errors.extend(f"{row_id}:{error}" for error in structure_errors)
            continue
        row.setdefault("metadata", {})["render_check_warnings"] = structure_warnings

        target = (row.get("messages") or [{}])[-1]
        target_calls = target.get("tool_calls") or []
        target_action = (row.get("metadata") or {}).get("target_action")
        if target_action == "assistant_text":
            if target_calls:
                reject(row, "text_target_has_tool_calls", rejected)
                errors.append(f"{row_id}:text_target_has_tool_calls")
                continue
            if not str(target.get("content") or "").strip():
                reject(row, "empty_text_target", rejected)
                errors.append(f"{row_id}:empty_text_target")
                continue
        elif target_action == "tool_call":
            if len(target_calls) != 1:
                reject(row, f"tool_target_call_count_{len(target_calls)}", rejected)
                errors.append(f"{row_id}:tool_target_call_count_{len(target_calls)}")
                continue
            grounding = validate_target_grounding(row)
            row.setdefault("metadata", {})["target_slot_grounding"] = grounding
            if not grounding.get("is_grounded"):
                reject(row, "target_not_grounded", rejected)
                continue
        else:
            reject(row, "unknown_target_action", rejected)
            errors.append(f"{row_id}:unknown_target_action")
            continue

        if tokenizer is not None:
            tokens, target_tokens = token_stats(row, tokenizer)
            row["metadata"]["qwen_tokens"] = tokens
            row["metadata"]["qwen_target_tokens"] = target_tokens
            row["metadata"]["truncated_at_stats_max_seq_len"] = tokens > max_sample_tokens
            if tokens > max_sample_tokens:
                reject(row, f"over_budget_{tokens}_gt_{max_sample_tokens}", rejected)
                errors.append(f"{row_id}:over_budget:{tokens}>{max_sample_tokens}")
                continue
            if target_tokens <= 0:
                reject(row, "target_tokens_zero", rejected)
                errors.append(f"{row_id}:target_tokens_zero")
                continue

        clean.append(row)
    return clean, rejected, errors


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter()
    sample_types = Counter()
    target_actions = Counter()
    tools = Counter()
    source_call_counts = Counter()
    token_counts: list[int] = []
    target_token_counts: list[int] = []
    dropped_prefix: list[int] = []
    text_chars: list[int] = []

    for row in samples:
        metadata = row.get("metadata") or {}
        tasks[str(metadata.get("task_id") or "")] += 1
        sample_types[str(row.get("sample_type") or "")] += 1
        target_actions[str(metadata.get("target_action") or "")] += 1
        source_call_counts[str(metadata.get("source_target_tool_call_count") or 0)] += 1
        for tool in metadata.get("target_tool_names") or []:
            tools[str(tool)] += 1
        if metadata.get("qwen_tokens") is not None:
            token_counts.append(int(metadata["qwen_tokens"]))
        if metadata.get("qwen_target_tokens") is not None:
            target_token_counts.append(int(metadata["qwen_target_tokens"]))
        trim = metadata.get("context_trim") or {}
        dropped_prefix.append(int(trim.get("dropped_prefix_message_count") or 0))
        text_chars.append(int(metadata.get("target_content_chars") or 0))

    return {
        "rows": len(samples),
        "tasks": dict(sorted(tasks.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[0]))),
        "sample_types": dict(sorted(sample_types.items())),
        "target_actions": dict(sorted(target_actions.items())),
        "tools": dict(sorted(tools.items())),
        "source_call_counts": dict(sorted(source_call_counts.items())),
        "mean_tokens": mean(token_counts) if token_counts else 0,
        "p90_tokens": percentile(token_counts, 0.9),
        "max_tokens": max(token_counts, default=0),
        "mean_target_tokens": mean(target_token_counts) if target_token_counts else 0,
        "max_target_tokens": max(target_token_counts, default=0),
        "trimmed_rows": sum(1 for value in dropped_prefix if value > 0),
        "max_dropped_prefix_messages": max(dropped_prefix, default=0),
        "mean_target_content_chars": mean(text_chars) if text_chars else 0,
    }


def summarize_rejections(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(str((row.get("metadata") or {}).get("mixed_policy_reject_reason") or "") for row in rows)
    tasks = Counter(str((row.get("metadata") or {}).get("task_id") or "") for row in rows)
    return {
        "rows": len(rows),
        "reasons": dict(sorted(reasons.items())),
        "tasks": dict(sorted(tasks.items(), key=lambda item: (int(item[0]) if item[0].isdigit() else 10**9, item[0]))),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mixed Dialogue Tool Policy Dataset v1",
        "",
        "## Goal",
        "",
        "Train the agent on full assistant policy decisions, not only next tool-call protocol. The dataset includes natural assistant text, grounded single-tool calls, and sequentialized single-tool targets converted from gold parallel tool-call turns.",
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
            f"- Tokenizer stats enabled: `{payload['tokenizer_stats_enabled']}`",
            f"- Max sample tokens: `{payload['max_sample_tokens']}`",
            f"- Include protocol variants: `{payload['include_protocol_variants']}`",
            "- Tool targets are required to be grounded in the online prefix.",
            "- Parallel gold tool turns are converted to sequential one-call targets.",
            "",
            "## Summary",
            "",
            "| Split | Rows | Rejected | Target actions | Sample types | Mean tokens | P90 tokens | Max tokens | Mean target tokens | Trimmed |",
            "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, summary in payload["summaries"].items():
        rejected = payload["rejected_summaries"][split]
        target_actions = ", ".join(f"{name}:{count}" for name, count in summary["target_actions"].items())
        sample_types = ", ".join(f"{name}:{count}" for name, count in summary["sample_types"].items())
        lines.append(
            f"| {split} | {summary['rows']} | {rejected['rows']} | `{target_actions}` | `{sample_types}` | "
            f"{summary['mean_tokens']:.1f} | {summary['p90_tokens']} | {summary['max_tokens']} | "
            f"{summary['mean_target_tokens']:.1f} | {summary['trimmed_rows']} |"
        )

    lines.extend(["", "## Tool Coverage", "", "| Split | Tools | Source call counts |", "| --- | --- | --- |"])
    for split, summary in payload["summaries"].items():
        tools = ", ".join(f"{name}:{count}" for name, count in summary["tools"].items())
        call_counts = ", ".join(f"{name}:{count}" for name, count in summary["source_call_counts"].items())
        lines.append(f"| {split} | `{tools}` | `{call_counts}` |")

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
            "- This is the Phase2H candidate dataset after Phase2G showed no full tau2 gain from protocol-only v4.",
            "- Train Qwen2.5-7B QLoRA with this dataset, then run mixed-policy behavior eval before another full tau2 pass.",
            "- Do not call the phase successful unless full tau2 pass^1 improves, even if teacher-forced loss drops.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed dialogue/tool-policy SFT samples.")
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

    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = load_tokenizer(str(repo_path(args.tokenizer_model)), local_files_only=not args.no_local_files_only)

    splits: dict[str, list[dict[str, Any]]] = {}
    rejected_splits: dict[str, list[dict[str, Any]]] = {}
    validation_errors: list[str] = []
    for split, input_path in input_paths.items():
        rows = load_jsonl(input_path)
        raw_samples, raw_rejected = build_split(rows, split, include_protocol_variants=args.include_protocol_variants)
        clean, rejected, errors = validate_and_finalize(raw_samples, raw_rejected, tokenizer, args.max_sample_tokens)
        validation_errors.extend(errors)
        splits[split] = clean
        rejected_splits[split] = rejected
        write_jsonl(output_paths[split], clean)
        write_jsonl(rejected_paths[split], rejected)

    summaries = {split: summarize(samples) for split, samples in splits.items()}
    rejected_summaries = {split: summarize_rejections(samples) for split, samples in rejected_splits.items()}
    manifest = {
        "format_version": "tau2_airline_mixed_dialogue_tool_policy_manifest_v1",
        "inputs": {split: display_path(path) for split, path in input_paths.items()},
        "outputs": {split: display_path(path) for split, path in output_paths.items()},
        "rejected_outputs": {split: display_path(path) for split, path in rejected_paths.items()},
        "report": display_path(repo_path(args.report)),
        "tokenizer_model": args.tokenizer_model,
        "tokenizer_stats_enabled": tokenizer is not None,
        "max_sample_tokens": args.max_sample_tokens,
        "include_protocol_variants": args.include_protocol_variants,
        "summaries": summaries,
        "rejected_summaries": rejected_summaries,
        "validation_errors": validation_errors,
    }
    write_json(repo_path(args.manifest), manifest)
    write_report(repo_path(args.report), manifest)

    print("Mixed dialogue tool-policy dataset complete")
    print("==========================================")
    for split, summary in summaries.items():
        rejected = rejected_summaries[split]
        print(
            f"{split}: rows={summary['rows']} rejected={rejected['rows']} "
            f"actions={summary['target_actions']} max_tokens={summary['max_tokens']}"
        )
        if rejected["reasons"]:
            print("  rejected: " + ", ".join(f"{k}={v}" for k, v in rejected["reasons"].items()))
    print(f"validation_errors: {len(validation_errors)}")
    print(f"wrote report: {repo_path(args.report)}")
    print(f"wrote manifest: {repo_path(args.manifest)}")
    return 0 if not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
