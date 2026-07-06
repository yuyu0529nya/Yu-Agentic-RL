from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_sft_render_mask import render_message, tool_call_payload  # noqa: E402
from train_sft_smoke import load_jsonl, load_tokenizer  # noqa: E402


DEFAULT_MODEL = "models/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = "data/train/tau2_airline_sft_heldout.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/sft_behavior_eval_phase1h"
DEFAULT_REPORT = "reports/sft_behavior_eval_phase1h.md"


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, stop_ids: list[torch.Tensor], prompt_len: int):
        self.stop_ids = [ids for ids in stop_ids if ids.numel() > 0]
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if not self.stop_ids:
            return False
        generated = input_ids[0, self.prompt_len :]
        for ids in self.stop_ids:
            ids = ids.to(generated.device)
            if generated.numel() >= ids.numel() and torch.equal(generated[-ids.numel() :], ids):
                return True
        return False


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


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_prefix(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        segment, _spans = render_message(
            message,
            target=False,
            assistant_content_loss=False,
            assistant_tool_call_loss=False,
        )
        parts.append(segment)
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def reference_assistant_body(message: dict[str, Any]) -> str:
    segment, _spans = render_message(
        message,
        target=False,
        assistant_content_loss=False,
        assistant_tool_call_loss=False,
    )
    prefix = "<|im_start|>assistant\n"
    suffix = "\n<|im_end|>\n"
    if segment.startswith(prefix):
        segment = segment[len(prefix) :]
    if segment.endswith(suffix):
        segment = segment[: -len(suffix)]
    return segment


def normalize_call(call: dict[str, Any]) -> dict[str, Any]:
    payload = tool_call_payload(call)
    args = payload.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return {
        "name": payload.get("name") or "",
        "arguments": args if args is not None else {},
    }


def extract_reference_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    return [normalize_call(call) for call in message.get("tool_calls") or []]


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_generated_body(text: str) -> str:
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    return text.strip()


def truncate_at_stop(text: str, stop_sequences: list[str]) -> str:
    first_idx: int | None = None
    first_stop = ""
    for stop in stop_sequences:
        if not stop:
            continue
        idx = text.find(stop)
        if idx == -1:
            continue
        if first_idx is None or idx < first_idx:
            first_idx = idx
            first_stop = stop
    if first_idx is None:
        return text
    return text[: first_idx + len(first_stop)]


def parse_tool_json(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    arguments = parsed.get("arguments")
    if arguments is None:
        arguments = parsed.get("params")
    if name is None and isinstance(parsed.get("function"), dict):
        function = parsed["function"]
        name = function.get("name")
        arguments = function.get("arguments")
        if arguments is None:
            arguments = function.get("params")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return {
        "name": str(name or ""),
        "arguments": arguments if arguments is not None else {},
    }


def iter_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def extract_generated_calls(text: str) -> tuple[list[dict[str, Any]], bool]:
    body = extract_generated_body(text)
    calls: list[dict[str, Any]] = []
    json_parse_failed = False
    for match in TOOL_RE.finditer(body):
        parsed = parse_tool_json(match.group(1))
        if parsed is None:
            json_parse_failed = True
        else:
            calls.append(parsed)

    seen = {json_text(call) for call in calls}
    for candidate in iter_balanced_json_objects(body):
        if '"name"' not in candidate and '"function"' not in candidate:
            continue
        parsed = parse_tool_json(candidate)
        if parsed is None:
            json_parse_failed = True
            continue
        key = json_text(parsed)
        if key not in seen:
            calls.append(parsed)
            seen.add(key)
    return calls, json_parse_failed


def argument_key_set(call: dict[str, Any]) -> set[str]:
    args = call.get("arguments")
    if isinstance(args, dict):
        return set(str(key) for key in args)
    return set()


def argument_exact_match(ref: dict[str, Any], pred: dict[str, Any]) -> bool:
    return ref.get("arguments") == pred.get("arguments")


def build_probes(rows: list[dict[str, Any]], max_probes: int, include_later_turns: bool) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages") or []
        metadata = row.get("metadata") or {}
        for idx, message in enumerate(messages):
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            calls = extract_reference_calls(message)
            probes.append(
                {
                    "probe_id": f"{row.get('id')}::turn{idx}",
                    "sample_id": row.get("id"),
                    "task_id": str(metadata.get("task_id") or ""),
                    "turn_index": idx,
                    "prefix_messages": messages[:idx],
                    "reference_body": reference_assistant_body(message),
                    "reference_calls": calls,
                    "reference_tool_names": [call["name"] for call in calls],
                }
            )
            if not include_later_turns:
                break
            if len(probes) >= max_probes:
                return probes
    return probes[:max_probes]


def load_model(base_model_path: Path, adapter_path: Path | None, fp16: bool) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        torch_dtype=torch.float16 if fp16 else None,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_seq_len: int,
    max_new_tokens: int,
    stop_sequences: list[str],
) -> dict[str, Any]:
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    stop_ids = [
        torch.tensor(tokenizer(stop, add_special_tokens=False)["input_ids"], dtype=torch.long)
        for stop in stop_sequences
    ]
    stopping_criteria = None
    if stop_ids:
        stopping_criteria = StoppingCriteriaList([StopOnTokenSequences(stop_ids, prompt_len=input_ids.shape[1])])
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    new_ids = output_ids[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    text = truncate_at_stop(text, stop_sequences)
    body = extract_generated_body(text)
    calls, json_failed = extract_generated_calls(text)
    return {
        "prompt_tokens": int(input_ids.shape[1]),
        "new_tokens": int(new_ids.shape[0]),
        "generated_text": text,
        "generated_body": body,
        "generated_calls": calls,
        "has_protocol_wrapper": bool(TOOL_RE.search(body)),
        "json_parse_failed": json_failed,
    }


def score_generation(probe: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    ref_calls = probe["reference_calls"]
    pred_calls = generation["generated_calls"]
    ref_first = ref_calls[0] if ref_calls else {"name": "", "arguments": {}}
    pred_first = pred_calls[0] if pred_calls else {"name": "", "arguments": {}}
    ref_keys = argument_key_set(ref_first)
    pred_keys = argument_key_set(pred_first)
    tool_name_match = bool(pred_calls) and pred_first.get("name") == ref_first.get("name")
    return {
        "has_tool_call": bool(pred_calls),
        "tool_name_match": tool_name_match,
        "arg_key_recall": len(ref_keys & pred_keys) / len(ref_keys) if ref_keys else 1.0,
        "arg_key_precision": len(ref_keys & pred_keys) / len(pred_keys) if pred_keys else (1.0 if not ref_keys else 0.0),
        "arg_exact_match": tool_name_match and argument_exact_match(ref_first, pred_first),
        "has_protocol_wrapper": generation["has_protocol_wrapper"],
        "json_parse_failed": generation["json_parse_failed"],
    }


def evaluate_model(
    model_name: str,
    base_model_path: Path,
    adapter_path: Path | None,
    tokenizer: Any,
    probes: list[dict[str, Any]],
    device: torch.device,
    max_seq_len: int,
    max_new_tokens: int,
    stop_sequences: list[str],
    fp16: bool,
) -> dict[str, Any]:
    started = time.time()
    model = load_model(base_model_path, adapter_path, fp16=fp16)
    model.to(device)
    model.eval()
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

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
        score = score_generation(probe, generation)
        row = {
            "probe_id": probe["probe_id"],
            "sample_id": probe["sample_id"],
            "task_id": probe["task_id"],
            "turn_index": probe["turn_index"],
            "reference_tool_names": probe["reference_tool_names"],
            "reference_calls": probe["reference_calls"],
            **generation,
            **score,
        }
        rows.append(row)
        print(
            f"{model_name} {idx}/{len(probes)} "
            f"tool_match={score['tool_name_match']} "
            f"ref={probe['reference_tool_names']} "
            f"pred={[call.get('name') for call in generation['generated_calls']]} "
            f"probe={probe['probe_id']}"
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_tool = Counter()
    call_counts: list[int] = []
    for row in rows:
        ref = row["reference_tool_names"][0] if row["reference_tool_names"] else ""
        by_tool[ref] += 1
        call_counts.append(len(row.get("generated_calls") or []))
    single_clean_tool = 0
    single_clean_exact = 0
    for row in rows:
        pred_calls = row.get("generated_calls") or []
        if len(pred_calls) != 1:
            continue
        if row["tool_name_match"]:
            single_clean_tool += 1
        if row["arg_exact_match"]:
            single_clean_exact += 1
    return {
        "probes": len(rows),
        "has_tool_call_rate": sum(1 for row in rows if row["has_tool_call"]) / len(rows),
        "protocol_wrapper_rate": sum(1 for row in rows if row["has_protocol_wrapper"]) / len(rows),
        "single_call_rate": sum(1 for count in call_counts if count == 1) / len(rows),
        "multi_call_rate": sum(1 for count in call_counts if count > 1) / len(rows),
        "mean_generated_calls": sum(call_counts) / len(rows),
        "max_generated_calls": max(call_counts) if call_counts else 0,
        "tool_name_accuracy": sum(1 for row in rows if row["tool_name_match"]) / len(rows),
        "arg_exact_accuracy": sum(1 for row in rows if row["arg_exact_match"]) / len(rows),
        "clean_single_tool_accuracy": single_clean_tool / len(rows),
        "clean_single_arg_exact_accuracy": single_clean_exact / len(rows),
        "mean_arg_key_recall": sum(float(row["arg_key_recall"]) for row in rows) / len(rows),
        "mean_arg_key_precision": sum(float(row["arg_key_precision"]) for row in rows) / len(rows),
        "json_parse_failures": sum(1 for row in rows if row["json_parse_failed"]),
        "reference_tool_counts": dict(sorted(by_tool.items())),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_generated(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1H SFT Behavior Evaluation",
        "",
        "## Goal",
        "",
        "Compare base and SFT adapters on heldout next-tool-call behavior, using deterministic generation from the same decision-point prompts.",
        "",
        "## Key Findings",
        "",
    ]
    if payload["results"]:
        base_summary = payload["results"][0]["summary"]
        for result in payload["results"][1:]:
            summary = result["summary"]
            lines.append(
                f"- `{result['name']}`: tool-name accuracy "
                f"{base_summary['tool_name_accuracy']:.3f} -> {summary['tool_name_accuracy']:.3f}; "
                f"exact call accuracy {base_summary['arg_exact_accuracy']:.3f} -> {summary['arg_exact_accuracy']:.3f}; "
                f"single-call rate {base_summary.get('single_call_rate', 0.0):.3f} -> {summary.get('single_call_rate', 0.0):.3f}."
            )
        best = max(
            payload["results"],
            key=lambda result: result["summary"]["tool_name_accuracy"],
        )
        lines.append(f"- Best free-form behavior adapter: `{best['name']}`.")
        lines.append("- Free-form generation is still weak; this motivates action-prefix SFT or constrained tool decoding.")
    lines.extend(
        [
        "",
        "## Setup",
        "",
        f"- Base model: `{payload['base_model']}`",
        f"- Data: `{payload['data']}`",
        f"- Probes: `{len(payload['probes'])}`",
        f"- Max sequence length: `{payload['max_seq_len']}`",
        f"- Max new tokens: `{payload['max_new_tokens']}`",
        f"- Stop sequences: `{payload['stop_sequences'] or []}`",
        f"- Device: `{payload['device']}`",
        "",
        "## Summary",
        "",
        "| Model | Tool-call rate | Protocol wrapper rate | Single-call rate | Multi-call rate | Tool-name acc | Arg exact acc | Clean single exact | Mean calls | JSON failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in payload["results"]:
        summary = result["summary"]
        lines.append(
            f"| `{result['name']}` | {summary['has_tool_call_rate']:.3f} | "
            f"{summary['protocol_wrapper_rate']:.3f} | "
            f"{summary.get('single_call_rate', 0.0):.3f} | "
            f"{summary.get('multi_call_rate', 0.0):.3f} | "
            f"{summary['tool_name_accuracy']:.3f} | {summary['arg_exact_accuracy']:.3f} | "
            f"{summary.get('clean_single_arg_exact_accuracy', 0.0):.3f} | "
            f"{summary.get('mean_generated_calls', 0.0):.2f} | "
            f"{summary['json_parse_failures']} |"
        )

    if len(payload["results"]) > 1:
        base = payload["results"][0]
        lines.extend(["", "## Delta Vs Base", "", "| Model | Tool-name acc delta | Arg exact delta | Single-call delta | Clean single exact delta |", "| --- | ---: | ---: | ---: | ---: |"])
        base_summary = base["summary"]
        for result in payload["results"][1:]:
            summary = result["summary"]
            lines.append(
                f"| `{result['name']}` | "
                f"{summary['tool_name_accuracy'] - base_summary['tool_name_accuracy']:+.3f} | "
                f"{summary['arg_exact_accuracy'] - base_summary['arg_exact_accuracy']:+.3f} | "
                f"{summary.get('single_call_rate', 0.0) - base_summary.get('single_call_rate', 0.0):+.3f} | "
                f"{summary.get('clean_single_arg_exact_accuracy', 0.0) - base_summary.get('clean_single_arg_exact_accuracy', 0.0):+.3f} |"
            )

    lines.extend(["", "## Per-Probe Results", ""])
    for result in payload["results"]:
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "| Probe | Task | Ref tool | Pred tool | # Calls | Wrapped | Tool ok | Arg exact | Generated preview |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in result["rows"]:
            ref_tool = ",".join(row["reference_tool_names"])
            pred_tool = ",".join(call.get("name", "") for call in row["generated_calls"]) or "-"
            lines.append(
                f"| `{row['probe_id']}` | {row['task_id']} | `{ref_tool}` | `{pred_tool}` | "
                f"{len(row['generated_calls'])} | "
                f"{'Y' if row['has_protocol_wrapper'] else 'N'} | "
                f"{'Y' if row['tool_name_match'] else 'N'} | {'Y' if row['arg_exact_match'] else 'N'} | "
                f"`{compact_generated(row['generated_body'])}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- This is a behavior-level proxy focused on next tool-call generation, not full tau2 pass rate.",
            "- A weak adapter can lower teacher-forced NLL without improving free-form tool-call generation.",
            "- If tool-name accuracy improves but protocol wrapper rate stays low, the model is learning JSON content without learning executable tool-call format.",
            "- If tool-name accuracy improves but single-call rate stays low, the model is learning the right action family without learning the runtime stop boundary.",
            "- If generation metrics are poor, the next step is to train on shorter action-decision prefixes or evaluate with constrained decoding.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate next-tool-call behavior for base and SFT adapters.")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--max-probes", type=int, default=12)
    parser.add_argument("--include-later-turns", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--stop-sequence", action="append", default=[])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_model = repo_path(args.base_model)
    data_path = repo_path(args.data)
    output_dir = repo_path(args.output_dir)
    report_path = repo_path(args.report)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = parse_adapter_specs(args.adapter)
    rows = load_jsonl(data_path)
    probes = build_probes(rows, max_probes=args.max_probes, include_later_turns=args.include_later_turns)
    if not probes:
        raise RuntimeError("No tool-call probes found.")

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
        write_json(output_dir / f"{name}_behavior.json", result)

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
