from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_sft_render_mask import render_with_spans, validate_structure  # noqa: E402


DEFAULT_TRAIN = "data/train/tau2_airline_sft_train.jsonl"
DEFAULT_VALID = "data/train/tau2_airline_sft_valid.jsonl"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PRETRAINED_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_smoke"
DEFAULT_REPORT = "reports/sft_smoke_v1.md"
DEFAULT_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class EncodedExample:
    sample_id: str
    task_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    token_count: int
    target_token_count: int
    truncated: bool
    warnings: list[str]


class SftSmokeDataset(Dataset):
    def __init__(self, examples: list[EncodedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.examples[idx]


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


def parse_sample_ids(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def cached_snapshot_path(model_name: str) -> str:
    explicit = Path(model_name)
    if explicit.exists():
        return str(explicit)
    if "/" not in model_name:
        return model_name

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + model_name.replace("/", "--")
    )
    snapshots = cache_root / "snapshots"
    if not snapshots.exists():
        return model_name

    ref_path = cache_root / "refs" / "main"
    if ref_path.exists():
        revision = ref_path.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if candidate.exists():
            return str(candidate)

    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    if len(candidates) == 1:
        return str(candidates[0])
    if candidates:
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        return str(newest)
    return model_name


def overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    start_a, end_a = span_a
    start_b, end_b = span_b
    return start_a < end_b and end_a > start_b


def encode_row(
    row: dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
) -> EncodedExample:
    errors, warnings = validate_structure(row)
    if errors:
        raise ValueError(f"{row.get('id')} structure errors: {errors}")

    render = render_with_spans(row)
    encoded = tokenizer(
        render.text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    offsets = list(encoded["offset_mapping"])

    labels = [-100] * len(input_ids)
    target_tokens = 0
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if any(overlap((start, end), span) for span in render.target_spans):
            labels[idx] = input_ids[idx]
            target_tokens += 1

    if target_tokens == 0:
        raise ValueError(f"{row.get('id')} has no target tokens after truncation")

    metadata = row.get("metadata") or {}
    return EncodedExample(
        sample_id=str(row.get("id")),
        task_id=str(metadata.get("task_id") or row.get("task_id") or ""),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        token_count=len(input_ids),
        target_token_count=target_tokens,
        truncated=len(input_ids) >= max_seq_len,
        warnings=warnings,
    )


def select_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
    max_samples: int,
    seed: int,
    prefer_shortest: bool,
    sample_ids: set[str] | None = None,
) -> list[EncodedExample]:
    if sample_ids is not None:
        rows = [row for row in rows if str(row.get("id")) in sample_ids]
        missing = sample_ids - {str(row.get("id")) for row in rows}
        if missing:
            raise ValueError("Missing requested sample ids: " + ", ".join(sorted(missing)))

    encoded: list[EncodedExample] = []
    skipped: list[str] = []
    for row in rows:
        try:
            encoded.append(encode_row(row, tokenizer, max_seq_len))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{row.get('id')}: {type(exc).__name__}: {exc}")

    if not encoded:
        raise RuntimeError("No encodable examples. First skipped rows: " + "; ".join(skipped[:3]))

    if prefer_shortest:
        encoded.sort(key=lambda item: (item.token_count, item.sample_id))
    else:
        random.Random(seed).shuffle(encoded)
    return encoded[:max_samples]


def collate_batch(examples: list[EncodedExample], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(example.input_ids) for example in examples)
    input_ids = []
    attention_mask = []
    labels = []
    for example in examples:
        pad = max_len - len(example.input_ids)
        input_ids.append(example.input_ids + [pad_token_id] * pad)
        attention_mask.append(example.attention_mask + [0] * pad)
        labels.append(example.labels + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": [example.sample_id for example in examples],
    }


def load_tokenizer(model_name: str, local_files_only: bool) -> Any:
    source = cached_snapshot_path(model_name) if local_files_only else model_name
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_tiny_model(tokenizer: Any, max_seq_len: int, args: argparse.Namespace) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=max_seq_len,
        n_ctx=max_seq_len,
        n_embd=args.tiny_hidden_size,
        n_layer=args.tiny_layers,
        n_head=args.tiny_heads,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GPT2LMHeadModel(config)


def load_model(tokenizer: Any, args: argparse.Namespace) -> torch.nn.Module:
    if args.model_init == "tiny-random":
        return build_tiny_model(tokenizer, args.max_seq_len, args)
    source = cached_snapshot_path(args.pretrained_model) if args.local_files_only else args.pretrained_model

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("4-bit loading requires a transformers build with BitsAndBytesConfig.") from exc
        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.bnb_4bit_compute_dtype]
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        if torch.cuda.is_available() and not args.cpu:
            model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["torch_dtype"] = torch.float16 if args.fp16 else None

    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.load_in_4bit and args.lora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.lora:
        target_modules = [
            item.strip()
            for item in args.lora_target_modules.split(",")
            if item.strip()
        ]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            model_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            outputs = model(**model_inputs)
            loss = float(outputs.loss.detach().cpu())
            if math.isfinite(loss):
                losses.append(loss)
    model.train()
    return mean(losses) if losses else float("nan")


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    result: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        result.append(mean(values[start : idx + 1]))
    return result


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.load_in_4bit:
        model.to(device)
    model.train()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )

    valid_before = evaluate(model, valid_loader, device, args.eval_batches)
    train_losses: list[float] = []
    step_trace: list[dict[str, Any]] = []
    eval_trace: list[dict[str, Any]] = [
        {"step": 0, "valid_loss": valid_before},
    ]
    started = time.time()
    step = 0

    while step < args.steps:
        for batch in train_loader:
            step += 1
            model_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            outputs = model(**model_inputs)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            step_trace.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "sample_ids": batch["sample_ids"],
                    "lr": scheduler.get_last_lr()[0],
                }
            )
            if args.eval_every > 0 and step % args.eval_every == 0:
                eval_loss = evaluate(model, valid_loader, device, args.eval_batches)
                eval_trace.append({"step": step, "valid_loss": eval_loss})
            if step % max(1, args.log_every) == 0 or step == 1 or step == args.steps:
                print(
                    f"step {step}/{args.steps} loss={loss_value:.4f} "
                    f"samples={batch['sample_ids']}"
                )
            if step >= args.steps:
                break

    valid_after = evaluate(model, valid_loader, device, args.eval_batches)
    if not eval_trace or eval_trace[-1]["step"] != step:
        eval_trace.append({"step": step, "valid_loss": valid_after})
    elapsed = time.time() - started
    max_memory_mb = None
    if device.type == "cuda":
        max_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    return {
        "train_losses": train_losses,
        "train_loss_min": min(train_losses) if train_losses else float("nan"),
        "train_loss_max": max(train_losses) if train_losses else float("nan"),
        "train_loss_rolling_mean": rolling_mean(train_losses, args.rolling_window),
        "step_trace": step_trace,
        "eval_trace": eval_trace,
        "valid_loss_before": valid_before,
        "valid_loss_after": valid_after,
        "elapsed_seconds": elapsed,
        "max_memory_mb": max_memory_mb,
    }


def summarize_examples(examples: list[EncodedExample]) -> dict[str, Any]:
    return {
        "rows": len(examples),
        "tasks": sorted({example.task_id for example in examples}),
        "mean_tokens": mean([example.token_count for example in examples]) if examples else 0,
        "max_tokens": max([example.token_count for example in examples], default=0),
        "mean_target_tokens": mean([example.target_token_count for example in examples]) if examples else 0,
        "max_target_tokens": max([example.target_token_count for example in examples], default=0),
        "truncated": sum(1 for example in examples if example.truncated),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_losses = payload["train"]["train_losses"]
    first_loss = train_losses[0] if train_losses else float("nan")
    final_loss = train_losses[-1] if train_losses else float("nan")
    is_gold_test = (
        payload.get("train_sample_ids")
        and payload.get("valid_sample_ids")
        and payload.get("train_sample_ids") == payload.get("valid_sample_ids")
        and not payload.get("shuffle")
    )
    is_small_scale = (
        payload.get("model_init") == "pretrained"
        and not payload.get("train_sample_ids")
        and payload.get("data", {}).get("train", {}).get("rows", 0) > 1
    )
    if is_gold_test:
        title = "SFT Mask Overfit Gold Test v1"
    elif is_small_scale:
        title = "SFT Small-Scale Train v1"
    else:
        title = "SFT Smoke Test v1"
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        f"- Status: `{'OK' if payload['ok'] else 'FAILED'}`",
        f"- Model init: `{payload['model_init']}`",
        f"- Tokenizer: `{payload['tokenizer_model']}`",
        f"- Device: `{payload['device']}`",
        f"- Max sequence length: `{payload['max_seq_len']}`",
        f"- Train sample ids: `{payload.get('train_sample_ids')}`",
        f"- Valid sample ids: `{payload.get('valid_sample_ids')}`",
        f"- Shuffle: `{payload.get('shuffle')}`",
        f"- Total parameters: `{payload['param_count']}`",
        f"- Trainable parameters: `{payload['trainable_param_count']}`",
        f"- Output dir: `{payload['output_dir']}`",
    ]
    if payload.get("lora", {}).get("enabled"):
        lora = payload["lora"]
        lines.extend(
            [
                f"- LoRA: `r={lora['r']}, alpha={lora['alpha']}, dropout={lora['dropout']}`",
                f"- LoRA targets: `{', '.join(lora['target_modules'])}`",
            ]
        )
    quant = payload.get("quantization") or {}
    if quant.get("load_in_4bit"):
        lines.extend(
            [
                f"- Quantization: `4bit {quant['quant_type']}, compute={quant['compute_dtype']}, double_quant={quant['use_double_quant']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Data",
            "",
            "| Split | Rows | Tasks | Mean tokens | Max tokens | Mean target tokens | Truncated |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ("train", "valid"):
        summary = payload["data"][split]
        tasks = ", ".join(summary["tasks"])
        lines.append(
            f"| {split} | {summary['rows']} | `{tasks}` | "
            f"{summary['mean_tokens']:.1f} | {summary['max_tokens']} | "
            f"{summary['mean_target_tokens']:.1f} | {summary['truncated']} |"
        )

    notes = [
        "- This smoke test validates tokenizer rendering, token-level assistant-only labels, CUDA execution, optimizer steps, and checkpoint saving.",
    ]
    if payload["model_init"] == "tiny-random":
        notes.extend(
            [
                "- The `tiny-random` model is intentionally small and randomly initialized; it is not a quality result.",
                "- Use `--model-init pretrained --pretrained-model Qwen/Qwen2.5-0.5B-Instruct` for the next realism check.",
            ]
        )
    else:
        notes.extend(
            [
                "- This run uses real pretrained model weights, not a random toy model.",
                "- The current short-context run is still a smoke test; longer trajectories need a larger `max_seq_len` and likely rented GPU.",
            ]
        )

    lines.extend(
        [
            "",
            "## Training",
            "",
            f"- Steps: `{payload['steps']}`",
            f"- First train loss: `{first_loss:.4f}`",
            f"- Final train loss: `{final_loss:.4f}`",
            f"- Min train loss: `{payload['train']['train_loss_min']:.4f}`",
            f"- Max train loss: `{payload['train']['train_loss_max']:.4f}`",
            f"- Final rolling mean: `{payload['train']['train_loss_rolling_mean'][-1]:.4f}`",
            f"- Valid loss before: `{payload['train']['valid_loss_before']:.4f}`",
            f"- Valid loss after: `{payload['train']['valid_loss_after']:.4f}`",
            f"- Elapsed seconds: `{payload['train']['elapsed_seconds']:.1f}`",
            f"- Max CUDA memory MB: `{payload['train']['max_memory_mb']}`",
        ]
    )
    if is_gold_test:
        reduction = payload["train"]["valid_loss_before"] / max(
            payload["train"]["valid_loss_after"],
            1e-12,
        )
        passed = (
            payload["train"]["valid_loss_after"] < 1e-3
            and payload["train"]["train_loss_rolling_mean"][-1] < 1e-3
        )
        lines.extend(
            [
                f"- Gold test verdict: `{'PASS' if passed else 'CHECK'}`",
                f"- Valid loss reduction: `{reduction:.1f}x`",
            ]
        )
    lines.extend(
        [
            "",
            "## Eval Trace",
            "",
            "| Step | Valid loss |",
            "| ---: | ---: |",
            *[
                f"| {item['step']} | {item['valid_loss']:.4f} |"
                for item in payload["train"].get("eval_trace", [])
            ],
            "",
            "## Loss Trace",
            "",
            "```text",
            "\n".join(f"{idx + 1}: {loss:.4f}" for idx, loss in enumerate(train_losses)),
            "```",
            "",
            "## Notes",
            "",
            *notes,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small SFT smoke test with assistant-only token labels.")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER)
    parser.add_argument("--pretrained-model", default=DEFAULT_PRETRAINED_MODEL)
    parser.add_argument("--model-init", choices=["tiny-random", "pretrained"], default="tiny-random")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="right")
    parser.add_argument("--max-train-samples", type=int, default=4)
    parser.add_argument("--max-valid-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefer-shortest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sample-id")
    parser.add_argument("--valid-sample-id")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bnb-4bit-quant-type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--bnb-4bit-use-double-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tiny-hidden-size", type=int, default=64)
    parser.add_argument("--tiny-layers", type=int, default=2)
    parser.add_argument("--tiny-heads", type=int, default=4)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_path = repo_path(args.train)
    valid_path = repo_path(args.valid)
    output_dir = repo_path(args.output_dir)
    report_path = repo_path(args.report)
    train_sample_ids = parse_sample_ids(args.train_sample_id)
    valid_sample_ids = parse_sample_ids(args.valid_sample_id)

    tokenizer = load_tokenizer(args.tokenizer_model, args.local_files_only)
    tokenizer.truncation_side = args.truncation_side
    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    train_examples = select_rows(
        train_rows,
        tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_train_samples,
        seed=args.seed,
        prefer_shortest=args.prefer_shortest,
        sample_ids=train_sample_ids,
    )
    valid_examples = select_rows(
        valid_rows,
        tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_valid_samples,
        seed=args.seed,
        prefer_shortest=args.prefer_shortest,
        sample_ids=valid_sample_ids,
    )

    pad_token_id = int(tokenizer.pad_token_id)
    train_loader = DataLoader(
        SftSmokeDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )
    valid_loader = DataLoader(
        SftSmokeDataset(valid_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = load_model(tokenizer, args)
    param_count = sum(param.numel() for param in model.parameters())
    trainable_param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"model_init={args.model_init} params={param_count:,} device={device}")
    print(f"trainable_params={trainable_param_count:,} lora={args.lora if args.model_init == 'pretrained' else False}")
    print(f"train_examples={len(train_examples)} valid_examples={len(valid_examples)} max_seq_len={args.max_seq_len}")

    train_result = train(model, train_loader, valid_loader, device, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoint:
        model.save_pretrained(output_dir / "checkpoint")
        tokenizer.save_pretrained(output_dir / "checkpoint")

    payload = {
        "ok": True,
        "model_init": args.model_init,
        "pretrained_model": args.pretrained_model if args.model_init == "pretrained" else None,
        "tokenizer_model": args.tokenizer_model,
        "device": str(device),
        "max_seq_len": args.max_seq_len,
        "steps": args.steps,
        "train_sample_ids": sorted(train_sample_ids) if train_sample_ids else None,
        "valid_sample_ids": sorted(valid_sample_ids) if valid_sample_ids else None,
        "shuffle": args.shuffle,
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "lora": {
            "enabled": bool(args.lora and args.model_init == "pretrained"),
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [
                item.strip()
                for item in args.lora_target_modules.split(",")
                if item.strip()
            ],
        },
        "quantization": {
            "load_in_4bit": bool(args.load_in_4bit),
            "quant_type": args.bnb_4bit_quant_type,
            "compute_dtype": args.bnb_4bit_compute_dtype,
            "use_double_quant": args.bnb_4bit_use_double_quant,
        },
        "output_dir": display_path(output_dir),
        "data": {
            "train": summarize_examples(train_examples),
            "valid": summarize_examples(valid_examples),
        },
        "train": train_result,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    trace_lines = ["step,loss,rolling_mean,lr,sample_ids"]
    rolling = train_result["train_loss_rolling_mean"]
    for item, roll in zip(train_result["step_trace"], rolling):
        sample_ids = "|".join(item["sample_ids"])
        trace_lines.append(
            f"{item['step']},{item['loss']:.8f},{roll:.8f},{item['lr']:.12g},{sample_ids}"
        )
    (output_dir / "loss_trace.csv").write_text(
        "\n".join(trace_lines) + "\n",
        encoding="utf-8",
    )
    write_report(report_path, payload)

    print("SFT smoke test complete")
    print(f"first_loss={train_result['train_losses'][0]:.4f}")
    print(f"final_loss={train_result['train_losses'][-1]:.4f}")
    print(f"valid_before={train_result['valid_loss_before']:.4f}")
    print(f"valid_after={train_result['valid_loss_after']:.4f}")
    print(f"wrote: {report_path}")
    print(f"wrote: {output_dir / 'metrics.json'}")
    print(f"wrote: {output_dir / 'loss_trace.csv'}")
    if args.save_checkpoint:
        print(f"wrote: {output_dir / 'checkpoint'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
