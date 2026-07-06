from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_PHASE2H = "autodl_artifacts/phase2h_mixed_dialogue_tool_policy_4090_20260616_134216/behavior_summary.json"
DEFAULT_PHASE2I = "autodl_artifacts/phase2i_decision_balanced_5090d_20260621_232029/behavior_summary.json"
DEFAULT_REPORT = "reports/phase2j_gated_policy_upper_bound.md"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_rows(summary: dict[str, Any], prefer_non_base: bool) -> list[dict[str, Any]]:
    results = summary["results"]
    if prefer_non_base:
        for result in results:
            if result["name"] != "base":
                return result["rows"]
    return results[0]["rows"]


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        probe_id = str(row["probe_id"])
        base_id = probe_id.replace("_single_tool_protocol", "_single_tool")
        groups[base_id].append(row)

    deduped: list[dict[str, Any]] = []
    for group in groups.values():
        non_protocol = [row for row in group if not str(row["probe_id"]).endswith("_protocol")]
        deduped.append(non_protocol[0] if non_protocol else group[0])
    return deduped


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_rows = [row for row in rows if row["target_action"] == "assistant_text"]
    tool_rows = [row for row in rows if row["target_action"] == "tool_call"]
    total = len(rows)

    def rate(count: int, denom: int) -> float:
        return count / denom if denom else 0.0

    return {
        "n": total,
        "text": len(text_rows),
        "tool": len(tool_rows),
        "action_type_acc": rate(sum(bool(row["action_type_match"]) for row in rows), total),
        "text_no_tool": rate(sum(bool(row["text_probe_no_tool"]) for row in text_rows), len(text_rows)),
        "text_nonempty": rate(sum(bool(row["text_probe_nonempty"]) for row in text_rows), len(text_rows)),
        "tool_call_rate": rate(sum(bool(row["tool_probe_has_tool"]) for row in tool_rows), len(tool_rows)),
        "wrapper": rate(sum(bool(row["has_protocol_wrapper"]) for row in tool_rows), len(tool_rows)),
        "tool_name_acc": rate(sum(bool(row["tool_name_match"]) for row in tool_rows), len(tool_rows)),
        "arg_exact": rate(sum(bool(row["arg_exact_match"]) for row in tool_rows), len(tool_rows)),
    }


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["probe_id"]): row for row in rows}


def build_oracle_gate_rows(tool_rows: list[dict[str, Any]], text_source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_by_probe = row_map(text_source_rows)
    tool_by_probe = row_map(tool_rows)
    combined: list[dict[str, Any]] = []
    for probe_id in sorted(text_by_probe):
        source = text_by_probe[probe_id]
        target_action = source["target_action"]
        item = dict(source)
        if target_action == "assistant_text":
            item.update(
                {
                    "predicted_action": "assistant_text",
                    "action_type_match": True,
                    "text_probe_no_tool": True,
                    "text_probe_nonempty": True,
                    "tool_probe_has_tool": False,
                    "has_protocol_wrapper": False,
                    "tool_name_match": False,
                    "arg_exact_match": False,
                }
            )
        else:
            item.update(tool_by_probe[probe_id])
        combined.append(item)
    return combined


def build_phase2i_gate_phase2h_tool_rows(
    phase2i_rows: list[dict[str, Any]],
    phase2h_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    h_by_probe = row_map(phase2h_rows)
    combined: list[dict[str, Any]] = []
    for gate_row in phase2i_rows:
        probe_id = str(gate_row["probe_id"])
        item = dict(gate_row)
        if gate_row["target_action"] == "assistant_text":
            combined.append(item)
            continue

        if gate_row["predicted_action"] != "tool_call":
            item.update(
                {
                    "action_type_match": False,
                    "tool_probe_has_tool": False,
                    "has_protocol_wrapper": False,
                    "tool_name_match": False,
                    "arg_exact_match": False,
                }
            )
            combined.append(item)
            continue

        tool_row = h_by_probe[probe_id]
        item.update(
            {
                "predicted_action": tool_row["predicted_action"],
                "action_type_match": bool(tool_row["tool_probe_has_tool"]),
                "tool_probe_has_tool": tool_row["tool_probe_has_tool"],
                "has_protocol_wrapper": tool_row["has_protocol_wrapper"],
                "tool_name_match": tool_row["tool_name_match"],
                "arg_exact_match": tool_row["arg_exact_match"],
            }
        )
        combined.append(item)
    return combined


def fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def write_report(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["base", "phase2h", "phase2i", "phase2i_gate_phase2h_tool", "oracle_gate_phase2h_tool"]
    labels = {
        "base": "base",
        "phase2h": "Phase2H",
        "phase2i": "Phase2I",
        "phase2i_gate_phase2h_tool": "Phase2I gate + Phase2H tool",
        "oracle_gate_phase2h_tool": "oracle gate + Phase2H tool",
    }
    lines = [
        "# Phase2J Gated Policy Upper Bound",
        "",
        "## Goal",
        "",
        "Estimate whether separating action decision from tool generation is worth training.",
        "",
        "## Reported 64-Probe Metrics",
        "",
        "| Policy | Action type acc | Text no-tool | Tool-call rate | Wrapper | Tool-name acc | Arg exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in order:
        metric = payload[key]["reported"]
        lines.append(
            "| "
            + labels[key]
            + " | "
            + " | ".join(
                [
                    fmt(metric["action_type_acc"]),
                    fmt(metric["text_no_tool"]),
                    fmt(metric["tool_call_rate"]),
                    fmt(metric["wrapper"]),
                    fmt(metric["tool_name_acc"]),
                    fmt(metric["arg_exact"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Deduplicated Metrics",
            "",
            "| Policy | N | Action type acc | Text no-tool | Tool-call rate | Wrapper | Tool-name acc | Arg exact |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in order:
        metric = payload[key]["dedup"]
        lines.append(
            "| "
            + labels[key]
            + " | "
            + " | ".join(
                [
                    fmt(metric["n"]),
                    fmt(metric["action_type_acc"]),
                    fmt(metric["text_no_tool"]),
                    fmt(metric["tool_call_rate"]),
                    fmt(metric["wrapper"]),
                    fmt(metric["tool_name_acc"]),
                    fmt(metric["arg_exact"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Phase2H is still the stronger tool generator.",
            "- Phase2I is a better decision signal than Phase2H for text turns, but it suppresses too many true tool calls.",
            "- The oracle gate shows the target: keep Phase2H tool accuracy while making text no-tool perfect.",
            "- Next implementation should train a decision-only gate instead of another one-stage mixed SFT.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2h", default=DEFAULT_PHASE2H)
    parser.add_argument("--phase2i", default=DEFAULT_PHASE2I)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()

    phase2h_summary = load_summary(repo_path(args.phase2h))
    phase2i_summary = load_summary(repo_path(args.phase2i))

    base_rows = model_rows(phase2h_summary, prefer_non_base=False)
    phase2h_rows = model_rows(phase2h_summary, prefer_non_base=True)
    phase2i_rows = model_rows(phase2i_summary, prefer_non_base=True)
    phase2i_gate_h_tool_rows = build_phase2i_gate_phase2h_tool_rows(phase2i_rows, phase2h_rows)
    oracle_gate_h_tool_rows = build_oracle_gate_rows(phase2h_rows, base_rows)

    payload = {
        "base": {"reported": metrics(base_rows), "dedup": metrics(dedup_rows(base_rows))},
        "phase2h": {"reported": metrics(phase2h_rows), "dedup": metrics(dedup_rows(phase2h_rows))},
        "phase2i": {"reported": metrics(phase2i_rows), "dedup": metrics(dedup_rows(phase2i_rows))},
        "phase2i_gate_phase2h_tool": {
            "reported": metrics(phase2i_gate_h_tool_rows),
            "dedup": metrics(dedup_rows(phase2i_gate_h_tool_rows)),
        },
        "oracle_gate_phase2h_tool": {
            "reported": metrics(oracle_gate_h_tool_rows),
            "dedup": metrics(dedup_rows(oracle_gate_h_tool_rows)),
        },
    }

    report_path = repo_path(args.report)
    write_report(report_path, payload)
    print(f"wrote: {report_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
