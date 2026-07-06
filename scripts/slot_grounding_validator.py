from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from summarize_tau2_results import load_results


WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}

USER_ID_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z][a-z0-9]*_\d+\b")
PAYMENT_ID_RE = re.compile(r"\b(?:credit_card|gift_card|certificate)_\d+\b")
FLIGHT_NUMBER_RE = re.compile(r"\bHAT\d{3}\b")
RESERVATION_ID_RE = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]{6}\b")


@dataclass(frozen=True)
class SlotEvidence:
    slot_type: str
    value: str
    source: str
    message_index: int
    detail: str


@dataclass(frozen=True)
class SlotIssue:
    component_name: str
    slot_type: str
    value: str
    tool_name: str
    message_index: int
    call_id: str
    severity: float
    detail: str
    tags: list[str]


@dataclass(frozen=True)
class SlotValidationResult:
    simulation_id: str
    task_id: str
    trial: int | None
    checked_tool_calls: int
    issue_count: int
    issues: list[SlotIssue]
    known_slot_counts: dict[str, int]


class SlotState:
    def __init__(self) -> None:
        self.evidence: dict[str, dict[str, SlotEvidence]] = defaultdict(dict)
        self.ungrounded_values: set[tuple[str, str]] = set()
        self.not_found_values: set[tuple[str, str]] = set()

    def add(self, slot_type: str, value: Any, source: str, message_index: int, detail: str) -> None:
        normalized = normalize_slot_value(value)
        if not normalized:
            return
        if slot_type == "reservation_id" and normalized.startswith("HAT"):
            return
        self.evidence[slot_type].setdefault(
            normalized,
            SlotEvidence(slot_type, normalized, source, message_index, detail),
        )

    def has(self, slot_type: str, value: Any) -> bool:
        normalized = normalize_slot_value(value)
        return bool(normalized and normalized in self.evidence.get(slot_type, {}))

    def remember_ungrounded(self, slot_type: str, value: Any) -> None:
        normalized = normalize_slot_value(value)
        if normalized:
            self.ungrounded_values.add((slot_type, normalized))

    def remember_not_found(self, slot_type: str, value: Any) -> None:
        normalized = normalize_slot_value(value)
        if normalized:
            self.not_found_values.add((slot_type, normalized))

    def counts(self) -> dict[str, int]:
        return {slot_type: len(values) for slot_type, values in sorted(self.evidence.items())}


def normalize_slot_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def text_blob(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def call_name_and_args(call: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(call.get("name") or function.get("name") or "")
    arguments = call.get("arguments")
    if arguments is None:
        arguments = function.get("arguments")
    arguments = parse_json_maybe(arguments)
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments, str(call.get("id") or "")


def add_slots_from_text(state: SlotState, text: str, source: str, message_index: int) -> None:
    for value in USER_ID_RE.findall(text or ""):
        state.add("user_id", value, source, message_index, "visible user id text")
    for value in PAYMENT_ID_RE.findall(text or ""):
        state.add("payment_id", value, source, message_index, "visible payment id text")
    for value in FLIGHT_NUMBER_RE.findall(text or ""):
        state.add("flight_number", value, source, message_index, "visible flight number text")
    for value in RESERVATION_ID_RE.findall(text or ""):
        if not value.startswith("HAT"):
            state.add("reservation_id", value, source, message_index, "visible reservation id text")


def iter_nested_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(iter_nested_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(iter_nested_dicts(child))
    return found


def add_slots_from_tool_result(
    state: SlotState,
    name: str,
    arguments: dict[str, Any],
    result: Any,
    message_index: int,
) -> None:
    if name == "get_user_details" and isinstance(result, dict):
        state.add("user_id", result.get("user_id"), "tool:get_user_details", message_index, "user profile")
        for reservation_id in result.get("reservations") or []:
            state.add(
                "reservation_id",
                reservation_id,
                "tool:get_user_details",
                message_index,
                "reservation listed in user profile",
            )
        payment_methods = result.get("payment_methods") or {}
        if isinstance(payment_methods, dict):
            for payment_id, method in payment_methods.items():
                state.add(
                    "payment_id",
                    payment_id,
                    "tool:get_user_details",
                    message_index,
                    "payment method in user profile",
                )
                if isinstance(method, dict):
                    state.add(
                        "payment_id",
                        method.get("id"),
                        "tool:get_user_details",
                        message_index,
                        "payment method id in user profile",
                    )

    if name == "get_reservation_details" and isinstance(result, dict):
        state.add(
            "reservation_id",
            result.get("reservation_id"),
            "tool:get_reservation_details",
            message_index,
            "reservation details",
        )
        state.add("user_id", result.get("user_id"), "tool:get_reservation_details", message_index, "reservation owner")
        for flight in result.get("flights") or []:
            if isinstance(flight, dict):
                state.add(
                    "flight_number",
                    flight.get("flight_number"),
                    "tool:get_reservation_details",
                    message_index,
                    "flight in reservation",
                )
        for payment in result.get("payment_history") or []:
            if isinstance(payment, dict):
                state.add(
                    "payment_id",
                    payment.get("payment_id"),
                    "tool:get_reservation_details",
                    message_index,
                    "payment history",
                )

    if name in {"search_direct_flight", "search_onestop_flight"}:
        for item in iter_nested_dicts(result):
            state.add(
                "flight_number",
                item.get("flight_number"),
                f"tool:{name}",
                message_index,
                "flight search result",
            )

    for slot_type, value in extract_argument_slots(arguments):
        if slot_type in {"user_id", "reservation_id", "payment_id", "flight_number"}:
            state.add(slot_type, value, f"tool_arg:{name}", message_index, "successful tool argument")


def extract_argument_slots(value: Any) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in {"user_id", "reservation_id", "payment_id", "flight_number"}:
                normalized = normalize_slot_value(child)
                if normalized:
                    slots.append((key_text, normalized))
            else:
                slots.extend(extract_argument_slots(child))
    elif isinstance(value, list):
        for child in value:
            slots.extend(extract_argument_slots(child))
    return slots


def slot_types_for_tool(name: str) -> set[str]:
    if name == "get_user_details":
        return {"user_id"}
    if name in {"get_reservation_details", "cancel_reservation"}:
        return {"reservation_id"}
    if name == "update_reservation_flights":
        return {"reservation_id", "payment_id"}
    if name in {"update_reservation_baggages", "update_reservation_passengers"}:
        return {"reservation_id", "payment_id"}
    if name == "book_reservation":
        return {"user_id", "payment_id"}
    if name == "send_certificate":
        return {"user_id"}
    return set()


def asks_for_identifier(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        phrase in lowered
        for phrase in (
            "provide your user id",
            "provide your user_id",
            "could you please provide your user id",
            "send your reservation id",
            "provide your reservation id",
            "double-check your user id",
        )
    )


def issue(
    component_name: str,
    slot_type: str,
    value: str,
    tool_name: str,
    message_index: int,
    call_id: str,
    severity: float,
    detail: str,
    extra_tags: list[str] | None = None,
) -> SlotIssue:
    tags = ["slot_grounding_error", component_name]
    if extra_tags:
        tags.extend(extra_tags)
    return SlotIssue(
        component_name=component_name,
        slot_type=slot_type,
        value=value,
        tool_name=tool_name,
        message_index=message_index,
        call_id=call_id,
        severity=severity,
        detail=detail,
        tags=sorted(set(tags)),
    )


def validate_call(
    state: SlotState,
    name: str,
    arguments: dict[str, Any],
    call_id: str,
    message_index: int,
    assistant_text: str,
) -> list[SlotIssue]:
    issues: list[SlotIssue] = []
    required_types = slot_types_for_tool(name)
    if required_types:
        for slot_type, value in extract_argument_slots(arguments):
            if slot_type not in required_types:
                continue
            if state.has(slot_type, value):
                continue
            component = f"ungrounded_{slot_type}"
            severity = 3.0 if slot_type == "user_id" else 2.0
            if slot_type in {"payment_id", "flight_number"}:
                severity = 1.5
            issues.append(
                issue(
                    component,
                    slot_type,
                    value,
                    name,
                    message_index,
                    call_id,
                    severity,
                    (
                        f"{name} used {slot_type}={value} before that value appeared "
                        "in prior user messages or successful tool results."
                    ),
                    ["argument_hallucination"],
                )
            )
            state.remember_ungrounded(slot_type, value)

    if name == "get_user_details" and asks_for_identifier(assistant_text):
        user_id = dict(extract_argument_slots(arguments)).get("user_id", "")
        issues.append(
            issue(
                "ask_identifier_and_call_tool_same_turn",
                "user_id",
                user_id,
                name,
                message_index,
                call_id,
                2.0,
                "Assistant asked the user for an identifier while also calling get_user_details in the same turn.",
                ["ask_then_tool_same_turn"],
            )
        )

    if name == "transfer_to_human_agents" and state.not_found_values:
        matching = state.not_found_values & state.ungrounded_values
        if matching:
            slot_type, value = sorted(matching)[0]
            issues.append(
                issue(
                    "transfer_after_ungrounded_slot_error",
                    slot_type,
                    value,
                    name,
                    message_index,
                    call_id,
                    2.5,
                    (
                        "Agent transferred after a not-found error caused by an earlier "
                        f"ungrounded {slot_type}={value}."
                    ),
                    ["premature_deferral"],
                )
            )
    return issues


def validate_simulation_slots(sim: dict[str, Any]) -> SlotValidationResult:
    state = SlotState()
    issues: list[SlotIssue] = []
    checked_tool_calls = 0
    call_by_id: dict[str, tuple[str, dict[str, Any]]] = {}

    for message_index, message in enumerate(sim.get("messages") or []):
        role = message.get("role")
        if role == "user":
            add_slots_from_text(state, str(message.get("content") or ""), "user", message_index)
            continue

        if role == "assistant":
            assistant_text = str(message.get("content") or "")
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name, arguments, call_id = call_name_and_args(call)
                checked_tool_calls += 1
                if call_id:
                    call_by_id[call_id] = (name, arguments)
                issues.extend(
                    validate_call(state, name, arguments, call_id, message_index, assistant_text)
                )
            continue

        if role == "tool":
            call_id = str(message.get("id") or message.get("tool_call_id") or "")
            name, arguments = call_by_id.get(call_id, ("", {}))
            content = message.get("content")
            content_text = str(content or "")
            if message.get("error") or content_text.lower().startswith("error:"):
                for slot_type, value in extract_slots_from_text(content_text):
                    state.remember_not_found(slot_type, value)
                continue
            add_slots_from_tool_result(
                state,
                name,
                arguments,
                parse_json_maybe(content),
                message_index,
            )

    return SlotValidationResult(
        simulation_id=str(sim.get("id") or ""),
        task_id=str(sim.get("task_id") or ""),
        trial=sim.get("trial"),
        checked_tool_calls=checked_tool_calls,
        issue_count=len(issues),
        issues=issues,
        known_slot_counts=state.counts(),
    )


def extract_slots_from_text(text: str) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    for value in USER_ID_RE.findall(text or ""):
        slots.append(("user_id", value))
    for value in PAYMENT_ID_RE.findall(text or ""):
        slots.append(("payment_id", value))
    for value in FLIGHT_NUMBER_RE.findall(text or ""):
        slots.append(("flight_number", value))
    for value in RESERVATION_ID_RE.findall(text or ""):
        if not value.startswith("HAT"):
            slots.append(("reservation_id", value))
    return slots


def result_to_dict(result: SlotValidationResult) -> dict[str, Any]:
    return {
        "simulation_id": result.simulation_id,
        "task_id": result.task_id,
        "trial": result.trial,
        "checked_tool_calls": result.checked_tool_calls,
        "issue_count": result.issue_count,
        "known_slot_counts": result.known_slot_counts,
        "issues": [issue.__dict__ for issue in result.issues],
    }


def summarize(results: list[SlotValidationResult]) -> dict[str, Any]:
    issue_counts = Counter(issue.component_name for result in results for issue in result.issues)
    tag_counts = Counter(tag for result in results for issue in result.issues for tag in issue.tags)
    by_task: dict[str, int] = Counter(result.task_id for result in results)
    issue_tasks: dict[str, int] = Counter(
        result.task_id for result in results if result.issues
    )
    return {
        "num_simulations": len(results),
        "checked_tool_calls": sum(result.checked_tool_calls for result in results),
        "simulations_with_issues": sum(1 for result in results if result.issues),
        "total_issues": sum(result.issue_count for result in results),
        "issue_counts": dict(sorted(issue_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "tasks": dict(sorted(by_task.items(), key=lambda item: int(item[0]))),
        "tasks_with_issues": dict(sorted(issue_tasks.items(), key=lambda item: int(item[0]))),
    }


def write_markdown(path: Path, run_name: str, summary: dict[str, Any], results: list[SlotValidationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Slot Grounding Report: {run_name}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Simulations | {summary['num_simulations']} |",
        f"| Checked tool calls | {summary['checked_tool_calls']} |",
        f"| Simulations with issues | {summary['simulations_with_issues']} |",
        f"| Total issues | {summary['total_issues']} |",
        "",
        "## Issue Counts",
        "",
    ]
    if summary["issue_counts"]:
        lines.extend(["| Issue | Count |", "| --- | ---: |"])
        for name, count in summary["issue_counts"].items():
            lines.append(f"| `{name}` | {count} |")
    else:
        lines.append("No slot-grounding issues found.")

    lines.extend(
        [
            "",
            "## Per Simulation",
            "",
            "| Task | Trial | Tool calls | Issues | Details |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in sorted(results, key=lambda item: (int(item.task_id or 0), item.trial or 0)):
        details = "<br>".join(
            f"`{issue.component_name}` {issue.tool_name}: {issue.slot_type}={issue.value}"
            for issue in result.issues
        )
        lines.append(
            f"| {result.task_id} | {result.trial} | {result.checked_tool_calls} | "
            f"{result.issue_count} | {details or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tau2 tool-call slot grounding.")
    parser.add_argument("path", type=Path, help="Path to tau2 run dir or results.json.")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    data = load_results(args.path)
    results = [validate_simulation_slots(sim) for sim in data.get("simulations") or []]
    summary = summarize(results)
    run_name = args.path.name if args.path.is_dir() else args.path.parent.name
    payload = {
        "run": run_name,
        "summary": summary,
        "simulations": [result_to_dict(result) for result in results],
    }

    print("Slot grounding summary")
    print("======================")
    print(f"run: {run_name}")
    print(f"simulations: {summary['num_simulations']}")
    print(f"checked_tool_calls: {summary['checked_tool_calls']}")
    print(f"simulations_with_issues: {summary['simulations_with_issues']}")
    print(f"total_issues: {summary['total_issues']}")
    if summary["issue_counts"]:
        print("\nissues:")
        for name, count in summary["issue_counts"].items():
            print(f"  {name}: {count}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.out_json}")
    if args.out_md:
        write_markdown(args.out_md, run_name, summary, results)
        print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
