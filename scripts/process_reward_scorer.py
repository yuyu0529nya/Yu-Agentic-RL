from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from summarize_tau2_results import is_success, load_results, reward_of
from slot_grounding_validator import validate_simulation_slots


WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}

RECENT_RESERVATION_PATTERNS = (
    "last reservation",
    "most recent reservation",
    "latest reservation",
)

SCHEDULE_SEARCH_TOOLS = {"search_direct_flight", "search_onestop_flight"}

SCHEDULE_NEEDLE_PATTERNS = (
    "duration",
    "durations",
    "longer than",
    "under or equal to",
    "3 hours",
    "4 hours",
    "including layovers",
)

DEFER_WITH_MISSING_TOOL_PATTERNS = (
    "unable to determine",
    "don't have access",
    "do not have access",
    "flight duration data isn't included",
    "flight duration information",
    "transfer to a human agent",
)

NO_CHANGE_OR_CANCEL_PATTERNS = (
    "not looking to change or cancel",
    "don't want to change or cancel",
    "do not want to change or cancel",
    "not want to change or cancel",
)


@dataclass
class ToolEvent:
    index: int
    message_index: int
    turn_idx: int | None
    name: str
    arguments: dict[str, Any]
    result: Any
    error: bool = False


@dataclass
class Component:
    name: str
    value: float
    detail: str
    tags: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    simulation_id: str
    task_id: str
    trial: int | None
    reward: float | None
    db_match: bool | None
    communicate_score: float | None
    process_score: float
    normalized_process_score: float
    tool_calls: int
    write_calls: int
    risk_tags: list[str]
    components: list[Component]


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    return {str(task.get("id")): task for task in tasks}


def parse_policy_current_time(path: Path | None) -> datetime:
    if path is None or not path.exists():
        return datetime(2024, 5, 15, 15, 0, 0)

    text = path.read_text(encoding="utf-8")
    match = re.search(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    if not match:
        return datetime(2024, 5, 15, 15, 0, 0)
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def parse_json_maybe(text: str | None) -> Any:
    if not text:
        return None
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def text_blob(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def task_text(task: dict[str, Any] | None) -> str:
    if not task:
        return ""
    parts: list[str] = []
    description = task.get("description") or {}
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions") or {}
    criteria = task.get("evaluation_criteria") or {}

    for value in description.values():
        parts.append(text_blob(value))
    for value in instructions.values():
        parts.append(text_blob(value))
    for value in criteria.get("nl_assertions") or []:
        parts.append(text_blob(value))
    for value in criteria.get("communicate_info") or []:
        parts.append(text_blob(value))
    return "\n".join(parts)


def transcript_text(sim: dict[str, Any]) -> str:
    return "\n".join(
        message.get("content") or ""
        for message in sim.get("messages") or []
        if message.get("role") in {"user", "assistant"}
    )


def assistant_transcript_text(sim: dict[str, Any]) -> str:
    return "\n".join(
        message.get("content") or ""
        for message in sim.get("messages") or []
        if message.get("role") == "assistant"
    )


def transcript_text_before(sim: dict[str, Any], message_index: int) -> str:
    return "\n".join(
        message.get("content") or ""
        for message in (sim.get("messages") or [])[:message_index]
        if message.get("role") in {"user", "assistant"}
    )


def extract_tool_events(sim: dict[str, Any]) -> tuple[list[ToolEvent], int]:
    call_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    total_calls = 0
    events: list[ToolEvent] = []

    for message_index, message in enumerate(sim.get("messages") or []):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                total_calls += 1
                call_id = str(call.get("id") or "")
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                arguments = call.get("arguments")
                if arguments is None:
                    arguments = function.get("arguments")
                arguments = parse_json_maybe(arguments)
                if not isinstance(arguments, dict):
                    arguments = {}
                call_by_id[call_id] = (
                    str(call.get("name") or function.get("name") or ""),
                    arguments,
                )
        elif message.get("role") == "tool":
            call_id = str(message.get("id") or "")
            name, arguments = call_by_id.get(call_id, ("<unknown>", {}))
            events.append(
                ToolEvent(
                    index=len(events),
                    message_index=message_index,
                    turn_idx=message.get("turn_idx"),
                    name=name,
                    arguments=arguments,
                    result=parse_json_maybe(message.get("content")),
                    error=bool(message.get("error")),
                )
            )

    return events, total_calls


def event_reservation_id(event: ToolEvent) -> str | None:
    value = event.arguments.get("reservation_id")
    if value:
        return str(value)
    if isinstance(event.result, dict) and event.result.get("reservation_id"):
        return str(event.result["reservation_id"])
    return None


def reservation_details_before(
    events: list[ToolEvent], reservation_id: str, before_index: int | None = None
) -> dict[str, Any] | None:
    upper = before_index if before_index is not None else len(events)
    for event in reversed(events[:upper]):
        if event.name != "get_reservation_details":
            continue
        if not isinstance(event.result, dict):
            continue
        if str(event.result.get("reservation_id")) == reservation_id:
            return event.result
    return None


def flight_status_confirmed_before(events: list[ToolEvent], before_index: int) -> set[str]:
    statuses: set[str] = set()
    for event in events[:before_index]:
        if event.name != "get_flight_status":
            continue
        if isinstance(event.result, str):
            statuses.add(event.result.lower())
    return statuses


def tool_names(events: list[ToolEvent]) -> set[str]:
    return {event.name for event in events}


def parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def flight_dates(details: dict[str, Any]) -> list[datetime]:
    dates: list[datetime] = []
    for flight in details.get("flights") or []:
        parsed = parse_date(flight.get("date"))
        if parsed is not None:
            dates.append(parsed)
    return dates


def all_flights_before(details: dict[str, Any], current_time: datetime) -> bool:
    dates = flight_dates(details)
    return bool(dates) and all(date.date() < current_time.date() for date in dates)


def expected_action_names(task: dict[str, Any] | None) -> set[str]:
    if not task:
        return set()
    criteria = task.get("evaluation_criteria") or {}
    return {str(action.get("name")) for action in criteria.get("actions") or []}


def task_forbids_db_changes(task: dict[str, Any] | None) -> bool:
    text = task_text(task).lower()
    return any(
        pattern in text
        for pattern in (
            "should not make any changes",
            "should make no changes",
            "do not make any changes",
            "not make any changes",
            "keep it as is",
            "leave the reservation unchanged",
        )
    )


def reward_db_match(sim: dict[str, Any]) -> bool | None:
    reward_info = sim.get("reward_info") or {}
    db_check = reward_info.get("db_check")
    if db_check is None:
        return None
    return bool(db_check.get("db_match"))


def communicate_score(sim: dict[str, Any]) -> float | None:
    breakdown = (sim.get("reward_info") or {}).get("reward_breakdown") or {}
    value = breakdown.get("COMMUNICATE")
    return float(value) if value is not None else None


def add_component(
    components: list[Component],
    name: str,
    value: float,
    detail: str,
    tags: list[str] | None = None,
) -> None:
    components.append(Component(name=name, value=value, detail=detail, tags=tags or []))


def score_reference_actions(sim: dict[str, Any], components: list[Component]) -> None:
    checks = (sim.get("reward_info") or {}).get("action_checks") or []
    if not checks:
        return

    matched = sum(1 for check in checks if check.get("action_match"))
    total = len(checks)
    if matched == total:
        add_component(
            components,
            "reference_actions_matched",
            1.0,
            f"All reference actions matched ({matched}/{total}).",
        )
    else:
        missing = total - matched
        add_component(
            components,
            "reference_action_mismatch",
            -1.0 * missing,
            f"Reference action mismatch ({matched}/{total}).",
            ["action_mismatch", "incomplete_evidence"],
        )


def normalize_numeric_text(value: str) -> str | None:
    cleaned = re.sub(r"[$,\s]", "", value)
    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return None
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def required_info_present(assistant_text: str, required: Any) -> bool:
    raw = text_blob(required).strip()
    if not raw:
        return True

    lowered = raw.lower()
    if lowered in assistant_text:
        return True

    required_number = normalize_numeric_text(raw)
    if required_number is None:
        return False

    for match in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?", assistant_text):
        observed = normalize_numeric_text(match.group(0))
        if observed == required_number:
            return True
    return False


def score_required_communication(
    sim: dict[str, Any],
    task: dict[str, Any] | None,
    components: list[Component],
) -> None:
    if not task:
        return

    criteria = task.get("evaluation_criteria") or {}
    required_items = list(criteria.get("communicate_info") or [])
    if not required_items:
        return

    assistant_text = assistant_transcript_text(sim).lower()
    missing = [
        text_blob(item)
        for item in required_items
        if not required_info_present(assistant_text, item)
    ]
    if missing:
        add_component(
            components,
            "missing_required_communication",
            -1.5 * len(missing),
            "Required communication info was not found in assistant messages: "
            + ", ".join(missing),
            ["communication_target_miss", "incomplete_evidence"],
        )
    else:
        add_component(
            components,
            "required_communication_met",
            1.0,
            f"All required communication info was found ({len(required_items)} item(s)).",
        )


def score_recent_reservation_evidence(
    sim: dict[str, Any],
    task: dict[str, Any] | None,
    events: list[ToolEvent],
    components: list[Component],
) -> None:
    combined_text = (transcript_text(sim) + "\n" + task_text(task)).lower()
    if not any(pattern in combined_text for pattern in RECENT_RESERVATION_PATTERNS):
        return

    user_reservations: list[str] = []
    for event in events:
        if event.name == "get_user_details" and isinstance(event.result, dict):
            reservations = event.result.get("reservations") or []
            user_reservations.extend(str(item) for item in reservations)

    user_reservations = list(dict.fromkeys(user_reservations))
    if len(user_reservations) <= 1:
        return

    observed_details = {
        str(event.result.get("reservation_id"))
        for event in events
        if event.name == "get_reservation_details" and isinstance(event.result, dict)
    }
    missing = [rid for rid in user_reservations if rid not in observed_details]
    if missing:
        add_component(
            components,
            "recent_reservation_candidates_missing",
            -2.0,
            "Recent/last reservation claim but not all candidate reservations were checked: "
            + ", ".join(missing),
            ["incomplete_evidence", "object_selection_error"],
        )
    else:
        add_component(
            components,
            "recent_reservation_candidates_checked",
            1.5,
            f"All {len(user_reservations)} candidate reservations were checked.",
        )


def score_tool_affordance(
    sim: dict[str, Any],
    task: dict[str, Any] | None,
    events: list[ToolEvent],
    components: list[Component],
) -> None:
    combined_text = (transcript_text(sim) + "\n" + task_text(task)).lower()
    expected_names = expected_action_names(task)
    names = tool_names(events)

    needs_schedule = any(pattern in combined_text for pattern in SCHEDULE_NEEDLE_PATTERNS)
    expected_schedule_search = bool(expected_names & SCHEDULE_SEARCH_TOOLS)
    used_status = "get_flight_status" in names
    used_schedule_search = bool(names & SCHEDULE_SEARCH_TOOLS)

    if (needs_schedule or expected_schedule_search) and used_status and not used_schedule_search:
        add_component(
            components,
            "wrong_tool_for_schedule_lookup",
            -2.0,
            "Schedule/duration information was needed, but only get_flight_status was used; "
            "search tools that return scheduled times were not called.",
            ["tool_affordance_error", "incomplete_evidence"],
        )

    if expected_schedule_search and not used_schedule_search:
        deferred = any(pattern in combined_text for pattern in DEFER_WITH_MISSING_TOOL_PATTERNS)
        if deferred:
            add_component(
                components,
                "premature_deferral",
                -1.5,
                "The agent deferred or claimed missing information even though task reference "
                "actions indicate schedule search tools were available.",
                ["tool_affordance_error", "premature_deferral", "incomplete_evidence"],
            )


def score_payment_planning(
    sim: dict[str, Any],
    task: dict[str, Any] | None,
    events: list[ToolEvent],
    components: list[Component],
) -> None:
    text = (transcript_text(sim) + "\n" + task_text(task)).lower()
    expected_names = expected_action_names(task)
    names = tool_names(events)

    payment_task = (
        ("certificate" in text or "certificates" in text)
        and ("gift card" in text or "gift cards" in text)
        and ("mastercard" in text or "master card" in text)
    )
    if not payment_task:
        return

    score_basic_economy_update_attempt(text, events, components)
    score_payment_write_errors(events, components)
    score_cancel_rebook_order(text, events, components)
    score_split_booking_payment_plan(events, components)

    expected_cancel_and_book = (
        "cancel_reservation" in expected_names and "book_reservation" in expected_names
    )
    if expected_cancel_and_book and "update_reservation_flights" in names:
        if "cancel_reservation" not in names or "book_reservation" not in names:
            add_component(
                components,
                "expected_write_plan_missing",
                -2.0,
                "Task expected a cancel-and-book payment plan, but the actual trajectory used "
                "an update-only or incomplete write strategy.",
                ["payment_planning_error", "premature_write"],
            )

    mastercard_amounts = extract_mastercard_amounts(transcript_text(sim))
    if len(mastercard_amounts) >= 2:
        add_component(
            components,
            "calculation_trace_inconsistent",
            -1.0,
            "Multiple Mastercard charge amounts were communicated: "
            + ", ".join(f"${amount}" for amount in sorted(mastercard_amounts)),
            ["payment_planning_error", "calculation_error"],
        )


def score_basic_economy_update_attempt(
    text: str,
    events: list[ToolEvent],
    components: list[Component],
) -> None:
    cancel_rebook_context = (
        "cancel the current one and book a new one" in text
        or "one-certificate-per-reservation" in text
        or "one certificate per reservation" in text
        or "three separate" in text
    )
    if not cancel_rebook_context:
        return

    for event in events:
        if event.name != "update_reservation_flights":
            continue
        reservation_id = event_reservation_id(event)
        if not reservation_id:
            continue
        details = reservation_details_before(events, reservation_id, event.index)
        if not details:
            continue
        if str(details.get("cabin", "")).lower() == "basic_economy":
            add_component(
                components,
                "basic_economy_update_attempt",
                -3.0,
                f"update_reservation_flights({reservation_id}) was attempted on a "
                "basic-economy reservation; these tasks require cancel-and-rebook planning.",
                ["payment_planning_error", "premature_write"],
            )


def score_payment_write_errors(
    events: list[ToolEvent],
    components: list[Component],
) -> None:
    error_events = [
        event
        for event in events
        if event.name in WRITE_TOOLS
        and (event.error or text_blob(event.result).lower().startswith("error:"))
    ]
    if not error_events:
        return

    names = ", ".join(f"{event.name}@{event.index}" for event in error_events)
    add_component(
        components,
        "write_tool_payment_error",
        -2.0,
        f"Write tool errors occurred during payment execution: {names}.",
        ["payment_planning_error", "premature_write"],
    )


def score_cancel_rebook_order(
    text: str,
    events: list[ToolEvent],
    components: list[Component],
) -> None:
    if "basic economy" not in text or "cancel" not in text or "book" not in text:
        return

    first_book = next((event for event in events if event.name == "book_reservation"), None)
    if first_book is None:
        return

    prior_cancel = any(
        event.name == "cancel_reservation" for event in events[: first_book.index]
    )
    if prior_cancel:
        return

    has_basic_reservation = any(
        event.name == "get_reservation_details"
        and isinstance(event.result, dict)
        and str(event.result.get("cabin", "")).lower() == "basic_economy"
        for event in events[: first_book.index]
    )
    if has_basic_reservation:
        add_component(
            components,
            "book_before_cancel_rebook_plan",
            -2.0,
            "A new booking was attempted before cancelling the existing basic-economy "
            "reservation in a cancel-and-rebook payment plan.",
            ["payment_planning_error", "premature_write"],
        )


def score_split_booking_payment_plan(
    events: list[ToolEvent],
    components: list[Component],
) -> None:
    book_events = successful_book_events(events)
    if len(book_events) < 2:
        return

    single_passenger_books = [
        event
        for event in book_events
        if len(event.arguments.get("passengers") or []) == 1
    ]
    cert_counts = [
        sum(1 for method in payment_methods(event) if payment_source(method) == "certificate")
        for event in book_events
    ]
    if len(single_passenger_books) == len(book_events) and all(count <= 1 for count in cert_counts):
        add_component(
            components,
            "split_booking_policy_supported",
            1.0,
            f"{len(book_events)} single-passenger bookings were made with at most one "
            "certificate per reservation.",
        )

    estimate = estimate_optimal_mastercard_charge(events, book_events)
    if estimate is None:
        return

    actual_card_total = round(sum_payment_amounts(book_events, "credit_card"))
    if actual_card_total == estimate:
        add_component(
            components,
            "optimal_mastercard_charge_matched",
            3.0,
            f"Tool payment plan matches the estimated minimum Mastercard charge: ${estimate}.",
        )
    else:
        add_component(
            components,
            "optimal_mastercard_charge_mismatch",
            -3.5,
            f"Tool payment plan charged ${actual_card_total} on credit card, but the "
            f"estimated minimum from searched business fares and stored balances is ${estimate}.",
            ["payment_planning_error", "calculation_error"],
        )


def successful_book_events(events: list[ToolEvent]) -> list[ToolEvent]:
    return [
        event
        for event in events
        if event.name == "book_reservation"
        and isinstance(event.result, dict)
        and not event.error
    ]


def payment_methods(event: ToolEvent) -> list[dict[str, Any]]:
    methods = event.arguments.get("payment_methods") or []
    return [method for method in methods if isinstance(method, dict)]


def payment_source(method: dict[str, Any]) -> str:
    payment_id = str(method.get("payment_id") or "")
    if payment_id.startswith("certificate_"):
        return "certificate"
    if payment_id.startswith("gift_card_"):
        return "gift_card"
    if payment_id.startswith("credit_card_"):
        return "credit_card"
    return ""


def payment_amount(method: dict[str, Any]) -> float:
    try:
        return float(method.get("amount") or 0)
    except (TypeError, ValueError):
        return 0.0


def sum_payment_amounts(events: list[ToolEvent], source: str) -> float:
    return sum(
        payment_amount(method)
        for event in events
        for method in payment_methods(event)
        if payment_source(method) == source
    )


def estimate_optimal_mastercard_charge(
    events: list[ToolEvent],
    book_events: list[ToolEvent],
) -> int | None:
    first = book_events[0]
    origin = str(first.arguments.get("origin") or "")
    destination = str(first.arguments.get("destination") or "")
    flight_dates = [
        str(flight.get("date"))
        for flight in first.arguments.get("flights") or []
        if flight.get("date")
    ]
    if not origin or not destination or len(flight_dates) < 2:
        return None

    search_prices = min_business_prices_by_search(events)
    outbound_price = search_prices.get((origin, destination, flight_dates[0]))
    return_price = search_prices.get((destination, origin, flight_dates[-1]))
    if outbound_price is None or return_price is None:
        return None

    per_passenger_price = outbound_price + return_price
    passenger_count = sum(len(event.arguments.get("passengers") or []) for event in book_events)
    if passenger_count <= 0:
        return None

    certificate_amounts, gift_card_total = user_stored_payment_balances(events)
    usable_certificate_total = sum(sorted(certificate_amounts, reverse=True)[:passenger_count])
    estimated = per_passenger_price * passenger_count - usable_certificate_total - gift_card_total
    return max(0, round(estimated))


def user_stored_payment_balances(events: list[ToolEvent]) -> tuple[list[float], float]:
    certificates: dict[str, float] = {}
    gift_cards: dict[str, float] = {}
    for event in events:
        if event.name != "get_user_details" or not isinstance(event.result, dict):
            continue
        methods = event.result.get("payment_methods") or {}
        if not isinstance(methods, dict):
            continue
        for payment_id, method in methods.items():
            if not isinstance(method, dict):
                continue
            source = str(method.get("source") or "")
            try:
                amount = float(method.get("amount") or 0)
            except (TypeError, ValueError):
                amount = 0.0
            if source == "certificate":
                certificates[str(payment_id)] = amount
            elif source == "gift_card":
                gift_cards[str(payment_id)] = amount
    return list(certificates.values()), sum(gift_cards.values())


def min_business_prices_by_search(events: list[ToolEvent]) -> dict[tuple[str, str, str], float]:
    prices: dict[tuple[str, str, str], float] = {}
    for event in events:
        if event.name not in SCHEDULE_SEARCH_TOOLS:
            continue
        origin = str(event.arguments.get("origin") or "")
        destination = str(event.arguments.get("destination") or "")
        date = str(event.arguments.get("date") or "")
        if not origin or not destination or not date:
            continue
        option_prices = [
            price
            for option in route_options(event.result)
            if (price := route_business_price(option)) is not None
        ]
        if not option_prices:
            continue
        key = (origin, destination, date)
        current = prices.get(key)
        best = min(option_prices)
        prices[key] = best if current is None else min(current, best)
    return prices


def route_options(result: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(result, list):
        return []
    options: list[list[dict[str, Any]]] = []
    for item in result:
        if isinstance(item, dict):
            options.append([item])
        elif isinstance(item, list):
            flights = [flight for flight in item if isinstance(flight, dict)]
            if flights:
                options.append(flights)
    return options


def route_business_price(option: list[dict[str, Any]]) -> float | None:
    total = 0.0
    for flight in option:
        prices = flight.get("prices") or {}
        if "business" not in prices:
            return None
        try:
            total += float(prices["business"])
        except (TypeError, ValueError):
            return None
    return total


def extract_mastercard_amounts(text: str) -> set[int]:
    amounts: set[int] = set()
    line_patterns = (
        r"(?:mastercard|master card)(?:\s+charge|\s+charged|[^$\n:]{0,60}:)\s*\$([0-9][0-9,]*)",
        r"(?:charged|charge|paid|pay|payment)[^$\n]{0,80}(?:mastercard|master card)[^$\n]{0,40}\$([0-9][0-9,]*)",
        r"\$([0-9][0-9,]*)[^$\n]{0,50}(?:on|to|from|with)\s+(?:my\s+)?(?:mastercard|master card)",
    )
    for line in text.splitlines():
        lower = line.lower()
        if "mastercard" not in lower and "master card" not in lower:
            continue
        for pattern in line_patterns:
            for match in re.finditer(pattern, line, flags=re.IGNORECASE):
                raw = match.group(1).replace(",", "")
                try:
                    amount = int(raw)
                except ValueError:
                    continue
                if amount >= 100:
                    amounts.add(amount)
    return amounts


def score_write_tools(
    sim: dict[str, Any],
    task: dict[str, Any] | None,
    events: list[ToolEvent],
    components: list[Component],
    current_time: datetime,
) -> None:
    expected_names = expected_action_names(task)
    combined_text = (transcript_text(sim) + "\n" + task_text(task)).lower()
    pressure_text = any(
        phrase in combined_text
        for phrase in (
            "representative",
            "approved",
            "told you",
            "told me",
            "customer support",
        )
    )
    write_events = [event for event in events if event.name in WRITE_TOOLS]

    if task_forbids_db_changes(task) and write_events:
        names = ", ".join(f"{event.name}@{event.index}" for event in write_events)
        add_component(
            components,
            "forbidden_db_write",
            -3.0 - 0.5 * (len(write_events) - 1),
            "Task assertions say the agent should not make changes, but write tools "
            f"were called: {names}.",
            ["premature_write", "policy_precedence_error"],
        )

    for event in write_events:
        if expected_names and event.name not in expected_names:
            add_component(
                components,
                "unexpected_write_tool",
                -1.5,
                f"Write tool {event.name} is not in the task reference actions.",
                ["premature_write"],
            )

        if event.name == "cancel_reservation":
            score_cancel_event(event, events, components, current_time, pressure_text)
        elif event.name == "send_certificate":
            score_certificate_event(event, sim, events, task, components)


def score_slot_grounding(sim: dict[str, Any], components: list[Component]) -> None:
    result = validate_simulation_slots(sim)
    for issue in result.issues:
        add_component(
            components,
            issue.component_name,
            -issue.severity,
            issue.detail,
            issue.tags,
        )

    if result.checked_tool_calls and not result.issues:
        add_component(
            components,
            "slot_grounding_clean",
            1.0,
            f"All {result.checked_tool_calls} tool call(s) used grounded identifiers.",
        )


def score_cancel_event(
    event: ToolEvent,
    events: list[ToolEvent],
    components: list[Component],
    current_time: datetime,
    pressure_text: bool,
) -> None:
    reservation_id = event_reservation_id(event)
    if not reservation_id:
        add_component(
            components,
            "cancel_missing_reservation_id",
            -2.0,
            "cancel_reservation was called without a reservation_id.",
            ["premature_write"],
        )
        return

    details = reservation_details_before(events, reservation_id, event.index)
    if details is None:
        add_component(
            components,
            "cancel_without_reservation_details",
            -2.0,
            f"cancel_reservation({reservation_id}) was called before reading reservation details.",
            ["incomplete_evidence", "premature_write"],
        )
        return

    created_at = details.get("created_at")
    if not created_at:
        return

    try:
        created = datetime.fromisoformat(str(created_at))
    except ValueError:
        return

    age_hours = (current_time - created).total_seconds() / 3600.0
    has_insurance = str(details.get("insurance", "")).lower() == "yes"
    statuses = flight_status_confirmed_before(events, event.index)
    has_airline_cancelled_flight = "cancelled" in statuses or "canceled" in statuses
    already_flown = all_flights_before(details, current_time)

    if already_flown:
        add_component(
            components,
            "past_flight_cancel_attempt",
            -3.0,
            f"{reservation_id} contains only flight dates before current time; "
            "already-flown reservations should not be cancelled.",
            ["temporal_policy_error", "policy_precedence_error", "premature_write"],
        )
        add_component(
            components,
            "policy_precedence_error",
            -1.0,
            "A permissive cancellation rule was applied while an already-flown blocking "
            "condition should have taken precedence.",
            ["policy_precedence_error"],
        )

    if age_hours > 24 and not has_insurance and not has_airline_cancelled_flight:
        add_component(
            components,
            "cancel_after_24h_without_policy_basis",
            -3.0,
            f"{reservation_id} was created {age_hours:.1f} hours before current time; "
            "no insurance or airline-cancelled status was confirmed.",
            ["temporal_policy_error", "premature_write"],
        )
        if pressure_text:
            add_component(
                components,
                "user_pressure_susceptibility",
                -1.0,
                "Cancellation occurred in a conversation containing approval/representative pressure.",
                ["user_pressure_susceptibility"],
            )
    else:
        basis = []
        if age_hours <= 24:
            basis.append(f"created {age_hours:.1f}h ago")
        if has_insurance:
            basis.append("insurance=yes")
        if has_airline_cancelled_flight:
            basis.append("airline-cancelled status confirmed")
        add_component(
            components,
            "cancel_precondition_supported",
            1.0,
            f"cancel_reservation({reservation_id}) had policy support: {', '.join(basis)}.",
        )


def score_certificate_event(
    event: ToolEvent,
    sim: dict[str, Any],
    events: list[ToolEvent],
    task: dict[str, Any] | None,
    components: list[Component],
) -> None:
    text = task_text(task).lower()
    forbids_certificate = (
        "should not offer a certificate" in text
        or "does not offer any compensation" in text
        or "should not get compensation" in text
    )
    if forbids_certificate:
        add_component(
            components,
            "certificate_forbidden_by_task",
            -3.0,
            "send_certificate was called on a task whose assertions forbid compensation/certificate.",
            ["premature_write", "compensation_policy_error"],
        )

    prior_statuses = flight_status_confirmed_before(events, event.index)
    if not ({"delayed", "cancelled", "canceled"} & prior_statuses):
        add_component(
            components,
            "certificate_without_status_confirmation",
            -1.0,
            "send_certificate was called before confirming delayed/cancelled flight status.",
            ["incomplete_evidence", "premature_write"],
        )

    prior_change_or_cancel = any(
        prior.name in {"cancel_reservation", "update_reservation_flights"}
        for prior in events[: event.index]
    )
    if "delayed" in text and not prior_change_or_cancel:
        add_component(
            components,
            "delayed_certificate_without_change_or_cancel",
            -1.5,
            "Delayed-flight certificate was issued without a prior change/cancellation action.",
            ["compensation_policy_error", "premature_write"],
        )

    prior_text = transcript_text_before(sim, event.message_index).lower()
    if "delayed" in text and any(pattern in prior_text for pattern in NO_CHANGE_OR_CANCEL_PATTERNS):
        add_component(
            components,
            "compensation_requires_user_goal",
            -2.0,
            "A delayed-flight certificate was issued after the user said they did not want "
            "to change or cancel the reservation.",
            ["compensation_policy_error", "premature_write"],
        )


def classify_tags(
    sim: dict[str, Any], components: list[Component], db_match: bool | None
) -> list[str]:
    tags = Counter({tag: 1 for tag in component_risk_tags(components)})
    comm = communicate_score(sim)
    if db_match is False and comm == 1.0:
        tags["communication_db_gap"] += 1
    if db_match is False and not tags:
        tags["unclassified_db_failure"] += 1
    return sorted(tags)


def component_risk_tags(components: list[Component]) -> list[str]:
    tags = Counter()
    for component in components:
        if component.value < 0:
            for tag in component.tags:
                tags[tag] += 1
    return sorted(tags)


def score_simulation(
    sim: dict[str, Any],
    task: dict[str, Any] | None,
    current_time: datetime,
) -> ScoreResult:
    events, total_tool_calls = extract_tool_events(sim)
    components: list[Component] = []

    score_reference_actions(sim, components)
    score_required_communication(sim, task, components)
    score_recent_reservation_evidence(sim, task, events, components)
    score_tool_affordance(sim, task, events, components)
    score_payment_planning(sim, task, events, components)
    score_write_tools(sim, task, events, components, current_time)
    score_slot_grounding(sim, components)

    process_score = sum(component.value for component in components)
    max_abs = sum(abs(component.value) for component in components) or 1.0
    normalized = process_score / max_abs
    db_match = reward_db_match(sim)

    return ScoreResult(
        simulation_id=str(sim.get("id")),
        task_id=str(sim.get("task_id")),
        trial=sim.get("trial"),
        reward=reward_of(sim),
        db_match=db_match,
        communicate_score=communicate_score(sim),
        process_score=process_score,
        normalized_process_score=normalized,
        tool_calls=total_tool_calls,
        write_calls=sum(1 for event in events if event.name in WRITE_TOOLS),
        risk_tags=component_risk_tags(components),
        components=components,
    )


def result_to_dict(result: ScoreResult) -> dict[str, Any]:
    return {
        "simulation_id": result.simulation_id,
        "task_id": result.task_id,
        "trial": result.trial,
        "reward": result.reward,
        "db_match": result.db_match,
        "communicate_score": result.communicate_score,
        "process_score": result.process_score,
        "normalized_process_score": result.normalized_process_score,
        "tool_calls": result.tool_calls,
        "write_calls": result.write_calls,
        "risk_tags": result.risk_tags,
        "components": [component.__dict__ for component in result.components],
    }


def summarize_scores(results: list[ScoreResult]) -> dict[str, Any]:
    rewards = [result.reward for result in results if result.reward is not None]
    scores = [result.process_score for result in results]
    success_scores = [
        result.process_score for result in results if is_success(result.reward)
    ]
    failure_scores = [
        result.process_score for result in results if not is_success(result.reward)
    ]
    tag_counts = Counter(tag for result in results for tag in result.risk_tags)

    return {
        "num_simulations": len(results),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "success_count": sum(1 for reward in rewards if is_success(reward)),
        "avg_process_score": sum(scores) / len(scores) if scores else 0.0,
        "avg_process_score_success": (
            sum(success_scores) / len(success_scores) if success_scores else None
        ),
        "avg_process_score_failure": (
            sum(failure_scores) / len(failure_scores) if failure_scores else None
        ),
        "risk_tags": dict(tag_counts),
    }


def write_csv(path: Path, results: list[ScoreResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "trial",
                "reward",
                "db_match",
                "communicate_score",
                "process_score",
                "normalized_process_score",
                "tool_calls",
                "write_calls",
                "risk_tags",
                "components",
                "simulation_id",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "task_id": result.task_id,
                    "trial": result.trial,
                    "reward": result.reward,
                    "db_match": result.db_match,
                    "communicate_score": result.communicate_score,
                    "process_score": f"{result.process_score:.4f}",
                    "normalized_process_score": f"{result.normalized_process_score:.4f}",
                    "tool_calls": result.tool_calls,
                    "write_calls": result.write_calls,
                    "risk_tags": "|".join(result.risk_tags),
                    "components": "|".join(
                        f"{component.name}:{component.value:+.1f}"
                        for component in result.components
                    ),
                    "simulation_id": result.simulation_id,
                }
            )


def write_markdown(
    path: Path,
    run_name: str,
    summary: dict[str, Any],
    results: list[ScoreResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# PRM-Lite Process Reward Report: {run_name}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Simulations | {summary['num_simulations']} |",
        f"| Avg reward | {summary['avg_reward']:.4f} |",
        f"| Success count | {summary['success_count']} |",
        f"| Avg process score | {summary['avg_process_score']:.4f} |",
        f"| Avg process score, success | {format_optional(summary['avg_process_score_success'])} |",
        f"| Avg process score, failure | {format_optional(summary['avg_process_score_failure'])} |",
        "",
        "## Risk Tags",
        "",
    ]
    if summary["risk_tags"]:
        lines += ["| Tag | Count |", "| --- | ---: |"]
        for tag, count in sorted(summary["risk_tags"].items()):
            lines.append(f"| `{tag}` | {count} |")
    else:
        lines.append("No risk tags triggered.")

    lines += [
        "",
        "## Per-task Scores",
        "",
        "| Task | Reward | DB | Process | Tags | Components |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for result in sorted(results, key=lambda item: (int(item.task_id), item.trial or 0)):
        components = "<br>".join(
            f"`{component.name}` {component.value:+.1f}" for component in result.components
        )
        tags = ", ".join(f"`{tag}`" for tag in result.risk_tags) or "-"
        lines.append(
            f"| {result.task_id} | {format_optional(result.reward)} | "
            f"{result.db_match} | {result.process_score:.1f} | {tags} | {components or '-'} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def default_tau2_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "tau2-bench"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score tau2 trajectories with a lightweight process reward model."
    )
    parser.add_argument("path", type=Path, help="Path to tau2 run dir or results.json.")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--tau2-root", type=Path, default=default_tau2_root())
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    data = load_results(args.path)
    domain_dir = args.tau2_root / "data" / "tau2" / "domains" / args.domain
    tasks = load_tasks(domain_dir / "tasks.json")
    current_time = parse_policy_current_time(domain_dir / "policy.md")

    results = [
        score_simulation(sim, tasks.get(str(sim.get("task_id"))), current_time)
        for sim in data.get("simulations") or []
    ]
    summary = summarize_scores(results)
    payload = {
        "run": args.path.name if args.path.is_dir() else args.path.parent.name,
        "summary": summary,
        "simulations": [result_to_dict(result) for result in results],
    }

    print("PRM-Lite process reward summary")
    print("===============================")
    print(f"run: {payload['run']}")
    print(f"simulations: {summary['num_simulations']}")
    print(f"avg_reward: {summary['avg_reward']:.4f}")
    print(f"avg_process_score: {summary['avg_process_score']:.4f}")
    print(
        "success/failure process score: "
        f"{format_optional(summary['avg_process_score_success'])} / "
        f"{format_optional(summary['avg_process_score_failure'])}"
    )
    if summary["risk_tags"]:
        print("\nrisk tags:")
        for tag, count in sorted(summary["risk_tags"].items()):
            print(f"  {tag}: {count}")

    print("\nper task:")
    for result in sorted(results, key=lambda item: (int(item.task_id), item.trial or 0)):
        tags = ",".join(result.risk_tags) or "-"
        print(
            f"  task {result.task_id}: reward={format_optional(result.reward)} "
            f"process={result.process_score:.1f} tags={tags}"
        )

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nwrote: {args.out_json}")

    if args.out_csv:
        write_csv(args.out_csv, results)
        print(f"wrote: {args.out_csv}")

    if args.out_md:
        write_markdown(args.out_md, payload["run"], summary, results)
        print(f"wrote: {args.out_md}")


if __name__ == "__main__":
    main()
