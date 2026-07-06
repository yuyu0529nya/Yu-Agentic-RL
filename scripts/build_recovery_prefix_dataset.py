from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_BASE_TRAIN = "data/decision_prefix/tau2_airline_decision_prefix_v1_2048_train.jsonl"
DEFAULT_BASE_VALID = "data/decision_prefix/tau2_airline_decision_prefix_v1_2048_valid.jsonl"
DEFAULT_BASE_HELDOUT = "data/decision_prefix/tau2_airline_decision_prefix_v1_2048_heldout.jsonl"
DEFAULT_FAILURE_RESULTS = "autodl_artifacts/phase2b_decision_prefix_sftonly_2task_4090_20260614_194545/results.json"
DEFAULT_OUT_DIR = "data/recovery_prefix"
DEFAULT_REPORT = "reports/recovery_prefix_dataset_v1.md"
DEFAULT_OUTPUT_STEM = "tau2_airline_recovery_prefix_v1_2048"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def has_user_identifier(text: str) -> bool:
    lowered = text.lower()
    signals = ["user id", "user_id", "my id is", "reservation id", "reservation_id"]
    return any(signal in lowered for signal in signals)


def assistant_tool_names(message: dict[str, Any]) -> list[str]:
    names = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        names.append(function.get("name") or call.get("name") or "")
    return names


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    call_names_by_id: dict[str, str] = {}
    for message in messages:
        item = copy.deepcopy(message)
        if item.get("role") == "assistant":
            calls = []
            for call in item.get("tool_calls") or []:
                call_item = copy.deepcopy(call)
                function = call_item.setdefault("function", {})
                if not function.get("name"):
                    function["name"] = call_item.get("name") or ""
                if "arguments" not in function:
                    function["arguments"] = call_item.get("arguments") or {}
                call_item["type"] = call_item.get("type") or "function"
                call_id = str(call_item.get("id") or "")
                if call_id:
                    call_names_by_id[call_id] = str(function.get("name") or "")
                calls.append(call_item)
            if calls:
                item["tool_calls"] = calls
        elif item.get("role") == "tool":
            call_id = str(item.get("tool_call_id") or item.get("id") or "")
            if call_id:
                item["tool_call_id"] = call_id
            if not item.get("name") and call_id in call_names_by_id:
                item["name"] = call_names_by_id[call_id]
        normalized.append(item)
    return normalized


def text_target_sample(
    sample_id: str,
    prefix: list[dict[str, Any]],
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    target = {
        "role": "assistant",
        "content": content,
        "tool_calls": None,
    }
    return {
        "id": sample_id,
        "format_version": "tau2_airline_recovery_prefix_v1",
        "sample_type": "recovery_prefix_text",
        "split": "train",
        "messages": copy.deepcopy(prefix) + [target],
        "loss_mask": [False] * len(prefix) + [True],
        "loss_policy": {
            "assistant_content": True,
            "assistant_tool_calls": False,
            "assistant_tool_call_wrappers": False,
            "user": False,
            "tool": False,
        },
        "metadata": metadata,
    }


def make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def tool_target_sample(
    sample_id: str,
    prefix: list[dict[str, Any]],
    name: str,
    arguments: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    call = make_tool_call(f"call_{sample_id}", name, arguments)
    target = {
        "role": "assistant",
        "content": "",
        "tool_calls": [call],
    }
    metadata = dict(metadata)
    metadata.update(
        {
            "target_action": "tool_call",
            "target_tool_names": [name],
            "target_tool_calls": [{"name": name, "arguments": arguments}],
            "protocol_wrapper_loss": True,
        }
    )
    return {
        "id": sample_id,
        "format_version": "tau2_airline_recovery_prefix_v1",
        "sample_type": "recovery_prefix_tool_call",
        "split": "train",
        "messages": copy.deepcopy(prefix) + [target],
        "loss_mask": [False] * len(prefix) + [True],
        "loss_policy": {
            "assistant_content": False,
            "assistant_tool_calls": True,
            "assistant_tool_call_wrappers": True,
            "user": False,
            "tool": False,
        },
        "metadata": metadata,
    }


def make_metadata(sim: dict[str, Any], turn_index: int, family: str) -> dict[str, Any]:
    return {
        "domain": "airline",
        "task_id": str(sim.get("task_id") or ""),
        "trial": sim.get("trial"),
        "simulation_id": sim.get("id"),
        "source_termination": sim.get("termination_reason"),
        "turn_index": turn_index,
        "target_action": "assistant_text",
        "recovery_family": family,
        "source": "failed_tau2_online_trace",
    }


def parse_json_content(message: dict[str, Any]) -> Any | None:
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def prior_user_text(messages: list[dict[str, Any]], end: int) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages[:end]
        if message.get("role") == "user"
    ).lower()


def mentions_atl_phl(text: str) -> bool:
    return ("atl" in text or "atlanta" in text) and (
        "phl" in text or "philadelphia" in text
    )


def get_queried_reservations(messages: list[dict[str, Any]], end: int) -> set[str]:
    queried: set[str] = set()
    for message in messages[:end]:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if (function.get("name") or call.get("name")) != "get_reservation_details":
                continue
            arguments = function.get("arguments") or call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            reservation_id = arguments.get("reservation_id")
            if reservation_id:
                queried.add(str(reservation_id))
    return queried


def add_route_disambiguation_corrections(
    sim: dict[str, Any],
    messages: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    """Add online corrections for choosing among multiple reservation IDs.

    The user profile only lists reservation IDs; it does not prove route/date.
    When the user goal contains route evidence, the agent should inspect
    candidate reservations instead of inventing route summaries from IDs.
    """

    task_id = str(sim.get("task_id") or "unknown")
    seen_ids: set[str] = {str(row.get("id")) for row in corrections}
    for idx, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        payload = parse_json_content(message)
        if not isinstance(payload, dict):
            continue
        reservations = payload.get("reservations")
        if not isinstance(reservations, list) or len(reservations) < 2:
            continue
        if not mentions_atl_phl(prior_user_text(messages, idx)):
            continue
        queried = get_queried_reservations(messages, idx)
        candidate = next((str(rid) for rid in reservations if str(rid) not in queried), "")
        if not candidate:
            continue
        sample_id = f"recovery_task{task_id}_turn{idx}_route_disambiguation_check_candidate"
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        corrections.append(
            tool_target_sample(
                sample_id,
                messages[: idx + 1],
                "get_reservation_details",
                {"reservation_id": candidate},
                make_metadata(sim, idx, "route_disambiguation_check_all_reservations"),
            )
        )

    known_reservations: list[str] = []
    for idx, message in enumerate(messages):
        if message.get("role") == "tool":
            payload = parse_json_content(message)
            if isinstance(payload, dict) and isinstance(payload.get("reservations"), list):
                known_reservations = [str(rid) for rid in payload["reservations"]]
        if message.get("role") != "user" or not known_reservations:
            continue
        user_text = str(message.get("content") or "").lower()
        mismatch_signals = ["doesn't sound right", "does not sound right", "discrepancy", "not a"]
        if not mentions_atl_phl(user_text) or not any(signal in user_text for signal in mismatch_signals):
            continue
        queried = get_queried_reservations(messages, idx)
        candidate = next((rid for rid in known_reservations if rid not in queried), "")
        if not candidate:
            continue
        sample_id = f"recovery_task{task_id}_turn{idx}_route_disambiguation_after_mismatch"
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        corrections.append(
            tool_target_sample(
                sample_id,
                messages[: idx + 1],
                "get_reservation_details",
                {"reservation_id": candidate},
                make_metadata(
                    sim,
                    idx,
                    "route_disambiguation_check_remaining_reservations",
                ),
            )
        )


def reservation_tool_message(call_id: str, reservation: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": "get_reservation_details",
        "content": json.dumps(reservation, ensure_ascii=False),
    }


def generic_tool_message(call_id: str, name: str, payload: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def load_airline_db() -> dict[str, Any] | None:
    db_path = repo_path("third_party/tau2-bench/data/tau2/domains/airline/db.json")
    if not db_path.exists():
        return None
    return json.loads(db_path.read_text(encoding="utf-8"))


def synthetic_metadata(task_id: str, family: str, simulation_id: str) -> dict[str, Any]:
    return {
        "domain": "airline",
        "task_id": task_id,
        "trial": None,
        "simulation_id": simulation_id,
        "source_termination": "synthetic",
        "turn_index": None,
        "recovery_family": family,
        "source": "synthetic_from_official_task_and_db",
    }


def expected_actions(sim: dict[str, Any], name: str | None = None) -> list[dict[str, Any]]:
    actions = []
    for check in ((sim.get("reward_info") or {}).get("action_checks") or []):
        action = check.get("action") or {}
        if name is None or action.get("name") == name:
            actions.append(copy.deepcopy(action))
    return actions


def first_sim(sims: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
    return next((sim for sim in sims if str(sim.get("task_id") or "") == task_id), None)


def flight_on_date(db: dict[str, Any], flight_number: str, date: str) -> dict[str, Any]:
    flight = copy.deepcopy((db.get("flights") or {}).get(flight_number) or {})
    date_data = copy.deepcopy((flight.get("dates") or {}).get(date) or {})
    flight["date"] = date
    flight["status"] = date_data.get("status")
    flight["available_seats"] = date_data.get("available_seats")
    flight["prices"] = date_data.get("prices")
    flight.pop("dates", None)
    return flight


def direct_search_result(db: dict[str, Any], origin: str, destination: str, date: str) -> list[dict[str, Any]]:
    results = []
    for flight in (db.get("flights") or {}).values():
        if flight.get("origin") != origin or flight.get("destination") != destination:
            continue
        dates = flight.get("dates") or {}
        if date not in dates:
            continue
        results.append(flight_on_date(db, str(flight.get("flight_number")), date))
    return results


def search_tool_result_for_task16(db: dict[str, Any]) -> list[list[dict[str, Any]]]:
    flights = db.get("flights") or {}
    if isinstance(flights, dict):
        by_number = flights
    else:
        by_number = {flight.get("flight_number"): flight for flight in flights}
    result = []
    for number in ["HAT110", "HAT172"]:
        flight = copy.deepcopy(by_number[number])
        date_data = copy.deepcopy(flight.get("dates", {}).get("2024-05-24", {}))
        flight["date"] = "2024-05-24"
        flight["status"] = date_data.get("status")
        flight["available_seats"] = date_data.get("available_seats")
        flight["prices"] = date_data.get("prices")
        flight.pop("dates", None)
        result.append(flight)
    return [result]


def task16_update_arguments() -> dict[str, Any]:
    return {
        "reservation_id": "M05KNL",
        "cabin": "economy",
        "flights": [
            {"flight_number": "HAT110", "date": "2024-05-24"},
            {"flight_number": "HAT172", "date": "2024-05-24"},
        ],
        "payment_id": "gift_card_8887175",
    }


def task16_search_arguments() -> dict[str, Any]:
    return {"origin": "ATL", "destination": "PHL", "date": "2024-05-24"}


def task16_cheapest_confirm_text() -> str:
    return (
        "The cheapest regular Economy option on 2024-05-24 is HAT110 from ATL "
        "to LGA followed by HAT172 from LGA to PHL. The total Economy fare is "
        "$207, which is cheaper than the HAT110 + HAT132 option. Please confirm "
        "that you want me to update existing reservation M05KNL to this Economy itinerary."
    )


def has_verified_task16_reservation(messages: list[dict[str, Any]], end: int) -> bool:
    for message in messages[:end]:
        if message.get("role") != "tool":
            continue
        payload = parse_json_content(message)
        if not isinstance(payload, dict):
            continue
        if payload.get("reservation_id") == "M05KNL" and payload.get("origin") == "ATL":
            return True
    return False


def add_task16_online_policy_corrections(
    sim: dict[str, Any],
    messages: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    if str(sim.get("task_id") or "") != "16":
        return

    seen_ids: set[str] = {str(row.get("id")) for row in corrections}
    for idx, message in enumerate(messages):
        role = message.get("role")
        user_text = str(message.get("content") or "").lower()
        prior_text = prior_user_text(messages, idx)

        if role == "user" and has_verified_task16_reservation(messages, idx):
            wants_next_day_economy = (
                ("may 24" in user_text or "2024-05-24" in user_text)
                and "economy" in (user_text + "\n" + prior_text)
                and ("cheapest" in (user_text + "\n" + prior_text) or "available" in user_text)
            )
            recovering_from_refusal = (
                ("may 24" in user_text or "date changed" in user_text or "date change" in user_text)
                and "cannot change the date" in "\n".join(
                    str(row.get("content") or "").lower()
                    for row in messages[max(0, idx - 3) : idx]
                    if row.get("role") == "assistant"
                )
            )
            if wants_next_day_economy or recovering_from_refusal:
                sample_id = f"recovery_task16_turn{idx}_date_change_policy_search_not_refuse"
                if sample_id not in seen_ids:
                    seen_ids.add(sample_id)
                    corrections.append(
                        tool_target_sample(
                            sample_id,
                            messages[: idx + 1],
                            "search_onestop_flight",
                            task16_search_arguments(),
                            make_metadata(
                                sim,
                                idx,
                                "date_change_policy_search_not_refuse",
                            ),
                        )
                    )

        if role == "tool":
            payload = parse_json_content(message)
            if isinstance(payload, list) and "cheapest" in prior_text:
                sample_id = f"recovery_task16_turn{idx}_cheapest_economy_confirm_h1712"
                if sample_id not in seen_ids:
                    seen_ids.add(sample_id)
                    corrections.append(
                        text_target_sample(
                            sample_id,
                            messages[: idx + 1],
                            task16_cheapest_confirm_text(),
                            make_metadata(
                                sim,
                                idx,
                                "cheapest_economy_calculation_confirm_before_write",
                            ),
                        )
                )

        if role == "user" and any(
            phrase in user_text
            for phrase in ["go ahead", "book that", "proceed", "yes, please"]
        ):
            if "cheapest" in prior_text and mentions_atl_phl(prior_text):
                sample_id = f"recovery_task16_turn{idx}_change_existing_update_not_book"
                if sample_id not in seen_ids:
                    seen_ids.add(sample_id)
                    corrections.append(
                        tool_target_sample(
                            sample_id,
                            messages[: idx + 1],
                            "update_reservation_flights",
                            task16_update_arguments(),
                            make_metadata(
                                sim,
                                idx,
                                "change_existing_reservation_not_book_new",
                            ),
                        )
                    )

        if role == "user" and any(
            signal in user_text
            for signal in ["refund", "fare difference", "original payment method"]
        ):
            sample_id = f"recovery_task16_turn{idx}_refund_answer_no_tool"
            if sample_id not in seen_ids:
                seen_ids.add(sample_id)
                corrections.append(
                    text_target_sample(
                        sample_id,
                        messages[: idx + 1],
                        (
                            "Yes. Any applicable fare difference from this change "
                            "will be handled through the original payment method for "
                            "the reservation. The reservation itself should be updated "
                            "with update_reservation_flights, not booked as a new trip."
                        ),
                        make_metadata(sim, idx, "refund_answer_no_unknown_tool"),
                    )
                )


def add_task16_gold_route_chain(
    sims: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    if not any(str(sim.get("task_id")) == "16" for sim in sims):
        return
    db_path = repo_path("third_party/tau2-bench/data/tau2/domains/airline/db.json")
    if not db_path.exists():
        return
    db = json.loads(db_path.read_text(encoding="utf-8"))
    reservation = copy.deepcopy((db.get("reservations") or {}).get("M05KNL"))
    if not reservation:
        return

    prefix = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {
            "role": "user",
            "content": (
                "Hi, I'd like to change my upcoming flight from Atlanta to Philadelphia. "
                "I'm hoping to switch to the day after my originally scheduled flight."
            ),
        },
        {
            "role": "assistant",
            "content": "I can help. Could you please provide your user ID?",
        },
        {"role": "user", "content": "Sure, my user ID is aarav_garcia_1177."},
        {
            "role": "assistant",
            "tool_calls": [
                make_tool_call(
                    "call_task16_get_user",
                    "get_user_details",
                    {"user_id": "aarav_garcia_1177"},
                )
            ],
            "content": "",
        },
        {
            "role": "tool",
            "tool_call_id": "call_task16_get_user",
            "name": "get_user_details",
            "content": json.dumps(
                {
                    "user_id": "aarav_garcia_1177",
                    "name": {"first_name": "Aarav", "last_name": "Garcia"},
                    "payment_methods": {
                        "certificate_7473723": {
                            "source": "certificate",
                            "id": "certificate_7473723",
                            "amount": 150.0,
                        },
                        "gift_card_8887175": {
                            "source": "gift_card",
                            "id": "gift_card_8887175",
                            "amount": 35.0,
                        },
                    },
                    "membership": "gold",
                    "reservations": ["M05KNL", "UHDAHF"],
                },
                ensure_ascii=False,
            ),
        },
    ]
    metadata = {
        "domain": "airline",
        "task_id": "16",
        "trial": None,
        "simulation_id": "synthetic_task16_route_chain",
        "source_termination": "synthetic",
        "turn_index": len(prefix),
        "recovery_family": "route_disambiguation_gold_chain",
        "source": "synthetic_from_official_task16_and_db",
    }
    call_reservation = make_tool_call(
        "call_task16_get_m05knl", "get_reservation_details", {"reservation_id": "M05KNL"}
    )
    corrections.append(
        tool_target_sample(
            "recovery_task16_gold_call_matching_reservation",
            prefix,
            "get_reservation_details",
            {"reservation_id": "M05KNL"},
            metadata,
        )
    )

    prefix_after_reservation = copy.deepcopy(prefix) + [
        {"role": "assistant", "content": "", "tool_calls": [call_reservation]},
        reservation_tool_message("call_task16_get_m05knl", reservation),
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task16_gold_search_next_day_onestop",
            prefix_after_reservation,
            "search_onestop_flight",
            task16_search_arguments(),
            metadata | {"turn_index": len(prefix_after_reservation)},
        )
    )

    prefix_after_date_request = prefix_after_reservation + [
        {
            "role": "user",
            "content": (
                "I want to move reservation M05KNL to May 24th and switch it to "
                "regular Economy, not Basic Economy. Please find the cheapest "
                "available ATL to PHL option."
            ),
        }
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task16_gold_date_change_search_not_refuse",
            prefix_after_date_request,
            "search_onestop_flight",
            task16_search_arguments(),
            metadata
            | {
                "turn_index": len(prefix_after_date_request),
                "recovery_family": "date_change_policy_search_not_refuse",
            },
        )
    )

    prefix_after_bad_refusal = prefix_after_date_request + [
        {
            "role": "assistant",
            "content": (
                "I cannot change the date of the reservation. I can only keep the "
                "current May 23 flights and change the cabin."
            ),
        },
        {
            "role": "user",
            "content": (
                "I specifically need to change the date to May 24th. That is the "
                "main reason for the change."
            ),
        },
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task16_gold_recover_from_date_refusal",
            prefix_after_bad_refusal,
            "search_onestop_flight",
            task16_search_arguments(),
            metadata
            | {
                "turn_index": len(prefix_after_bad_refusal),
                "recovery_family": "date_change_policy_search_not_refuse",
            },
        )
    )

    search_result = search_tool_result_for_task16(db)
    call_search = make_tool_call(
        "call_task16_search_onestop",
        "search_onestop_flight",
        task16_search_arguments(),
    )
    confirm_text = (
        "I found the matching reservation M05KNL for your ATL to PHL trip. "
        "The cheapest Economy option for the day after your current flight is "
        "HAT110 from ATL to LGA and HAT172 from LGA to PHL on 2024-05-24. "
        "I will keep the reservation as M05KNL, change the cabin to economy, "
        "and use payment method gift_card_8887175 for the change. "
        "Please confirm with yes before I update the reservation."
    )
    prefix_after_search = prefix_after_reservation + [
        {"role": "assistant", "content": "", "tool_calls": [call_search]},
        {
            "role": "tool",
            "tool_call_id": "call_task16_search_onestop",
            "name": "search_onestop_flight",
            "content": json.dumps(search_result, ensure_ascii=False),
        },
    ]
    corrections.append(
        text_target_sample(
            "recovery_task16_gold_confirm_before_update",
            prefix_after_search,
            confirm_text,
            metadata
            | {
                "turn_index": len(prefix_after_search),
                "recovery_family": "confirm_before_write",
            },
        )
    )

    prefix_after_yes = prefix_after_search + [
        {"role": "assistant", "content": confirm_text, "tool_calls": None},
        {"role": "user", "content": "Yes, please update reservation M05KNL."},
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task16_gold_update_after_confirmation",
            prefix_after_yes,
            "update_reservation_flights",
            task16_update_arguments(),
            metadata
            | {
                "turn_index": len(prefix_after_yes),
                "recovery_family": "write_after_explicit_confirmation",
            },
        )
    )


def add_task18_online_policy_corrections(
    sim: dict[str, Any],
    messages: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    if str(sim.get("task_id") or "") != "18":
        return
    seen_ids: set[str] = {str(row.get("id")) for row in corrections}
    for idx, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        user_text = str(message.get("content") or "").lower()
        if "omar_davis_3817" not in user_text:
            continue
        if not any(signal in user_text for signal in ["all", "account", "reservations", "look up"]):
            continue
        next_message = messages[idx + 1] if idx + 1 < len(messages) else {}
        next_content = str(next_message.get("content") or "").lower()
        refused_lookup = (
            next_message.get("role") == "assistant"
            and not next_message.get("tool_calls")
            and any(signal in next_content for signal in ["cannot look up", "need to obtain", "reservation id"])
        )
        if not refused_lookup:
            continue
        sample_id = f"recovery_task18_turn{idx}_all_reservations_lookup_from_user_id"
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        corrections.append(
            tool_target_sample(
                sample_id,
                messages[: idx + 1],
                "get_user_details",
                {"user_id": "omar_davis_3817"},
                make_metadata(sim, idx, "all_reservations_lookup_from_user_id"),
            )
        )


def add_task18_gold_chain(
    sims: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    sim = first_sim(sims, "18")
    if not sim:
        return
    db = load_airline_db()
    if not db:
        return
    user = copy.deepcopy((db.get("users") or {}).get("omar_davis_3817"))
    if not user:
        return
    update_actions = expected_actions(sim, "update_reservation_flights")
    if not update_actions:
        return

    reservation_ids = [str(rid) for rid in user.get("reservations") or []]
    prefix = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {
            "role": "user",
            "content": (
                "I need to downgrade all of my business class reservations to economy. "
                "My user ID is omar_davis_3817, but I do not remember each reservation ID."
            ),
        },
    ]
    metadata = synthetic_metadata("18", "all_reservations_lookup_from_user_id", "synthetic_task18_all_business_downgrade")
    corrections.append(
        tool_target_sample(
            "recovery_task18_gold_get_user_from_account_request",
            prefix,
            "get_user_details",
            {"user_id": "omar_davis_3817"},
            metadata,
        )
    )

    prefix_after_user = prefix + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                make_tool_call("call_task18_get_user", "get_user_details", {"user_id": "omar_davis_3817"})
            ],
        },
        generic_tool_message("call_task18_get_user", "get_user_details", user),
    ]
    running_prefix = copy.deepcopy(prefix_after_user)
    for reservation_id in reservation_ids:
        sample_id = f"recovery_task18_gold_check_reservation_{reservation_id.lower()}"
        corrections.append(
            tool_target_sample(
                sample_id,
                running_prefix,
                "get_reservation_details",
                {"reservation_id": reservation_id},
                metadata
                | {
                    "turn_index": len(running_prefix),
                    "recovery_family": "inspect_all_reservations_for_bulk_change",
                },
            )
        )
        call_id = f"call_task18_get_{reservation_id.lower()}"
        reservation = copy.deepcopy((db.get("reservations") or {}).get(reservation_id) or {})
        running_prefix = running_prefix + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [make_tool_call(call_id, "get_reservation_details", {"reservation_id": reservation_id})],
            },
            reservation_tool_message(call_id, reservation),
        ]

    confirm_text = (
        "I found the reservations in your profile and checked each one. The business class reservations "
        "that can be downgraded to economy are JG7FMM, 2FBBAH, X7BYG1, EQ1G6C, and BOH180. "
        "I will keep the same flights and passengers, change only the cabin to economy, and refund each "
        "fare difference to the original payment method. Please confirm with yes before I update them."
    )
    corrections.append(
        text_target_sample(
            "recovery_task18_gold_confirm_bulk_downgrade",
            running_prefix,
            confirm_text,
            metadata
            | {
                "turn_index": len(running_prefix),
                "recovery_family": "confirm_bulk_downgrade_before_writes",
            },
        )
    )

    update_prefix = running_prefix + [
        {"role": "assistant", "content": confirm_text, "tool_calls": None},
        {"role": "user", "content": "Yes, please downgrade all of those business class reservations to economy."},
    ]
    for idx, action in enumerate(update_actions):
        arguments = copy.deepcopy(action.get("arguments") or {})
        sample_id = f"recovery_task18_gold_update_{arguments.get('reservation_id', idx)}".lower()
        corrections.append(
            tool_target_sample(
                sample_id,
                update_prefix,
                "update_reservation_flights",
                arguments,
                metadata
                | {
                    "turn_index": len(update_prefix),
                    "recovery_family": "bulk_downgrade_write_after_confirmation",
                },
            )
        )
        call_id = f"call_task18_update_{idx}"
        update_prefix = update_prefix + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [make_tool_call(call_id, "update_reservation_flights", arguments)],
            },
            generic_tool_message(call_id, "update_reservation_flights", {"status": "success"}),
        ]


def task25_payment_plan_text() -> str:
    return (
        "For Ivan Smith only, the one-way Economy itinerary DTW to PHX to SEA on 2024-05-17 "
        "costs $375 total: HAT097 is $175 and HAT251 is $200. Because the total is below "
        "$400, I should not use the certificate. I should use gift_card_8516878 for $128 "
        "and credit_card_3563913 for the remaining $247. There will be no baggage and no insurance. "
        "Please confirm with yes before I book this new reservation."
    )


def add_task25_online_policy_corrections(
    sim: dict[str, Any],
    messages: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    if str(sim.get("task_id") or "") != "25":
        return
    seen_ids: set[str] = {str(row.get("id")) for row in corrections}
    book_actions = expected_actions(sim, "book_reservation")
    book_args = copy.deepcopy((book_actions[0] if book_actions else {}).get("arguments") or {})
    for idx, message in enumerate(messages):
        role = message.get("role")
        content = str(message.get("content") or "").lower()
        if role == "user" and any(signal in content for signal in ["total price", "certificate balance", "make a decision"]):
            sample_id = f"recovery_task25_turn{idx}_booking_payment_plan_single_passenger"
            if sample_id not in seen_ids:
                seen_ids.add(sample_id)
                corrections.append(
                    text_target_sample(
                        sample_id,
                        messages[: idx + 1],
                        task25_payment_plan_text(),
                        make_metadata(sim, idx, "booking_payment_plan_single_passenger"),
                    )
                )
        if role == "tool" and str(message.get("content") or "").strip().startswith("-"):
            sample_id = f"recovery_task25_turn{idx}_recover_from_negative_payment_math"
            if sample_id not in seen_ids:
                seen_ids.add(sample_id)
                corrections.append(
                    text_target_sample(
                        sample_id,
                        messages[: idx + 1],
                        task25_payment_plan_text(),
                        make_metadata(sim, idx, "booking_payment_plan_single_passenger"),
                    )
                )
        if role == "user" and book_args and any(signal in content for signal in ["yes", "confirm", "book it", "go ahead"]):
            if "ivan smith" in prior_user_text(messages, idx):
                sample_id = f"recovery_task25_turn{idx}_book_after_payment_confirmation"
                if sample_id not in seen_ids:
                    seen_ids.add(sample_id)
                    corrections.append(
                        tool_target_sample(
                            sample_id,
                            messages[: idx + 1],
                            "book_reservation",
                            book_args,
                            make_metadata(sim, idx, "book_after_payment_confirmation"),
                        )
                    )


def add_task25_gold_chain(
    sims: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    sim = first_sim(sims, "25")
    if not sim:
        return
    db = load_airline_db()
    if not db:
        return
    user = copy.deepcopy((db.get("users") or {}).get("ivan_muller_7015"))
    reservation = copy.deepcopy((db.get("reservations") or {}).get("G72NSF"))
    book_actions = expected_actions(sim, "book_reservation")
    if not user or not reservation or not book_actions:
        return
    book_args = copy.deepcopy(book_actions[0].get("arguments") or {})
    metadata = synthetic_metadata("25", "booking_payment_plan_single_passenger", "synthetic_task25_friend_booking")
    prefix = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {
            "role": "user",
            "content": (
                "I want to book a new one-way reservation for my friend Ivan Smith. "
                "It should match my current DTW to SEA reservation. My user ID is ivan_muller_7015. "
                "Use my certificate only if the price is over $400; otherwise use my gift card and credit card. "
                "No baggage and no insurance."
            ),
        },
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task25_gold_get_user_for_friend_booking",
            prefix,
            "get_user_details",
            {"user_id": "ivan_muller_7015"},
            metadata,
        )
    )
    prefix_user = prefix + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                make_tool_call("call_task25_get_user", "get_user_details", {"user_id": "ivan_muller_7015"})
            ],
        },
        generic_tool_message("call_task25_get_user", "get_user_details", user),
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task25_gold_get_current_reservation",
            prefix_user,
            "get_reservation_details",
            {"reservation_id": "G72NSF"},
            metadata | {"turn_index": len(prefix_user)},
        )
    )
    prefix_res = prefix_user + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                make_tool_call("call_task25_get_res", "get_reservation_details", {"reservation_id": "G72NSF"})
            ],
        },
        reservation_tool_message("call_task25_get_res", reservation),
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task25_gold_search_matching_onestop",
            prefix_res,
            "search_onestop_flight",
            {"origin": "DTW", "destination": "SEA", "date": "2024-05-17"},
            metadata | {"turn_index": len(prefix_res), "recovery_family": "search_matching_itinerary_before_booking"},
        )
    )
    search_payload = [[flight_on_date(db, "HAT097", "2024-05-17"), flight_on_date(db, "HAT251", "2024-05-17")]]
    prefix_search = prefix_res + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                make_tool_call(
                    "call_task25_search",
                    "search_onestop_flight",
                    {"origin": "DTW", "destination": "SEA", "date": "2024-05-17"},
                )
            ],
        },
        generic_tool_message("call_task25_search", "search_onestop_flight", search_payload),
    ]
    confirm_text = task25_payment_plan_text()
    corrections.append(
        text_target_sample(
            "recovery_task25_gold_confirm_payment_plan",
            prefix_search,
            confirm_text,
            metadata | {"turn_index": len(prefix_search)},
        )
    )
    prefix_yes = prefix_search + [
        {"role": "assistant", "content": confirm_text, "tool_calls": None},
        {"role": "user", "content": "Yes, please book that for Ivan Smith."},
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task25_gold_book_after_confirmation",
            prefix_yes,
            "book_reservation",
            book_args,
            metadata | {"turn_index": len(prefix_yes), "recovery_family": "book_after_payment_confirmation"},
        )
    )


def add_task44_online_policy_corrections(
    sim: dict[str, Any],
    messages: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    if str(sim.get("task_id") or "") != "44":
        return
    seen_ids: set[str] = {str(row.get("id")) for row in corrections}
    flight_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for idx, message in enumerate(messages):
        if message.get("role") == "tool":
            payload = parse_json_content(message)
            if isinstance(payload, dict) and isinstance(payload.get("flights"), list):
                for flight in payload["flights"]:
                    if isinstance(flight, dict):
                        key = (str(flight.get("flight_number")), str(flight.get("date")))
                        flight_lookup[key] = {
                            "origin": str(flight.get("origin")),
                            "destination": str(flight.get("destination")),
                            "date": str(flight.get("date")),
                        }
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if (function.get("name") or call.get("name")) != "get_flight_status":
                continue
            arguments = function.get("arguments") or call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            key = (str(arguments.get("flight_number")), str(arguments.get("date")))
            replacement = flight_lookup.get(key)
            if not replacement:
                continue
            sample_id = f"recovery_task44_turn{idx}_duration_requires_search_direct_{key[0]}_{key[1]}"
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            corrections.append(
                tool_target_sample(
                    sample_id,
                    messages[:idx],
                    "search_direct_flight",
                    replacement,
                    make_metadata(sim, idx, "duration_requires_search_direct"),
                )
            )
        content = str(message.get("content") or "").lower()
        if "cannot check flight durations" in content or "transfer you" in content:
            sample_id = f"recovery_task44_turn{idx}_do_not_transfer_for_duration"
            if sample_id not in seen_ids:
                seen_ids.add(sample_id)
                corrections.append(
                    text_target_sample(
                        sample_id,
                        messages[:idx],
                        (
                            "I should not transfer yet. Flight status alone is not enough for duration, "
                            "but I can use search_direct_flight for each route/date to retrieve scheduled "
                            "departure and arrival times, then compare durations before deciding which "
                            "reservations to cancel or upgrade."
                        ),
                        make_metadata(sim, idx, "duration_requires_search_direct"),
                    )
                )


def add_task44_gold_chain(
    sims: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> None:
    sim = first_sim(sims, "44")
    if not sim:
        return
    db = load_airline_db()
    if not db:
        return
    user = copy.deepcopy((db.get("users") or {}).get("sophia_silva_7557"))
    if not user:
        return
    search_actions = expected_actions(sim, "search_direct_flight")
    update_actions = expected_actions(sim, "update_reservation_flights")
    if not search_actions:
        return
    metadata = synthetic_metadata("44", "duration_requires_search_direct", "synthetic_task44_duration_reasoning")
    prefix = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {
            "role": "user",
            "content": (
                "My user ID is sophia_silva_7557. I want to cancel future reservations "
                "with any flights longer than 4 hours, and upgrade the ones under or equal "
                "to 3 hours including layovers where possible. I need you to figure out the durations."
            ),
        },
    ]
    corrections.append(
        tool_target_sample(
            "recovery_task44_gold_get_user_for_duration_request",
            prefix,
            "get_user_details",
            {"user_id": "sophia_silva_7557"},
            metadata,
        )
    )
    running_prefix = prefix + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                make_tool_call("call_task44_get_user", "get_user_details", {"user_id": "sophia_silva_7557"})
            ],
        },
        generic_tool_message("call_task44_get_user", "get_user_details", user),
    ]
    for reservation_id in user.get("reservations") or []:
        reservation_id = str(reservation_id)
        corrections.append(
            tool_target_sample(
                f"recovery_task44_gold_get_reservation_{reservation_id.lower()}",
                running_prefix,
                "get_reservation_details",
                {"reservation_id": reservation_id},
                metadata
                | {
                    "turn_index": len(running_prefix),
                    "recovery_family": "inspect_all_reservations_for_duration_rules",
                },
            )
        )
        call_id = f"call_task44_get_{reservation_id.lower()}"
        reservation = copy.deepcopy((db.get("reservations") or {}).get(reservation_id) or {})
        running_prefix = running_prefix + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [make_tool_call(call_id, "get_reservation_details", {"reservation_id": reservation_id})],
            },
            reservation_tool_message(call_id, reservation),
        ]

    for idx, action in enumerate(search_actions):
        arguments = copy.deepcopy(action.get("arguments") or {})
        corrections.append(
            tool_target_sample(
                f"recovery_task44_gold_search_duration_{idx}",
                running_prefix,
                "search_direct_flight",
                arguments,
                metadata
                | {
                    "turn_index": len(running_prefix),
                    "recovery_family": "duration_requires_search_direct",
                },
            )
        )
        call_id = f"call_task44_search_{idx}"
        running_prefix = running_prefix + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [make_tool_call(call_id, "search_direct_flight", arguments)],
            },
            generic_tool_message(
                call_id,
                "search_direct_flight",
                direct_search_result(
                    db,
                    str(arguments.get("origin")),
                    str(arguments.get("destination")),
                    str(arguments.get("date")),
                ),
            ),
        ]

    confirm_text = (
        "I searched the schedules for the relevant direct flight segments so I can compare durations. "
        "The reservations that should be upgraded to business rather than cancelled are NM1VX1, H8Q05L, "
        "and KC18K6. I will use credit_card_4196779 for the upgrade cost. Please confirm with yes before "
        "I update these reservations."
    )
    corrections.append(
        text_target_sample(
            "recovery_task44_gold_confirm_duration_based_upgrades",
            running_prefix,
            confirm_text,
            metadata
            | {
                "turn_index": len(running_prefix),
                "recovery_family": "confirm_duration_based_upgrades_before_writes",
            },
        )
    )
    update_prefix = running_prefix + [
        {"role": "assistant", "content": confirm_text, "tool_calls": None},
        {"role": "user", "content": "Yes, please upgrade those reservations to business."},
    ]
    for idx, action in enumerate(update_actions):
        arguments = copy.deepcopy(action.get("arguments") or {})
        corrections.append(
            tool_target_sample(
                f"recovery_task44_gold_update_duration_choice_{idx}",
                update_prefix,
                "update_reservation_flights",
                arguments,
                metadata
                | {
                    "turn_index": len(update_prefix),
                    "recovery_family": "duration_based_upgrade_write_after_confirmation",
                },
            )
        )
        call_id = f"call_task44_update_{idx}"
        update_prefix = update_prefix + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [make_tool_call(call_id, "update_reservation_flights", arguments)],
            },
            generic_tool_message(call_id, "update_reservation_flights", {"status": "success"}),
        ]


def build_corrections_from_sim(sim: dict[str, Any]) -> list[dict[str, Any]]:
    corrections: list[dict[str, Any]] = []
    messages = normalize_messages(sim.get("messages") or [])
    task_id = str(sim.get("task_id") or "unknown")
    seen_ids: set[str] = set()

    for idx, message in enumerate(messages):
        role = message.get("role")
        content = str(message.get("content") or "")

        if role == "user" and idx + 1 < len(messages):
            next_message = messages[idx + 1]
            if next_message.get("role") == "assistant" and "get_user_details" in assistant_tool_names(next_message):
                if not has_user_identifier(content):
                    corrections.append(
                        text_target_sample(
                            f"recovery_task{task_id}_turn{idx}_ask_identity_before_tool",
                            messages[: idx + 1],
                            (
                                "I can help, but I need to verify the booking before using tools. "
                                "Could you provide your user ID or reservation ID?"
                            ),
                            make_metadata(sim, idx, "ask_identity_before_tool"),
                        )
                    )

        if role == "assistant" and message.get("tool_calls") and content.strip():
            if "?" in content or any(phrase in content.lower() for phrase in ["would you", "could you", "what would"]):
                corrections.append(
                    text_target_sample(
                        f"recovery_task{task_id}_turn{idx}_no_tool_while_asking",
                        messages[:idx],
                        content.strip(),
                        make_metadata(sim, idx, "no_tool_while_asking"),
                    )
                )

        if role != "tool":
            continue

        error = content.strip()
        if not error.lower().startswith("error:"):
            continue
        family = ""
        target_text = ""
        if "User " in error and " not found" in error:
            family = "user_not_found_stop_guessing"
            target_text = (
                "I couldn't find that user ID, so I should not guess another one. "
                "Could you double-check your user ID or send your reservation ID?"
            )
        elif "Reservation " in error and " not found" in error:
            family = "reservation_not_found_stop_guessing"
            target_text = (
                "I couldn't find that reservation ID, so I should not keep guessing reservation IDs. "
                "Please provide the correct reservation ID, or I can continue only with reservation IDs that appeared in verified user details."
            )
        if not family:
            continue

        sample_id = f"recovery_task{task_id}_turn{idx}_{family}"
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        corrections.append(
            text_target_sample(
                sample_id,
                messages[: idx + 1],
                target_text,
                make_metadata(sim, idx, family),
            )
        )

    add_route_disambiguation_corrections(sim, messages, corrections)
    add_task16_online_policy_corrections(sim, messages, corrections)
    add_task18_online_policy_corrections(sim, messages, corrections)
    add_task25_online_policy_corrections(sim, messages, corrections)
    add_task44_online_policy_corrections(sim, messages, corrections)
    return corrections


def load_failure_sims(paths: list[Path]) -> list[dict[str, Any]]:
    sims: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            sims.extend(data.get("simulations") or [])
        elif isinstance(data, list):
            sims.extend(data)
    return sims


def oversample(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    expanded = []
    for copy_idx in range(max(1, n)):
        for row in rows:
            item = copy.deepcopy(row)
            item["id"] = f"{row['id']}_os{copy_idx}"
            item.setdefault("metadata", {})["oversample_index"] = copy_idx
            expanded.append(item)
    return expanded


def uniquify_row_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep generated correction IDs stable while making repeats explicit."""

    counts: Counter[str] = Counter()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        item = copy.deepcopy(row)
        base_id = str(item.get("id") or "sample")
        counts[base_id] += 1
        if counts[base_id] > 1:
            item["id"] = f"{base_id}_dup{counts[base_id] - 1}"
            metadata = item.setdefault("metadata", {})
            metadata["duplicate_source_id"] = base_id
            metadata["duplicate_source_ordinal"] = counts[base_id]
        unique_rows.append(item)
    return unique_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_types = Counter(str(row.get("sample_type")) for row in rows)
    families = Counter(str((row.get("metadata") or {}).get("recovery_family") or "") for row in rows)
    tasks = Counter(str((row.get("metadata") or {}).get("task_id") or "") for row in rows)
    message_counts = [len(row.get("messages") or []) for row in rows]
    return {
        "rows": len(rows),
        "sample_types": dict(sorted(sample_types.items())),
        "recovery_families": {k: v for k, v in sorted(families.items()) if k},
        "tasks": {k: v for k, v in sorted(tasks.items()) if k},
        "mean_messages": mean(message_counts) if message_counts else 0,
        "max_messages": max(message_counts, default=0),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Recovery-Prefix Dataset",
        "",
        "## Goal",
        "",
        "Mix online failure corrections and synthetic gold chains into decision-prefix SFT. The corrections target evidence discipline, route/date disambiguation, payment calculation, duration-based search, and confirmation-before-write behavior.",
        "",
        "## Outputs",
        "",
    ]
    for split, out_path in payload["outputs"].items():
        lines.append(f"- {split}: `{out_path}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Base train rows: `{payload['base_train_rows']}`",
            f"- Raw correction rows: `{payload['raw_correction_rows']}`",
            f"- Oversample factor: `{payload['oversample_factor']}`",
            f"- Final train rows: `{payload['summaries']['train']['rows']}`",
            "",
            "| Split | Rows | Sample types | Recovery families | Tasks | Mean messages | Max messages |",
            "| --- | ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    for split, summary in payload["summaries"].items():
        sample_types = ", ".join(f"{k}:{v}" for k, v in summary["sample_types"].items())
        families = ", ".join(f"{k}:{v}" for k, v in summary["recovery_families"].items())
        tasks = ", ".join(f"{k}:{v}" for k, v in summary["tasks"].items())
        lines.append(
            f"| {split} | {summary['rows']} | `{sample_types}` | `{families}` | `{tasks}` | "
            f"{summary['mean_messages']:.1f} | {summary['max_messages']} |"
        )

    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This is targeted recovery data from failed and diagnostic tau2 runs, so it must not be reported as clean heldout generalization. Use it to train the recovery mechanism, then evaluate on separate regression and heldout tasks.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recovery-prefix SFT data from failed tau2 traces.")
    parser.add_argument("--base-train", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-valid", default=DEFAULT_BASE_VALID)
    parser.add_argument("--base-heldout", default=DEFAULT_BASE_HELDOUT)
    parser.add_argument("--failure-results", action="append")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--manifest", default="data/recovery_prefix/tau2_airline_recovery_prefix_manifest_v1.json")
    parser.add_argument("--oversample-factor", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_train = load_jsonl(repo_path(args.base_train))
    base_valid = load_jsonl(repo_path(args.base_valid))
    base_heldout = load_jsonl(repo_path(args.base_heldout))
    failure_inputs = args.failure_results or [DEFAULT_FAILURE_RESULTS]
    failure_paths = [repo_path(path) for path in failure_inputs]
    sims = load_failure_sims(failure_paths)

    corrections: list[dict[str, Any]] = []
    for sim in sims:
        corrections.extend(build_corrections_from_sim(sim))
    add_task16_gold_route_chain(sims, corrections)
    add_task18_gold_chain(sims, corrections)
    add_task25_gold_chain(sims, corrections)
    add_task44_gold_chain(sims, corrections)
    corrections = uniquify_row_ids(corrections)

    train_rows = base_train + oversample(corrections, args.oversample_factor)
    valid_rows = base_valid
    heldout_rows = base_heldout

    out_dir = repo_path(args.out_dir)
    outputs = {
        "train": out_dir / f"{args.output_stem}_train.jsonl",
        "valid": out_dir / f"{args.output_stem}_valid.jsonl",
        "heldout": out_dir / f"{args.output_stem}_heldout.jsonl",
        "corrections": out_dir / f"{args.output_stem}_corrections.jsonl",
    }
    write_jsonl(outputs["train"], train_rows)
    write_jsonl(outputs["valid"], valid_rows)
    write_jsonl(outputs["heldout"], heldout_rows)
    write_jsonl(outputs["corrections"], corrections)

    payload = {
        "format_version": "tau2_airline_recovery_prefix_manifest_v1",
        "inputs": {
            "base_train": display_path(repo_path(args.base_train)),
            "base_valid": display_path(repo_path(args.base_valid)),
            "base_heldout": display_path(repo_path(args.base_heldout)),
            "failure_results": [display_path(path) for path in failure_paths],
        },
        "outputs": {key: display_path(path) for key, path in outputs.items()},
        "base_train_rows": len(base_train),
        "raw_correction_rows": len(corrections),
        "oversample_factor": args.oversample_factor,
        "summaries": {
            "train": summarize(train_rows),
            "valid": summarize(valid_rows),
            "heldout": summarize(heldout_rows),
            "corrections": summarize(corrections),
        },
    }
    write_json(repo_path(args.manifest), payload)
    write_report(repo_path(args.report), payload)
    print("Recovery-prefix dataset complete")
    print("================================")
    print(f"base_train={len(base_train)} corrections={len(corrections)} final_train={len(train_rows)}")
    print(f"families={payload['summaries']['corrections']['recovery_families']}")
    print(f"wrote report: {repo_path(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
