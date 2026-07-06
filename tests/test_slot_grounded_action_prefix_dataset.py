from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_slot_grounded_action_prefix_dataset import (  # noqa: E402
    split_clean_rejected,
    validate_target_grounding,
)


def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def row_with_messages(messages: list[dict]) -> dict:
    return {
        "id": "sample",
        "messages": messages,
        "metadata": {
            "task_id": "1",
            "target_tool_names": [
                call["function"]["name"]
                for call in messages[-1].get("tool_calls", [])
            ],
        },
    }


def test_user_id_from_user_message_is_grounded() -> None:
    row = row_with_messages(
        [
            {"role": "user", "content": "My user id is sara_doe_496."},
            {
                "role": "assistant",
                "content": "Let me check your profile.",
                "tool_calls": [tool_call("get_user_details", {"user_id": "sara_doe_496"})],
            },
        ]
    )

    result = validate_target_grounding(row)

    assert result["is_grounded"] is True
    assert result["issue_count"] == 0


def test_reservation_id_from_tool_result_is_grounded() -> None:
    row = row_with_messages(
        [
            {"role": "user", "content": "My user id is sara_doe_496."},
            {
                "role": "assistant",
                "content": "Let me check your profile.",
                "tool_calls": [tool_call("get_user_details", {"user_id": "sara_doe_496"})],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_user_details",
                "content": '{"user_id":"sara_doe_496","reservations":["Q69X3R"]}',
            },
            {
                "role": "assistant",
                "content": "I found the reservation. Let me inspect it.",
                "tool_calls": [tool_call("get_reservation_details", {"reservation_id": "Q69X3R"}, "call_2")],
            },
        ]
    )

    result = validate_target_grounding(row)

    assert result["is_grounded"] is True
    assert result["issue_count"] == 0


def test_hallucinated_reservation_id_is_rejected() -> None:
    row = row_with_messages(
        [
            {"role": "user", "content": "I need help with a trip, but I do not know the reservation id."},
            {
                "role": "assistant",
                "content": "Let me inspect the reservation.",
                "tool_calls": [tool_call("get_reservation_details", {"reservation_id": "Q69X3R"})],
            },
        ]
    )

    clean, rejected = split_clean_rejected([row])
    result = rejected[0]["metadata"]["target_slot_grounding"]

    assert clean == []
    assert len(rejected) == 1
    assert result["is_grounded"] is False
    assert result["issues"][0]["component_name"] == "ungrounded_reservation_id"
