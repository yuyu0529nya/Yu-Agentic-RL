from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_single_tool_protocol_dataset import convert_row  # noqa: E402
from check_sft_render_mask import render_with_spans  # noqa: E402


def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def row_with_target(tool_calls: list[dict]) -> dict:
    return {
        "id": "action_prefix_slot_grounded_v3_train_sample",
        "format_version": "tau2_airline_action_prefix_slot_grounded_v3",
        "messages": [
            {"role": "user", "content": "My user id is sara_doe_496."},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": tool_calls,
            },
        ],
        "loss_mask": [False, True],
        "loss_policy": {
            "assistant_content": True,
            "assistant_tool_calls": True,
            "user": False,
            "tool": False,
        },
        "metadata": {
            "task_id": "1",
            "target_tool_names": [call["function"]["name"] for call in tool_calls],
            "target_tool_call_count": len(tool_calls),
            "target_slot_grounding": {
                "is_grounded": True,
                "issue_count": 0,
                "issues": [],
            },
        },
    }


def test_single_tool_row_converts_to_protocol_only_target() -> None:
    row = row_with_target([tool_call("get_user_details", {"user_id": "sara_doe_496"})])

    converted, rejected = convert_row(row, "train")

    assert rejected is None
    assert converted is not None
    assert converted["sample_type"] == "single_tool_protocol"
    assert converted["messages"][-1]["content"] == ""
    assert len(converted["messages"][-1]["tool_calls"]) == 1
    assert converted["loss_policy"]["assistant_tool_call_wrappers"] is True
    assert converted["loss_policy"]["assistant_content"] is False
    assert converted["metadata"]["target_slot_grounding"]["is_grounded"] is True

    rendered = render_with_spans(converted)
    assert "<tool_call>" in rendered.text
    assert "</tool_call>" in rendered.text
    assert len(rendered.target_spans) == 3


def test_multi_tool_target_is_rejected() -> None:
    row = row_with_target(
        [
            tool_call("get_reservation_details", {"reservation_id": "Q69X3R"}, "call_1"),
            tool_call("get_reservation_details", {"reservation_id": "MZDDS4"}, "call_2"),
        ]
    )

    converted, rejected = convert_row(row, "train")

    assert converted is None
    assert rejected is not None
    assert rejected["metadata"]["single_tool_protocol_reject_reason"] == "target_tool_call_count_2"
