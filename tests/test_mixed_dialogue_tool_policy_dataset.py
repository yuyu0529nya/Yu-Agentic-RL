from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_mixed_dialogue_tool_policy_dataset import build_split, validate_and_finalize  # noqa: E402
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


def source_row(messages: list[dict], loss_mask: list[bool] | None = None) -> dict:
    if loss_mask is None:
        loss_mask = [message.get("role") == "assistant" for message in messages]
    return {
        "id": "sft_success_task1_trial0",
        "format_version": "tau2_airline_sft_training_v1",
        "messages": messages,
        "loss_mask": loss_mask,
        "metadata": {
            "domain": "airline",
            "task_id": "1",
            "trial": 0,
            "reward": 1.0,
            "process_score": 1.0,
        },
    }


def finalize(samples: list[dict], rejected: list[dict]) -> tuple[list[dict], list[dict]]:
    clean, rejected, errors = validate_and_finalize(samples, rejected, tokenizer=None, max_sample_tokens=2048)
    assert errors == []
    return clean, rejected


def test_text_turn_keeps_only_assistant_content_loss() -> None:
    row = source_row(
        [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ],
        [False, True],
    )

    raw, rejected = build_split([row], "train", include_protocol_variants=True)
    clean, rejected = finalize(raw, rejected)

    assert rejected == []
    assert len(clean) == 1
    sample = clean[0]
    assert sample["sample_type"] == "mixed_policy_text"
    assert sample["metadata"]["target_action"] == "assistant_text"
    assert sample["loss_policy"]["assistant_content"] is True
    assert sample["loss_policy"]["assistant_tool_calls"] is False
    rendered = render_with_spans(sample)
    assert "Hi! How can I help?" in rendered.marked_text
    assert "<tool_call>" not in rendered.text


def test_single_tool_turn_has_mixed_and_protocol_variants() -> None:
    row = source_row(
        [
            {"role": "user", "content": "My user id is sara_doe_496."},
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [tool_call("get_user_details", {"user_id": "sara_doe_496"})],
            },
        ],
        [False, True],
    )

    raw, rejected = build_split([row], "train", include_protocol_variants=True)
    clean, rejected = finalize(raw, rejected)

    assert rejected == []
    assert [sample["sample_type"] for sample in clean] == [
        "mixed_policy_single_tool",
        "mixed_policy_protocol_tool",
    ]
    mixed, protocol = clean
    assert mixed["messages"][-1]["content"] == "Let me check that."
    assert protocol["messages"][-1]["content"] == ""
    assert mixed["loss_policy"]["assistant_content"] is True
    assert protocol["loss_policy"]["assistant_content"] is False
    assert mixed["loss_policy"]["assistant_tool_call_wrappers"] is True
    assert protocol["loss_policy"]["assistant_tool_call_wrappers"] is True
    assert len(render_with_spans(protocol).target_spans) == 3


def test_parallel_tool_turn_is_sequentialized() -> None:
    row = source_row(
        [
            {"role": "user", "content": "My user id is sara_doe_496."},
            {
                "role": "assistant",
                "content": "Let me check your profile.",
                "tool_calls": [tool_call("get_user_details", {"user_id": "sara_doe_496"}, "call_user")],
            },
            {
                "role": "tool",
                "tool_call_id": "call_user",
                "name": "get_user_details",
                "content": '{"user_id":"sara_doe_496","reservations":["ABC123","XYZ789"]}',
            },
            {
                "role": "assistant",
                "content": "Let me inspect both reservations.",
                "tool_calls": [
                    tool_call("get_reservation_details", {"reservation_id": "ABC123"}, "call_a"),
                    tool_call("get_reservation_details", {"reservation_id": "XYZ789"}, "call_b"),
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_a",
                "name": "get_reservation_details",
                "content": '{"reservation_id":"ABC123","user_id":"sara_doe_496","origin":"PHL","destination":"LGA"}',
            },
            {
                "role": "tool",
                "tool_call_id": "call_b",
                "name": "get_reservation_details",
                "content": '{"reservation_id":"XYZ789","user_id":"sara_doe_496","origin":"ATL","destination":"PHL"}',
            },
        ],
        [False, False, False, True, False, False],
    )

    raw, rejected = build_split([row], "train", include_protocol_variants=True)
    clean, rejected = finalize(raw, rejected)

    assert rejected == []
    assert len(clean) == 2
    assert all(sample["sample_type"] == "mixed_policy_sequential_tool" for sample in clean)
    assert clean[0]["messages"][-1]["tool_calls"][0]["id"] == "call_a"
    assert clean[1]["messages"][-1]["tool_calls"][0]["id"] == "call_b"
    assert clean[1]["messages"][-3]["role"] == "assistant"
    assert clean[1]["messages"][-2]["role"] == "tool"
    assert clean[1]["messages"][-2]["tool_call_id"] == "call_a"
    assert clean[1]["metadata"]["sequentialized_from_parallel_tool_calls"] is True
