from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_decision_gate_dataset import build_split, make_gate_sample, validate_samples  # noqa: E402
from evaluate_decision_gate_behavior import classify_gate_label  # noqa: E402


def mixed_row(
    row_id: str,
    target_action: str,
    *,
    turn_index: int = 3,
    sample_type: str = "mixed_policy_text",
    protocol_only: bool = False,
    source_id: str = "sft_success_task1_trial0",
) -> dict:
    assistant_target = {"role": "assistant", "content": "I can help with that."}
    if target_action == "tool_call":
        assistant_target = {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_user_details",
                        "arguments": {"user_id": "sara_doe_496"},
                    },
                }
            ],
        }

    return {
        "id": row_id,
        "format_version": "tau2_airline_mixed_dialogue_tool_policy_v1",
        "sample_type": sample_type,
        "messages": [
            {"role": "user", "content": "My user id is sara_doe_496."},
            assistant_target,
        ],
        "loss_mask": [False, True],
        "metadata": {
            "domain": "airline",
            "task_id": "1",
            "trial": 0,
            "source_id": source_id,
            "turn_index": turn_index,
            "source_call_index": 0,
            "prefix_message_count": 1,
            "target_action": target_action,
            "target_tool_names": ["get_user_details"] if target_action == "tool_call" else [],
            "target_tool_call_count": 1 if target_action == "tool_call" else 0,
            "target_content_chars": 11,
            "protocol_only": protocol_only,
        },
    }


def test_gate_sample_targets_only_short_label() -> None:
    sample = make_gate_sample(mixed_row("tool_a", "tool_call"), "train")

    assert sample["messages"][-1] == {"role": "assistant", "content": "tool_call"}
    assert sample["loss_mask"] == [False, True]
    assert sample["loss_policy"]["assistant_content"] is True
    assert sample["loss_policy"]["assistant_tool_calls"] is False
    assert sample["metadata"]["gate_label"] == "tool_call"
    assert validate_samples([sample]) == []


def test_initial_greeting_is_skipped_by_default() -> None:
    samples, dropped, skipped = build_split(
        [mixed_row("greeting", "assistant_text", turn_index=0)],
        "train",
        include_initial_greeting=False,
    )

    assert samples == []
    assert dropped == []
    assert skipped["initial_greeting"] == 1


def test_protocol_duplicate_prefers_non_protocol_row() -> None:
    protocol = mixed_row(
        "protocol_tool",
        "tool_call",
        sample_type="mixed_policy_protocol_tool",
        protocol_only=True,
    )
    non_protocol = mixed_row(
        "single_tool",
        "tool_call",
        sample_type="mixed_policy_single_tool",
        protocol_only=False,
    )

    samples, dropped, skipped = build_split([protocol, non_protocol], "train", include_initial_greeting=False)

    assert skipped == {}
    assert [row["id"] for row in dropped] == ["protocol_tool"]
    assert len(samples) == 1
    assert samples[0]["metadata"]["source_id"] == "single_tool"
    assert samples[0]["metadata"]["source_protocol_only"] is False


def test_gate_label_classifier_accepts_short_outputs() -> None:
    assert classify_gate_label("tool_call<|im_end|>") == "tool_call"
    assert classify_gate_label("assistant_text\n") == "assistant_text"
    assert classify_gate_label("Here is a call <tool_call>{}</tool_call>") == "tool_call"
    assert classify_gate_label("I should inspect the booking first.") == "unknown"
