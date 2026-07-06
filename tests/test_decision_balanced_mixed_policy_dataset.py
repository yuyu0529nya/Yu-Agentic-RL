from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_decision_balanced_mixed_policy_dataset import balance_train_samples  # noqa: E402


def row(row_id: str, sample_type: str, target_action: str, protocol_only: bool = False) -> dict:
    return {
        "id": row_id,
        "sample_type": sample_type,
        "metadata": {
            "target_action": target_action,
            "protocol_only": protocol_only,
        },
    }


def test_text_repeat_duplicates_text_rows_with_unique_ids() -> None:
    rows = [
        row("text_a", "mixed_policy_text", "assistant_text"),
        row("tool_a", "mixed_policy_single_tool", "tool_call"),
    ]

    balanced, dropped, stats = balance_train_samples(rows, text_repeat=3, protocol_keep_ratio=1.0, seed=11)

    assert dropped == []
    assert [item["id"] for item in balanced] == [
        "text_a",
        "text_a__phase2i_text_repeat1",
        "text_a__phase2i_text_repeat2",
        "tool_a",
    ]
    assert stats["output_target_actions"] == {"assistant_text": 3, "tool_call": 1}
    assert balanced[1]["metadata"]["phase2i_original_id"] == "text_a"
    assert balanced[1]["metadata"]["phase2i_repeat_index"] == 1


def test_protocol_keep_ratio_zero_drops_protocol_only_rows() -> None:
    rows = [
        row("text_a", "mixed_policy_text", "assistant_text"),
        row("single_a", "mixed_policy_single_tool", "tool_call"),
        row("seq_a", "mixed_policy_sequential_tool", "tool_call"),
        row("protocol_a", "mixed_policy_protocol_tool", "tool_call", protocol_only=True),
    ]

    balanced, dropped, stats = balance_train_samples(rows, text_repeat=1, protocol_keep_ratio=0.0, seed=11)

    assert [item["id"] for item in balanced] == ["text_a", "single_a", "seq_a"]
    assert [item["id"] for item in dropped] == ["protocol_a"]
    assert dropped[0]["metadata"]["mixed_policy_reject_reason"] == "phase2i_protocol_downsampled"
    assert stats["balance_dropped_rows"] == 1


def test_protocol_keep_ratio_half_keeps_exact_deterministic_count() -> None:
    rows = [
        row(f"protocol_{idx}", "mixed_policy_protocol_tool", "tool_call", protocol_only=True)
        for idx in range(5)
    ]

    balanced_a, dropped_a, stats_a = balance_train_samples(rows, text_repeat=1, protocol_keep_ratio=0.5, seed=11)
    balanced_b, dropped_b, stats_b = balance_train_samples(rows, text_repeat=1, protocol_keep_ratio=0.5, seed=11)

    assert len(balanced_a) == 3
    assert len(dropped_a) == 2
    assert [item["id"] for item in balanced_a] == [item["id"] for item in balanced_b]
    assert [item["id"] for item in dropped_a] == [item["id"] for item in dropped_b]
    assert stats_a["output_rows"] == stats_b["output_rows"] == 3
