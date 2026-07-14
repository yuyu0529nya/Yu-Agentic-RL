"""
CPU unit tests for prm_lite_adapter — no torch, no GPU. Run: python3 test_prm_lite_adapter.py

Shows (1) PRM-Lite densifies a binary reward so same-outcome trajectories differ, and
(2) that densification creates phantom variance on all-fail groups which grpo_gated's
binary gate correctly kills — i.e. PRM-Lite + grpo_gated are the intended pair.
"""

import numpy as np

from prm_lite_adapter import shape_tau2_reward
from grpo_gated import standard_grpo_advantage, gated_grpo_advantage

_passed = _failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if cond: _passed += 1
    else: _failed += 1


def asst(text, name, args):
    return {"role": "assistant", "content": text,
            "tool_calls": [{"id": f"c{abs(hash((name,str(args))))%9999}",
                            "function": {"name": name, "arguments": args}}]}

def think():
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": "t1", "function": {"name": "think", "arguments": "{}"}}]}

def tool(content):
    return {"role": "tool", "content": content}


# a GOOD trajectory: two diverse reads, data-chained write, a think step
GOOD = [
    asst("Let me pull up your account.", "get_user_details", '{"user_id": "yusuf_rossi_9876"}'),
    tool('{"user_id": "yusuf_rossi_9876", "reservations": ["ZFA04Y"]}'),
    asst("Now checking that reservation.", "get_reservation_details", '{"reservation_id": "ZFA04Y"}'),
    tool('{"reservation_id": "ZFA04Y", "flight": "HAT045", "cabin": "economy"}'),
    think(),
    asst("I'll add the checked bag now.", "update_reservation_baggages",
         '{"reservation_id": "ZFA04Y", "total_baggages": 2, "nonfree_baggages": 1}'),
    tool('{"status": "ok"}'),
]

# a BAD trajectory: escalate with no prior read, placeholder param, redundant repeated error
BAD = [
    asst("", "transfer_to_human_agents", '{}'),
    tool('{"error": "cannot transfer"}'),
    asst("ok", "get_reservation_details", '{"reservation_id": "previous"}'),
    tool('{"error": "reservation not found"}'),
    asst("ok", "get_reservation_details", '{"reservation_id": "previous"}'),
    tool('{"error": "reservation not found"}'),
]


def test_process_scores_spread():
    """Good process >> bad process — the dense signal exists and is ordered sensibly."""
    print("test_process_scores_spread")
    g = shape_tau2_reward(0.0, GOOD)["process_score"]
    b = shape_tau2_reward(0.0, BAD)["process_score"]
    check("good trajectory scores higher process than bad", g > b, f"good={g:+.3f}  bad={b:+.3f}")
    check("bad trajectory is penalized (negative process)", b < 0, f"bad process={b:+.3f}")


def test_densifies_same_outcome():
    """Two FAILED trajectories (outcome 0) get DIFFERENT shaped rewards — binary would tie them."""
    print("test_densifies_same_outcome")
    rg = shape_tau2_reward(0.0, GOOD, beta=0.3)["reward"]
    rb = shape_tau2_reward(0.0, BAD, beta=0.3)["reward"]
    check("raw binary reward ties the two failures at 0.0",
          shape_tau2_reward(0.0, GOOD, enable=False)["reward"] ==
          shape_tau2_reward(0.0, BAD, enable=False)["reward"] == 0.0)
    check("PRM-Lite separates them (dense reward)", abs(rg - rb) > 0.02,
          f"good-fail={rg:+.3f}  bad-fail={rb:+.3f}  gap={rg-rb:+.3f}")


def test_prm_plus_gate_on_all_fail_group():
    """An all-FAIL group, densified by PRM-Lite. Standard GRPO fabricates a phantom advantage;
    grpo_gated (gate on the binary outcome) zeroes the whole group. The intended pairing."""
    print("test_prm_plus_gate_on_all_fail_group")
    trajs = [GOOD, BAD, GOOD, BAD]          # 4 rollouts, ALL failed the task
    outcome = np.array([0, 0, 0, 0])
    shaped = np.array([shape_tau2_reward(0.0, t, beta=0.3)["reward"] for t in trajs])
    index = np.array([0, 0, 0, 0])

    std_adv = standard_grpo_advantage(shaped, index)
    gated_adv, keep = gated_grpo_advantage(shaped, outcome, index)

    check("PRM-Lite gave the all-fail group non-zero reward variance",
          shaped.std() > 0, f"shaped={shaped.round(3).tolist()}")
    check("standard GRPO fabricates a phantom advantage on it",
          np.abs(std_adv).max() > 0.1, f"phantom adv={std_adv.round(2).tolist()}")
    check("grpo_gated zeroes the all-fail group (gate on binary outcome)",
          np.allclose(gated_adv, 0.0) and (~keep).all())


def test_contrastful_group_richer_signal():
    """A contrastful group (2 solved, 2 failed). PRM-Lite makes the advantages richer than the
    flat ±1 of a pure binary reward, while grpo_gated keeps the group (real contrast)."""
    print("test_contrastful_group_richer_signal")
    trajs =   [GOOD,  BAD,  GOOD,  BAD]
    outcome = np.array([1,    1,    0,    0])       # first two solved, last two failed
    shaped = np.array([shape_tau2_reward(o, t, beta=0.3)["reward"]
                       for o, t in zip(outcome, trajs)])
    index = np.array([0, 0, 0, 0])
    adv, keep = gated_grpo_advantage(shaped, outcome, index)

    check("group is kept (has real success/fail contrast)", keep.all())
    check("solved-with-good-process gets the top advantage", adv.argmax() == 0,
          f"adv={adv.round(2).tolist()} shaped={shaped.round(3).tolist()}")
    check("the two solved rollouts are separated by process quality (not tied)",
          shaped[0] != shaped[1] if outcome[0] == outcome[1] else True)


class _FakeMsg:
    """Mimics a tau2 pydantic Message: attribute access + model_dump()."""
    def __init__(self, d): self._d = d
    def model_dump(self): return self._d


def test_accepts_pydantic_like_objects():
    """The adapter must accept tau2 Message objects (traj.tau2_messages), not just dicts —
    it normalizes via model_dump(). Same trajectory as dicts vs as objects must match."""
    print("test_accepts_pydantic_like_objects")
    as_objects = [_FakeMsg(m) for m in GOOD]
    r_dict = shape_tau2_reward(0.0, GOOD)["process_score"]
    r_obj = shape_tau2_reward(0.0, as_objects)["process_score"]
    check("object-form messages parse identically to dict-form",
          abs(r_dict - r_obj) < 1e-9, f"dict={r_dict:+.4f}  obj={r_obj:+.4f}")


if __name__ == "__main__":
    print("=" * 78)
    print("prm_lite_adapter tests  (dense process reward  +  binary-outcome gate)")
    print("=" * 78)
    for t in (test_process_scores_spread, test_densifies_same_outcome,
              test_prm_plus_gate_on_all_fail_group, test_contrastful_group_richer_signal,
              test_accepts_pydantic_like_objects):
        t(); print()
    print("=" * 78)
    print(f"RESULT: {_passed} passed, {_failed} failed")
    print("=" * 78)
    raise SystemExit(1 if _failed else 0)
