"""
prm_lite_adapter — wire the hand-written PRM-Lite dense process reward into veRL's tau2
reward path, while preserving the RAW binary outcome for grpo_gated's gate.

Why (the flat-curve fix, 2026-07-13):
  tau2-airline's reward is binary (task solved / not). At ~20% success with n=8, most groups
  are all-fail -> zero variance -> no gradient -> flat GRPO. PRM-Lite turns the reward
  CONTINUOUS (reward = outcome + beta * process_score), so even same-outcome trajectories are
  differentiated by process quality -> far fewer zero-variance groups -> a signal to climb.

The catch (why we ALSO need grpo_gated): once the reward is shaped, an all-FAIL group has
non-zero variance, so vanilla GRPO fabricates a "phantom advantage" on it (see grpo_gated.py).
So this adapter returns BOTH the shaped reward (for token_level_rewards) AND the raw binary
outcome (for the gate). PRM-Lite gives the signal; the binary gate keeps it honest.

Deploy: place prm_lite_reward.py (the hand-written module) + this file on the tau2_integration
PYTHONPATH, and patch tau2_bridge.py::compute_reward per `BRIDGE_PATCH` below.
"""

from __future__ import annotations

import os
import sys

# locate the hand-written prm_lite_reward.py across local dev and the GPU box layouts
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "..", "sft_assets"),
    os.path.join(_HERE, "..", "..", "sft_assets"),
    "/root/autodl-tmp/verl-work/tau2_integration/sft_assets",
    "/root/autodl-tmp/verl-work/tau2_integration",
]
for _c in _CANDIDATES:
    if os.path.isfile(os.path.join(_c, "prm_lite_reward.py")):
        sys.path.insert(0, os.path.abspath(_c))
        break
import prm_lite_reward as _prm  # noqa: E402

DEFAULT_BETA = 0.3


def _tc_to_dict(tc):
    if isinstance(tc, dict):
        return tc
    if hasattr(tc, "model_dump"):
        return tc.model_dump()
    return {"id": getattr(tc, "id", None), "name": getattr(tc, "name", None),
            "arguments": getattr(tc, "arguments", None)}


def _normalize_messages(messages) -> list:
    """Accept tau2 pydantic Message objects OR OpenAI-format dicts; return dicts that
    prm_lite_reward.action_history_from_messages can parse. tau2 tool_calls are flat
    ({id,name,arguments}) which that parser already handles."""
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            out.append(m)
            continue
        if hasattr(m, "model_dump"):
            d = m.model_dump()
        else:
            d = {"role": getattr(m, "role", None), "content": getattr(m, "content", None)}
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                d["tool_calls"] = [_tc_to_dict(tc) for tc in tcs]
            tid = getattr(m, "tool_call_id", None) or getattr(m, "id", None)
            if tid is not None:
                d["id"] = tid
        out.append(d)
    return out


def shape_tau2_reward(outcome_binary, messages, domain: str = "airline",
                      beta: float = DEFAULT_BETA, enable: bool = True) -> dict:
    """Compute the (optionally shaped) reward for one finished tau2 trajectory.

    `messages` may be tau2 pydantic Message objects (traj.tau2_messages) or OpenAI dicts.

    Returns:
      {
        "reward":         float,  # outcome + beta*process   (== outcome if enable=False)
        "outcome_binary": float,  # raw task success in {0,1} — the GATE signal
        "process_score":  float,  # dense process score in [-0.5, 0.5]
      }
    """
    ob = float(outcome_binary)
    if not enable:
        return {"reward": ob, "outcome_binary": ob, "process_score": 0.0}
    hist = _prm.action_history_from_messages(_normalize_messages(messages))
    ps = _prm.compute_process_score(hist, domain=domain)
    return {"reward": ob + beta * float(ps), "outcome_binary": ob, "process_score": float(ps)}


# The exact edit to enable PRM-Lite in the tau2 bridge (kept here so deployment is unambiguous).
BRIDGE_PATCH = r'''
# in tau2_bridge.py::compute_reward, after `reward_info = evaluate_simulation(...)`:
#
#   from prm_lite_adapter import shape_tau2_reward
#   outcome = float(reward_info.reward)                       # raw binary 0/1 (unchanged tau2 score)
#   shaped  = shape_tau2_reward(outcome, traj.tau2_messages,  # dense reward + preserved outcome
#                               domain=self.domain,
#                               beta=float(os.environ.get("PRM_BETA", "0.3")),
#                               enable=os.environ.get("USE_PRM_LITE", "0") == "1")
#   return {"reward": shaped["reward"],
#           "reward_breakdown": {**(reward_info.reward_breakdown or {}),
#                                "outcome_binary": shaped["outcome_binary"],   # <- grpo_gated reads this
#                                "process_score":  shaped["process_score"]},
#           "termination": termination_reason.value}
#
# Then in tau2_agent_loop.py the breakdown already flows into extra_fields
# ("tau2_reward_breakdown"); grpo_gated's veRL wrapper gates on outcome_binary from there.
'''
