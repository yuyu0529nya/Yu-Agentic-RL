"""Build a rejection-sampling fine-tune (RFT/STaR) dataset from already-collected GRPO rollouts.

Filters SUCCESSFUL rollouts (reward>=1), dedups near-identical trajectories per task, caps
per-task to avoid one task dominating, and writes a rollouts-format JSONL (reward forced 1.0)
that `grpo_update.py --rft` consumes directly. No GPU and no new rollouts needed.

Example:
  python build_rft_dataset.py \
      --rollouts outputs/grpo_airline_r4_*/rollouts_iter1.jsonl outputs/grpo_airline_r4_*/rollouts_iter2.jsonl outputs/grpo_airline_r4_*/rollouts_iter3.jsonl \
      --out outputs/rft_airline.jsonl --cap-per-task 8
  python grpo_update.py --rft --rollouts outputs/rft_airline.jsonl --reward-mode binary \
      --base-model <model> --out-adapter outputs/rft_airline_adapter --epochs 3
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path


def traj_sig(msgs: list[dict]) -> str:
    """Dedup signature: the sequence of assistant tool-call names / text prefixes."""
    parts = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if tcs:
            parts.append("T:" + ",".join((tc.get("function") or {}).get("name", "") for tc in tcs))
        else:
            parts.append("S:" + (m.get("content") or "")[:40])
    return "|".join(parts)


def _key(t: str):
    return int(t) if t.lstrip("-").isdigit() else t


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an RFT/STaR success-only dataset from rollouts.")
    ap.add_argument("--rollouts", nargs="+", required=True, help="one or more rollout jsonls")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-per-task", type=int, default=8)
    ap.add_argument("--min-reward", type=float, default=1.0)
    args = ap.parse_args()

    rows = []
    for p in args.rollouts:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    succ = [r for r in rows if float(r.get("reward", 0.0)) >= args.min_reward - 1e-6]
    print(f"[rft] {len(rows)} rollouts -> {len(succ)} successful (reward >= {args.min_reward})")

    by_task: dict[str, list[dict]] = defaultdict(list)
    seen: set = set()
    for r in succ:
        sig = (str(r.get("task_id")), traj_sig(r.get("messages") or []))
        if sig in seen:
            continue
        seen.add(sig)
        by_task[str(r.get("task_id"))].append(r)

    out_rows = []
    for t, lst in sorted(by_task.items(), key=lambda kv: _key(kv[0])):
        keep = lst[:args.cap_per_task]
        out_rows.extend(keep)
        print(f"[rft] task {t:>3}: {len(lst)} unique successes -> keep {len(keep)}")

    for r in out_rows:
        r["reward"] = 1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[rft] wrote {len(out_rows)} RFT examples from {len(by_task)} tasks -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
