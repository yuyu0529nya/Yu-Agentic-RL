#!/usr/bin/env python3
"""Multi-trial paired analysis of tau2 eval rollouts.
Each eval jsonl row = {task_id, trial, reward, ...}. pass^1 = mean success over all (task,trial).
Paired delta vs base is bootstrapped over TASKS (resample task set) so the CI accounts for
task difficulty correlation. Usage: tau2_eval_analyze.py base.jsonl name1=a.jsonl name2=b.jsonl"""
import json, sys

def load(p):
    by = {}
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        by.setdefault(str(r["task_id"]), []).append(1.0 if float(r.get("reward") or 0.0) >= 1 - 1e-6 else 0.0)
    return {t: sum(v) / len(v) for t, v in by.items()}  # task -> mean pass over its trials

def boot_ci(deltas, n=10000):
    # deterministic LCG (no random module dependence on seed reproducibility across machines)
    ts = sorted(deltas); xs = [deltas[t] for t in ts]; m = len(xs)
    if m == 0:
        return 0.0, 0.0
    seed = 12345; means = []
    for _ in range(n):
        s = 0.0
        for _ in range(m):
            seed = (1103515245 * seed + 12345) & 0x7fffffff
            s += xs[seed % m]
        means.append(s / m)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]

base = load(sys.argv[1])
nb = len(base)
print(f"base pass^1 = {sum(base.values())/nb:.3f}  (n_tasks={nb})")
print(f"{'method':<18} {'pass^1':>7} {'Δ_vs_base':>10} {'paired_bootCI':>22} sig")
for spec in sys.argv[2:]:
    name, p = spec.split("=", 1)
    m = load(p)
    common = sorted(set(base) & set(m))
    bp = sum(base[t] for t in common) / len(common)
    mp = sum(m[t] for t in common) / len(common)
    deltas = {t: m[t] - base[t] for t in common}
    lo, hi = boot_ci(deltas)
    sig = "*" if (lo > 0 or hi < 0) else ""
    print(f"{name:<18} {mp:>7.3f} {mp-bp:>+10.3f}   [{lo:+.3f},{hi:+.3f}]   {sig}  (n={len(common)})")
