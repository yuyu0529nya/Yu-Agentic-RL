"""
sim_dynamic_sampling — CPU-only demonstration that dynamic sampling un-flattens GRPO under
a sparse (tau2-airline-like) reward. No torch, no GPU. A MECHANISM demo, not a tau2 number.

It reproduces the exact effect diagnosed on 2026-07-13: at a low success rate many GRPO
groups are zero-variance (all-fail), carry no gradient, yet under veRL's standard trainer
stay in the batch and DILUTE the averaged update (their tokens sit in the denominator). The
policy is SHARED across tasks, so this dilution shrinks the effective learning rate — the
flat-curve mechanism. Dynamic sampling (DAPO / ours) re-generates groups until the batch is
full of contrastful groups, so every weight update lands full-strength on informative data.

Model: contextual bandit. Task t has feature x_t; a SHARED linear policy W maps x_t -> action
logits; correct action = argmax(W* x_t) for a fixed random teacher W* (so it's learnable).
Base success at W=0 is 1/A. GRPO update = group-relative advantage + REINFORCE, the mean
gradient taken over ALL rollouts in the batch (dead ones included) — exactly veRL's dilution.

  * vanilla          : B groups/step, dead groups dilute the mean gradient  (veRL standard GRPO)
  * dynamic_sampling : re-sample groups until B are contrastful, then update (DAPO / ours)
  * gated_no_resample: drop dead groups but keep the smaller batch (filtering without refill)
"""

import numpy as np

A = 25           # actions per task -> base success 1/A = 4%  (very sparse -> >80% dead groups)
D = 20           # feature dim (shared policy has A*D params)
POOL = 60        # distinct tasks
N = 8            # rollouts per group (our rollout.n)
B = 16           # groups per update step (our train_batch_size)
LR = 4.0         # shared-policy lr (gradient is a mean over rollouts, so magnitudes are small)
STEPS = 80
EPS = 1e-6
MAX_TRIES = 40 * B   # safety cap on resampling


def softmax_rows(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


def make_world(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((POOL, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Wstar = rng.standard_normal((A, D))
    correct = (X @ Wstar.T).argmax(axis=1)     # learnable target
    return rng, X, correct


def greedy_success(W, X, correct):
    return float(((X @ W.T).argmax(axis=1) == correct).mean())


def sample_group(W, X, correct, task, rng):
    p = softmax_rows((X[task] @ W.T)[None, :])[0]
    actions = rng.choice(A, size=N, p=p)
    rewards = (actions == correct[task]).astype(float)
    return actions, rewards, p


def group_grad(W, X, task, actions, rewards, p):
    """Sum_i adv_i * (onehot(a_i) - pi) outer x_t  ->  (A, D). Also return contrastful flag
    and the count of rollouts that carry a non-zero advantage."""
    mean, std = rewards.mean(), rewards.std()
    if std == 0.0:
        return np.zeros((A, D)), False           # dead group: zero gradient contribution
    adv = (rewards - mean) / (std + EPS)
    G = np.zeros((A, D))
    x = X[task]
    for a, ad in zip(actions, adv):
        oh = np.zeros(A); oh[a] = 1.0
        G += ad * np.outer(oh - p, x)
    return G, True


def train(mode, seed=0):
    rng, X, correct = make_world(seed)
    W = np.zeros((A, D))
    curve, rollouts = [], 0

    for _ in range(STEPS):
        Gsum = np.zeros((A, D))
        slots = 0                      # groups counted toward the batch (the denominator)
        live_groups = 0
        tries = 0
        while slots < B and tries < MAX_TRIES:
            task = int(rng.integers(0, POOL)); tries += 1
            actions, rewards, p = sample_group(W, X, correct, task, rng)
            rollouts += N
            G, contrastful = group_grad(W, X, task, actions, rewards, p)

            if mode == "vanilla":
                Gsum += G; slots += 1                    # dead group: adds 0 but fills a slot
            elif mode == "gated_no_resample":
                if contrastful:
                    Gsum += G; live_groups += 1
                slots += 1                               # slot consumed either way (no refill)
            elif mode == "dynamic_sampling":
                if contrastful:
                    Gsum += G; slots += 1; live_groups += 1
            else:
                raise ValueError(mode)

        denom = (B if mode != "gated_no_resample" else max(live_groups, 1)) * N
        W += LR * Gsum / denom
        curve.append(greedy_success(W, X, correct))

    return np.array(curve), rollouts


def spark(vals):
    b = "▁▂▃▄▅▆▇█"
    return "".join(b[min(7, int(v * 7 + 0.5))] for v in vals)


if __name__ == "__main__":
    SEEDS = range(8)
    labels = {"vanilla": "vanilla GRPO", "gated_no_resample": "gated (no resample)",
              "dynamic_sampling": "dynamic sampling"}
    res = {}
    for mode in labels:
        cs, cost = [], 0
        for s in SEEDS:
            c, r = train(mode, seed=s); cs.append(c); cost += r
        res[mode] = (np.mean(cs, axis=0), cost / len(list(SEEDS)))

    print("=" * 80)
    print(f"GRPO dynamic-sampling demo  (A={A} -> base {1/A:.0%}, n={N}, batch={B} groups, "
          f"shared linear policy, {len(list(SEEDS))} seeds)")
    print("=" * 80)
    show = [0, 3, 6, 10, 20, 40, 60, STEPS - 1]
    print(f"\n{'':>20} | " + "  ".join(f"{s:>4}" for s in show) + f"  | learning curve (step 0->{STEPS-1})")
    print("-" * 80)
    for mode, (curve, _) in res.items():
        row = "  ".join(f"{curve[s]:.2f}" for s in show)
        print(f"{labels[mode]:>20} | {row} | {spark(curve)}")
    print(f"\n{'':>20}   greedy success at step ^")
    print("\ncost axis (honest): total rollouts generated over training")
    for mode, (curve, cost) in res.items():
        print(f"  {labels[mode]:>20} : {cost:>7.0f} rollouts | final {curve[-1]:.2f} | "
              f"step-15 {curve[15]:.2f}")
    v, d = res["vanilla"][0], res["dynamic_sampling"][0]
    print(f"\n=> at update step 15: dynamic sampling {d[15]:.2f} vs vanilla {v[15]:.2f} "
          f"({d[15]-v[15]:+.2f}).")
    print("   Same GRPO, same lr — the ONLY difference is dropping+refilling zero-variance")
    print("   groups so each weight update isn't diluted by dead weight. That is the fix for")
    print("   the flat tau2 curve, validated here before spending a single GPU-hour.")
