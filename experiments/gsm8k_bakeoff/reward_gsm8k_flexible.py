"""
reward_gsm8k_flexible — GSM8K exact-match reward that an *Instruct* model can actually earn.

Why this file exists
--------------------
veRL's built-in gsm8k scorer defaults to method="strict", which only accepts the fine-tuned
format `#### <number>`. An Instruct model does not emit that unprompted -- it writes
"The answer is 42." or "**42**" or "\\boxed{42}". Under strict extraction every rollout scores 0,
so the reward is a constant, every group is all-fail, the group-relative advantage is identically
zero, and GRPO has no gradient at all.

That is not hypothetical: a GSM8K run on Qwen2.5-1.5B-Instruct scored 0.0796-0.0895 val accuracy
across four evals with strict extraction, while the same model is known to be around 60%. The
number being measured was the extractor, not the model.

Wired in via verl's public hook, so no verl source changes:

    custom_reward_function.path=<this file>
    custom_reward_function.name=compute_score

Extraction policy
-----------------
Ordered, most-explicit-first. The order matters: a permissive rule applied first would swallow
cases a stricter rule would have gotten right.

  1. `#### N`                     the canonical GSM8K format, if the model happens to use it
  2. `\\boxed{N}` / `$\\boxed{N}$` common in math-tuned models
  3. "the answer is N" and friends (case-insensitive, tolerant of markdown/punctuation)
  4. last number in the string   the fallback

Rule 4 is the risky one and is deliberately last. "Last number" is the standard flexible baseline
(it is what verl's own method="flexible" does), and it works because a solved GSM8K answer almost
always ends with the result. It does misfire on trailing units ("...so 3 boxes at $5 = $15 total,
which is 15 dollars") -- but that lands on the right number anyway. The real failure mode is a
model that keeps talking after answering; rules 1-3 exist to catch those first.

Numbers are normalized before comparison: strip `$ , % ` and trailing `.`, then compare as floats
with a tight tolerance so "42", "42.0", "$42", and "42," all match ground truth "42".
"""

from __future__ import annotations

import re

# ---- extraction patterns, tried in this order -------------------------------------------------

_NUM = r"-?\$?\d[\d,]*\.?\d*"

_PATTERNS = (
    ("hash", re.compile(r"####\s*(" + _NUM + r")")),
    ("boxed", re.compile(r"\\boxed\{\s*(" + _NUM + r")\s*\}")),
    (
        "phrase",
        re.compile(
            r"(?:answer|result|total|equals?)\s*(?:is|:|=)?\s*[\*\s\$]*(" + _NUM + r")",
            re.IGNORECASE,
        ),
    ),
)

_ANY_NUM = re.compile(_NUM)

# Guard against pathological rollouts (a model that degenerates into a wall of digits).
_MAX_CHARS = 8000


def _normalize(s: str):
    """'$1,234.0' -> 1234.0 ; returns None if it is not a number."""
    if s is None:
        return None
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    if not s or s in {"-", "."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(text: str, method: str = "flexible"):
    """Return (value, which_rule_fired). value is None when nothing parses.

    method="strict" restricts to rule 1, matching verl's built-in behaviour, so the two can be
    compared on the same rollouts.
    """
    if not text:
        return None, "empty"
    if len(text) > _MAX_CHARS:
        text = text[-_MAX_CHARS:]

    for name, pat in _PATTERNS:
        if method == "strict" and name != "hash":
            continue
        hits = pat.findall(text)
        if hits:
            # last hit: models often restate the answer at the end
            v = _normalize(hits[-1])
            if v is not None:
                return v, name
    if method == "strict":
        return None, "none"

    hits = _ANY_NUM.findall(text)
    for raw in reversed(hits):
        v = _normalize(raw)
        if v is not None:
            return v, "last_number"
    return None, "none"


def _gold(ground_truth) -> float | None:
    """GSM8K gold answers arrive as '... #### 42' or already as '42'."""
    if ground_truth is None:
        return None
    s = str(ground_truth)
    if "####" in s:
        s = s.split("####")[-1]
    v, _ = extract_answer(s, method="flexible")
    return v


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None,
                  method: str = "flexible", format_score: float = 0.0, score: float = 1.0,
                  **kwargs) -> float:
    """verl custom_reward_function entry point.

    Binary reward: 1.0 for a correct final answer, 0.0 otherwise. Deliberately NOT shaped --
    the bake-off this feeds compares group-filtering rules, and a shaped reward would change what
    "an all-fail group" even means (see experiments/gsm8k_bakeoff/PREREGISTRATION.md).

    Signature is permissive because verl has passed these positionally and by keyword across
    versions; **kwargs absorbs the difference rather than pinning us to one.
    """
    # tolerate the older positional convention compute_score(solution_str, ground_truth)
    if solution_str is None and isinstance(data_source, str) and ground_truth is not None:
        solution_str, data_source = data_source, None

    pred, _rule = extract_answer(solution_str or "", method=method)
    gold = _gold(ground_truth)
    if pred is None or gold is None:
        return 0.0
    return score if abs(pred - gold) < 1e-4 else format_score


# ---- self-test: run this file directly ---------------------------------------------------------

if __name__ == "__main__":
    CASES = [
        # (text, gold, expect_correct, note)
        ("Janet has 3 apples. #### 3", "3", True, "canonical hash format"),
        ("So the total is \\boxed{18}.", "#### 18", True, "boxed"),
        ("Therefore, the answer is 72.", "72", True, "phrase"),
        ("The answer is **1,250** dollars.", "1250", True, "phrase + markdown + comma"),
        ("...he pays $15 for them.", "15", True, "fallback: last number, with $"),
        ("Let me compute: 5 * 3 = 15. So 15 boxes.", "15", True, "fallback picks the last"),
        ("First 10, then 20, so the answer is 30.", "30", True, "phrase beats last-number order"),
        ("The answer is 30, not 40.", "30", True, "phrase rule beats the trailing distractor 40"),
        ("I cannot solve this.", "42", False, "no number at all"),
        ("", "42", False, "empty rollout"),
        ("#### 42", "42.0", True, "gold given as float"),
        ("The result equals 3.5", "3.5", True, "decimal"),
    ]
    ok = fail = 0
    print("=== flexible extraction ===")
    for text, gold, expect, note in CASES:
        got = compute_score(solution_str=text, ground_truth=gold)
        good = (got == 1.0) == expect
        pred, rule = extract_answer(text)
        print(f"  [{'PASS' if good else 'FAIL'}] {note:42} pred={pred} rule={rule} score={got}")
        ok, fail = (ok + 1, fail) if good else (ok, fail + 1)

    print("\n=== strict, on the same cases (this is what scored 8%) ===")
    strict_hits = sum(
        1 for text, gold, expect, _ in CASES
        if expect and compute_score(solution_str=text, ground_truth=gold, method="strict") == 1.0
    )
    solvable = sum(1 for _, _, expect, _ in CASES if expect)
    print(f"  strict got {strict_hits}/{solvable} of the answerable cases; flexible is the point.")

    print("\n" + "=" * 78)
    print(f"RESULT: {ok} passed, {fail} failed")
    print("=" * 78)
    raise SystemExit(1 if fail else 0)
