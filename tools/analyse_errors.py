#!/usr/bin/env python3
"""
Classify the reasoning model's wrong answers, and score them leniently.

WHY THIS EXISTS
---------------
`tools/evaluate_reasoning.py` reports exact-match accuracy, which is the metric
the specification asks for. Exact match answers "how often is the model right",
but not "when it is wrong, what kind of wrong is it". Two models can score
identically and be failing for completely different reasons, and the fix for one
failure is not the fix for the other.

This script answers the second question by sorting every prediction into one of
four buckets, using only information the dataset already fixes:

  test-pool name        a name that appears in test prompts - a legal answer,
                        whether or not it is the right one
  train-pool name       a name that appears ONLY in training prompts. The entity
                        pools are disjoint, so such an answer cannot be correct
                        on any test item. Producing one means the model is
                        reciting a name it memorised rather than reading the
                        prompt in front of it.
  malformed variant     a string that is a prefix or extension of exactly one
                        test-pool name and of no other - मीर for मीरा, कृष्णा
                        for कृष्ण. The model picked an entity that is actually
                        in the prompt and failed to emit its exact tokens.
  equality word         the invariant समान, or Konkani's gender-agreeing
                        सारकी / सारकें.

THE LENIENT SCORE, AND WHY IT IS SECONDARY
------------------------------------------
The "malformed variant" bucket raises a scoring question. If a model answers
मीर where the gold is मीरा, exact match counts it wrong - correctly, because the
specification asks for exact match. But that conflates two different failures:
not knowing the answer, and not being able to spell it.

So a second, clearly-labelled figure is computed alongside: a prediction counts
if it is a prefix or extension of the gold AND of no other candidate. The
uniqueness requirement is what keeps this honest - a prediction that could refer
to two different entities is never credited, so the rule cannot manufacture
accuracy out of ambiguity.

Exact match remains the headline number everywhere. The lenient figure is
reported beside it as a diagnostic, and it was defined AFTER seeing the error
distribution, which is recorded in D-060 because a reader has to know that.

USAGE
    python3 tools/analyse_errors.py --language marathi
    python3 tools/analyse_errors.py --language konkani --out-dir report
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_pools(language: str, report_dir: Path) -> tuple[set[str], set[str]]:
    """The train and test entity pools this language's dataset was built from.

    These are written by tools/make_reasoning_data.py at generation time, so the
    classification below uses the same pools the data actually used rather than
    anything re-derived here.
    """
    meta = json.loads(
        (report_dir / f"phase3_reasoning_data_{language}.json").read_text(encoding="utf-8"))
    return set(meta["entity_pool"]["train"]), set(meta["entity_pool"]["test"])


def unique_prefix_match(pred: str, gold: str, candidates: set[str]) -> bool:
    """True when `pred` can only have been an attempt at `gold`.

    A prediction qualifies if it is a prefix or an extension of the gold, and of
    no other candidate. The second half is the important half: without it,
    a truncation short enough to match several names would be credited to
    whichever one happened to be correct, which would inflate the score exactly
    where the model is least certain.
    """
    if not pred:
        return False
    if pred == gold:
        return True
    matches = [c for c in candidates if c.startswith(pred) or pred.startswith(c)]
    return matches == [gold]


def classify(rows: list[dict], train_pool: set[str], test_pool: set[str],
             equality_words: set[str]) -> dict:
    """Sort every parseable prediction into one of the four buckets."""
    answered = [r for r in rows if r["emitted_marker"] and r["predicted"]]
    buckets: Counter = Counter()
    examples: dict[str, Counter] = {}

    for r in answered:
        p = r["predicted"]
        if p in test_pool:
            key = "test_pool_name"
        elif p in train_pool:
            key = "train_pool_name"
        elif p in equality_words:
            key = "equality_word"
        elif any(c.startswith(p) or p.startswith(c) for c in test_pool):
            key = "malformed_variant"
        else:
            key = "other_string"
        buckets[key] += 1
        examples.setdefault(key, Counter())[p] += 1

    n = len(answered)
    return {
        "answered": n,
        "unparseable": len(rows) - n,
        "buckets": {k: {"count": v,
                        "percent": round(100.0 * v / n, 2),
                        "most_common": examples[k].most_common(6)}
                    for k, v in buckets.most_common()},
    }


def score(rows: list[dict], candidates: set[str]) -> dict:
    """Exact-match and lenient accuracy, each against the per-item chance floor.

    Chance is 1/(k+1) per item and k varies by family, so the test sums the
    per-item variance rather than assuming one p across the set.
    """
    n = len(rows)
    exact = sum(r["correct"] for r in rows)
    lenient = sum(1 for r in rows
                  if unique_prefix_match(r["predicted"], r["gold"], candidates))
    ps = [r["chance"] for r in rows]
    mu, var = sum(ps), sum(p * (1 - p) for p in ps)

    def z_and_p(k: int) -> tuple[float, float]:
        if var <= 0:
            return 0.0, 1.0
        z = (k - mu) / math.sqrt(var)
        return z, 0.5 * math.erfc(z / math.sqrt(2))

    z_e, p_e = z_and_p(exact)
    z_l, p_l = z_and_p(lenient)
    return {
        "n": n,
        "exact_match_percent": round(100.0 * exact / n, 2),
        "lenient_percent": round(100.0 * lenient / n, 2),
        "uniform_chance_percent": round(100.0 * mu / n, 2),
        "exact_z": round(z_e, 3), "exact_p_one_tailed": round(p_e, 4),
        "lenient_z": round(z_l, 3), "lenient_p_one_tailed": round(p_l, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["marathi", "konkani"])
    ap.add_argument("--eval-json", default=None,
                    help="defaults to report/phase3_reasoning_eval_<lang>_final.json")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    report_dir = Path(args.report_dir) if args.report_dir else REPO_ROOT / "report"
    eval_path = Path(args.eval_json) if args.eval_json else (
        report_dir / f"phase3_reasoning_eval_{args.language}_final.json")
    rows = json.loads(eval_path.read_text(encoding="utf-8"))["finetuned"]["rows"]

    train_pool, test_pool = load_pools(args.language, report_dir)
    # The equality families answer with an adjective rather than a name, so those
    # strings are legal answers too and must not be classed as hallucinations.
    equality_words = {r["gold"] for r in rows if r["family"] == "equality"}
    candidates = test_pool | equality_words

    print("=" * 74)
    print(f"ERROR ANALYSIS - {args.language}")
    print("=" * 74)
    print(f"  train pool  {len(train_pool)} names")
    print(f"  test pool   {len(test_pool)} names  (disjoint from train)")
    print(f"  equality    {sorted(equality_words)}")

    overall = classify(rows, train_pool, test_pool, equality_words)
    print(f"\n--- what the {overall['answered']} parseable answers are ---")
    for key, b in overall["buckets"].items():
        top = ", ".join(f"{w}({c})" for w, c in b["most_common"][:4])
        print(f"  {key:<20}{b['count']:>5}  {b['percent']:>5.1f}%   {top}")

    per_family = {}
    print(f"\n--- per family ---")
    print(f"  {'family':<20}{'exact%':>8}{'lenient%':>10}{'chance%':>9}{'lenient z':>11}")
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        s = score(sub, candidates)
        per_family[fam] = s
        print(f"  {fam:<20}{s['exact_match_percent']:>8.2f}{s['lenient_percent']:>10.2f}"
              f"{s['uniform_chance_percent']:>9.2f}{s['lenient_z']:>11.2f}")

    all_rows = score(rows, candidates)
    no_eq = score([r for r in rows if r["family"] != "equality"], test_pool)
    print(f"\n--- overall ---")
    for label, s in (("all families", all_rows), ("excluding equality", no_eq)):
        print(f"  {label:<20} exact {s['exact_match_percent']:6.2f}%  "
              f"lenient {s['lenient_percent']:6.2f}%  chance {s['uniform_chance_percent']:6.2f}%"
              f"   exact z {s['exact_z']:+6.2f}   lenient z {s['lenient_z']:+6.2f}"
              f"  (p={s['lenient_p_one_tailed']:.4f})")

    payload = {"language": args.language,
               "source": str(eval_path.name),
               "entity_pool_sizes": {"train": len(train_pool), "test": len(test_pool)},
               "equality_words": sorted(equality_words),
               "prediction_classes": overall,
               "scores": {"all_families": all_rows, "excluding_equality": no_eq,
                          "by_family": per_family}}
    out_dir = Path(args.out_dir) if args.out_dir else report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"phase3_error_analysis_{args.language}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
