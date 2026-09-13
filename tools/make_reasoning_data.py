#!/usr/bin/env python3
"""
Build the Phase 3 synthetic reasoning dataset for one language.

The specification asks for comparative and basic reasoning - transitive and
ordering comparisons - generated programmatically so the ground-truth label is
known by construction, with train/validation/test splits and a stated defence
against train-test leakage. This script is that generator; the language
resources it draws on are in `common/reasoning.py`.

    python3 tools/make_reasoning_data.py --language konkani \\
        --n-train 8000 --n-val 500 --n-test 1000

HOW LEAKAGE IS PREVENTED, ON TWO AXES
-------------------------------------
The specification names two options - held-out entity names or held-out relation
patterns - and this dataset uses both, which lets the two be scored separately.

Entities. The name pool is split before anything is generated, so no name in the
test split ever appears in training. A model that memorised "रामा -> राहुल"
scores zero on test.

Wording. The relation `A > B` can be stated from either side: "A's height is
more than B's" (p_more) or "B's height is less than A's" (p_less). Training and
validation see only p_more. Test contains both. The p_less rows therefore ask
exactly the relations the model was trained on, in wording it has never seen,
and they are solvable only if the model represents the ordering rather than the
surface string. Reporting accuracy separately on the two makes the difference
visible instead of averaging it away.

Because facts inside an item are shuffled, the answer is also never readable
from sentence position alone.

WHAT IS CHECKED BEFORE THE DATA IS WRITTEN
------------------------------------------
Language. Every Konkani item is scored with the Phase 1 Marathi/Konkani
discriminator (`common/scriptid.py`). The two languages share a script and much
vocabulary, and generated Konkani that drifts into Marathi would quietly
undermine the independence claim the whole project rests on. The label
distribution goes into the statistics file.

Tokenization. Items are encoded with that language's frozen Phase 1 tokenizer
and the byte-fallback rate is reported. A high rate would mean the generator is
emitting characters the tokenizer never learned, which would waste context and
make the task harder than intended.

Overlap. Prompts are deduplicated within and across splits, so the same question
cannot appear in both training and test.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common import reasoning as R                                  # noqa: E402
from common.scriptid import identify_marathi_konkani               # noqa: E402

TRAIN_PATTERNS = ["p_value", "p_more"]
TEST_PATTERNS = ["p_value", "p_more", "p_less"]


def write_jsonl(items: list[R.Item], path: Path, split: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({
                "id": it.item_id, "split": split, "language": it.language,
                "family": it.family, "pattern": it.pattern,
                "attribute": it.attribute, "entities": it.entities,
                "values": it.values, "prompt": it.prompt, "answer": it.answer,
                "answer_marker": it.answer_marker, "rationale": it.rationale,
                "text_plain": it.text(False), "text_rationale": it.text(True),
                **it.metadata,
            }, ensure_ascii=False) + "\n")


def language_check(items: list[R.Item], language: str) -> dict:
    """Score every item with the Phase 1 discriminator.

    Items are short, so some fall below the discriminator's five-marker
    evidence gate and come back undecided. That is the gate working as designed,
    not a failure - what would matter is items confidently labelled as the
    *other* language.
    """
    labels = collections.Counter()
    scores = []
    wrong = []
    want = "mr" if language == "marathi" else "kok"
    for it in items:
        res = identify_marathi_konkani(it.text(True))
        labels[res.label] += 1
        scores.append(res.score)
        if res.label not in (want, "undecided"):
            wrong.append({"id": it.item_id, "label": res.label,
                          "score": res.score, "text": it.text(True)})
    return {
        "expected_label": want,
        "labels": dict(labels),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "misassigned": len(wrong),
        "misassigned_examples": wrong[:5],
    }


def token_check(items: list[R.Item], language: str) -> dict:
    """Encode with the frozen Phase 1 tokenizer and report length and fallback."""
    import sentencepiece as spm
    model = REPO_ROOT / language / "tokenizer" / f"{language}_bpe.model"
    sp = spm.SentencePieceProcessor(model_file=str(model))

    lengths, fallback, total = [], 0, 0
    for it in items:
        ids = sp.encode(it.text(True))
        lengths.append(len(ids))
        for i in ids:
            total += 1
            if sp.id_to_piece(i).startswith("<0x"):
                fallback += 1
    lengths.sort()
    return {
        "vocab_size": sp.get_piece_size(),
        "tokens_min": lengths[0],
        "tokens_median": lengths[len(lengths) // 2],
        "tokens_max": lengths[-1],
        "tokens_mean": round(sum(lengths) / len(lengths), 2),
        "byte_fallback_tokens": fallback,
        "byte_fallback_rate": round(fallback / max(1, total), 8),
    }


def composition(items: list[R.Item]) -> dict:
    return {
        "count": len(items),
        "by_family": dict(collections.Counter(i.family for i in items)),
        "by_pattern": dict(collections.Counter(i.pattern for i in items)),
        "by_attribute": dict(collections.Counter(i.attribute for i in items)),
        "by_extreme": dict(collections.Counter(i.metadata["extreme"] for i in items)),
        "distinct_answers": len({i.answer for i in items}),
        "answer_distribution": dict(collections.Counter(i.answer for i in items)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=["marathi", "konkani"])
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--test-name-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    lang = args.language
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / lang / "data" / "reasoning"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"REASONING DATASET - {lang}")
    print("=" * 74)

    train_names, test_names = R.split_names(lang, args.test_name_fraction, args.seed)
    print(f"  entity pool     {len(train_names)} train / {len(test_names)} test "
          f"(disjoint)")
    print(f"  train patterns  {TRAIN_PATTERNS}")
    print(f"  test patterns   {TEST_PATTERNS}  (held out: {R.HELD_OUT_PATTERNS})")

    # Different seeds per split, then a global prompt-level dedup, so that an
    # item generated for training can never reappear in validation or test.
    train = R.generate(lang, args.n_train, train_names, TRAIN_PATTERNS, args.seed)
    val = R.generate(lang, args.n_val, train_names, TRAIN_PATTERNS, args.seed + 1)
    test = R.generate(lang, args.n_test, test_names, TEST_PATTERNS, args.seed + 2)

    seen = {i.prompt for i in train}
    val = [i for i in val if i.prompt not in seen]
    seen |= {i.prompt for i in val}
    test = [i for i in test if i.prompt not in seen]

    splits = {"train": train, "val": val, "test": test}
    for name, items in splits.items():
        write_jsonl(items, out_dir / f"{name}.jsonl", name)
        print(f"  {name:<6} {len(items):>6,} items -> {name}.jsonl")

    # The leakage claim, verified rather than asserted.
    train_entities = {e for i in train for e in i.entities}
    test_entities = {e for i in test for e in i.entities}
    overlap_entities = sorted(train_entities & test_entities)
    overlap_prompts = len({i.prompt for i in train} & {i.prompt for i in test})

    print("\n--- leakage ---")
    print(f"  entity overlap train/test   {len(overlap_entities)}")
    print(f"  prompt overlap train/test   {overlap_prompts}")

    print("\n--- language check (Phase 1 discriminator) ---")
    lang_stats = {n: language_check(items, lang) for n, items in splits.items()}
    for n, d in lang_stats.items():
        print(f"  {n:<6} labels={d['labels']}  mean score={d['mean_score']:+.3f}  "
              f"misassigned={d['misassigned']}")

    print("\n--- tokenization (frozen Phase 1 tokenizer) ---")
    tok_stats = {n: token_check(items, lang) for n, items in splits.items()}
    for n, d in tok_stats.items():
        print(f"  {n:<6} tokens min/med/max {d['tokens_min']}/"
              f"{d['tokens_median']}/{d['tokens_max']}  "
              f"byte-fallback {d['byte_fallback_rate']:.6%}")

    stats = {
        "language": lang,
        "seed": args.seed,
        "entity_pool": {"train": train_names, "test": test_names,
                        "train_size": len(train_names),
                        "test_size": len(test_names)},
        "patterns": {"train": TRAIN_PATTERNS, "test": TEST_PATTERNS,
                     "held_out": R.HELD_OUT_PATTERNS},
        "leakage": {"entity_overlap_train_test": overlap_entities,
                    "prompt_overlap_train_test": overlap_prompts},
        "composition": {n: composition(i) for n, i in splits.items()},
        "language_check": lang_stats,
        "tokenization": tok_stats,
    }
    stats_path = REPO_ROOT / "report" / f"phase3_reasoning_data_{lang}.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n  {stats_path.relative_to(REPO_ROOT)}")

    print("\n--- samples ---")
    shown = set()
    for it in test:
        key = (it.family, it.pattern)
        if key in shown:
            continue
        shown.add(key)
        print(f"\n  [{it.family} / {it.pattern} / {it.attribute}]")
        print(f"    {it.prompt}")
        print(f"    -> {it.completion(True)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
