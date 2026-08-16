#!/usr/bin/env python3
"""
Prove that the Marathi and Konkani corpora share no documents.

WHY THIS IS A REQUIRED CHECK, NOT A NICETY
------------------------------------------
The specification says: "Do not share documents across corpora or concatenate
languages", and "The two resulting models must not share data, tokenizer,
vocabulary, or weights."

For most language pairs this is trivially satisfied. For Marathi and Konkani it
is not. They are closely related, both written in Devanagari, and share a large
amount of vocabulary. Measured during the Phase 1 audit: the *Konkani* tokenizer
processed Marathi text at 0% unknown tokens and 3.81 characters per token -
statistically indistinguishable from its behaviour on Konkani. Corpora labelled
"Konkani" routinely contain Marathi, and the reverse.

So the claim "our two corpora are independent" needs evidence, and this script
produces it.

WHAT IT MEASURES
----------------
1. EXACT overlap - documents whose canonical form hashes identically. Required
   result: zero.

2. NEAR-DUPLICATE overlap - documents that are not byte-identical but carry the
   same content. A translated or lightly-edited article would show up here and
   nowhere else. Exact hashing cannot catch it.

3. LANGUAGE PURITY - for each corpus, how many documents the function-word
   discriminator assigns to the *other* language. This is different from 1 and
   2: it catches contamination that is not duplication. A Marathi article that
   appears only in the Konkani corpus is not a shared document, but it is still
   Marathi text in Model L's training data.

All three can pass or fail independently, which is why all three are reported.

USAGE
-----
    python3 tools/cross_corpus_check.py
    python3 tools/cross_corpus_check.py --sample 50000    # faster spot check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.dedup import MinHasher, exact_hash, jaccard                # noqa: E402
from common.manifest import atomic_write_json                           # noqa: E402
from common.scriptid import identify_marathi_konkani                    # noqa: E402

LANGUAGES = ("marathi", "konkani")
OTHER_LABEL = {"marathi": "kok", "konkani": "mr"}
OWN_LABEL = {"marathi": "mr", "konkani": "kok"}


def load_documents(language: str, sample: int = 0) -> list[str]:
    """Load documents from the language's shards."""
    data_dir = REPO_ROOT / language / "data"
    shards = sorted(data_dir.rglob("shard_*.txt"))
    if not shards:
        return []

    docs: list[str] = []
    for shard in shards:
        with open(shard, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(line)
                    if sample and len(docs) >= sample:
                        return docs
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-corpus independence check.")
    parser.add_argument("--sample", type=int, default=0,
                        help="documents per language (0 = all)")
    parser.add_argument("--near-threshold", type=float, default=0.80,
                        help="Jaccard threshold for near-duplicate overlap")
    parser.add_argument("--bands", type=int, default=32)
    args = parser.parse_args()

    print("=" * 72)
    print("CROSS-CORPUS INDEPENDENCE CHECK")
    print("Requirement: the two corpora must share no documents")
    print("=" * 72)

    corpora = {}
    for lang in LANGUAGES:
        docs = load_documents(lang, args.sample)
        if not docs:
            print(f"\n!! No documents found for {lang}. Run its collectors first.")
            return 1
        corpora[lang] = docs
        words = sum(len(d.split()) for d in docs)
        print(f"\n{lang:9s} {len(docs):>10,} documents  {words:>14,} words"
              + ("   (sampled)" if args.sample else ""))

    mr_docs, kok_docs = corpora["marathi"], corpora["konkani"]

    # ---- 1. exact overlap ------------------------------------------------
    print("\n" + "-" * 72)
    print("1. EXACT OVERLAP (identical canonical form)")
    print("-" * 72)

    mr_hashes = {exact_hash(d): i for i, d in enumerate(mr_docs)}
    kok_hashes = {exact_hash(d): i for i, d in enumerate(kok_docs)}
    shared = set(mr_hashes) & set(kok_hashes)

    print(f"  marathi hashes      {len(mr_hashes):>10,}")
    print(f"  konkani hashes      {len(kok_hashes):>10,}")
    print(f"  shared documents    {len(shared):>10,}"
          + ("   PASS" if not shared else "   FAIL"))

    if shared:
        print("\n  Examples of shared documents:")
        for h in list(shared)[:3]:
            print(f"    {mr_docs[mr_hashes[h]][:100]}")

    # ---- 2. near-duplicate overlap ---------------------------------------
    print("\n" + "-" * 72)
    print(f"2. NEAR-DUPLICATE OVERLAP (Jaccard >= {args.near_threshold})")
    print("-" * 72)
    print("  building signatures ...", flush=True)

    hasher = MinHasher(128)
    rows = 128 // args.bands

    def band_keys(sig):
        for b in range(args.bands):
            yield b, sig[b * rows:(b + 1) * rows]

    # Index the smaller corpus, stream the larger one against it.
    if len(kok_docs) <= len(mr_docs):
        index_docs, index_name = kok_docs, "konkani"
        probe_docs, probe_name = mr_docs, "marathi"
    else:
        index_docs, index_name = mr_docs, "marathi"
        probe_docs, probe_name = kok_docs, "konkani"

    buckets: dict = {}
    index_sigs: list = []
    for i, doc in enumerate(index_docs):
        sig = hasher.signature(doc)
        index_sigs.append(sig)
        for key in band_keys(sig):
            buckets.setdefault(key, []).append(i)

    print(f"  indexed {len(index_docs):,} {index_name} documents", flush=True)

    near_pairs = []
    for j, doc in enumerate(probe_docs):
        sig = hasher.signature(doc)
        candidates = set()
        for key in band_keys(sig):
            candidates.update(buckets.get(key, ()))
        for i in candidates:
            score = jaccard(sig, index_sigs[i])
            if score >= args.near_threshold:
                near_pairs.append((score, j, i))
                break

    print(f"  probed  {len(probe_docs):,} {probe_name} documents")
    print(f"  near-duplicate pairs {len(near_pairs):>7,}"
          + ("   PASS" if not near_pairs else "   REVIEW"))

    if near_pairs:
        near_pairs.sort(reverse=True)
        print("\n  Most similar cross-corpus pairs:")
        for score, j, i in near_pairs[:3]:
            print(f"    jaccard={score:.3f}")
            print(f"      {probe_name}: {probe_docs[j][:80]}")
            print(f"      {index_name}: {index_docs[i][:80]}")

    # ---- 3. language purity ----------------------------------------------
    print("\n" + "-" * 72)
    print("3. LANGUAGE PURITY (contamination that is not duplication)")
    print("-" * 72)

    purity = {}
    for lang, docs in corpora.items():
        counts = {"own": 0, "other": 0, "undecided": 0}
        for doc in docs:
            label = identify_marathi_konkani(doc).label
            if label == OWN_LABEL[lang]:
                counts["own"] += 1
            elif label == OTHER_LABEL[lang]:
                counts["other"] += 1
            else:
                counts["undecided"] += 1
        total = max(len(docs), 1)
        purity[lang] = {
            **counts,
            "total": len(docs),
            "own_rate": counts["own"] / total,
            "contamination_rate": counts["other"] / total,
        }
        print(f"  {lang:9s} own={counts['own']:>9,} "
              f"other={counts['other']:>7,} "
              f"undecided={counts['undecided']:>7,}   "
              f"contamination {counts['other'] / total:.3%}")

    # ---- verdict ----------------------------------------------------------
    exact_ok = not shared
    near_ok = not near_pairs
    contamination_ok = all(p["contamination_rate"] < 0.01 for p in purity.values())

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  exact overlap          {'PASS' if exact_ok else 'FAIL'}  "
          f"({len(shared):,} shared documents)")
    print(f"  near-duplicate overlap {'PASS' if near_ok else 'REVIEW'}  "
          f"({len(near_pairs):,} pairs)")
    print(f"  language purity        {'PASS' if contamination_ok else 'REVIEW'}  "
          f"(<1% cross-language in each corpus)")

    if exact_ok and near_ok and contamination_ok:
        print("\n  The two corpora are independent. This is the evidence for the")
        print("  specification's requirement that Model H and Model L share no data.")
    else:
        print("\n  Independence is NOT established. Remove the offending documents")
        print("  before building tokenizers - a contaminated corpus invalidates")
        print("  every downstream number.")

    payload = {
        "sampled": bool(args.sample),
        "sample_size": args.sample or None,
        "documents": {k: len(v) for k, v in corpora.items()},
        "exact_shared_documents": len(shared),
        "near_duplicate_threshold": args.near_threshold,
        "near_duplicate_pairs": len(near_pairs),
        "language_purity": purity,
        "passes": {"exact": exact_ok, "near_duplicate": near_ok,
                   "language_purity": contamination_ok},
        "independent": exact_ok and near_ok and contamination_ok,
    }
    out = REPO_ROOT / "report" / "phase1_cross_corpus_check.json"
    atomic_write_json(out, payload)
    print(f"\n  {out.relative_to(REPO_ROOT)}")
    print("=" * 72)
    return 0 if (exact_ok and near_ok and contamination_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
