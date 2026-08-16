#!/usr/bin/env python3
"""
Build document-level train / validation / test splits for one language.

WHY DOCUMENT-LEVEL AND NOT LINE-LEVEL
-------------------------------------
This is the single decision that makes the splits trustworthy.

Consecutive lines of a document share vocabulary, named entities, topic and
phrasing. If a document is split across train and test, the model sees part of a
news article during training and is then evaluated on the rest of the same
article. Perplexity drops, the number looks good, and it measures nothing except
leakage. Every document therefore lands entirely in exactly one split.

Our shards already store one document per line, so a document is a line - but
the guarantee comes from *how the corpus was written*, not from an assumption
made here, which is why the collectors flatten newlines when they write shards.

WHY STRATIFY BY SOURCE
----------------------
The corpus is deliberately heterogeneous: government resolutions read nothing
like news, which reads nothing like digitised books. An unstratified random
split can easily give validation a different source mix from training, and then
validation loss measures domain shift rather than model quality. Splitting each
source separately and then concatenating keeps the mix identical across splits.

WHY MANUAL AND DOWNLOADED ARE TRACKED PER SPLIT
-----------------------------------------------
The 20% manual requirement applies to the **training** tokens. Reporting the
manual fraction of each split makes that verifiable rather than assumed, and
catches the case where manual data accidentally concentrates in one split.

CROSS-SOURCE DEDUPLICATION HAPPENS HERE
---------------------------------------
Each collector deduplicates within itself, but nothing has yet compared sources
against each other. A news article can appear in both the scraped corpus and a
downloaded web crawl. That comparison has to happen once, over the whole corpus,
immediately before splitting - which is here.

USAGE
-----
    python3 tools/make_splits.py --language marathi
    python3 tools/make_splits.py --language konkani --val 0.02 --test 0.02
    python3 tools/make_splits.py --language marathi --no-dedup    # faster re-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.dedup import Deduplicator, exact_hash                     # noqa: E402
from common.manifest import atomic_write_json                          # noqa: E402

SEED = 20260819          # the Phase 1 deadline; arbitrary but fixed and recorded


def classify(path: Path) -> tuple[str, bool]:
    """Return (source_name, is_manual) for a shard file.

    Provenance is read from the directory layout the collectors write:
        <lang>/data/manual/<source>/shard_*.txt      -> manual
        <lang>/data/processed/<source>/shard_*.txt   -> downloaded

    This is deliberate. Recomputing it from the manifests would be fragile,
    because shards flatten newlines and so their hashes no longer match the
    manifest's content_hash for multi-paragraph documents.
    """
    parts = path.parts
    is_manual = "manual" in parts
    source = path.parent.name
    if source.startswith("shard"):
        source = path.parent.parent.name
    return source, is_manual


def gather(language: str) -> dict[tuple[str, bool], list[str]]:
    """Load every document, grouped by (source, is_manual)."""
    data_dir = REPO_ROOT / language / "data"
    shards = sorted(data_dir.rglob("shard_*.txt"))
    if not shards:
        raise SystemExit(f"No shards under {data_dir}. Run the collectors first.")

    groups: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for shard in shards:
        source, is_manual = classify(shard)
        with open(shard, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    groups[(source, is_manual)].append(line)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Build document-level splits.")
    parser.add_argument("--language", required=True, choices=["marathi", "konkani"])
    parser.add_argument("--val", type=float, default=0.01)
    parser.add_argument("--test", type=float, default=0.01)
    parser.add_argument("--no-dedup", action="store_true",
                        help="skip the cross-source deduplication pass")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    lang = args.language
    if args.val + args.test >= 0.5:
        raise SystemExit("val + test must leave a majority for training.")

    out_dir = REPO_ROOT / lang / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SPLITS - {lang}")
    print(f"train/val/test = {1 - args.val - args.test:.2%} / "
          f"{args.val:.2%} / {args.test:.2%}   seed={SEED}")
    print("=" * 70)

    groups = gather(lang)
    total_docs = sum(len(v) for v in groups.values())
    print(f"\nLoaded {total_docs:,} documents from {len(groups)} source group(s):")
    for (source, is_manual), docs in sorted(groups.items(),
                                            key=lambda kv: -len(kv[1])):
        words = sum(len(d.split()) for d in docs)
        tag = "manual" if is_manual else "downloaded"
        print(f"  {source:38s} {len(docs):>9,} docs {words:>13,} words [{tag}]")

    # ---- cross-source deduplication -------------------------------------
    if not args.no_dedup:
        print("\n--- Cross-source deduplication ---")
        deduper = Deduplicator(threshold=args.dedup_threshold)
        removed_by_source: dict[str, int] = defaultdict(int)
        cleaned: dict[tuple[str, bool], list[str]] = {}

        # Manual sources are processed FIRST so that when a document exists in
        # both a manual and a downloaded source, the copy that survives is the
        # manual one. Dropping the manual copy instead would silently reduce the
        # manual ratio - the one number the whole corpus design protects.
        ordered = sorted(groups.items(), key=lambda kv: (not kv[0][1], kv[0][0]))
        for key, docs in ordered:
            keep = []
            for doc in docs:
                if deduper.is_duplicate(doc):
                    removed_by_source[key[0]] += 1
                else:
                    keep.append(doc)
            cleaned[key] = keep

        groups = cleaned
        print(f"  {deduper.stats.to_dict()}")
        if removed_by_source:
            print("  removed per source:")
            for source, count in sorted(removed_by_source.items(),
                                        key=lambda kv: -kv[1]):
                print(f"    {source:38s} {count:,}")
    else:
        print("\n--- Cross-source deduplication SKIPPED (--no-dedup) ---")

    # ---- stratified, document-level split -------------------------------
    rng = random.Random(SEED)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    per_split_stats: dict[str, dict] = {
        name: {"docs": 0, "words": 0, "manual_docs": 0, "manual_words": 0,
               "downloaded_docs": 0, "downloaded_words": 0, "by_source": {}}
        for name in splits
    }

    for (source, is_manual), docs in sorted(groups.items()):
        if not docs:
            continue
        shuffled = docs[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_val = max(1, int(n * args.val)) if n >= 10 else 0
        n_test = max(1, int(n * args.test)) if n >= 10 else 0

        assignment = [
            ("val", shuffled[:n_val]),
            ("test", shuffled[n_val:n_val + n_test]),
            ("train", shuffled[n_val + n_test:]),
        ]

        for name, chunk in assignment:
            splits[name].extend(chunk)
            words = sum(len(d.split()) for d in chunk)
            stats = per_split_stats[name]
            stats["docs"] += len(chunk)
            stats["words"] += words
            if is_manual:
                stats["manual_docs"] += len(chunk)
                stats["manual_words"] += words
            else:
                stats["downloaded_docs"] += len(chunk)
                stats["downloaded_words"] += words
            stats["by_source"][source] = {
                "docs": len(chunk), "words": words, "is_manual": is_manual}

    # Shuffle within each split so sources are interleaved rather than blocked.
    for name in splits:
        rng.shuffle(splits[name])
        path = out_dir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as fh:
            for doc in splits[name]:
                fh.write(doc + "\n")

    # ---- verification ----------------------------------------------------
    print("\n--- Leakage check ---")
    hashes = {name: {exact_hash(d) for d in docs} for name, docs in splits.items()}
    leaks = 0
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = hashes[a] & hashes[b]
        leaks += len(shared)
        status = "OK" if not shared else f"!! {len(shared):,} SHARED"
        print(f"  {a:6s} vs {b:6s}  {status}")

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    print(f"{'split':>7} {'docs':>11} {'words':>14} {'manual':>14} {'manual %':>10}")
    print("-" * 70)
    for name in ("train", "val", "test"):
        s = per_split_stats[name]
        share = s["manual_words"] / s["words"] if s["words"] else 0.0
        print(f"{name:>7} {s['docs']:>11,} {s['words']:>14,} "
              f"{s['manual_words']:>14,} {share:>9.1%}")

    train = per_split_stats["train"]
    train_share = train["manual_words"] / train["words"] if train["words"] else 0
    print("-" * 70)
    if train_share >= 0.20:
        print(f"  Training manual share {train_share:.1%} - MEETS the 20% requirement.")
    else:
        print(f"  !! Training manual share {train_share:.1%} - BELOW the 20% "
              f"requirement.")
        print(f"  !! Collect more manual data, or reduce downloaded data by "
              f"{train['words'] - train['manual_words'] * 5:,} words.")

    payload = {
        "language": lang,
        "seed": SEED,
        "ratios": {"train": 1 - args.val - args.test,
                   "val": args.val, "test": args.test},
        "granularity": "document",
        "stratified_by": "source",
        "cross_source_dedup": not args.no_dedup,
        "leaked_documents": leaks,
        "splits": per_split_stats,
        "train_manual_share": train_share,
        "meets_manual_requirement": train_share >= 0.20,
    }
    report_path = REPO_ROOT / "report" / f"phase1_splits_{lang}.json"
    atomic_write_json(report_path, payload)

    print(f"\n  {out_dir.relative_to(REPO_ROOT)}/train.txt")
    print(f"  {out_dir.relative_to(REPO_ROOT)}/val.txt")
    print(f"  {out_dir.relative_to(REPO_ROOT)}/test.txt")
    print(f"  {report_path.relative_to(REPO_ROOT)}")
    print("=" * 70)
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
