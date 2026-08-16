#!/usr/bin/env python3
"""
Konkani source K1: bring the existing self-collected Wikipedia corpus into the
manifest, as manual data, Devanagari only.

WHY THIS SCRIPT IS WORTH RUNNING FIRST
--------------------------------------
This collects nothing new. The data already exists on disk - 4,933 pages were
scraped by `collect_wikipedia_sample.py`, and 3,999 survived filtering - but it
predates the manifest and accounting layer, so **not one word of it currently
counts toward the manual requirement**.

Given that `total <= 5 x manual`, and that the Konkani probe found only 1 usable
site out of 8, this is the highest-leverage action available for Model L: it
converts work already done into roughly 1-2M manual words at zero collection
cost, which in turn permits 5-10M words of the books corpus into the corpus.

WHY ONLY DEVANAGARI
-------------------
Decision D-001 fixed the Konkani corpus as Devanagari-only. The original
collection is mixed: 2,613 Devanagari pages, 1,727 Roman, 593 mixed. The Roman
and mixed pages are excluded here and reported separately, so the excluded
sub-corpus is documented rather than silently dropped.

WHY THIS COUNTS AS MANUAL - AND THE CAVEAT
-------------------------------------------
We wrote the crawler, called the MediaWiki API ourselves, cleaned the wikitext
ourselves and filtered the result. Under the brief's definition ("scraping +
cleaning pages you gather yourself") that is manual collection.

The caveat, stated openly because it will be asked: Wikipedia is a well-known
public dataset, and the TAs advised against relying on it. It is therefore
recorded as a **secondary** manual source, not the primary one, and the source
inventory should say so. Its licence is CC BY-SA 4.0, which requires
attribution in the report.

USAGE
-----
    python3 konkani/scripts/ingest_wikipedia_manual.py --dry-run
    python3 konkani/scripts/ingest_wikipedia_manual.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.dedup import Deduplicator                                 # noqa: E402
from common.manifest import (                                         # noqa: E402
    CollectionType,
    ManifestWriter,
    make_record,
)
from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import NORMALIZATION_STEPS, normalize            # noqa: E402

SOURCE_NAME = "konkani_wikipedia_selfcollected"
JOB_NAME = "konkani_wikipedia"
LANGUAGE = "kok"

DATA_DIR = REPO_ROOT / "konkani" / "data"
OUT_DIR = DATA_DIR / "manual" / SOURCE_NAME
MANIFEST_PATH = DATA_DIR / "manifests" / f"{JOB_NAME}.jsonl"

# Prefer the filtered output; fall back to the raw collection.
CANDIDATE_INPUTS = [
    (DATA_DIR / "processed" / "konkani_wikipedia_filtered.txt",
     DATA_DIR / "processed" / "konkani_wikipedia_filtered_metadata.csv"),
    (DATA_DIR / "manual" / "konkani_wikipedia_sample.txt",
     DATA_DIR / "manual" / "konkani_wikipedia_sample_metadata.csv"),
]

MIN_WORDS = 50
MIN_DEVANAGARI_RATIO = 0.80     # Devanagari-only per decision D-001
SHARD_SIZE = 2000
LICENSE_NOTE = "CC BY-SA 4.0; attribution required; source: gom.wikipedia.org"


def pick_input() -> tuple[Path, Path]:
    for text_path, meta_path in CANDIDATE_INPUTS:
        if text_path.exists():
            return text_path, meta_path
    raise SystemExit(
        "No Konkani Wikipedia collection found. Expected one of:\n  "
        + "\n  ".join(str(p) for p, _ in CANDIDATE_INPUTS)
    )


def load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dry-run", action="store_true",
                        help="measure and report, write nothing")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    text_path, meta_path = pick_input()
    rows = load_metadata(meta_path)

    print("=" * 68)
    print(f"INGESTING: {SOURCE_NAME}")
    print(f"Language: {LANGUAGE}   Collection type: MANUAL (self-scraped)")
    print(f"Text:     {text_path.relative_to(REPO_ROOT)}")
    print(f"Metadata: {len(rows):,} rows"
          + ("" if rows else "  (absent - urls will be generic)"))
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    print("=" * 68)

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = None if args.dry_run else ManifestWriter(MANIFEST_PATH)

    rejected: dict[str, int] = {}
    accepted = words_total = 0
    script_counts: dict[str, int] = {}
    shard_index = 0
    shard = None
    if not args.dry_run:
        shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a", encoding="utf-8")

    def note(reason: str):
        rejected[reason] = rejected.get(reason, 0) + 1

    with open(text_path, "r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue

            meta = rows[index] if index < len(rows) else {}
            title = meta.get("title", "")
            url = meta.get("url") or (
                "https://gom.wikipedia.org/wiki/" + title.replace(" ", "_")
                if title else "https://gom.wikipedia.org/"
            )

            text = normalize(raw, keep_paragraphs=False)

            if len(text.split()) < MIN_WORDS:
                note("too_short")
                continue

            profile = profile_script(text)
            script_counts[profile.script] = script_counts.get(profile.script, 0) + 1

            # Roman and mixed-script pages are excluded by decision D-001. They
            # are counted above so the excluded sub-corpus is reported, not lost.
            if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
                note("not_devanagari_excluded_by_D001")
                continue

            langid = identify_marathi_konkani(text)
            if langid.label == "mr":
                note("langid_marathi_rejected")
                continue

            if deduper.is_duplicate(text):
                note("duplicate")
                continue

            accepted += 1
            words_total += len(text.split())

            if manifest:
                manifest.write(make_record(
                    text=text,
                    raw_text=raw,
                    source_name=SOURCE_NAME,
                    source_url=url,
                    collection_type=CollectionType.MANUAL_SCRAPE,
                    language=LANGUAGE,
                    preprocessing_applied=NORMALIZATION_STEPS + [
                        "wikitext_clean", "length_filter", "devanagari_only_D001"],
                    script=profile.script,
                    langid_score=langid.score,
                    langid_label=langid.label,
                    devanagari_ratio=profile.devanagari_ratio,
                    doc_id=meta.get("page_id", f"wp_{index:06d}"),
                    notes=f"title={title}; {LICENSE_NOTE}; secondary_source",
                ))

            if shard:
                shard.write(text + "\n")
                if accepted % SHARD_SIZE == 0:
                    shard.close()
                    shard_index += 1
                    shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt",
                                 "a", encoding="utf-8")

    if shard:
        shard.close()
    if manifest:
        manifest.close()

    print("\n" + "=" * 68)
    print("RUN SUMMARY")
    print("=" * 68)
    print(f"Pages accepted (manual):  {accepted:,}")
    print(f"Words accepted:           {words_total:,}")
    if accepted:
        print(f"Words per page:           {words_total / accepted:,.0f}")
    print(f"\nDeduplication:            {deduper.stats.to_dict()}")

    print("\nScript distribution of the source collection:")
    for script, count in sorted(script_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {script:14s} {count:,}")

    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:34s} {count:,}")

    print("\n" + "-" * 68)
    print("WHAT THIS UNLOCKS")
    print("-" * 68)
    print(f"  Manual words added:     {words_total:,}")
    print(f"  Downloaded words this")
    print(f"  permits (5x manual):    {words_total * 5:,} total corpus words")
    print("  The Konkani corpus is capped at 5x the manual total, so this")
    print("  number is the ceiling on how much of the books corpus may be used.")
    if not args.dry_run:
        print(f"\nManifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
        print(f"Shards:   {OUT_DIR.relative_to(REPO_ROOT)}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
