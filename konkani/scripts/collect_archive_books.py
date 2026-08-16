#!/usr/bin/env python3
"""
Konkani source K3: OCR text of Konkani books from the Internet Archive.

WHY THIS MATTERS MORE THAN ITS SIZE SUGGESTS
--------------------------------------------
This source is small - the Internet Archive holds only 44 items tagged Konkani,
of which roughly 13-15 are real books. It will contribute on the order of
100k-300k words, not millions.

It is still worth collecting, for two reasons that have nothing to do with size:

1. **It diversifies the manual claim.** Konkani manual collection currently
   stands at ~1.13M words, of which 99.95% is self-scraped Wikipedia. The TAs
   advised specifically against leaning on Wikipedia. A manual corpus that is
   almost entirely one well-known public dataset is a weak claim, however
   legitimately it was scraped. Digitised books are a genuinely different kind
   of source.

2. **OCR from books is the strongest form of manual collection.** The project
   brief names it first: "OCR from books/PDFs". Nobody can argue that fetching
   individual scanned books and extracting their text layer is anything other
   than manual work.

And because the Konkani corpus is capped at `5 x manual`, every word here earns
five words of the books corpus back into the corpus. 250k words of OCR raises
the Konkani ceiling by ~1.25M words.

WHY THE ITEM COUNT IS ITSELF A RESULT
-------------------------------------
Measured 14 Aug 2026 via the archive.org search API:

    language:Konkani                     ->      44 items
    identifier:in.gov.maharashtra.gr.*   -> 170,796 items

That ratio, alongside Sangraha holding 10.1M Konkani tokens in total, is the
evidence for the shortfall the specification explicitly permits for the
lower-resource language. This script records it rather than just citing it.

USAGE
-----
    python3 konkani/scripts/collect_archive_books.py --dry-run
    python3 konkani/scripts/collect_archive_books.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.checkpoint import Checkpoint                             # noqa: E402
from common.dedup import Deduplicator                                 # noqa: E402
from common.manifest import (                                         # noqa: E402
    CollectionType,
    ManifestWriter,
    make_record,
)
from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import NORMALIZATION_STEPS, normalize            # noqa: E402

SOURCE_NAME = "archive_org_konkani_books"
JOB_NAME = "konkani_archive_books"
LANGUAGE = "kok"

SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
METADATA_URL = "https://archive.org/metadata/{ident}"
DOWNLOAD_URL = "https://archive.org/download/{ident}/{name}"
QUERY = 'language:(Konkani OR Konknni OR Concani) AND mediatype:texts'

DATA_DIR = REPO_ROOT / "konkani" / "data"
RAW_DIR = DATA_DIR / "manual" / SOURCE_NAME
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / f"{JOB_NAME}.json"
MANIFEST_PATH = DATA_DIR / "manifests" / f"{JOB_NAME}.jsonl"

# Books are long; split into segments so dedup and shuffling stay meaningful.
MAX_SEGMENT_WORDS = 1200
MIN_WORDS = 60
MIN_DEVANAGARI_RATIO = 0.70    # these books mix English and Kannada script
SHARD_SIZE = 1000

USER_AGENT = "lma-phase1-research/1.0 (student project; contact: pakkabhabad@gmail.com)"
TIMEOUT = (5, 30)              # books are large files; allow a longer read

# Wikipedia-derived items carry the `wikipedia_` / `gomwiki-` prefixes. They are
# excluded: we already hold that content from our own Wikipedia collection, and
# counting it twice would inflate the manual total with duplicates.
EXCLUDE_PREFIXES = ("wikipedia_", "wiktionary_", "wikiquote_", "wikibooks_",
                    "gomwiki-", "dni.ncaa")


def get(url: str, **kwargs):
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": USER_AGENT}, **kwargs)
        return r if r.status_code == 200 else None
    except requests.RequestException:
        return None


def list_items() -> list[str]:
    """Enumerate candidate Konkani text items."""
    response = get(SCRAPE_URL, params={"q": QUERY, "fields": "identifier",
                                       "count": 1000})
    if response is None:
        print("  [error] archive.org search unavailable")
        return []
    payload = response.json()
    total = payload.get("total", 0)
    items = [i["identifier"] for i in payload.get("items", []) if i.get("identifier")]
    print(f"  archive.org reports {total:,} items matching the Konkani query")
    kept = [i for i in items if not i.startswith(EXCLUDE_PREFIXES)]
    print(f"  {len(items):,} returned, {len(kept):,} after excluding "
          f"Wikipedia-derived and non-text items")
    return kept


def fetch_text(ident: str) -> tuple[str | None, str]:
    """Resolve the OCR text file through item metadata, then download it."""
    meta_response = get(METADATA_URL.format(ident=ident))
    if meta_response is None:
        return None, ""
    try:
        meta = meta_response.json()
    except ValueError:
        return None, ""

    files = meta.get("files", []) or []
    name = next((f.get("name") for f in files if f.get("format") == "DjVuTXT"), None)
    if not name:
        name = next((f.get("name") for f in files
                     if str(f.get("name", "")).endswith("_djvu.txt")), None)
    if not name:
        return None, ""

    item_meta = meta.get("metadata", {}) or {}
    note = "; ".join(
        f"{k}={item_meta.get(k) or 'not_stated'}"
        for k in ("title", "licenseurl", "rights", "possible-copyright-status")
    )

    response = get(DOWNLOAD_URL.format(ident=ident, name=name))
    if response is None:
        return None, ""
    response.encoding = "utf-8"
    return response.text, f"{note}; text_file={name}"


def split_segments(text: str, max_words: int = MAX_SEGMENT_WORDS):
    paragraphs = [p for p in text.split("\n") if p.strip()]
    segment, count = [], 0
    for para in paragraphs:
        words = len(para.split())
        if count + words > max_words and segment:
            yield "\n".join(segment)
            segment, count = [], 0
        segment.append(para)
        count += words
    if segment:
        yield "\n".join(segment)


def evaluate(raw_segment: str):
    text = normalize(raw_segment, keep_paragraphs=True)
    if len(text.split()) < MIN_WORDS:
        return text, None, None, "too_short"

    profile = profile_script(text)
    if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
        # These volumes genuinely mix English and Kannada script; per decision
        # D-001 only Devanagari enters the corpus.
        return text, profile, None, "not_devanagari_excluded_by_D001"

    langid = identify_marathi_konkani(text)
    if langid.label == "mr":
        return text, profile, langid, "langid_marathi_rejected"
    return text, profile, langid, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Konkani books via OCR text.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    print("=" * 70)
    print(f"COLLECTING: {SOURCE_NAME}")
    print(f"Language: {LANGUAGE}   Collection type: MANUAL (OCR text extraction)")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    print("=" * 70)

    identifiers = list_items()
    if not identifiers:
        print("\nNo items found. archive.org may be unavailable - retry later.")
        return 1

    if not args.dry_run:
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    # A DRY RUN MUST NOT PERSIST CHECKPOINT STATE.
    #
    # The first version wrote to the real checkpoint even under --dry-run, so
    # the dry run marked all 14 items as "seen" and the subsequent real run
    # skipped every one of them and collected 0 documents. A preview that
    # silently consumes the work it was previewing is worse than no preview.
    #
    # Dry runs now checkpoint into a throwaway directory.
    if args.dry_run:
        import tempfile
        checkpoint_path = Path(tempfile.mkdtemp(prefix="dryrun_")) / f"{JOB_NAME}.json"
    else:
        checkpoint_path = CHECKPOINT_PATH

    checkpoint = Checkpoint(checkpoint_path, JOB_NAME)
    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = None if args.dry_run else ManifestWriter(MANIFEST_PATH)

    rejected: dict[str, int] = {}
    accepted = words_total = books_used = 0
    shard_index = 0
    shard = None
    if not args.dry_run:
        shard = open(RAW_DIR / f"shard_{shard_index:05d}.txt", "a", encoding="utf-8")
    start = time.time()

    try:
        for n, ident in enumerate(identifiers, 1):
            if checkpoint.has_seen(ident):
                continue

            print(f"  [{n}/{len(identifiers)}] {ident}", flush=True)
            raw, provenance = fetch_text(ident)
            time.sleep(1.0)

            if raw is None:
                rejected["no_text_layer"] = rejected.get("no_text_layer", 0) + 1
                checkpoint.mark_seen(ident)
                continue

            book_segments = 0
            for segment in split_segments(raw):
                text, profile, langid, reason = evaluate(segment)
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                if deduper.is_duplicate(text):
                    rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                    continue

                accepted += 1
                book_segments += 1
                words_total += len(text.split())

                if manifest:
                    manifest.write(make_record(
                        text=text, raw_text=segment,
                        source_name=SOURCE_NAME,
                        source_url=f"https://archive.org/details/{ident}",
                        collection_type=CollectionType.MANUAL_OCR,
                        language=LANGUAGE,
                        preprocessing_applied=NORMALIZATION_STEPS + [
                            "ocr_text_layer", "segment_split", "devanagari_only_D001"],
                        script=profile.script,
                        langid_score=langid.score,
                        langid_label=langid.label,
                        devanagari_ratio=profile.devanagari_ratio,
                        doc_id=f"{ident}_seg_{book_segments:05d}",
                        notes=provenance,
                    ))
                if shard:
                    shard.write(text.replace("\n", " ") + "\n")
                    if accepted % SHARD_SIZE == 0:
                        shard.close()
                        shard_index += 1
                        shard = open(RAW_DIR / f"shard_{shard_index:05d}.txt",
                                     "a", encoding="utf-8")

            if book_segments:
                books_used += 1
            print(f"        {book_segments} usable segment(s)", flush=True)

            checkpoint.mark_seen(ident)
            checkpoint.save(collected=accepted)

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved; re-run to resume.")
    finally:
        if shard:
            shard.close()
        if manifest:
            manifest.close()
        checkpoint.save(collected=accepted, dedup_stats=deduper.stats.to_dict(),
                        rejection_reasons=rejected)
        checkpoint.close()

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"Items examined:      {len(identifiers):,}")
    print(f"Books contributing:  {books_used:,}")
    print(f"Segments accepted:   {accepted:,}")
    print(f"Words accepted:      {words_total:,}")
    print(f"Elapsed:             {elapsed / 60:.1f} min")
    print(f"\nDeduplication:       {deduper.stats.to_dict()}")
    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:34s} {count:,}")

    print("\n" + "-" * 70)
    print("EFFECT ON THE KONKANI CORPUS CEILING")
    print("-" * 70)
    print(f"  manual words added        {words_total:>12,}")
    print(f"  corpus ceiling raised by  {words_total * 5:>12,} words")
    print("  (the Konkani corpus is capped at 5x its manual total)")
    if not args.dry_run:
        print(f"\nManifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
