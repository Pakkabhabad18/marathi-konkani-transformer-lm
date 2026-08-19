#!/usr/bin/env python3
"""
Konkani source K3: OCR text of Konkani books from the Internet Archive.

STATUS: REVISED 16 Aug 2026. READ THIS FIRST.
---------------------------------------------
Everything below the line "WHY THE ITEM COUNT IS ITSELF A RESULT" was written
against a query that was wrong. This script originally reported that the
Internet Archive holds **44** Konkani items, and that figure was then used as
headline evidence that Konkani data barely exists.

The archive actually holds **5,093** items matching `language:kok AND
mediatype:texts` (verified 16 Aug 2026). The old query asked for the English
name of the language and never for `kok`, the ISO 639-2 code that cataloguers
actually use. See D-018 and `konkani/scripts/discover_sources.py`.

The original text is left in place rather than rewritten, per the standing rule
that superseded results stay visible.

WHY THIS MATTERS MORE THAN ITS SIZE SUGGESTS  [superseded - see above]
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
# CORRECTED 16 Aug 2026 (see D-018). The previous query was
#     'language:(Konkani OR Konknni OR Concani) AND mediatype:texts'
# which returned 44 items and was reported as "the Internet Archive's entire
# Konkani holdings". It is not. Archive.org's `language` field is free text and
# cataloguers overwhelmingly write the ISO 639-2 code, not the English name:
#
#     language:(Konkani OR Konknni OR Concani)  ->     44 items
#     language:kok AND mediatype:texts          ->  5,093 items   (verified)
#
# The old query measured our spelling list, not the archive.
# BROADENED 19 Aug 2026 (D-033). The language-field queries above are the
# reliable ones, but discovery measured that `subject:Konkani AND
# mediatype:texts` returns 848 items, most overlapping and a residue that the
# language field never covers - items whose cataloguer filled in the subject but
# left `language` blank or wrong. Enumeration deduplicates identifiers and the
# .seen checkpoint skips anything already collected, so adding this term costs
# only the new items.
QUERY = ('(language:(kok OR gom OR Konkani OR Konknni OR Concani) '
         'OR subject:Konkani OR subject:Konknni) '
         'AND mediatype:texts')

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
    """Enumerate candidate Konkani text items, following the scrape cursor.

    CORRECTED 16 Aug 2026 (D-018). The previous implementation issued a single
    request with `count=1000` and no cursor, so it saw at most one page. With
    the old 44-item query that happened to be the whole result set, which is
    why the bug was invisible: two independent faults - a wrong query and a
    missing cursor - cancelled out to produce a plausible number. With the
    corrected query the population is ~5,000 items and pagination is required.

    Absence of a `cursor` in the response is the only correct stop condition.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    cursor, page, total = None, 0, 0

    while True:
        params = {"q": QUERY, "fields": "identifier", "count": 1000}
        if cursor:
            params["cursor"] = cursor
        response = get(SCRAPE_URL, params=params)
        if response is None:
            print(f"  [error] archive.org search unavailable at page {page}")
            break
        payload = response.json()
        if page == 0:
            total = payload.get("total", 0)
            print(f"  archive.org reports {total:,} items matching the "
                  f"Konkani query")
        for item in payload.get("items", []):
            ident = item.get("identifier")
            if ident and ident not in seen:
                seen.add(ident)
                ordered.append(ident)
        cursor = payload.get("cursor")
        page += 1
        if not cursor:
            break
        time.sleep(0.2)

    kept = [i for i in ordered if not i.startswith(EXCLUDE_PREFIXES)]
    print(f"  {len(ordered):,} identifiers enumerated over {page} page(s), "
          f"{len(kept):,} after excluding Wikipedia-derived and non-text items")
    return kept


def fetch_text(ident: str, attempts: int = 3) -> tuple[str | None, str, str]:
    """Resolve the OCR text file through item metadata, then download it.

    Returns (text, provenance, status) where status is one of:
        "ok"            - text retrieved
        "no_text_layer" - the item genuinely has no djvu.txt   (DEFINITIVE)
        "transient"     - the request failed                   (RETRYABLE)

    CORRECTED (D-027). The previous version returned a bare `None` for all four
    failure paths - metadata request failed, JSON did not parse, no djvu.txt
    exists, download failed - and the caller recorded every one as
    `no_text_layer` and then called `mark_seen()`. A network hiccup was thereby
    written down as a permanent property of the item and never retried.

    Measured consequence: the 12-hour full run reported 3,734 items (73.1%) as
    `no_text_layer`, while a 120-item random probe two days earlier found ZERO.
    A 60-item metadata recheck then found **98.3% of them do have a text
    layer** - roughly 13.4M words discarded by a misclassification.

    This is the same failure class already recorded for the Marathi GR
    collector, where marking identifiers seen before a definitive outcome would
    have discarded 61% of that collection. It was caught there before the run
    and missed here. Hence the explicit three-valued status: the caller can no
    longer conflate "we know there is nothing" with "we failed to ask".
    """
    meta_response = None
    for attempt in range(attempts):
        meta_response = get(METADATA_URL.format(ident=ident))
        if meta_response is not None:
            break
        time.sleep(2.0 * (attempt + 1))      # linear backoff, deliberately gentle
    if meta_response is None:
        return None, "", "transient"
    try:
        meta = meta_response.json()
    except ValueError:
        return None, "", "transient"

    files = meta.get("files", []) or []
    name = next((f.get("name") for f in files if f.get("format") == "DjVuTXT"), None)
    if not name:
        name = next((f.get("name") for f in files
                     if str(f.get("name", "")).endswith("_djvu.txt")), None)
    if not name:
        return None, "", "no_text_layer"

    item_meta = meta.get("metadata", {}) or {}
    note = "; ".join(
        f"{k}={item_meta.get(k) or 'not_stated'}"
        for k in ("title", "licenseurl", "rights", "possible-copyright-status")
    )

    response = None
    for attempt in range(attempts):
        response = get(DOWNLOAD_URL.format(ident=ident, name=name))
        if response is not None:
            break
        time.sleep(2.0 * (attempt + 1))
    if response is None:
        return None, "", "transient"
    response.encoding = "utf-8"
    return response.text, f"{note}; text_file={name}", "ok"


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
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N newly-fetched items (0 = all)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="walk the enumeration in archive.org's order "
                             "instead of a seeded shuffle (not recommended)")
    parser.add_argument("--shuffle-seed", type=int, default=20260816)
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="politeness delay between items, seconds. Default "
                             "lowered from 1.0 to 0.3: the 120-item discovery "
                             "probe ran at 0.3s with ZERO fetch failures, and "
                             "at 5,178 items 1.0s costs 1.4 h of pure sleep.")
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

    # SHUFFLE, with a fixed seed. This matters for two reasons.
    #
    # 1. A partial run stays representative. This source is ~5,100 items at
    #    ~7 s each, so a full pass is ~10 hours. If it has to be stopped early -
    #    deadline, laptop sleeping, network - what we keep should be a random
    #    sample of Konkani books, not "everything whose identifier sorts first".
    #    Alphabetical order correlates with uploader, collection and era, so a
    #    truncated in-order run yields a biased corpus, and the bias is silent.
    #
    # 2. It explains the smoke test. The 120-item discovery probe drew a random
    #    sample and found 0% of items missing an OCR text layer; the first 30
    #    identifiers in enumeration order gave 43%. Same population, different
    #    sampling - so the head of the list is measurably unrepresentative.
    #
    # The seed is fixed so the order is identical across resumes; combined with
    # the .seen checkpoint, an interrupted run continues exactly where it left
    # off rather than re-walking a different permutation.
    if not args.no_shuffle:
        import random as _random
        _random.Random(args.shuffle_seed).shuffle(identifiers)
        print(f"  order: seeded shuffle (seed {args.shuffle_seed}) so that a "
              f"partial run stays representative")

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
    fetched = 0
    interrupted = False
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
            raw, provenance, status = fetch_text(ident)
            time.sleep(args.sleep)
            fetched += 1

            if status == "transient":
                # NOT marked seen: we never got a definitive answer, so this
                # identifier stays eligible for the next run.
                rejected["transient_fetch_failure"] = rejected.get(
                    "transient_fetch_failure", 0) + 1
                continue
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

            if args.limit and fetched >= args.limit:
                print(f"\n  --limit {args.limit} reached; stopping cleanly.")
                interrupted = True
                break

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Checkpoint saved; re-run to resume.")
    finally:
        if shard:
            shard.close()
        if manifest:
            manifest.close()
        # This source is finite, so record COMPLETED rather than leaving the
        # health monitor to infer a stall from a checkpoint that will never be
        # written again.
        #
        # CORRECTED: `finished=True` was previously set unconditionally in this
        # `finally` block, so a Ctrl+C - or hitting --limit - also marked the
        # job complete. A resumable job that reports itself finished after an
        # interrupt is how a half-collected source gets treated as done.
        checkpoint.save(collected=accepted, dedup_stats=deduper.stats.to_dict(),
                        rejection_reasons=rejected,
                        finished=not interrupted)
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
