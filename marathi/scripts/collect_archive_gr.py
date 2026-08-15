#!/usr/bin/env python3
"""
Pilot source M1: Maharashtra Government Resolutions from the Internet Archive.

WHAT THIS IS
------------
The Internet Archive holds 170,725 items under the identifier prefix
`in.gov.maharashtra.gr.*`. Each is a scanned Maharashtra Government Resolution
that the Archive has already OCR'd, and each exposes a plain-text derivative at

    https://archive.org/download/<identifier>/<identifier>_djvu.txt

Verified during source selection on 14 Aug 2026:
  - total items matching the prefix: 170,725 (archive.org scrape API `total`)
  - sample item in.gov.maharashtra.gr.202607071620477816 has a 34.5 KB
    `_djvu.txt`, language metadata "Marathi, English", OCR by Tesseract 5.3.0

WHY THIS COUNTS AS MANUAL COLLECTION
------------------------------------
The project brief counts OCR from books/PDFs and pages we gather and clean
ourselves as manual, and explicitly refuses to count a ready-made corpus as
manual merely because we downloaded and cleaned it. This source is the former:
there is no "Marathi GR corpus" dataset to download. We enumerate items through
a search API, fetch each document individually, extract the OCR text layer, and
do all segmentation, normalization, language filtering and deduplication
ourselves. The manifest records every document's own source URL.

WHAT TO WATCH OUT FOR (this is why we pilot before scaling)
-----------------------------------------------------------
1. These documents are bilingual. Archive metadata says "Marathi, English" and
   the header/footer furniture is often English. Per-document language filtering
   is mandatory, not optional.
2. They are formulaic. Departmental headers, reference-number blocks and closing
   signature paragraphs repeat across thousands of documents. Expect a high
   near-duplicate rate; that is what the MinHash pass is for.
3. They are OCR of scans, so there will be recognition noise.
4. The register is narrow - bureaucratic prose only. This source can supply
   volume but must not be the only manual source, or Model H will be fluent in
   government circulars and nothing else.

The pilot exists to measure 1-4 as numbers before committing to a full crawl.

USAGE
-----
    # Pilot: 300 documents, then stop and report.
    python3 marathi/scripts/collect_archive_gr.py --limit 300

    # Full run; safe to Ctrl-C and re-run, it resumes from the checkpoint.
    python3 marathi/scripts/collect_archive_gr.py

    # Inspect progress from another terminal without touching the job:
    python3 tools/health_check.py --job marathi_archive_gr
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.checkpoint import Checkpoint                      # noqa: E402
from common.dedup import Deduplicator                          # noqa: E402
from common.manifest import (                                  # noqa: E402
    CollectionType,
    ManifestWriter,
    make_record,
)
from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import NORMALIZATION_STEPS, normalize     # noqa: E402

SOURCE_NAME = "archive_org_maharashtra_gr"
JOB_NAME = "marathi_archive_gr"
LANGUAGE = "mr"

SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
QUERY = "identifier:in.gov.maharashtra.gr.*"
ADVANCED_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{ident}"
DOWNLOAD_URL = "https://archive.org/download/{ident}/{name}"

DATA_DIR = REPO_ROOT / "marathi" / "data"
RAW_DIR = DATA_DIR / "manual" / SOURCE_NAME
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / f"{JOB_NAME}.json"
MANIFEST_PATH = DATA_DIR / "manifests" / f"{JOB_NAME}.jsonl"

# Quality gates. Chosen conservatively for the pilot and re-tuned from its output.
MIN_WORDS = 60                 # below this a GR is a cover page or a stub
MIN_DEVANAGARI_RATIO = 0.55    # bilingual documents that are mostly English are dropped
LANGID_MIN_SCORE = 0.30        # must lean Marathi, not Konkani, not undecidable
SHARD_SIZE = 2000              # documents per output file

# A contactable User-Agent is the convention archive.org asks for. Keep it short:
# during the first pilot the longer parenthesised variant coincided with repeated
# HTTP 500s from the scrape API. Run tools/diagnose_source.py to re-test.
USER_AGENT = "lma-phase1-research/1.0 (student project; contact: pakkabhabad@gmail.com)"
REQUEST_TIMEOUT = 90
POLITE_DELAY = 0.7             # seconds between document fetches
MAX_RETRIES = 8                # archive.org sheds load with 5xx; be patient
MAX_BACKOFF = 300


class RateLimitedSession:
    """requests session with backoff, 429 handling and error accounting."""

    def __init__(self, checkpoint: Checkpoint):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.checkpoint = checkpoint

    def get(self, url: str, **kwargs) -> requests.Response | None:
        """GET with exponential backoff. Returns None if permanently failed.

        Every retry message prints the attempt number. This matters: without it,
        two consecutive '5s' messages are ambiguous between one call retrying and
        two separate calls each on their first attempt, and those two situations
        have completely different causes.
        """
        delay = 5
        for attempt in range(1, MAX_RETRIES + 1):
            tag = f"{attempt}/{MAX_RETRIES}"
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)

                if response.status_code == 429:
                    self.checkpoint.state.rate_limit_hits += 1
                    wait = int(response.headers.get("Retry-After", delay))
                    print(f"  [429 {tag}] rate limited, waiting {wait}s", flush=True)
                    time.sleep(wait)
                    delay = min(delay * 2, MAX_BACKOFF)
                    continue

                if response.status_code == 404:
                    return None          # item has no text derivative; not an error

                if response.status_code >= 500:
                    # Print the body: a 500 with an error message is a different
                    # problem from a 500 with an empty body (load shedding).
                    body = response.text[:200].replace("\n", " ").strip()
                    print(f"  [{response.status_code} {tag}] server error, "
                          f"retry in {delay}s | body: {body or '(empty)'}", flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, MAX_BACKOFF)
                    continue

                response.raise_for_status()
                return response

            except (requests.Timeout, requests.ConnectionError) as exc:
                print(f"  [net {tag}] {type(exc).__name__}, retry in {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, MAX_BACKOFF)
            except requests.RequestException as exc:
                print(f"  [err {tag}] {exc}", flush=True)
                self.checkpoint.state.errors += 1
                return None

        self.checkpoint.state.errors += 1
        return None


def iter_identifiers(session: RateLimitedSession, checkpoint: Checkpoint, page_size: int):
    """Yield item identifiers, resuming from the persisted cursor.

    This is the piece the original Wikipedia collector was missing. The cursor is
    written to the checkpoint after every page, so a restart continues from the
    exact page it stopped on instead of replaying the traversal from item zero.
    """
    while True:
        params = {"q": QUERY, "fields": "identifier", "count": page_size}
        if checkpoint.state.cursor:
            params["cursor"] = checkpoint.state.cursor

        response = session.get(SCRAPE_URL, params=params)

        if response is None:
            # The scrape API is the correct tool for 170k items, but it sheds
            # load with 5xx. Rather than abandoning the run, fall back to the
            # advancedsearch endpoint so the pilot can still produce numbers.
            print("  [warn] scrape API unavailable; falling back to advancedsearch",
                  flush=True)
            yield from _iter_identifiers_advancedsearch(session, checkpoint, page_size)
            return

        payload = response.json()
        items = payload.get("items", [])
        if not items:
            return

        for item in items:
            ident = item.get("identifier")
            if ident:
                yield ident

        cursor = payload.get("cursor")
        checkpoint.save(cursor=cursor, page=checkpoint.state.page + 1)

        if not cursor:
            print("  [done] no further cursor; source exhausted", flush=True)
            return


def _iter_identifiers_advancedsearch(
    session: RateLimitedSession, checkpoint: Checkpoint, page_size: int
):
    """Fallback enumerator using advancedsearch.php page-based paging.

    LIMITATION, stated because it matters: advancedsearch caps deep paging at a
    few thousand rows, so this cannot enumerate all 170,725 items. It is enough
    to run a pilot and get real numbers. The full run needs the scrape API, or
    the query must be partitioned (e.g. by year) to stay under the paging cap.
    """
    page = int(checkpoint.state.extra.get("advsearch_page", 1))

    while True:
        response = session.get(
            ADVANCED_URL,
            params={"q": QUERY, "fl[]": "identifier", "rows": page_size,
                    "page": page, "output": "json"},
        )
        if response is None:
            print("  [fatal] advancedsearch fallback also failed", flush=True)
            return

        docs = response.json().get("response", {}).get("docs", [])
        if not docs:
            print("  [done] advancedsearch returned no further results", flush=True)
            return

        for doc in docs:
            ident = doc.get("identifier")
            if ident:
                yield ident

        page += 1
        checkpoint.save(advsearch_page=page, page=checkpoint.state.page + 1)


def fetch_document(session: RateLimitedSession, ident: str) -> tuple[str | None, str]:
    """Resolve the OCR text file via item metadata, then download it.

    WHY THE EXTRA REQUEST
    ---------------------
    The text file's basename is NOT reliably the item identifier, and assuming
    it is produces a 404 on every single document. Two real examples:

        item in.gov.maharashtra.gr.202607071620477816
          -> file 202607071620477816_djvu.txt          (numeric part only)

        item konkanibhashaman0000jbmo
          -> file konkanibhashaman0000jbmo_djvu.txt    (full identifier)

    The convention differs between collections, so the name cannot be
    constructed - it has to be looked up. `https://archive.org/metadata/<id>`
    is authoritative and cheap, and it also carries the rights fields we want
    recorded as provenance.

    This bug was caught by tools/diagnose_source.py PROBE 4 before the full run.

    Returns:
        (text, provenance_note). text is None if the item has no text layer.
    """
    meta_response = session.get(METADATA_URL.format(ident=ident))
    if meta_response is None:
        return None, ""

    try:
        meta = meta_response.json()
    except ValueError:
        return None, ""

    files = meta.get("files", []) or []

    # Prefer the declared format; fall back to the filename suffix.
    name = next((f.get("name") for f in files if f.get("format") == "DjVuTXT"), None)
    if not name:
        name = next(
            (f.get("name") for f in files
             if str(f.get("name", "")).endswith("_djvu.txt")),
            None,
        )
    if not name:
        return None, ""

    # Record rights provenance. These fields are often absent on government
    # items; "not stated" is itself information and is logged rather than
    # silently assumed to mean "public domain".
    item_meta = meta.get("metadata", {}) or {}
    note_parts = []
    for field in ("licenseurl", "rights", "possible-copyright-status"):
        value = item_meta.get(field)
        note_parts.append(f"{field}={value}" if value else f"{field}=not_stated")
    note_parts.append(f"text_file={name}")

    response = session.get(DOWNLOAD_URL.format(ident=ident, name=name))
    if response is None:
        return None, ""

    response.encoding = "utf-8"
    return response.text, "; ".join(note_parts)


def evaluate(raw_text: str):
    """Normalize and apply the quality gates. Returns (text, profile, langid, reason)."""
    text = normalize(raw_text, keep_paragraphs=True)

    if len(text.split()) < MIN_WORDS:
        return text, None, None, "too_short"

    profile = profile_script(text)
    if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
        return text, profile, None, "not_enough_devanagari"

    langid = identify_marathi_konkani(text)
    if langid.label != "mr" or langid.score < LANGID_MIN_SCORE:
        return text, profile, langid, f"langid_{langid.label}"

    return text, profile, langid, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N accepted documents (0 = no limit)")
    parser.add_argument("--page-size", type=int, default=200,
                        help="identifiers per scrape-API page")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"COLLECTING: {SOURCE_NAME}")
    print(f"Language: {LANGUAGE}   Collection type: manual (OCR text extraction)")
    print(f"Mode: {'PILOT (limit=%d)' % args.limit if args.limit else 'FULL RUN'}")
    print("=" * 68)

    checkpoint = Checkpoint(CHECKPOINT_PATH, JOB_NAME)
    session = RateLimitedSession(checkpoint)
    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = ManifestWriter(MANIFEST_PATH)

    if checkpoint.state.collected:
        print(f"Resuming: {checkpoint.state.collected:,} already collected, "
              f"{len(checkpoint.seen):,} identifiers seen, "
              f"cursor={'set' if checkpoint.state.cursor else 'none'}")

    rejected: dict[str, int] = {}
    accepted_this_run = 0
    shard_index = checkpoint.state.collected // SHARD_SIZE
    shard_file = open(RAW_DIR / f"shard_{shard_index:05d}.txt", "a", encoding="utf-8")
    start = time.time()

    try:
        for ident in iter_identifiers(session, checkpoint, args.page_size):
            if checkpoint.has_seen(ident):
                continue
            checkpoint.mark_seen(ident)

            raw, provenance_note = fetch_document(session, ident)
            time.sleep(POLITE_DELAY)

            if raw is None:
                rejected["no_text_layer"] = rejected.get("no_text_layer", 0) + 1
                checkpoint.state.skipped += 1
                continue

            text, profile, langid, reason = evaluate(raw)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                checkpoint.state.skipped += 1
                continue

            if deduper.is_duplicate(text):
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                checkpoint.state.skipped += 1
                continue

            url = f"https://archive.org/details/{ident}"
            record = make_record(
                text=text,
                raw_text=raw,
                source_name=SOURCE_NAME,
                source_url=url,
                collection_type=CollectionType.MANUAL_OCR,
                language=LANGUAGE,
                preprocessing_applied=NORMALIZATION_STEPS,
                script=profile.script,
                langid_score=langid.score,
                langid_label=langid.label,
                devanagari_ratio=profile.devanagari_ratio,
                doc_id=ident,
                notes=provenance_note,
            )
            manifest.write(record)

            shard_file.write(text.replace("\n", " ") + "\n")
            shard_file.flush()

            checkpoint.state.collected += 1
            accepted_this_run += 1

            if checkpoint.state.collected % SHARD_SIZE == 0:
                shard_file.close()
                shard_index += 1
                shard_file = open(RAW_DIR / f"shard_{shard_index:05d}.txt", "a",
                                  encoding="utf-8")

            if accepted_this_run % 25 == 0:
                checkpoint.save()
                rate = accepted_this_run / max(time.time() - start, 1e-6)
                print(f"  accepted={checkpoint.state.collected:,} "
                      f"skipped={checkpoint.state.skipped:,} "
                      f"dup_rate={deduper.stats.duplicate_rate:.1%} "
                      f"rate={rate * 60:.1f}/min", flush=True)

            if args.limit and accepted_this_run >= args.limit:
                print("\nPilot limit reached.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved; re-run to resume.")
    finally:
        shard_file.close()
        checkpoint.save(dedup_stats=deduper.stats.to_dict(),
                        rejection_reasons=rejected)
        checkpoint.close()
        manifest.close()

    elapsed = time.time() - start
    print("\n" + "=" * 68)
    print("RUN SUMMARY")
    print("=" * 68)
    print(f"Accepted this run:   {accepted_this_run:,}")
    print(f"Accepted total:      {checkpoint.state.collected:,}")
    print(f"Skipped total:       {checkpoint.state.skipped:,}")
    print(f"Rate-limit hits:     {checkpoint.state.rate_limit_hits:,}")
    print(f"Errors:              {checkpoint.state.errors:,}")
    print(f"Elapsed:             {elapsed / 60:.1f} min")
    print(f"\nDeduplication:       {deduper.stats.to_dict()}")
    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:24s} {count:,}")
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Shards:   {RAW_DIR}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
