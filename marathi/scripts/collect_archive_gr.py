#!/usr/bin/env python3
"""
Pilot source M1: Maharashtra Government Resolutions from the Internet Archive.

WHAT THIS IS
------------
The Internet Archive holds ~170,796 items under the identifier prefix
`in.gov.maharashtra.gr.*`. Each is a scanned Maharashtra Government Resolution
that the Archive has already OCR'd, and each exposes a plain-text derivative.

Measured on a real item during source verification:
  in.gov.maharashtra.gr.202607071620477816
    -> file 202607071620477816_djvu.txt, 34,505 bytes
    -> 14,336 characters, 1,710 words, 81.6% Devanagari of non-whitespace

WHY THIS COUNTS AS MANUAL COLLECTION
------------------------------------
The brief counts OCR from books/PDFs and pages we gather and clean ourselves as
manual, and refuses to count a ready-made corpus as manual merely because we
downloaded it. There is no "Marathi GR corpus" to download. We enumerate items
through a search API, fetch each document individually, extract the OCR text
layer, and do all segmentation, normalization, language filtering and dedup
ourselves.

THROUGHPUT - WHY THIS VERSION EXISTS
------------------------------------
The first pilot worked correctly but ran at **1.7 documents/minute**. At that
rate 50,000 documents would take 20 days. Three causes, all now addressed:

1. **Retry storms dominated the clock.** archive.org sheds load with HTTP 5xx,
   and the old policy retried a single failing item up to 8 times with backoff
   growing to 300s - several minutes spent on one document. With ~170k items
   available, *skipping* a failing item is nearly free and retrying it is
   ruinously expensive. Documents now get 2 quick retries, then are abandoned
   and recorded. The identifier listing still retries patiently, because that
   is the one request that cannot be skipped.

2. **Two requests per document.** Resolving the filename through
   `/metadata/<id>` doubled the request count. The naming pattern is now known:
   the file is the identifier with the `in.gov.maharashtra.gr.` prefix removed,
   plus `_djvu.txt`. We construct that directly and fall back to the metadata
   lookup only on 404, so the expensive path runs only when the cheap one fails.

3. **Fetching was serial.** Document fetches now run in a small thread pool.
   Parsing, deduplication and manifest writing stay single-threaded, because
   the deduplicator holds shared state and correctness there matters more than
   speed.

Politeness is preserved: a modest worker count, a real contactable User-Agent,
and 429 responses always honoured with the full Retry-After wait.

USAGE
-----
    python3 marathi/scripts/collect_archive_gr.py --limit 300      # pilot
    python3 marathi/scripts/collect_archive_gr.py                  # full run
    python3 marathi/scripts/collect_archive_gr.py --workers 10     # push harder

Safe to Ctrl-C at any time; re-running resumes from the checkpoint.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

SOURCE_NAME = "archive_org_maharashtra_gr"
JOB_NAME = "marathi_archive_gr"
LANGUAGE = "mr"

SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
ADVANCED_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{ident}"
DOWNLOAD_URL = "https://archive.org/download/{ident}/{name}"
QUERY = "identifier:in.gov.maharashtra.gr.*"
IDENT_PREFIX = "in.gov.maharashtra.gr."

DATA_DIR = REPO_ROOT / "marathi" / "data"
RAW_DIR = DATA_DIR / "manual" / SOURCE_NAME
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / f"{JOB_NAME}.json"
MANIFEST_PATH = DATA_DIR / "manifests" / f"{JOB_NAME}.jsonl"

# Quality gates. Tuned from pilot measurements: a real GR measured 81.6%
# Devanagari, so 0.55 keeps bilingual documents while dropping English-only ones.
MIN_WORDS = 60
MIN_DEVANAGARI_RATIO = 0.55
LANGID_MIN_SCORE = 0.30
SHARD_SIZE = 2000

USER_AGENT = "lma-phase1-research/1.0 (student project; contact: pakkabhabad@gmail.com)"

# TIMEOUTS ARE THE THROUGHPUT LEVER.
# Measured: with a 45s timeout and 61% of fetches failing, a 120-document batch
# took ~13 minutes on 6 workers - one hung request blocked a worker for 45s, and
# with a retry that is 90s spent on a single document. A healthy archive.org
# response arrives in 1-3s, so anything slower is not worth waiting for when
# there are 170,796 items and failures can be retried on a later pass.
# (connect timeout, read timeout)
REQUEST_TIMEOUT = (5, 12)
DOC_RETRIES = 1               # one attempt, no grinding - failures are retried later
LIST_RETRIES = 8              # the listing is not skippable - be patient
DEFAULT_WORKERS = 12
BATCH = 240                   # identifiers fetched per concurrent wave

_thread_local = threading.local()
_HTML_WS = re.compile(r"\s+")


def _session() -> requests.Session:
    """One requests.Session per thread; Sessions are not safely shared."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = session
    return session


class Stats:
    """Counters mutated only from the main thread."""

    def __init__(self):
        self.errors = 0
        self.rate_limits = 0
        self.no_text = 0
        self.fetch_failed = 0


def _tidy(text: str, limit: int = 120) -> str:
    """Flatten an HTML error body to one short line.

    The previous version printed raw bodies containing carriage returns, which
    overwrote each other on the terminal and produced unreadable garbage like
    `</html>nter>nginx</center>erver Error</h1></center>d>`. Collapsing all
    whitespace first makes the log legible.
    """
    return _HTML_WS.sub(" ", text)[:limit].strip()


def http_get(url: str, *, retries: int, stats: Stats, quiet: bool = False, **kwargs):
    """GET with bounded retries. Returns a Response, or None if it gave up."""
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            response = _session().get(url, timeout=REQUEST_TIMEOUT, **kwargs)

            if response.status_code == 429:
                stats.rate_limits += 1
                wait = int(response.headers.get("Retry-After", 30))
                if not quiet:
                    print(f"  [429] honouring Retry-After {wait}s", flush=True)
                time.sleep(wait)
                continue

            if response.status_code == 404:
                return response          # meaningful; caller decides

            if response.status_code >= 500:
                if attempt == retries:
                    if not quiet:
                        print(f"  [{response.status_code}] giving up after {retries}"
                              f" | {_tidy(response.text)}", flush=True)
                    return None
                time.sleep(delay)
                delay *= 2
                continue

            response.raise_for_status()
            return response

        except (requests.Timeout, requests.ConnectionError):
            if attempt == retries:
                return None
            time.sleep(delay)
            delay *= 2
        except requests.RequestException:
            return None

    return None


def resolve_text_filename(ident: str, stats: Stats) -> str | None:
    """Ask the metadata endpoint which file holds the OCR text.

    Only called when the constructed name 404s. See fetch_document.
    """
    response = http_get(METADATA_URL.format(ident=ident),
                        retries=DOC_RETRIES, stats=stats, quiet=True)
    if response is None or response.status_code != 200:
        return None
    try:
        files = response.json().get("files", []) or []
    except ValueError:
        return None

    name = next((f.get("name") for f in files if f.get("format") == "DjVuTXT"), None)
    if not name:
        name = next((f.get("name") for f in files
                     if str(f.get("name", "")).endswith("_djvu.txt")), None)
    return name


def fetch_document(ident: str, stats: Stats) -> tuple[str, str | None, str]:
    """Fetch one item's OCR text. Runs inside the thread pool.

    Filename strategy, cheapest first:
      1. Construct `<numeric>_djvu.txt` by stripping the collection prefix.
         Verified against a real item; this is the pattern for this collection.
      2. Only if that 404s, ask /metadata/<id> for the real filename.

    The naive construction is NOT safe in general - across archive.org the file
    is sometimes named after the full identifier instead - which is why the
    fallback exists rather than being dropped.

    Returns (identifier, text_or_None, provenance_note).
    """
    numeric = ident[len(IDENT_PREFIX):] if ident.startswith(IDENT_PREFIX) else ident
    name = f"{numeric}_djvu.txt"

    response = http_get(DOWNLOAD_URL.format(ident=ident, name=name),
                        retries=DOC_RETRIES, stats=stats, quiet=True)

    if response is not None and response.status_code == 404:
        resolved = resolve_text_filename(ident, stats)
        if not resolved:
            return ident, None, "no_text_layer"
        name = resolved
        response = http_get(DOWNLOAD_URL.format(ident=ident, name=name),
                            retries=DOC_RETRIES, stats=stats, quiet=True)

    if response is None:
        return ident, None, "fetch_failed"
    if response.status_code != 200:
        return ident, None, "no_text_layer"

    response.encoding = "utf-8"
    return ident, response.text, f"text_file={name}; rights=not_stated"


def iter_identifiers(checkpoint: Checkpoint, stats: Stats, page_size: int):
    """Yield item identifiers, resuming from the persisted cursor."""
    while True:
        params = {"q": QUERY, "fields": "identifier", "count": page_size}
        if checkpoint.state.cursor:
            params["cursor"] = checkpoint.state.cursor

        response = http_get(SCRAPE_URL, retries=LIST_RETRIES, stats=stats, params=params)

        if response is None or response.status_code != 200:
            print("  [warn] scrape API unavailable; falling back to advancedsearch",
                  flush=True)
            yield from _iter_advancedsearch(checkpoint, stats, page_size)
            return

        payload = response.json()
        items = payload.get("items", [])
        if not items:
            return

        for item in items:
            if item.get("identifier"):
                yield item["identifier"]

        cursor = payload.get("cursor")
        checkpoint.save(cursor=cursor, page=checkpoint.state.page + 1)
        if not cursor:
            print("  [done] source exhausted", flush=True)
            return


def _iter_advancedsearch(checkpoint: Checkpoint, stats: Stats, page_size: int):
    """Fallback enumerator. Deep paging is capped, so this cannot reach all items."""
    page = int(checkpoint.state.extra.get("advsearch_page", 1))
    while True:
        response = http_get(
            ADVANCED_URL, retries=LIST_RETRIES, stats=stats,
            params={"q": QUERY, "fl[]": "identifier", "rows": page_size,
                    "page": page, "output": "json"},
        )
        if response is None or response.status_code != 200:
            return
        docs = response.json().get("response", {}).get("docs", [])
        if not docs:
            return
        for doc in docs:
            if doc.get("identifier"):
                yield doc["identifier"]
        page += 1
        checkpoint.save(advsearch_page=page, page=checkpoint.state.page + 1)


def evaluate(raw_text: str):
    """Normalize and apply quality gates. Returns (text, profile, langid, reason)."""
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


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Maharashtra GRs.")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N accepted documents (0 = no limit)")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"COLLECTING: {SOURCE_NAME}")
    print(f"Language: {LANGUAGE}   Collection type: manual (OCR text extraction)")
    print(f"Mode: {'PILOT limit=%d' % args.limit if args.limit else 'FULL RUN'} "
          f"| workers={args.workers}")
    print("=" * 68)

    checkpoint = Checkpoint(CHECKPOINT_PATH, JOB_NAME)
    stats = Stats()
    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = ManifestWriter(MANIFEST_PATH)

    if checkpoint.state.collected:
        print(f"Resuming: {checkpoint.state.collected:,} collected, "
              f"{len(checkpoint.seen):,} identifiers already seen")

    rejected: dict[str, int] = {}
    accepted_run = 0
    words_run = 0
    shard_index = checkpoint.state.collected // SHARD_SIZE
    shard = open(RAW_DIR / f"shard_{shard_index:05d}.txt", "a", encoding="utf-8")
    start = time.time()
    stop = False

    def note(reason: str):
        rejected[reason] = rejected.get(reason, 0) + 1
        checkpoint.state.skipped += 1

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            identifiers = iter_identifiers(checkpoint, stats, args.page_size)
            fresh = (i for i in identifiers if not checkpoint.has_seen(i))

            for batch in batched(fresh, BATCH):
                # NOTE: identifiers are marked seen only after a DEFINITIVE
                # outcome, never before the fetch. Marking upfront was a real
                # bug: archive.org was failing ~61% of fetches while degraded,
                # and pre-marking would have permanently discarded 61% of the
                # collection on a transient outage. Items that fail to fetch
                # stay unmarked, so a later pass retries them. Re-fetching a
                # document after a crash is harmless - dedup catches it.
                for ident, raw, provenance in pool.map(
                    lambda i: fetch_document(i, stats), batch
                ):
                    if raw is None:
                        if provenance == "fetch_failed":
                            stats.fetch_failed += 1
                            # deliberately NOT marked seen - retry on next pass
                            continue
                        checkpoint.mark_seen(ident)
                        note(provenance)
                        stats.no_text += 1
                        continue

                    checkpoint.mark_seen(ident)

                    text, profile, langid, reason = evaluate(raw)
                    if reason:
                        note(reason)
                        continue

                    if deduper.is_duplicate(text):
                        note("duplicate")
                        continue

                    manifest.write(make_record(
                        text=text,
                        raw_text=raw,
                        source_name=SOURCE_NAME,
                        source_url=f"https://archive.org/details/{ident}",
                        collection_type=CollectionType.MANUAL_OCR,
                        language=LANGUAGE,
                        preprocessing_applied=NORMALIZATION_STEPS,
                        script=profile.script,
                        langid_score=langid.score,
                        langid_label=langid.label,
                        devanagari_ratio=profile.devanagari_ratio,
                        doc_id=ident,
                        notes=provenance,
                    ))

                    shard.write(text.replace("\n", " ") + "\n")
                    checkpoint.state.collected += 1
                    accepted_run += 1
                    words_run += len(text.split())

                    if checkpoint.state.collected % SHARD_SIZE == 0:
                        shard.close()
                        shard_index += 1
                        shard = open(RAW_DIR / f"shard_{shard_index:05d}.txt",
                                     "a", encoding="utf-8")

                    if args.limit and accepted_run >= args.limit:
                        stop = True
                        break

                shard.flush()
                checkpoint.save(errors=stats.errors, rate_limit_hits=stats.rate_limits)

                elapsed = max(time.time() - start, 1e-6)
                print(f"  accepted={checkpoint.state.collected:,} "
                      f"skipped={checkpoint.state.skipped:,} "
                      f"dup={deduper.stats.duplicate_rate:.1%} "
                      f"words={words_run:,} "
                      f"rate={accepted_run / elapsed * 60:.1f}/min "
                      f"no_text={stats.no_text:,} failed={stats.fetch_failed:,}",
                      flush=True)

                if stop:
                    print("\nLimit reached.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved; re-run to resume.")
    finally:
        shard.close()
        manifest.close()
        checkpoint.save(errors=stats.errors, rate_limit_hits=stats.rate_limits,
                        dedup_stats=deduper.stats.to_dict(),
                        rejection_reasons=rejected)
        checkpoint.close()

    elapsed = time.time() - start
    print("\n" + "=" * 68)
    print("RUN SUMMARY")
    print("=" * 68)
    print(f"Accepted this run:   {accepted_run:,}")
    print(f"Accepted total:      {checkpoint.state.collected:,}")
    print(f"Words this run:      {words_run:,}")
    if accepted_run:
        print(f"Words per document:  {words_run / accepted_run:,.0f}")
    print(f"Skipped total:       {checkpoint.state.skipped:,}")
    print(f"No text layer:       {stats.no_text:,}")
    print(f"Fetch failed:        {stats.fetch_failed:,}")
    print(f"Rate-limit hits:     {stats.rate_limits:,}")
    print(f"Elapsed:             {elapsed / 60:.1f} min")
    if elapsed > 0 and accepted_run:
        per_min = accepted_run / elapsed * 60
        print(f"Throughput:          {per_min:.1f} docs/min")
        print(f"  -> 50,000 docs in  {50000 / max(per_min, 1e-9) / 60:.1f} hours")
    print(f"\nDeduplication:       {deduper.stats.to_dict()}")
    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:24s} {count:,}")
    print(f"\nManifest: {MANIFEST_PATH}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
