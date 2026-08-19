#!/usr/bin/env python3
"""
Were the 3,734 `no_text_layer` rejections real, or transient network failures?

WHY THIS MATTERS ENOUGH TO CHECK BEFORE FINALISING
--------------------------------------------------
The full Konkani book collection reported:

    items examined      5,110
    no_text_layer       3,734   = 73.1%
    books contributing    733   = 14.3%

The 120-item discovery probe, drawn at random from the same population two days
earlier, found **0 items (0.0%)** with no OCR text layer, and predicted a 48.3%
contribution rate. A 0% versus 73% split on the same population is not sampling
variance. Something changed between the probe and the run.

The likely explanation is in `fetch_text()`, which returns `(None, "")` for FOUR
different situations:

    * the metadata request failed          (network / rate limit)
    * the metadata JSON did not parse      (truncated response)
    * the item genuinely has no djvu.txt   (a real "no text layer")
    * the text download itself failed      (network / rate limit)

The caller cannot tell them apart, records all four as `no_text_layer`, and then
calls `checkpoint.mark_seen(ident)` - so a transient failure is recorded as a
permanent property of the item and is never retried. Over a 12-hour run,
archive.org rate-limiting would produce exactly this signature.

This is the same failure class as the Marathi GR bug recorded earlier in
`phase1_decisions.md`, where marking identifiers seen before a definitive
outcome would have permanently discarded 61% of that collection. There it was
caught before the run. Here it was not.

WHAT THIS SCRIPT DOES
---------------------
`--diagnose` (default): take a random sample of the identifiers that were marked
seen but contributed nothing, and issue ONE cheap metadata request each to ask a
single question - does this item have a djvu.txt file at all? It downloads no
text and changes no state. Roughly one minute per 100 items.

If most sampled items DO have a text layer, the rejections were transient and
re-running recovers real data. If most genuinely lack one, the run was correct
and the corpus is complete as it stands.

`--unmark` rewrites the checkpoint's `.seen` file to drop the non-contributing
identifiers, so a subsequent normal run retries exactly those and nothing else.
It writes a timestamped backup first, because a corrupted seen-set would force a
full 12-hour recollection.

USAGE
-----
    python3 konkani/scripts/recheck_failed_items.py --diagnose --sample 60
    python3 konkani/scripts/recheck_failed_items.py --unmark
    python3 konkani/scripts/collect_archive_books.py      # retries only those
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.manifest import read_manifest        # noqa: E402

DATA_DIR = REPO_ROOT / "konkani" / "data"
SEEN_PATH = DATA_DIR / "checkpoints" / "konkani_archive_books.seen"
MANIFEST_PATH = DATA_DIR / "manifests" / "konkani_archive_books.jsonl"

METADATA_URL = "https://archive.org/metadata/{ident}"
USER_AGENT = ("lma-phase1-research/1.0 "
              "(student project; contact: pakkabhabad@gmail.com)")
TIMEOUT = (5, 20)


def contributing_identifiers() -> set[str]:
    """Identifiers that produced at least one accepted segment."""
    out: set[str] = set()
    for row in read_manifest(MANIFEST_PATH):
        url = row.get("source_url") or ""
        if "/details/" in url:
            out.add(url.rsplit("/details/", 1)[1])
        doc = row.get("doc_id") or ""
        if "_seg_" in doc:
            out.add(doc.rsplit("_seg_", 1)[0])
    return out


def seen_identifiers() -> list[str]:
    if not SEEN_PATH.exists():
        return []
    return [l.strip() for l in SEEN_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def has_text_layer(session: requests.Session, ident: str) -> str:
    """One cheap metadata call. Returns a classification, downloads nothing."""
    try:
        r = session.get(METADATA_URL.format(ident=ident), timeout=TIMEOUT)
    except requests.RequestException as exc:
        return f"request_failed:{type(exc).__name__}"
    if r.status_code != 200:
        return f"http_{r.status_code}"
    try:
        meta = r.json()
    except ValueError:
        return "json_parse_failed"
    files = meta.get("files") or []
    if not files:
        return "no_files_listed"
    for f in files:
        if f.get("format") == "DjVuTXT" or str(f.get("name", "")).endswith("_djvu.txt"):
            return "HAS_TEXT_LAYER"
    return "genuinely_no_text_layer"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check whether no_text_layer rejections were transient.")
    ap.add_argument("--diagnose", action="store_true", default=True)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--unmark", action="store_true",
                    help="drop non-contributing ids from .seen so they retry")
    args = ap.parse_args()

    seen = seen_identifiers()
    contributed = contributing_identifiers()
    failed = [i for i in seen if i not in contributed]

    print("=" * 72)
    print("RECHECK: were the no_text_layer rejections real?")
    print("=" * 72)
    print(f"  identifiers marked seen      {len(seen):,}")
    print(f"  contributed >=1 segment      {len(contributed):,}")
    print(f"  seen but contributed nothing {len(failed):,}")

    if not failed:
        print("\n  Nothing to recheck.")
        return 0

    if args.unmark:
        backup = SEEN_PATH.with_suffix(
            f".seen.backup_{int(time.time())}")
        shutil.copy2(SEEN_PATH, backup)
        SEEN_PATH.write_text("\n".join(sorted(contributed)) + "\n",
                             encoding="utf-8")
        print(f"\n  backup written : {backup.name}")
        print(f"  .seen rewritten: {len(contributed):,} kept, "
              f"{len(failed):,} cleared for retry")
        print("\n  Now re-run:  python3 konkani/scripts/collect_archive_books.py")
        print("  It will retry ONLY the cleared identifiers.")
        return 0

    rng = random.Random(args.seed)
    sample = rng.sample(failed, min(args.sample, len(failed)))
    print(f"\n  Probing {len(sample)} of them (metadata only, no downloads)\n")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    counts: dict[str, int] = {}
    for i, ident in enumerate(sample, 1):
        verdict = has_text_layer(session, ident)
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"  {i:>4}/{len(sample)}  {ident[:46]:<46s} {verdict}")
        time.sleep(0.25)

    print("\n" + "-" * 72)
    for verdict, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {verdict:<32}{count:>5}  ({count/len(sample):.1%})")
    print("-" * 72)

    recoverable = counts.get("HAS_TEXT_LAYER", 0) / len(sample)
    print(f"\n  RECOVERABLE SHARE: {recoverable:.1%}")
    if recoverable >= 0.30:
        est = len(failed) * recoverable * 0.143 * 21_800
        print(f"  -> the rejections were largely TRANSIENT.")
        print(f"     Retrying ~{len(failed):,} items could recover on the order")
        print(f"     of {est/1e6:.1f}M words. Run with --unmark, then re-run the")
        print(f"     collector. Budget roughly {len(failed)*7/3600:.1f} h.")
    else:
        print(f"  -> the rejections were largely REAL. The corpus is complete")
        print(f"     as collected; record this measurement and move on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
