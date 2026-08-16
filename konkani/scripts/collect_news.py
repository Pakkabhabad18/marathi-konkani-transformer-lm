#!/usr/bin/env python3
"""
Konkani source K2: manual collection from Konkani web publications.

WHY THIS IS THE MOST IMPORTANT SCRIPT FOR MODEL L
--------------------------------------------------
The manual requirement is a ratio: `manual / total >= 0.20`, so
`total <= 5 x manual`. Konkani manual collection is therefore not a box to tick
- it is the hard cap on how large the Konkani corpus is allowed to be.

The arithmetic as it stands:

  downloaded  books corpus            ~70M words available (measured by pilot)
  manual      Wikipedia, Devanagari   ~1-2M words
  =>  total allowed = 5 x manual      ~5-10M words

That would mean discarding roughly 85-90% of the books corpus, not because it is
bad data but because there is not enough manual data to balance it. Every
additional manual word this script collects lets **five** more words of the
books corpus into the corpus. Nothing else in the Konkani pipeline has that
leverage.

REALISTIC EXPECTATIONS
----------------------
Konkani is genuinely low-resource - that is why it is Model L. Measured during
source selection: the Internet Archive holds **44** items tagged Konkani, of
which only ~13-15 are real books, and Sangraha - the largest systematic Indic
corpus effort - contains **10.1M tokens** of Konkani in total.

So this crawl may return far less than the Marathi equivalent, and that is a
*result*, not a failure. The specification explicitly permits the lower-resource
language to fall short of the token target provided the exact count is reported
and the shortfall justified. These numbers are that justification.

SITE LIST
---------
Every entry below is an **unverified candidate**. They could not be checked from
the environment this script was written in. `--probe` is what turns them into
evidence: it reads robots.txt, walks the sitemaps, fetches real pages, and runs
them through the full extraction and language pipeline.

Sites that fail the probe must be recorded as failed in the source inventory,
not silently dropped.

USAGE
-----
    python3 konkani/scripts/collect_news.py --probe        # verify first
    python3 konkani/scripts/collect_news.py --limit 200    # pilot
    python3 konkani/scripts/collect_news.py                # full run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.newscrawl import CrawlConfig, run_crawl, run_probe   # noqa: E402

# Unverified candidates. Konkani in Devanagari is published by a small number of
# Goa-based outlets and institutions; several publish primarily as e-paper
# images, which carry no extractable text and will fail the probe.
SITES = [
    {"name": "sunaparant",       "home": "https://www.sunaparant.com"},
    {"name": "bhaangarbhuin",    "home": "https://www.bhaangarbhuin.com"},
    {"name": "konkani_akademi",  "home": "https://konkaniakademi.goa.gov.in"},
    {"name": "goa_gov",          "home": "https://www.goa.gov.in"},
    {"name": "dol_goa",          "home": "https://dol.goa.gov.in"},
    {"name": "vishwakonkani",    "home": "https://vishwakonkani.org"},
    {"name": "goanvarta",        "home": "https://www.goanvarta.net"},
    {"name": "herald_goa",       "home": "https://www.heraldgoa.in"},
]


def build_config(since: datetime) -> CrawlConfig:
    return CrawlConfig(
        job_name="konkani_news",
        language="kok",
        accept_label="kok",
        # Marathi is rejected outright. Konkani and Marathi are closely related
        # and share Devanagari, so without this the Konkani corpus would fill up
        # with Marathi - the specification forbids the two corpora sharing
        # documents, and this is where that is enforced.
        reject_label="mr",
        sites=SITES,
        data_dir=REPO_ROOT / "konkani" / "data",
        source_prefix="news",
        # Konkani pages are typically shorter than Marathi news articles, and
        # the corpus is scarce enough that a 120-word floor would discard usable
        # text. Lowered deliberately, and recorded here so the difference between
        # the two languages' filters is visible rather than accidental.
        min_words=60,
        min_devanagari_ratio=0.60,
        langid_min_abs_score=0.30,
        since=since,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Konkani web text.")
    parser.add_argument("--probe", action="store_true",
                        help="verify sites only; collect nothing")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-site", type=int, default=200000)
    parser.add_argument("--since", default="2015-01-01",
                        help="publication-date floor (default is early: Konkani "
                             "text is scarce, so we take what exists)")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    cfg = build_config(since)

    if args.probe:
        return run_probe(cfg)

    return run_crawl(cfg, limit=args.limit, workers=args.workers,
                     per_site=args.per_site,
                     dedup_threshold=args.dedup_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
