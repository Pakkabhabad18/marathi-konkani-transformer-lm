#!/usr/bin/env python3
"""
Marathi source M2: sitemap-driven news and long-form article collection.

WHY THIS IS THE PRIMARY MANUAL SOURCE
-------------------------------------
Source M1 (Maharashtra GRs on the Internet Archive) measured 2.1 documents per
minute with 61-68% of fetches failing - 393 hours for the required volume. This
source measured **758 documents per minute at 98.5% acceptance** on the same
machine. M1 is kept and still runs in the background; M2 does the heavy lifting.

WHY THIS COUNTS AS MANUAL COLLECTION
------------------------------------
The brief counts "scraping + cleaning pages you gather yourself" as manual. We
discover URLs from each site's own sitemap, fetch each article, strip the site
furniture ourselves, and normalize, language-check and deduplicate the result.
No ready-made corpus is involved.

THE OVERLAP PROBLEM, AND HOW THE DATE FILTER SOLVES IT
------------------------------------------------------
IndicCorpV2 and Sangraha - our downloaded corpora - are themselves built from
Marathi news crawls, so scraping the same sites and calling the output "manual"
would be self-deception. Both are **fixed snapshots**, so an article published
after their release cannot be in them. Collecting only recent articles makes the
manual claim true by construction; the cross-corpus hash check then proves it.

HISTORY OF THIS FILE (worth knowing before changing it)
--------------------------------------------------------
This was originally a standalone script containing its own copy of the crawling
machinery. It has been refactored onto `common/newscrawl.py`, which Konkani also
uses, so a fix in one place now benefits both languages. The refactor was
deliberately deferred while a live collection job was running on the old file.

The refactor also carried a real fix. The first full run exited after only 4,411
documents across seven sites and *looked* like a completed job. It was not: the
sitemap traversal was capped at 40 files per site, and a news site publishes
hundreds - one index pointing at one sitemap per month per section. The crawler
had walked a few percent of the available URLs. See `CrawlConfig.max_sitemaps`.

USAGE
-----
    python3 marathi/scripts/collect_news.py --probe        # verify sites
    python3 marathi/scripts/collect_news.py --limit 200    # pilot
    python3 marathi/scripts/collect_news.py                # full run, resumable
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.newscrawl import CrawlConfig, run_crawl, run_probe   # noqa: E402

# Verified by --probe on 16 Aug 2026: 7 of 8 usable.
# marathivishwakosh.org was unreachable and is recorded as failed rather than
# quietly removed.
SITES = [
    {"name": "loksatta",          "home": "https://www.loksatta.com"},
    {"name": "maharashtratimes",  "home": "https://maharashtratimes.com"},
    {"name": "lokmat",            "home": "https://www.lokmat.com"},
    {"name": "esakal",            "home": "https://www.esakal.com"},
    {"name": "abplive_marathi",   "home": "https://marathi.abplive.com"},
    {"name": "tv9marathi",        "home": "https://www.tv9marathi.com"},
    {"name": "divyamarathi",      "home": "https://divyamarathi.bhaskar.com"},
]


def build_config(since: datetime, max_sitemaps: int) -> CrawlConfig:
    return CrawlConfig(
        job_name="marathi_news",
        language="mr",
        accept_label="mr",
        # Konkani is rejected outright: the two languages are closely related
        # and share Devanagari, and the specification forbids the two corpora
        # sharing documents. Enforced here at collection time.
        reject_label="kok",
        sites=SITES,
        data_dir=REPO_ROOT / "marathi" / "data",
        source_prefix="news",
        min_words=120,
        min_devanagari_ratio=0.60,
        langid_min_abs_score=0.30,
        max_sitemaps=max_sitemaps,
        since=since,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Marathi news articles.")
    parser.add_argument("--probe", action="store_true",
                        help="verify sites only; collect nothing")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-site", type=int, default=200000)
    parser.add_argument("--max-sitemaps", type=int, default=3000,
                        help="sitemap files to walk per site (was 40, far too low)")
    parser.add_argument("--since", default="2025-01-01",
                        help="publication-date floor, YYYY-MM-DD")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    cfg = build_config(since, args.max_sitemaps)

    if args.probe:
        return run_probe(cfg)

    return run_crawl(cfg, limit=args.limit, workers=args.workers,
                     per_site=args.per_site,
                     dedup_threshold=args.dedup_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
