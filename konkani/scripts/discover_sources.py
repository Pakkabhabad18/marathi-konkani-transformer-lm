#!/usr/bin/env python3
"""
Konkani Phase 1 SOURCE DISCOVERY + PROBE.  Read-only: collects nothing.

WHY THIS SCRIPT EXISTS
----------------------
Phase 1 recorded "the Internet Archive holds 44 Konkani items". That number is
wrong, and the way it was wrong matters more than the number itself.

`collect_archive_books.py` asked archive.org:

    language:(Konkani OR Konknni OR Concani) AND mediatype:texts

Archive.org's `language` field is not a controlled vocabulary. Cataloguers write
the ISO 639-2 code far more often than the English name, in whatever case they
please. Measured 16 Aug 2026 against the same API:

    language:(Konkani OR Konknni OR Concani)   ->     44 items
    language:kok AND mediatype:texts           ->  5,093 items

So the earlier figure measured our spelling list, not the Internet Archive. A
single query with a plausible-looking term returned a confident, precise, wrong
answer, and that answer was then used as evidence that Konkani data does not
exist. This script exists so that the replacement number is produced by
enumeration rather than by one lucky or unlucky string.

WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT DO
--------------------------------------------------
Does:
  * enumerate candidate items across every spelling of the language field,
    with cursor pagination, deduplicating identifiers across queries;
  * read each sampled item's metadata to find the real OCR text filename and
    its size in bytes (never guessing the filename - see the 404 correction
    recorded in phase1_decisions.md);
  * download a random sample of items, run the project's own normalization,
    script profiling and language identification over them, and measure
    Devanagari share, langid label, OCR noise and words-per-item;
  * extrapolate a yield estimate with the sample size stated alongside it;
  * write a machine-readable inventory plus a human-readable summary.

Does NOT:
  * write to any manifest, or to `<lang>/data/manual/**`;
  * touch any checkpoint or `.seen` file. Discovery must never consume the work
    it previews - that bug already cost us one real collection run
    (`--dry-run` marked all 14 books seen, and the real run then got zero).
    The safest fix is architectural: this script has no seen-set at all.

BODY SCRIPT IS NOT PREDICTED BY METADATA - THE CENTRAL PROBE FINDING
--------------------------------------------------------------------
Konkani is written in Devanagari (Goa) and Kannada (coastal Karnataka). Both are
catalogued `language: kok`. Worse, two sampled items carried a *Devanagari
title* over a *Kannada body*:

    20veashekddeachy0000drje   title "20व्या शेक्ड्यांचे कोंकणी म्हान मनिस्"
                               body  Kannada script, 96.2% Kannada characters
    27kavitha0000step          title "27 कविता"
                               body  Kannada script

Filtering on `language`, or on the script of the title, would therefore have
quietly admitted Kannada-script Konkani into a Devanagari-only corpus. Only the
body text decides. That is why this probe downloads text instead of trusting
the catalogue, and why the estimated yield is discounted by a measured
Devanagari share rather than assumed to be 100%.

USAGE
-----
    python3 konkani/scripts/discover_sources.py --enumerate-only
    python3 konkani/scripts/discover_sources.py --sample 40
    python3 konkani/scripts/discover_sources.py --sample 120 --seed 7

Output:
    konkani/data/discovery/archive_enumeration.json   every identifier found
    konkani/data/discovery/probe_results.json         per-item measurements
    report/phase1_konkani_source_discovery_measured.md   generated summary
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import normalize                                 # noqa: E402

SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
METADATA_URL = "https://archive.org/metadata/{ident}"

OUT_DIR = REPO_ROOT / "konkani" / "data" / "discovery"
ENUM_PATH = OUT_DIR / "archive_enumeration.json"
PROBE_PATH = OUT_DIR / "probe_results.json"
REPORT_PATH = REPO_ROOT / "report" / "phase1_konkani_source_discovery_measured.md"

USER_AGENT = ("lma-phase1-research/1.0 "
              "(student project; contact: pakkabhabad@gmail.com)")
TIMEOUT = (5, 30)

# Every spelling of the language field that a cataloguer might plausibly use.
# The scrape API lowercases terms, so `kok` also matches `Kok` and `KOK`; the
# variants are listed anyway so the query is self-documenting and so that a
# future reader can see exactly what was and was not asked for.
LANGUAGE_QUERIES = [
    'language:kok AND mediatype:texts',
    'language:Konkani AND mediatype:texts',
    'language:Konknni AND mediatype:texts',
    'language:Concani AND mediatype:texts',
    'language:"Konkani (Devanagari)" AND mediatype:texts',
    'language:gom AND mediatype:texts',
    'subject:Konkani AND mediatype:texts',
]

# Excluded at enumeration time, with the reason recorded rather than implied.
EXCLUDE_PREFIXES = {
    "wikipedia_": "wikipedia_derived",
    "wiktionary_": "wikipedia_derived",
    "wikiquote_": "wikipedia_derived",
    "wikibooks_": "wikipedia_derived",
    "gomwiki-": "wikipedia_derived",
    "dni.ncaa": "not_konkani",
}

MIN_DEVANAGARI_RATIO = 0.70


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def enumerate_items(session: requests.Session, verbose: bool = True) -> dict:
    """Cursor-paginated enumeration across every language spelling.

    The previous implementation issued one request with `count=1000` and no
    cursor, so it saw at most one page even when more existed. The scrape API
    returns a `cursor` whenever results remain; absence of a cursor is the only
    correct stop condition.
    """
    found: dict[str, dict] = {}
    per_query: dict[str, int] = {}

    for query in LANGUAGE_QUERIES:
        cursor, page, seen_here = None, 0, 0
        while True:
            params = {"q": query, "fields": "identifier,title,language,collection",
                      "count": 1000}
            if cursor:
                params["cursor"] = cursor
            try:
                r = session.get(SCRAPE_URL, params=params, timeout=TIMEOUT)
            except requests.RequestException as exc:
                print(f"  [error] {query!r} page {page}: {exc}")
                break
            if r.status_code != 200:
                print(f"  [error] {query!r} page {page}: HTTP {r.status_code}")
                break
            payload = r.json()
            if page == 0 and verbose:
                print(f"  {query:<52s} total={payload.get('total', 0):,}")
            for item in payload.get("items", []):
                ident = item.get("identifier")
                if not ident:
                    continue
                seen_here += 1
                if ident not in found:
                    found[ident] = {
                        "identifier": ident,
                        "title": item.get("title"),
                        "language": item.get("language"),
                        "collection": item.get("collection"),
                        "first_seen_query": query,
                    }
            cursor = payload.get("cursor")
            page += 1
            if not cursor:
                break
            time.sleep(0.2)
        per_query[query] = seen_here

    # Apply exclusions, keeping the reason on the record.
    excluded = {}
    for ident in list(found):
        for prefix, reason in EXCLUDE_PREFIXES.items():
            if ident.startswith(prefix):
                excluded[ident] = reason
                found.pop(ident)
                break

    collections = Counter()
    for rec in found.values():
        col = rec.get("collection")
        for c in (col if isinstance(col, list) else [col]):
            if c:
                collections[c] += 1

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "per_query_items_returned": per_query,
        "unique_candidates": len(found),
        "excluded": excluded,
        "excluded_count": len(excluded),
        "top_collections": collections.most_common(40),
        "items": list(found.values()),
    }


def item_text_file(session: requests.Session, ident: str) -> tuple[str | None, int, dict]:
    """Resolve the OCR text filename from metadata. Never guess it.

    Guessing `<identifier>_djvu.txt` 404'd on every Marathi GR item, because the
    real name there is a numeric string. One Konkani book happens to use the
    identifier form, which is exactly why the wrong assumption looked safe.
    """
    try:
        r = session.get(METADATA_URL.format(ident=ident), timeout=TIMEOUT)
    except requests.RequestException:
        return None, 0, {}
    if r.status_code != 200:
        return None, 0, {}
    try:
        meta = r.json()
    except ValueError:
        return None, 0, {}

    files = meta.get("files", []) or []
    chosen = next((f for f in files if f.get("format") == "DjVuTXT"), None)
    if chosen is None:
        chosen = next((f for f in files
                       if str(f.get("name", "")).endswith("_djvu.txt")), None)
    if chosen is None:
        return None, 0, meta.get("metadata", {}) or {}

    try:
        size = int(chosen.get("size") or 0)
    except (TypeError, ValueError):
        size = 0

    server = meta.get("server")
    directory = meta.get("dir")
    name = chosen.get("name")
    if server and directory:
        url = f"https://{server}{directory}/{name}"
    else:
        url = f"https://archive.org/download/{ident}/{name}"
    return url, size, meta.get("metadata", {}) or {}


def probe_item(session: requests.Session, rec: dict) -> dict:
    """Download one item's OCR text and measure it with the project pipeline."""
    ident = rec["identifier"]
    out = dict(rec)
    url, size, meta = item_text_file(session, ident)
    out["djvu_bytes"] = size
    out["text_url"] = url
    out["imagecount"] = meta.get("imagecount")
    out["licenseurl"] = meta.get("licenseurl") or "not_stated"
    out["rights"] = meta.get("rights") or "not_stated"
    out["access_restricted"] = meta.get("access-restricted-item")

    if not url:
        out["verdict"] = "no_ocr_text_layer"
        return out

    try:
        r = session.get(url, timeout=(5, 60))
    except requests.RequestException as exc:
        out["verdict"] = f"fetch_failed:{type(exc).__name__}"
        return out
    if r.status_code != 200:
        out["verdict"] = f"fetch_failed:http_{r.status_code}"
        return out

    r.encoding = "utf-8"
    text = normalize(r.text, keep_paragraphs=True)
    words = text.split()
    out["words"] = len(words)

    profile = profile_script(text)
    out["devanagari_ratio"] = round(profile.devanagari_ratio, 4)
    out["kannada_ratio"] = round(profile.kannada_ratio, 4)
    out["latin_ratio"] = round(profile.latin_ratio, 4)
    out["script"] = profile.script

    if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
        out["verdict"] = "rejected_not_devanagari"
        out["langid_label"] = None
        return out

    langid = identify_marathi_konkani(text)
    out["langid_label"] = langid.label
    out["langid_score"] = round(langid.score, 4)
    out["langid_confident"] = langid.confident
    out["verdict"] = ("rejected_langid_marathi" if langid.label == "mr"
                      else "accepted")
    return out


def summarize(enum: dict, probes: list[dict]) -> dict:
    """Extrapolate from the probe, keeping the sample size attached to it."""
    population = enum["unique_candidates"]
    fetched = [p for p in probes if p.get("words")]
    accepted = [p for p in fetched if p.get("verdict") == "accepted"]

    n = len(fetched)
    deva_share = len(accepted) / n if n else 0.0
    mean_words = sum(p["words"] for p in accepted) / len(accepted) if accepted else 0
    median_words = 0
    if accepted:
        ws = sorted(p["words"] for p in accepted)
        median_words = ws[len(ws) // 2]

    # CORRECTED (D-021). Both figures are reported, but note what each IS:
    #
    #   mean   x N  is the unbiased estimator of a SUM. This is the estimate.
    #   median x N  is NOT an estimator of the sum at all. On a right-skewed
    #               distribution it is systematically low, so it serves as a
    #               conservative planning floor and nothing more.
    #
    # An earlier version of this file called the median-based figure "the one
    # to quote", which is wrong: describing a deliberate underestimate as the
    # better estimate is not conservatism, it is a mislabelled statistic.
    est_items = population * deva_share
    return {
        "population_candidates": population,
        "probe_sample_fetched": n,
        "probe_accepted": len(accepted),
        "devanagari_accept_share": round(deva_share, 4),
        "mean_words_per_accepted_item": int(mean_words),
        "median_words_per_accepted_item": int(median_words),
        "estimated_usable_items": int(est_items),
        "estimated_words_median_based": int(est_items * median_words),
        "estimated_words_mean_based": int(est_items * mean_words),
        "verdict_breakdown": dict(Counter(p.get("verdict") for p in probes)),
    }


def write_report(enum: dict, probes: list[dict], summary: dict) -> None:
    lines = [
        "# Phase 1 - Konkani source discovery: measured results",
        "",
        f"Generated {enum['generated_utc']} by `konkani/scripts/discover_sources.py`.",
        "This file is produced by the script. Do not hand-edit it; the narrative",
        "inventory lives in `report/phase1_konkani_source_discovery.md`.",
        "",
        "## Enumeration",
        "",
        "| query | items returned |",
        "|---|---:|",
    ]
    for q, c in enum["per_query_items_returned"].items():
        lines.append(f"| `{q}` | {c:,} |")
    lines += [
        "",
        f"**Unique candidates after deduplication and exclusions: "
        f"{enum['unique_candidates']:,}** "
        f"({enum['excluded_count']:,} excluded by prefix rule)",
        "",
        "### Largest parent collections",
        "",
        "| collection | items |",
        "|---|---:|",
    ]
    for name, count in enum["top_collections"][:20]:
        lines.append(f"| `{name}` | {count:,} |")

    lines += [
        "",
        "## Probe",
        "",
        f"- items fetched and measured: **{summary['probe_sample_fetched']}**",
        f"- accepted (Devanagari body, langid not Marathi): "
        f"**{summary['probe_accepted']}**",
        f"- measured Devanagari accept share: "
        f"**{summary['devanagari_accept_share']:.1%}**",
        f"- median words per accepted item: "
        f"**{summary['median_words_per_accepted_item']:,}**",
        f"- mean words per accepted item: "
        f"**{summary['mean_words_per_accepted_item']:,}**",
        "",
        "### Verdict breakdown",
        "",
        "| verdict | items |",
        "|---|---:|",
    ]
    for verdict, count in sorted(summary["verdict_breakdown"].items(),
                                 key=lambda kv: -kv[1]):
        lines.append(f"| {verdict} | {count} |")

    lines += [
        "",
        "## Extrapolated yield",
        "",
        f"- estimated usable items: **{summary['estimated_usable_items']:,}** "
        f"of {summary['population_candidates']:,} candidates",
        f"- **estimated total words (mean x N — the estimate): "
        f"{summary['estimated_words_mean_based']:,}**",
        f"- conservative planning floor (median x N): "
        f"**{summary['estimated_words_median_based']:,}**",
        "",
        "`mean x N` is the unbiased estimator of a sum and is the number to",
        "quote as the expected yield. `median x N` is *not* an estimator of a",
        "sum: item sizes are strongly right-skewed (a four-volume encyclopedia",
        "is ~700k words, a poetry booklet ~5k), so on this distribution it sits",
        "well below the true total. Use it as a floor for planning, never as",
        "the headline estimate.",
        "",
        f"Sample size is {summary['probe_sample_fetched']} items. The Devanagari",
        "share is the dominant uncertainty; re-run with a larger `--sample` to",
        "tighten it before committing to the full collection.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Konkani source discovery and probe. Collects nothing.")
    parser.add_argument("--sample", type=int, default=40,
                        help="how many items to download and measure")
    parser.add_argument("--seed", type=int, default=20260816,
                        help="RNG seed, so the sample is reproducible")
    parser.add_argument("--enumerate-only", action="store_true",
                        help="list items, download none")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = _session()

    print("=" * 74)
    print("KONKANI SOURCE DISCOVERY - PROBE ONLY, NOTHING IS COLLECTED")
    print("=" * 74)
    print("\n[1/3] Enumerating archive.org across every language spelling")
    enum = enumerate_items(session)
    ENUM_PATH.write_text(json.dumps(enum, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\n  unique candidates: {enum['unique_candidates']:,}"
          f"   excluded: {enum['excluded_count']:,}")
    print(f"  written to {ENUM_PATH.relative_to(REPO_ROOT)}")

    if args.enumerate_only:
        print("\n--enumerate-only: stopping before the download probe.")
        return 0

    items = enum["items"]
    if not items:
        print("\nNo candidates enumerated; cannot probe. Check connectivity.")
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(items, min(args.sample, len(items)))
    print(f"\n[2/3] Probing {len(sample)} randomly sampled items "
          f"(seed {args.seed})")

    probes = []
    for i, rec in enumerate(sample, 1):
        result = probe_item(session, rec)
        probes.append(result)
        print(f"  {i:>4}/{len(sample)}  {result['identifier'][:44]:<44s} "
              f"{str(result.get('script') or '-'):<11s} "
              f"{result.get('words') or 0:>8,}w  {result['verdict']}")
        time.sleep(0.3)

    PROBE_PATH.write_text(json.dumps(probes, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    print("\n[3/3] Summarizing")
    summary = summarize(enum, probes)
    write_report(enum, probes, summary)

    print("-" * 74)
    print(f"  candidates                {summary['population_candidates']:,}")
    print(f"  probed / accepted         {summary['probe_sample_fetched']} / "
          f"{summary['probe_accepted']}")
    print(f"  Devanagari accept share   {summary['devanagari_accept_share']:.1%}")
    print(f"  median words per item     "
          f"{summary['median_words_per_accepted_item']:,}")
    print(f"  ESTIMATED USABLE WORDS    "
          f"{summary['estimated_words_median_based']:,}  (median-based)")
    print("-" * 74)
    print(f"\nReport: {REPORT_PATH.relative_to(REPO_ROOT)}")
    print("Nothing was collected. Review the report before approving the run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
