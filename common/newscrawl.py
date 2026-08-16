"""
Reusable sitemap-driven web crawling, parameterised by language.

WHY THIS MODULE EXISTS
----------------------
`marathi/scripts/collect_news.py` proved the approach works: 758 documents per
minute at 98.5% acceptance. Konkani needs exactly the same machinery with a
different site list and the language gate reversed.

Copying that file and editing two constants would violate the standing rule
against multiple scripts doing essentially the same thing, so the machinery
lives here and each language supplies only its configuration.

NOTE ON THE MARATHI SCRIPT
--------------------------
`marathi/scripts/collect_news.py` still contains its own copy of this logic. It
was left untouched **because it was running a live collection job at the time
this module was written**, and editing a script mid-run to remove duplication is
a bad trade. It should be refactored onto this module once its run completes.
This is recorded rather than quietly left as drift.

LANGUAGE GATING
---------------
The `accept` and `reject` labels are explicit configuration, not defaults. For
Marathi, Konkani text is rejected; for Konkani, Marathi text is rejected. That
symmetry is what enforces the specification's requirement that the two corpora
share no documents - it is applied at collection time, not audited afterwards.
"""

from __future__ import annotations

import gzip
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests

from common.checkpoint import Checkpoint
from common.dedup import Deduplicator
from common.manifest import CollectionType, ManifestWriter, make_record
from common.scriptid import identify_marathi_konkani, profile_script
from common.textnorm import NORMALIZATION_STEPS, normalize

USER_AGENT = "lma-phase1-research/1.0 (student project; contact: pakkabhabad@gmail.com)"
REQUEST_TIMEOUT = (5, 15)

_thread_local = threading.local()

_TAG_STRIP = re.compile(
    r"<(script|style|nav|header|footer|aside|form|noscript|svg|iframe)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAGS = re.compile(r"<[^>]+>")
_PARA = re.compile(r"<p\b[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)
_DEVA = re.compile(r"[ऀ-ॿ]")

EXCLUDE_URL_RE = re.compile(
    r"/(ampstories|web-story|webstory|photo|photos|gallery|video|videos|"
    r"astro|astrology|horoscope|rashi|panchang|rashifal|live-blog|liveblog|"
    r"podcast|shorts|quiz|poll|cartoon|epaper)(/|$|-)",
    re.IGNORECASE,
)

_ISO_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+\-]\d{1,2}:\d{2}")
_UPDATED = re.compile(
    r"(Updated|Published|Created)\s*:?\s*[A-Za-z]+\s+\d{1,2},?\s*\d{4}[^|\-ऀ]*"
    r"(IST|GMT|UTC)?", re.IGNORECASE)
_BYLINE = re.compile(r"^\s*(By|द्वारा|लेखक|बरोवपी)\b[^|]{0,80}\|\s*", re.IGNORECASE)
_LEAD_DASH = re.compile(r"^\s*[-–—|:]\s*")


@dataclass
class CrawlConfig:
    """Everything that differs between the two languages."""

    job_name: str
    language: str                       # "mr" or "kok"
    accept_label: str                   # langid label required to keep a document
    reject_label: str                   # langid label that is an outright rejection
    sites: list                         # [{"name": ..., "home": ...}, ...]
    data_dir: Path
    source_prefix: str = "news"
    min_words: int = 120
    min_devanagari_ratio: float = 0.60
    langid_min_abs_score: float = 0.30
    shard_size: int = 2000
    sample_urls: int = 8
    # How many sitemap files to walk per site before giving up.
    #
    # This was 40, and 40 was badly wrong. A news site publishes a sitemap
    # *index* pointing at one sitemap per month per section - often several
    # hundred files. Stopping at 40 meant reading a sliver of each site: the
    # first Marathi run exhausted all seven sites after only 4,411 documents
    # and exited looking like a completed job, when in fact it had walked a
    # few percent of the available URLs.
    #
    # A sitemap file is one cheap request that yields thousands of URLs, so a
    # high ceiling costs very little and the `cap` on yielded URLs is the real
    # limiter.
    max_sitemaps: int = 3000
    since: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "mr,kok,en;q=0.5"})
        _thread_local.session = s
    return s


def get(url: str, **kwargs):
    """Single-attempt GET. With millions of candidate URLs, retrying is waste."""
    try:
        r = _session().get(url, timeout=REQUEST_TIMEOUT, **kwargs)
        return r if r.status_code == 200 else None
    except requests.RequestException:
        return None


def strip_byline(text: str) -> str:
    text = _ISO_STAMP.sub(" ", text)
    text = _UPDATED.sub(" ", text)
    text = _BYLINE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _LEAD_DASH.sub("", text).strip()


def extract_article_text(html_text: str) -> str:
    """Pull article prose out of a page: drop furniture tags, keep <p> content."""
    import html as _html

    cleaned = _TAG_STRIP.sub(" ", html_text)
    paragraphs = []

    for raw in _PARA.findall(cleaned):
        text = _html.unescape(_TAGS.sub(" ", raw))
        text = strip_byline(re.sub(r"\s+", " ", text).strip())
        if len(text.split()) < 6 or not _DEVA.search(text):
            continue
        non_ws = sum(1 for c in text if not c.isspace())
        if non_ws and len(_DEVA.findall(text)) / non_ws < 0.45:
            continue
        paragraphs.append(text)

    return "\n\n".join(paragraphs)


def read_robots(home: str):
    response = get(urljoin(home, "/robots.txt"))
    if response is None:
        return None, [], 0.0

    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    sitemaps = [l.split(":", 1)[1].strip() for l in response.text.splitlines()
                if l.lower().startswith("sitemap:")]
    try:
        value = parser.crawl_delay(USER_AGENT)
        delay = float(value) if value else 0.0
    except Exception:
        delay = 0.0
    return parser, sitemaps, delay


def parse_sitemap(url: str):
    response = get(url)
    if response is None:
        return [], []

    body = response.content
    if url.endswith(".gz") or body[:2] == b"\x1f\x8b":
        try:
            body = gzip.decompress(body)
        except OSError:
            return [], []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return [], []

    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    children = [n.find("s:loc", ns).text.strip()
                for n in root.findall("s:sitemap", ns)
                if n.find("s:loc", ns) is not None and n.find("s:loc", ns).text]
    pages = []
    for n in root.findall("s:url", ns):
        loc = n.find("s:loc", ns)
        if loc is None or not loc.text:
            continue
        mod = n.find("s:lastmod", ns)
        pages.append((loc.text.strip(),
                      (mod.text or "").strip() if mod is not None else ""))
    return children, pages


def lastmod_ok(lastmod: str, since: datetime) -> bool:
    if not lastmod:
        return True
    try:
        stamp = datetime.fromisoformat(lastmod.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp >= since
    except ValueError:
        return True


def discover_urls(site: dict, cfg: CrawlConfig, cap: int, max_sitemaps: int = 0):
    parser, sitemaps, _ = read_robots(site["home"])
    if not sitemaps:
        sitemaps = [urljoin(site["home"], "/sitemap.xml")]

    max_sitemaps = max_sitemaps or cfg.max_sitemaps
    seen, queue, yielded = set(), list(sitemaps), 0
    while queue and yielded < cap and len(seen) < max_sitemaps:
        sm = queue.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        children, pages = parse_sitemap(sm)
        queue.extend(c for c in children if c not in seen)

        for url, lastmod in pages:
            if not lastmod_ok(lastmod, cfg.since) or EXCLUDE_URL_RE.search(url):
                continue
            if parser is not None and not parser.can_fetch(USER_AGENT, url):
                continue
            yield url
            yielded += 1
            if yielded >= cap:
                return


def evaluate(raw_html: str, cfg: CrawlConfig):
    """Extract, normalize and gate one page against this language's config."""
    body = extract_article_text(raw_html)
    if not body:
        return "", None, None, "no_article_text"

    text = normalize(body, keep_paragraphs=True)
    if len(text.split()) < cfg.min_words:
        return text, None, None, "too_short"

    profile = profile_script(text)
    if profile.devanagari_ratio < cfg.min_devanagari_ratio:
        return text, profile, None, "not_enough_devanagari"

    langid = identify_marathi_konkani(text)

    # The other language is always an outright rejection. This is the
    # cross-corpus independence requirement enforced at collection time.
    if langid.label == cfg.reject_label:
        return text, profile, langid, f"langid_{cfg.reject_label}_rejected"

    if langid.label != cfg.accept_label:
        return text, profile, langid, f"langid_{langid.label}"

    if abs(langid.score) < cfg.langid_min_abs_score:
        return text, profile, langid, "langid_low_confidence"

    return text, profile, langid, None


def run_probe(cfg: CrawlConfig) -> int:
    """Verify each candidate site without collecting anything."""
    print("=" * 72)
    print(f"SITE PROBE - {cfg.language} | date floor {cfg.since.date()}")
    print(f"Sampling up to {cfg.sample_urls} urls per site")
    print("=" * 72)

    usable = 0
    for site in cfg.sites:
        print(f"\n{site['name']}  ({site['home']})")
        parser, sitemaps, delay = read_robots(site["home"])
        if parser is None:
            print("   robots.txt   UNREACHABLE -> skip")
            continue
        print(f"   robots.txt   ok | {len(sitemaps)} sitemap(s)"
              f"{f' | crawl-delay {delay}s' if delay else ''}")

        urls = list(discover_urls(site, cfg, cap=cfg.sample_urls))
        if not urls:
            print("   sitemaps     no urls passed date/exclusion filters -> skip")
            continue
        print(f"   sitemaps     ok | {len(urls)} candidate url(s)")

        results, failures = [], {}
        for url in urls:
            page = get(url)
            if page is None:
                failures["fetch_failed"] = failures.get("fetch_failed", 0) + 1
                continue
            text, profile, langid, reason = evaluate(page.text, cfg)
            if reason:
                failures[reason] = failures.get(reason, 0) + 1
                continue
            results.append((len(text.split()), text, profile, langid))

        if not results:
            summary = ", ".join(f"{k} x{v}" for k, v in failures.items())
            print(f"   article      0/{len(urls)} usable ({summary}) -> skip")
            continue

        results.sort(key=lambda r: -r[0])
        words, text, profile, langid = results[0]
        median = sorted(r[0] for r in results)[len(results) // 2]
        print(f"   article      {len(results)}/{len(urls)} usable | "
              f"best {words} words, median {median} | "
              f"devanagari {profile.devanagari_ratio:.1%} | "
              f"langid {langid.label} {langid.score:+.2f}")
        print(f"   preview      {' '.join(text.split())[:110]}")
        usable += 1

    print("\n" + "=" * 72)
    print(f"VERDICT: {usable} of {len(cfg.sites)} sites usable")
    print("=" * 72)
    return 0 if usable else 1


def run_crawl(cfg: CrawlConfig, *, limit: int = 0, workers: int = 8,
              per_site: int = 200000, dedup_threshold: float = 0.85,
              batch: int = 200) -> int:
    """Collect from every configured site. Resumable and checkpointed."""
    raw_dir = cfg.data_dir / "manual" / cfg.source_prefix
    raw_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(cfg.data_dir / "checkpoints" / f"{cfg.job_name}.json",
                            cfg.job_name)
    deduper = Deduplicator(threshold=dedup_threshold)
    manifest = ManifestWriter(cfg.data_dir / "manifests" / f"{cfg.job_name}.jsonl")

    print("=" * 68)
    print(f"COLLECTING: {cfg.job_name} ({cfg.language}, manual scrape)")
    print(f"Date floor: {cfg.since.date()} | workers={workers}")
    print("=" * 68)
    if checkpoint.state.collected:
        print(f"Resuming: {checkpoint.state.collected:,} collected, "
              f"{len(checkpoint.seen):,} urls seen")

    rejected: dict = {}
    per_site_counts: dict = {}
    accepted_run = words_run = 0
    shard_index = checkpoint.state.collected // cfg.shard_size
    shard = open(raw_dir / f"shard_{shard_index:05d}.txt", "a", encoding="utf-8")
    start = time.time()
    stop = False

    def note(reason):
        rejected[reason] = rejected.get(reason, 0) + 1
        checkpoint.state.skipped += 1

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for site in cfg.sites:
                if stop:
                    break
                name = site["name"]
                print(f"\n--- {name} ---", flush=True)
                pending: list = []

                urls = (u for u in discover_urls(site, cfg, per_site)
                        if not checkpoint.has_seen(u))

                for url in urls:
                    pending.append(url)
                    if len(pending) < batch:
                        continue

                    for source_url, page in zip(pending, pool.map(get, pending)):
                        if page is None:
                            note("fetch_failed")
                            continue
                        checkpoint.mark_seen(source_url)

                        text, profile, langid, reason = evaluate(page.text, cfg)
                        if reason:
                            note(reason)
                            continue
                        if deduper.is_duplicate(text):
                            note("duplicate")
                            continue

                        manifest.write(make_record(
                            text=text, raw_text=page.text,
                            source_name=f"{cfg.source_prefix}_{name}",
                            source_url=source_url,
                            collection_type=CollectionType.MANUAL_SCRAPE,
                            language=cfg.language,
                            preprocessing_applied=NORMALIZATION_STEPS + [
                                "html_boilerplate_strip", "paragraph_extract"],
                            script=profile.script,
                            langid_score=langid.score,
                            langid_label=langid.label,
                            devanagari_ratio=profile.devanagari_ratio,
                            doc_id=source_url,
                            notes=f"site={name}; date_floor={cfg.since.date()}",
                        ))
                        shard.write(text.replace("\n", " ") + "\n")
                        checkpoint.state.collected += 1
                        accepted_run += 1
                        words_run += len(text.split())
                        per_site_counts[name] = per_site_counts.get(name, 0) + 1

                        if checkpoint.state.collected % cfg.shard_size == 0:
                            shard.close()
                            shard_index += 1
                            shard = open(raw_dir / f"shard_{shard_index:05d}.txt",
                                         "a", encoding="utf-8")
                        if limit and accepted_run >= limit:
                            stop = True
                            break

                    pending = []
                    shard.flush()
                    checkpoint.save()
                    elapsed = max(time.time() - start, 1e-6)
                    print(f"  accepted={checkpoint.state.collected:,} "
                          f"skipped={checkpoint.state.skipped:,} "
                          f"dup={deduper.stats.duplicate_rate:.1%} "
                          f"words={words_run:,} "
                          f"rate={accepted_run / elapsed * 60:.0f}/min", flush=True)
                    if stop:
                        break

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved; re-run to resume.")
    finally:
        shard.close()
        manifest.close()
        checkpoint.save(dedup_stats=deduper.stats.to_dict(),
                        rejection_reasons=rejected, per_site=per_site_counts)
        checkpoint.close()

    elapsed = max(time.time() - start, 1e-6)
    print("\n" + "=" * 68)
    print("RUN SUMMARY")
    print("=" * 68)
    print(f"Accepted this run:   {accepted_run:,}")
    print(f"Accepted total:      {checkpoint.state.collected:,}")
    print(f"Words this run:      {words_run:,}")
    if accepted_run:
        print(f"Words per document:  {words_run / accepted_run:,.0f}")
        print(f"Throughput:          {accepted_run / elapsed * 60:.0f} docs/min")
    print(f"Elapsed:             {elapsed / 60:.1f} min")
    print(f"\nDeduplication:       {deduper.stats.to_dict()}")
    print("\nPer site:")
    for n, c in sorted(per_site_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:24s} {c:,}")
    print("\nRejection reasons:")
    for r, c in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {r:28s} {c:,}")
    print("=" * 68)
    return 0
