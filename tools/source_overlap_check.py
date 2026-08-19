#!/usr/bin/env python3
"""
Does the downloaded Konkani books corpus derive from the archive.org scans we
are about to collect ourselves?

WHY THIS BLOCKS THE COLLECTION RUN
----------------------------------
The whole of Phase 1 turns on one ratio:

    manual_words / total_words >= 0.20

Both sides of that fraction have to describe *different text*. If
`omdeep22/Konkani_books_corpus-v2` was built by dumping archive.org `_djvu.txt`
files, and we then collect those same scans and label them MANUAL, the same
sentences appear on both sides: they inflate the numerator as "our manual work"
and the denominator as "downloaded data". The ratio would still compute, the
pipeline would still run, and the number would be meaningless.

That failure is invisible downstream. Nothing crashes. The corpus looks larger
and the manual share looks better. It is exactly the class of bug that has
already cost us twice in this project - the doubled Wikipedia manifest and the
dry run that consumed its own work - so it gets checked before, not after.

WHY THE SUSPICION IS REASONABLE
-------------------------------
Measured, not guessed:

  * the dataset card names no provenance at all - no scanning source, no book
    count, no title list, only "digitized books";
  * its rows average 7.52 words, i.e. they are OCR *lines*, not documents;
  * it carries `--- SOURCE:` marker rows that delimit books by title;
  * archive.org `_djvu.txt` files are line-oriented OCR dumps of scanned books.

Those four facts are consistent with the corpus being a repackaging of exactly
the material we just enumerated. They do not prove it. This script decides.

HOW IT DECIDES - THREE INDEPENDENT SIGNALS
------------------------------------------
1. TITLE OVERLAP. Match `--- SOURCE:` book titles from the corpus against
   archive.org item titles from the enumeration. Cheap, and needs no downloads.
   Titles are normalized aggressively (case, punctuation, Devanagari/Latin
   transliteration noise) because the same book is catalogued inconsistently.

2. EXACT CONTENT OVERLAP. Hash the canonical form of paragraphs from each side
   and intersect. Any non-trivial intersection is decisive: identical OCR of
   identical scans.

3. NEAR-DUPLICATE OVERLAP. MinHash/LSH between the two sides. Catches the case
   where the corpus re-wrapped lines into different paragraph boundaries, which
   would defeat exact hashing entirely. This is the signal that matters most,
   because line re-wrapping is the single most likely difference between a raw
   `_djvu.txt` and a cleaned redistribution of it.

All three are reported separately. They can disagree, and how they disagree is
informative: titles matching with no content overlap suggests independent
digitisations of the same books (different scans, different OCR) - which is
still a duplication problem for the corpus, but a different one.

WHAT TO DO WITH THE ANSWER
--------------------------
If overlap is high, drop the DOWNLOADED corpus, not ours. Collecting the
originals ourselves is legitimately manual, better attributed and better
cleaned, and 43M+ manual words clears the 20% rule with no downloaded component
at all. Keeping both is the only option that is actually wrong.

USAGE
-----
    python3 tools/source_overlap_check.py --sample-archive 60
    python3 tools/source_overlap_check.py --sample-archive 150 --titles-only
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.dedup import MinHasher                                     # noqa: E402
from common.manifest import content_hash, read_manifest                # noqa: E402
from common.textnorm import normalize                                  # noqa: E402

ENUM_PATH = REPO_ROOT / "konkani" / "data" / "discovery" / "archive_enumeration.json"
CORPUS_MANIFEST = (REPO_ROOT / "konkani" / "data" / "manifests"
                   / "konkani_books_corpus.jsonl")
RAW_CORPUS_DIR = REPO_ROOT / "konkani" / "data" / "downloaded"
OUT_PATH = REPO_ROOT / "report" / "phase1_konkani_overlap_check.json"
REPORT_PATH = REPO_ROOT / "report" / "phase1_konkani_overlap_check.md"

METADATA_URL = "https://archive.org/metadata/{ident}"
USER_AGENT = ("lma-phase1-research/1.0 "
              "(student project; contact: pakkabhabad@gmail.com)")
TIMEOUT = (5, 60)

# CORRECTED (D-028). This was 25 words, which is wrong for the data on both
# sides: archive.org `_djvu.txt` files and the HF corpus are BOTH line-oriented
# OCR (the corpus averages 7.52 words per row - that is the premise of the whole
# suspicion). A 25-word floor therefore discarded nearly everything: the first
# real run compared just **26 archive paragraphs from 59 items** and still
# printed a confident "LOW OVERLAP" verdict.
#
# 12 words is long enough that random collisions stay negligible, short enough
# that line-oriented OCR actually passes.
MIN_PARAGRAPH_WORDS = 12
NEAR_DUP_THRESHOLD = 0.80


def norm_title(text: str) -> str:
    """Aggressive title normalization: the same book is catalogued many ways."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text)).lower()
    t = re.sub(r"[^\w\sऀ-ॿ]", " ", t)
    t = re.sub(r"\b(vol|volume|part|bhag|khand|no|ed|edition)\b", " ", t)
    t = re.sub(r"\d+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_archive_titles() -> dict[str, str]:
    if not ENUM_PATH.exists():
        print(f"  [error] {ENUM_PATH} not found - run discover_sources.py first")
        return {}
    data = json.loads(ENUM_PATH.read_text(encoding="utf-8"))
    out = {}
    for item in data.get("items", []):
        nt = norm_title(item.get("title"))
        if nt:
            out.setdefault(nt, item["identifier"])
    return out


def corpus_shards() -> list[Path]:
    """Locate the downloaded corpus shards.

    CORRECTED (D-024). This script was written before the real repository
    layout was known and hard-coded `konkani/data/downloaded/`. The corpus is
    actually at `konkani/data/processed/hf_konkani_books_corpus_v2/`, so
    `corpus_paragraphs()` loaded zero paragraphs and the content comparison -
    the whole point of the check - silently did not run. Same bug class as
    D-022; it was fixed in token_budget.py and not carried over here.

    Discovery is now by search rather than by constant, keyed off the manifest
    source name so it survives the next reorganisation too.
    """
    root = REPO_ROOT / "konkani" / "data"
    names = {row.get("source_name") for row in read_manifest(CORPUS_MANIFEST)}
    names.discard(None)
    shards: list[Path] = []
    for name in names:
        for d in root.rglob(name):
            if d.is_dir():
                shards.extend(sorted(d.glob("shard_*.txt")))
    if not shards:                      # last resort: anything under processed/
        shards = sorted((root / "processed").rglob("shard_*.txt"))
    return shards


def load_corpus_titles() -> set[str]:
    """`--- SOURCE: <title>` marker rows delimit books inside the corpus."""
    titles: set[str] = set()
    pattern = re.compile(r"---\s*SOURCE:\s*(.+?)\s*$", re.MULTILINE)
    for path in corpus_shards():
        try:
            with path.open(encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if "SOURCE:" not in line:
                        continue
                    for m in pattern.finditer(line):
                        nt = norm_title(m.group(1))
                        if nt:
                            titles.add(nt)
        except OSError:
            continue
    return titles


def corpus_paragraphs(limit: int) -> list[str]:
    """Paragraph-level text sampled from the downloaded corpus shards."""
    out: list[str] = []
    for path in corpus_shards():
        try:
            with path.open(encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if len(line.split()) >= MIN_PARAGRAPH_WORDS:
                        out.append(normalize(line.strip()))
                        if len(out) >= limit:
                            return out
        except OSError:
            continue
    return out


def archive_paragraphs(identifiers: list[str], session: requests.Session,
                       per_item: int) -> tuple[list[str], list[str]]:
    """Download OCR text for sampled items and return their paragraphs."""
    paras, fetched = [], []
    for i, ident in enumerate(identifiers, 1):
        try:
            r = session.get(METADATA_URL.format(ident=ident), timeout=TIMEOUT)
            meta = r.json() if r.status_code == 200 else {}
        except (requests.RequestException, ValueError):
            continue
        files = meta.get("files", []) or []
        chosen = next((f for f in files if f.get("format") == "DjVuTXT"), None)
        if chosen is None:
            chosen = next((f for f in files
                           if str(f.get("name", "")).endswith("_djvu.txt")), None)
        if chosen is None:
            continue
        server, directory = meta.get("server"), meta.get("dir")
        url = (f"https://{server}{directory}/{chosen['name']}" if server and directory
               else f"https://archive.org/download/{ident}/{chosen['name']}")
        try:
            tr = session.get(url, timeout=TIMEOUT)
        except requests.RequestException:
            continue
        if tr.status_code != 200:
            continue
        tr.encoding = "utf-8"
        fetched.append(ident)
        kept = 0
        for para in tr.text.split("\n"):
            if len(para.split()) >= MIN_PARAGRAPH_WORDS:
                paras.append(normalize(para))
                kept += 1
                if kept >= per_item:
                    break
        print(f"  {i:>4}/{len(identifiers)}  {ident[:46]:<46s} {kept:>4} paragraphs")
        time.sleep(0.3)
    return paras, fetched


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Test whether the downloaded Konkani corpus derives from "
                    "the archive.org scans.")
    ap.add_argument("--sample-archive", type=int, default=60,
                    help="how many archive.org items to download for comparison")
    ap.add_argument("--per-item", type=int, default=120,
                    help="paragraphs to take from each archive item")
    ap.add_argument("--corpus-paragraphs", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--titles-only", action="store_true",
                    help="skip downloads; run signal 1 only")
    args = ap.parse_args()

    print("=" * 74)
    print("KONKANI SOURCE OVERLAP CHECK")
    print("  downloaded corpus  vs  archive.org scans")
    print("=" * 74)

    print("\n[1/3] Title overlap")
    arch_titles = load_archive_titles()
    corp_titles = load_corpus_titles()
    shared_titles = sorted(set(arch_titles) & corp_titles)
    denom = min(len(arch_titles), len(corp_titles)) or 1
    title_rate = len(shared_titles) / denom
    print(f"  archive.org titles       {len(arch_titles):,}")
    print(f"  corpus book titles       {len(corp_titles):,}")
    print(f"  shared                   {len(shared_titles):,}  "
          f"({title_rate:.1%} of the smaller set)")
    for t in shared_titles[:10]:
        print(f"      - {t[:66]}")
    if not shared_titles:
        print("  0 shared. Check the two namespaces are comparable before "
              "concluding independence:")
        print("    archive.org examples:")
        for t in sorted(arch_titles)[:4]:
            print(f"      - {t[:64]}")
        print("    corpus examples:")
        for t in sorted(corp_titles)[:4]:
            print(f"      - {t[:64]}")
        print("  If one side is romanized and the other Devanagari, a 0% title")
        print("  overlap says nothing - rely on the content signals below.")

    result = {
        "archive_titles": len(arch_titles),
        "corpus_titles": len(corp_titles),
        "shared_titles": len(shared_titles),
        "shared_title_rate": round(title_rate, 4),
        "shared_title_examples": shared_titles[:40],
    }

    if args.titles_only:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print("\n--titles-only: stopping before the content comparison.")
        return 0

    print(f"\n[2/3] Loading up to {args.corpus_paragraphs:,} corpus paragraphs")
    corp_paras = corpus_paragraphs(args.corpus_paragraphs)
    print(f"  loaded {len(corp_paras):,}")
    if not corp_paras:
        print("  [error] no corpus paragraphs found - check CORPUS_MANIFEST "
              "and RAW_CORPUS_DIR paths")
        return 1

    rng = random.Random(args.seed)
    idents = list(arch_titles.values())
    sample = rng.sample(idents, min(args.sample_archive, len(idents)))
    print(f"\n[3/3] Downloading {len(sample)} archive.org items for comparison")
    arch_paras, fetched = archive_paragraphs(sample, requests.Session(),
                                             args.per_item)
    print(f"  {len(arch_paras):,} paragraphs from {len(fetched)} items")

    corp_hashes = {content_hash(p) for p in corp_paras}
    arch_hashes = [content_hash(p) for p in arch_paras]
    exact_shared = sum(1 for h in arch_hashes if h in corp_hashes)
    exact_rate = exact_shared / len(arch_hashes) if arch_hashes else 0.0

    hasher = MinHasher()
    index: dict[tuple, list[int]] = {}
    for i, p in enumerate(corp_paras):
        sig = hasher.signature_from_canon(p)
        for b in range(0, len(sig), 4):
            index.setdefault((b, sig[b:b + 4]), []).append(i)

    near = 0
    for p in arch_paras:
        sig = hasher.signature_from_canon(p)
        cands: set[int] = set()
        for b in range(0, len(sig), 4):
            cands.update(index.get((b, sig[b:b + 4]), ()))
        if not cands:
            continue
        for c in list(cands)[:40]:
            other = hasher.signature_from_canon(corp_paras[c])
            same = sum(1 for a, b_ in zip(sig, other) if a == b_) / len(sig)
            if same >= NEAR_DUP_THRESHOLD:
                near += 1
                break
    near_rate = near / len(arch_paras) if arch_paras else 0.0

    result.update({
        "archive_items_fetched": len(fetched),
        "archive_paragraphs": len(arch_paras),
        "corpus_paragraphs": len(corp_paras),
        "exact_shared_paragraphs": exact_shared,
        "exact_overlap_rate": round(exact_rate, 4),
        "near_duplicate_paragraphs": near,
        "near_duplicate_rate": round(near_rate, 4),
    })

    # Statistical power gate. A "no overlap" conclusion drawn from a handful of
    # paragraphs is not a finding, it is an absence of measurement, and the
    # first run of this script reported exactly that.
    UNDERPOWERED = 400
    if len(arch_paras) < UNDERPOWERED or len(corp_paras) < UNDERPOWERED:
        verdict = (f"INCONCLUSIVE - too little data to decide "
                   f"({len(arch_paras):,} archive vs {len(corp_paras):,} corpus "
                   f"paragraphs; need >={UNDERPOWERED} each). This is NOT "
                   f"evidence of independence. Re-run with a larger "
                   f"--sample-archive and check MIN_PARAGRAPH_WORDS.")
    elif near_rate >= 0.25 or exact_rate >= 0.10:
        verdict = ("HIGH OVERLAP - the downloaded corpus appears to derive from "
                   "these scans. Drop the DOWNLOADED corpus and keep our own "
                   "collection.")
    elif near_rate >= 0.05:
        verdict = ("PARTIAL OVERLAP - some shared books. Collect ours, then "
                   "deduplicate the downloaded corpus against it before "
                   "admitting any of it.")
    else:
        verdict = ("LOW OVERLAP - the two sources are substantially independent. "
                   "Both may be used, with normal deduplication.")
    result["verdict"] = verdict

    print("\n" + "-" * 74)
    print(f"  shared titles            {len(shared_titles):,} ({title_rate:.1%})")
    print(f"  exact paragraph overlap  {exact_shared:,} ({exact_rate:.1%})")
    print(f"  near-duplicate overlap   {near:,} ({near_rate:.1%})")
    print("-" * 74)
    print(f"\n  {verdict}\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    REPORT_PATH.write_text(
        "# Phase 1 - Konkani source overlap check\n\n"
        "Generated by `tools/source_overlap_check.py`.\n\n"
        "| signal | value |\n|---|---:|\n"
        f"| archive.org titles | {len(arch_titles):,} |\n"
        f"| corpus book titles | {len(corp_titles):,} |\n"
        f"| shared titles | {len(shared_titles):,} ({title_rate:.1%}) |\n"
        f"| archive items fetched | {len(fetched)} |\n"
        f"| exact paragraph overlap | {exact_shared:,} ({exact_rate:.1%}) |\n"
        f"| near-duplicate overlap | {near:,} ({near_rate:.1%}) |\n\n"
        f"**Verdict:** {verdict}\n",
        encoding="utf-8")
    print(f"Written: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
