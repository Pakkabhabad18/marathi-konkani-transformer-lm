#!/usr/bin/env python3
"""
Look at the actual text. Every other check in this project measured counts.

WHY THIS EXISTS
---------------
Everything verified so far - documents/minute, words collected, duplicate rate,
manual ratio, langid distribution - is a count. Counts can all look healthy while
the corpus is full of OCR garbage, navigation boilerplate, or repeated headers.
Discovering that after training would be expensive; discovering it after the
deadline would be fatal.

So this reads real documents and reports, per source:

  * SAMPLES you can actually read. No metric substitutes for looking.
  * OCR noise - the share of "words" that are one or two stray characters.
    Scanned government forms produce a lot of these.
  * Junk characters - anything that is not letters, digits, or normal punctuation.
  * Boilerplate - lines and opening phrases that repeat across many documents.
    A phrase appearing at the start of 40% of documents is site furniture that
    the extractor missed, and it will be memorised by the model.
  * Repetition within documents - a document that is one phrase repeated is OCR
    failure or a stuck template.
  * Length distribution - to expose truncation or runaway outliers.
  * Script and language purity per source.

WHAT TO DO WITH THE OUTPUT
--------------------------
Read the samples first. If they read like natural language in the right
language, the corpus is probably fine and the metrics tell you how fine. If they
do not, no metric will rescue it and the source needs fixing before more time is
spent collecting from it.

USAGE
-----
    python3 tools/audit_quality.py --language marathi
    python3 tools/audit_quality.py --language konkani --samples 5
    python3 tools/audit_quality.py --language marathi --full   # scan every doc
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.scriptid import identify_marathi_konkani, profile_script   # noqa: E402

SEED = 20260819
SCAN_PER_SOURCE = 4000          # documents scanned per source unless --full

_DEVA = re.compile(r"[ऀ-ॿ]")
_ALLOWED_PUNCT = set(" .,;:!?()[]{}\"'`-–—/\\|%&*+=<>@#$^~\n\t।॥…‘’“”")


def classify(path: Path) -> tuple[str, bool]:
    parts = path.parts
    is_manual = "manual" in parts
    source = path.parent.name
    if source.startswith("shard"):
        source = path.parent.parent.name
    return source, is_manual


def ocr_noise_ratio(text: str) -> float:
    """Share of whitespace tokens that are 1-2 characters and not real words.

    OCR of scanned documents fragments words into stray characters. Devanagari
    matras and single consonants appearing as standalone tokens is the classic
    signature. Clean prose sits well under 10%; heavily damaged OCR runs 30%+.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    stray = sum(1 for t in tokens if len(t) <= 2 and not t.isdigit()
                and t not in {"।", "॥", "आणि", "आनी", "तो", "ती", "ना", "व", "अ"})
    return stray / len(tokens)


def junk_char_ratio(text: str) -> float:
    """Share of characters that are neither letters/digits nor normal punctuation."""
    if not text:
        return 0.0
    junk = 0
    for ch in text:
        if ch in _ALLOWED_PUNCT or ch.isalnum():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("L", "N", "M", "Z")):
            continue
        junk += 1
    return junk / len(text)


def internal_repetition(text: str) -> float:
    """1 - (unique 5-grams / total 5-grams). High means the document repeats itself."""
    words = text.split()
    if len(words) < 20:
        return 0.0
    grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    return 1 - (len(set(grams)) / len(grams))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect real corpus text.")
    parser.add_argument("--language", required=True, choices=["marathi", "konkani"])
    parser.add_argument("--samples", type=int, default=3,
                        help="documents to print per source")
    parser.add_argument("--full", action="store_true",
                        help="scan every document (slow) instead of a sample")
    parser.add_argument("--chars", type=int, default=420,
                        help="characters of each sample to print")
    args = parser.parse_args()

    lang = args.language
    rng = random.Random(SEED)
    data_dir = REPO_ROOT / lang / "data"
    shards = sorted(data_dir.rglob("shard_*.txt"))
    if not shards:
        raise SystemExit(f"No shards under {data_dir}.")

    by_source: dict[str, list[Path]] = defaultdict(list)
    manual_flag: dict[str, bool] = {}
    for shard in shards:
        source, is_manual = classify(shard)
        by_source[source].append(shard)
        manual_flag[source] = is_manual

    print("=" * 78)
    print(f"CORPUS QUALITY AUDIT - {lang}")
    print(f"{len(shards)} shard file(s) across {len(by_source)} source(s)")
    print("=" * 78)

    overall_problems = []

    for source, paths in sorted(by_source.items()):
        docs: list[str] = []
        limit = 0 if args.full else SCAN_PER_SOURCE
        for path in paths:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        docs.append(line)
                        if limit and len(docs) >= limit:
                            break
            if limit and len(docs) >= limit:
                break

        if not docs:
            continue

        tag = "manual" if manual_flag[source] else "downloaded"
        print(f"\n{'=' * 78}")
        print(f"SOURCE: {source}   [{tag}]   {len(docs):,} documents scanned")
        print("=" * 78)

        lengths = [len(d.split()) for d in docs]
        noise = [ocr_noise_ratio(d) for d in docs]
        junk = [junk_char_ratio(d) for d in docs]
        repet = [internal_repetition(d) for d in docs]
        deva = [profile_script(d).devanagari_ratio for d in docs]

        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        ordered = sorted(lengths)
        print(f"\n  length (words)   min {ordered[0]:,}  "
              f"median {ordered[len(ordered) // 2]:,}  "
              f"mean {mean(lengths):,.0f}  "
              f"p95 {ordered[int(len(ordered) * .95)]:,}  max {ordered[-1]:,}")
        print(f"  devanagari       mean {mean(deva):.1%}  "
              f"min {min(deva):.1%}")
        print(f"  OCR noise        mean {mean(noise):.1%}  "
              f"worst {max(noise):.1%}   (clean prose < 10%)")
        print(f"  junk characters  mean {mean(junk):.3%}          (clean < 0.5%)")
        print(f"  self-repetition  mean {mean(repet):.1%}  "
              f"worst {max(repet):.1%}   (normal < 15%)")

        # language check
        labels = Counter(identify_marathi_konkani(d).label for d in docs[:1500])
        total_lab = sum(labels.values())
        label_str = "  ".join(f"{k}={v / total_lab:.1%}" for k, v in labels.most_common())
        print(f"  language         {label_str}")

        # ---- boilerplate: repeated opening phrases -----------------------
        openings = Counter(" ".join(d.split()[:8]) for d in docs)
        common = [(p, c) for p, c in openings.most_common(5) if c > 1]
        if common:
            print(f"\n  most repeated opening phrases:")
            for phrase, count in common:
                share = count / len(docs)
                marker = "  <-- BOILERPLATE" if share > 0.05 else ""
                print(f"    {share:>6.1%}  {phrase[:64]}{marker}")
                if share > 0.05:
                    overall_problems.append(
                        f"{source}: {share:.1%} of documents start with "
                        f"'{phrase[:44]}'")
        else:
            print("\n  most repeated opening phrases: none repeat (good)")

        # ---- flags --------------------------------------------------------
        if mean(noise) > 0.25:
            overall_problems.append(
                f"{source}: OCR noise {mean(noise):.1%} - text is fragmented")
        if mean(junk) > 0.01:
            overall_problems.append(
                f"{source}: junk characters {mean(junk):.2%} - encoding or OCR damage")
        if mean(repet) > 0.30:
            overall_problems.append(
                f"{source}: self-repetition {mean(repet):.1%} - templated or stuck text")
        if mean(deva) < 0.75:
            overall_problems.append(
                f"{source}: Devanagari only {mean(deva):.1%} - script contamination")

        # ---- samples ------------------------------------------------------
        print(f"\n  {'-' * 74}")
        print(f"  SAMPLES - read these; no metric replaces looking at the text")
        print(f"  {'-' * 74}")
        for i, doc in enumerate(rng.sample(docs, min(args.samples, len(docs))), 1):
            words = len(doc.split())
            print(f"\n  [{i}] {words:,} words | devanagari "
                  f"{profile_script(doc).devanagari_ratio:.1%} | "
                  f"noise {ocr_noise_ratio(doc):.1%}")
            print(f"      {doc[:args.chars]}")
            if len(doc) > args.chars:
                print(f"      ... (+{len(doc) - args.chars:,} more characters)")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if overall_problems:
        print(f"  {len(overall_problems)} issue(s) worth attention:\n")
        for problem in overall_problems:
            print(f"    - {problem}")
        print("\n  These are flags, not verdicts. Read the samples above and decide")
        print("  whether the text is usable. A high OCR-noise score on scanned")
        print("  government forms may be acceptable; boilerplate never is.")
    else:
        print("  No automated flags raised.")
        print("  Still read the samples - a corpus can pass every metric and")
        print("  still be the wrong text.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
