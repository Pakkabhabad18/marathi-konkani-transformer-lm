#!/usr/bin/env python3
"""
Is IndicCorp v2's `gom.txt` really Konkani? (D-035)

THE QUESTION
------------
`ai4bharat/IndicCorpV2` ships `data/gom.txt`, 533,108,246 bytes labelled Goan
Konkani. Packed into 300-word documents and run through our closed-class
discriminator, 33,976 of 48,628 documents (70%) come back labelled Marathi.

The same discriminator, the same day, on three other Konkani sources:

    GlotCC-V1  gom-Deva         0 Marathi rejections of 1,049
    MADLAD-400 gom noisy       14 Marathi rejections of 4,602   (0.3%)
    Sangraha   verified gom    11 Marathi rejections of 14,491  (0.1%)

Two explanations fit, and they demand opposite actions:

    (A) `gom.txt` really is mostly Marathi. Then the gate is protecting the
        corpus, we keep the 4.3M words that pass, and the rejection rate is a
        finding worth reporting.

    (B) The discriminator misfires on this particular text. Then we are
        discarding ~25M genuine Konkani words for no reason.

Asserting either without evidence is exactly the failure mode this project has
already hit several times. This script decides it by CALIBRATION.

THE METHOD
----------
Run the identical discriminator over three populations:

    reference Konkani   our own manual archive.org Konkani books - text whose
                        language is not in doubt, since it was collected from
                        catalogued Konkani publications
    reference Marathi   our Marathi corpus - likewise not in doubt
    the population under test   packed documents from gom.txt

Documents are packed to the SAME target length in all three, because the
discriminator's confidence depends on document length and comparing a 340-word
population against a 2,000-word one would confound length with language.

If gom.txt's score distribution sits on top of the Marathi reference, (A) holds.
If it sits on top of the Konkani reference while still being labelled Marathi,
the discriminator is broken and (B) holds.

The script also prints the actual marker words driving the verdict on sampled
rejected documents, so the conclusion can be eyeballed rather than trusted.

USAGE
-----
    python3 tools/verify_gom_langid.py
    python3 tools/verify_gom_langid.py --sample 3000 --examples 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.scriptid import (                                         # noqa: E402
    KONKANI_MARKERS,
    MARATHI_MARKERS,
    _WORD_RE,
    identify_marathi_konkani,
)
from common.textnorm import normalize                                 # noqa: E402

TARGET_WORDS = 300
SEED = 20260819


def pack_lines(lines, target_words: int, limit: int):
    """Pack whitespace-separated lines into ~target_words documents."""
    docs, buf, count = [], [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        n = len(line.split())
        buf.append(line)
        count += n
        if count >= target_words:
            docs.append(" ".join(buf))
            buf, count = [], 0
            if len(docs) >= limit:
                return docs
    if buf:
        docs.append(" ".join(buf))
    return docs


def read_shard_docs(directory: Path, limit: int, max_shards: int = 40):
    """Sample packed documents from a directory of shard_*.txt files.

    Individual shards are read defensively. On this machine the repository sits
    under an iCloud-synced Desktop, so a shard can be a dataless placeholder
    whose first read blocks and then fails with ETIMEDOUT (errno 60). One
    unavailable shard must not abort a calibration that only needs a sample -
    it is skipped, counted, and reported.
    """
    shards = sorted(directory.rglob("shard_*.txt"))[:max_shards]
    if not shards:
        return []
    docs: list[str] = []
    skipped = 0
    for shard in shards:
        try:
            with shard.open("r", encoding="utf-8", errors="replace") as handle:
                docs.extend(pack_lines(handle, TARGET_WORDS, limit - len(docs)))
        except OSError as exc:
            skipped += 1
            print(f"    [skip] {shard.name}: {exc.__class__.__name__} {exc}")
            continue
        if len(docs) >= limit:
            break
    if skipped:
        print(f"    {skipped} shard(s) unreadable and skipped; "
              f"{len(docs):,} documents sampled from the rest.")
    return docs[:limit]


def read_gom_docs(limit: int):
    """Packed documents from the cached IndicCorp gom.txt."""
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id="ai4bharat/IndicCorpV2",
                                filename="data/gom.txt",
                                repo_type="dataset"))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return pack_lines(handle, TARGET_WORDS, limit)


def profile(name: str, docs: list[str]) -> dict:
    """Score a population and summarise the label distribution."""
    labels = {"mr": 0, "kok": 0, "undecided": 0}
    scores = []
    rejected_examples = []
    for text in docs:
        result = identify_marathi_konkani(normalize(text, keep_paragraphs=False))
        labels[result.label] = labels.get(result.label, 0) + 1
        scores.append(result.score)
        if result.label == "mr":
            rejected_examples.append((text, result))
    total = max(len(docs), 1)
    scores.sort()
    return {
        "name": name,
        "n": len(docs),
        "mr_pct": labels["mr"] / total,
        "kok_pct": labels["kok"] / total,
        "undecided_pct": labels["undecided"] / total,
        "median_score": scores[len(scores) // 2] if scores else 0.0,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "rejected_examples": rejected_examples,
    }


def marker_breakdown(text: str, top: int = 8):
    """Which marker words actually fired, and how often."""
    counts_m: dict[str, int] = {}
    counts_k: dict[str, int] = {}
    for word in _WORD_RE.findall(text):
        if word in MARATHI_MARKERS:
            counts_m[word] = counts_m.get(word, 0) + 1
        elif word in KONKANI_MARKERS:
            counts_k[word] = counts_k.get(word, 0) + 1
    fmt = lambda d: ", ".join(  # noqa: E731
        f"{w}×{c}" for w, c in sorted(d.items(), key=lambda kv: -kv[1])[:top]
    ) or "(none)"
    return fmt(counts_m), fmt(counts_k)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate the Marathi/Konkani discriminator on gom.txt.")
    ap.add_argument("--sample", type=int, default=2000,
                    help="documents per population")
    ap.add_argument("--examples", type=int, default=4,
                    help="rejected gom.txt documents to print in full detail")
    ap.add_argument("--marathi-dir", default="",
                    help="override the reference-Marathi shard directory")
    args = ap.parse_args()

    random.seed(SEED)

    populations = []

    kok_ref = REPO_ROOT / "konkani" / "data" / "manual" / "archive_org_konkani_books"
    # One specific directory, not the whole marathi/data tree: that tree holds
    # the IndicCorp shards and walking it costs minutes before a single
    # document is scored. Any Marathi shard directory serves as the reference.
    mr_ref = REPO_ROOT / "marathi" / "data" / "manual" / "archive_org_maharashtra_gr"
    if args.marathi_dir:
        mr_ref = Path(args.marathi_dir)

    print("=" * 78)
    print("LANGUAGE-ID CALIBRATION - is IndicCorp v2 gom.txt actually Konkani?")
    print(f"All populations packed to ~{TARGET_WORDS} words so that document")
    print("length cannot be confounded with language.")
    print("=" * 78)

    print(f"\nreading reference Konkani  ({kok_ref.name}) ...", flush=True)
    docs = read_shard_docs(kok_ref, args.sample)
    if docs:
        populations.append(profile("reference Konkani (our manual books)", docs))
    else:
        print("  WARNING: no shards found - reference Konkani unavailable.")

    print(f"reading reference Marathi  ({mr_ref.name}/...) ...", flush=True)
    docs = read_shard_docs(mr_ref, args.sample)
    if docs:
        populations.append(profile("reference Marathi (our Marathi corpus)", docs))
    else:
        print("  WARNING: no shards found - reference Marathi unavailable.")

    print("reading IndicCorp v2 gom.txt (cached) ...", flush=True)
    gom = read_gom_docs(args.sample)
    gom_profile = profile("IndicCorp v2 gom.txt  <- UNDER TEST", gom)
    populations.append(gom_profile)

    print("\n" + "=" * 78)
    print("LABEL DISTRIBUTION")
    print("=" * 78)
    print(f"  {'population':<42}{'n':>7}{'mr':>9}{'kok':>9}{'undec':>9}"
          f"{'median':>9}")
    print("  " + "-" * 74)
    for p in populations:
        print(f"  {p['name']:<42}{p['n']:>7,}{p['mr_pct']:>8.1%}"
              f"{p['kok_pct']:>9.1%}{p['undecided_pct']:>9.1%}"
              f"{p['median_score']:>9.2f}")

    print("\n  score: +1 = fully Marathi markers, -1 = fully Konkani markers")

    print("\n" + "=" * 78)
    print(f"SAMPLED gom.txt DOCUMENTS THE GATE LABELLED MARATHI")
    print("=" * 78)
    examples = gom_profile["rejected_examples"]
    if not examples:
        print("  None - the gate did not reject any sampled gom.txt document.")
    for text, result in examples[:args.examples]:
        m_hits, k_hits = marker_breakdown(text)
        print(f"\n  score {result.score:+.2f}   marathi_hits {result.marathi_hits}"
              f"   konkani_hits {result.konkani_hits}"
              f"   words {result.total_words}")
        print(f"    Marathi markers found: {m_hits}")
        print(f"    Konkani markers found: {k_hits}")
        snippet = " ".join(text.split()[:34])
        print(f"    text: {snippet} ...")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  If gom.txt's row sits beside the reference MARATHI row, the file is")
    print("  genuinely contaminated and the gate is doing its job - keep the")
    print("  documents that pass and report the rejection rate as a finding.")
    print()
    print("  If gom.txt's row sits beside the reference KONKANI row yet the gate")
    print("  still labels it Marathi, the discriminator is at fault and roughly")
    print("  25M words are being discarded wrongly - do NOT ship that result.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
