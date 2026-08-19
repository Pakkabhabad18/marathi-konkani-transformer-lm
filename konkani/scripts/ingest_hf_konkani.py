#!/usr/bin/env python3
"""
Additional DOWNLOADED Konkani corpora from Hugging Face.

WHY THIS EXISTS
---------------
Konkani reached 165,815,092 training tokens - 33.2% of the ~500M target - with
69.4% of those tokens manually collected. The manual requirement is met more
than three times over; what is short is the *total*, and the only way to raise a
total without touching the manual side is more DOWNLOADED text.

A systematic re-search of Hugging Face (19 Aug 2026) found two datasets of real
size that earlier searches missed, because earlier searches queried the string
"konkani" against dataset *cards* rather than the language filter and the full
dataset index.

WHAT IS INGESTED, AND WHAT IS NOT
---------------------------------
`cfilt/RoundTripOCR-konkani` - IIT Bombay CFILT. 1.44 GB, size category
1M<n<10M. Three columns: `ocr` (text with OCR errors), `correct` (clean text),
`font`. We take **only `correct`**.

    IMPORTANT: this dataset renders each source sentence in many different
    fonts, so the same `correct` string recurs once per font. The row count
    therefore massively overstates the unique text. Exact-hash deduplication
    collapses those repeats, and the run summary reports unique yield rather
    than rows read - the distinction matters, and quoting the row count would
    be misleading.

`praveenkumar99/Konkani_Raw` - 1.37 GB of scraped pages. Ingested SELECTIVELY:

    included   konkani_page_*.txt, vishwa_konkani_page_*.txt, konkani_set_*
    excluded   translated_konkani_*.txt   - machine translated. The TAs were
                                            explicit that MT data is not
                                            appreciated, and our own language
                                            gate exists to keep translated
                                            Marathi out of the Konkani corpus.
    excluded   konkani_wikipedia_*        - we already hold Wikipedia from our
                                            own collection; ingesting it again
                                            as DOWNLOADED would double-count the
                                            same text on both sides of the ratio.

CLASSIFICATION
--------------
Both are `DOWNLOADED_DATASET`. Per TA guidance - "anything already organized on
HuggingFace which is then used is not Manual" - no amount of cleaning we apply
changes that, and the CollectionType is passed at the call site so it cannot
drift into being labelled manual later.

QUALITY GATES (identical to every other Konkani source)
-------------------------------------------------------
Unicode NFC, Devanagari-ratio floor (D-001), Marathi rejection by the
closed-class function-word discriminator, minimum length, exact + near-duplicate
removal. Konkani and Marathi share a script and much vocabulary, so the language
gate is load-bearing here: a corpus labelled "Konkani" is not evidence that its
rows are Konkani.

USAGE
-----
    python3 konkani/scripts/ingest_hf_konkani.py --source roundtripocr --pilot 50000
    python3 konkani/scripts/ingest_hf_konkani.py --source roundtripocr
    python3 konkani/scripts/ingest_hf_konkani.py --source konkani_raw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.checkpoint import Checkpoint                              # noqa: E402
from common.dedup import Deduplicator, exact_hash                     # noqa: E402
from common.manifest import (                                         # noqa: E402
    CollectionType,
    ManifestWriter,
    make_record,
)
from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import NORMALIZATION_STEPS, normalize            # noqa: E402

DATA_DIR = REPO_ROOT / "konkani" / "data"
LANGUAGE = "kok"

MIN_WORDS = 25
MIN_DEVANAGARI_RATIO = 0.70
SHARD_SIZE = 2000
MAX_SEGMENT_WORDS = 1200

SOURCES = {
    "roundtripocr": {
        "dataset": "cfilt/RoundTripOCR-konkani",
        "source_name": "hf_cfilt_roundtripocr_konkani",
        "text_column": "correct",
        "note": "IIT Bombay CFILT. Only the `correct` column is used; the same "
                "sentence recurs once per font, so dedup does the real work.",
    },
    "konkani_raw": {
        "dataset": "praveenkumar99/Konkani_Raw",
        "source_name": "hf_konkani_raw_scrape",
        "text_column": None,               # plain-text files, not columnar
        "include_prefixes": ("konkani_page_", "vishwa_konkani_page_",
                             "konkani_set_"),
        "exclude_prefixes": ("translated_konkani_", "konkani_wikipedia"),
        "note": "Selective: MT-translated and Wikipedia files excluded.",
    },
}


def segments(text: str, max_words: int = MAX_SEGMENT_WORDS):
    """Split long text at paragraph boundaries so dedup stays meaningful."""
    paras = [p for p in text.split("\n") if p.strip()]
    buf, count = [], 0
    for para in paras:
        w = len(para.split())
        if count + w > max_words and buf:
            yield "\n".join(buf)
            buf, count = [], 0
        buf.append(para)
        count += w
    if buf:
        yield "\n".join(buf)


def evaluate(raw: str):
    """Apply the same gates every Konkani source passes through."""
    text = normalize(raw, keep_paragraphs=True)
    if len(text.split()) < MIN_WORDS:
        return text, None, None, "too_short"
    profile = profile_script(text)
    if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
        return text, profile, None, "not_devanagari_excluded_by_D001"
    langid = identify_marathi_konkani(text)
    if langid.label == "mr":
        return text, profile, langid, "langid_marathi_rejected"
    return text, profile, langid, None


def iter_rows(cfg: dict, limit: int):
    """Yield raw text strings from the configured Hugging Face dataset."""
    from datasets import load_dataset

    name = cfg["dataset"]
    col = cfg["text_column"]

    if col:
        ds = load_dataset(name, split="train", streaming=True)
        for i, row in enumerate(ds):
            if limit and i >= limit:
                return
            value = row.get(col)
            if value:
                yield str(value)
        return

    # File-based dataset: pull the repo and read only the permitted files.
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=name, repo_type="dataset")
    inc = cfg.get("include_prefixes", ())
    exc = cfg.get("exclude_prefixes", ())
    seen_rows = 0
    for path in sorted(Path(local).rglob("*.txt")):
        stem = path.name
        if exc and stem.startswith(exc):
            print(f"    [skip] {stem}  (excluded by policy)")
            continue
        if inc and not stem.startswith(inc):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for seg in segments(text):
            if limit and seen_rows >= limit:
                return
            seen_rows += 1
            yield seg


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ingest additional DOWNLOADED Konkani corpora from HF.")
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--pilot", type=int, default=0,
                    help="process only N rows and report; writes nothing")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dedup-threshold", type=float, default=0.85)
    args = ap.parse_args()

    cfg = SOURCES[args.source]
    source_name = cfg["source_name"]
    out_dir = DATA_DIR / "processed" / source_name
    manifest_path = DATA_DIR / "manifests" / f"{source_name}.jsonl"
    checkpoint_path = DATA_DIR / "checkpoints" / f"{source_name}.json"
    dry = args.pilot > 0

    print("=" * 74)
    print(f"INGESTING (DOWNLOADED): {source_name}")
    print(f"  dataset : {cfg['dataset']}")
    print(f"  note    : {cfg['note']}")
    print(f"  mode    : {'PILOT (writes nothing)' if dry else 'WRITE'}")
    print("=" * 74)

    if dry:
        import tempfile
        checkpoint_path = Path(tempfile.mkdtemp(prefix="pilot_")) / "cp.json"
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(checkpoint_path, source_name)
    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = None if dry else ManifestWriter(manifest_path)

    seen_exact: set[str] = set()
    rejected: dict[str, int] = {}
    accepted = words_total = rows_read = 0
    shard_index, shard = 0, None
    if not dry:
        shard = open(out_dir / f"shard_{shard_index:05d}.txt", "a",
                     encoding="utf-8")
    start = time.time()

    limit = args.pilot or args.limit

    try:
        for raw in iter_rows(cfg, limit):
            rows_read += 1

            # Exact hash BEFORE the expensive gates: RoundTripOCR repeats each
            # sentence once per font, so most rows are cheap rejects.
            h = exact_hash(raw)
            if h in seen_exact:
                rejected["exact_duplicate"] = rejected.get("exact_duplicate", 0) + 1
                continue
            seen_exact.add(h)

            text, profile, langid, reason = evaluate(raw)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            if deduper.is_duplicate(text):
                rejected["near_duplicate"] = rejected.get("near_duplicate", 0) + 1
                continue

            accepted += 1
            words_total += len(text.split())

            if manifest:
                manifest.write(make_record(
                    text=text, raw_text=raw,
                    source_name=source_name,
                    source_url=f"https://huggingface.co/datasets/{cfg['dataset']}",
                    collection_type=CollectionType.DOWNLOADED_DATASET,
                    language=LANGUAGE,
                    preprocessing_applied=NORMALIZATION_STEPS + [
                        "hf_ingest", "devanagari_only_D001", "exact_dedup"],
                    script=profile.script,
                    langid_score=langid.score,
                    langid_label=langid.label,
                    devanagari_ratio=profile.devanagari_ratio,
                    doc_id=f"{source_name}_{accepted:08d}",
                    notes=cfg["note"],
                ))
            if shard:
                shard.write(text.replace("\n", " ") + "\n")
                if accepted % SHARD_SIZE == 0:
                    shard.close()
                    shard_index += 1
                    shard = open(out_dir / f"shard_{shard_index:05d}.txt", "a",
                                 encoding="utf-8")

            if rows_read % 100_000 == 0:
                el = time.time() - start
                print(f"  rows={rows_read:,} accepted={accepted:,} "
                      f"words={words_total:,} "
                      f"({rows_read/max(el,1e-9):,.0f} rows/s)", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved; re-run to resume.")
    finally:
        if shard:
            shard.close()
        if manifest:
            manifest.close()
        checkpoint.save(collected=accepted, rejection_reasons=rejected,
                        dedup_stats=deduper.stats.to_dict())
        checkpoint.close()

    el = (time.time() - start) / 60
    print("\n" + "=" * 74)
    print("RUN SUMMARY")
    print("=" * 74)
    print(f"Rows read:            {rows_read:,}")
    print(f"Documents accepted:   {accepted:,}")
    print(f"Words accepted:       {words_total:,}")
    print(f"Elapsed:              {el:.1f} min")
    if rows_read:
        print(f"\nUnique yield: {accepted/rows_read:.1%} of rows survived. "
              f"Row count alone would OVERSTATE this corpus.")
    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<34}{count:>12,}")
    print("\n" + "-" * 74)
    print("EFFECT ON THE KONKANI CORPUS")
    print("-" * 74)
    print(f"  downloaded words added   {words_total:>14,}")
    print("  (DOWNLOADED - does not count toward the 20% manual requirement)")
    if dry:
        print("\nPILOT: nothing was written. Re-run without --pilot to collect.")
    else:
        print(f"\nManifest: {manifest_path.relative_to(REPO_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
