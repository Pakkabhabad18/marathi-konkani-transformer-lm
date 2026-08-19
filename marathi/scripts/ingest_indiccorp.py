#!/usr/bin/env python3
"""
Marathi source M3: IndicCorp v2 (DOWNLOADED, not manual).

WHAT THIS IS
------------
`ai4bharat/IndicCorpV2`, split `mar_Deva`. Verified from the dataset card:
~27.8M rows for Marathi, released under **CC-0** (public domain) - the cleanest
licensing available for a public Indic corpus, which is why it is preferred over
Sangraha (CC-BY-4.0) as the primary downloaded source.

THE POINT OF THIS SCRIPT IS TO STOP AT THE RIGHT PLACE
------------------------------------------------------
Every other collector in this project tries to gather as much as possible. This
one is the opposite: it must NOT gather as much as possible.

The requirement is `manual / total >= 0.20`, i.e. `total <= 5 x manual`, so:

    downloaded_allowed = 4 x manual_words

Past that point, every additional downloaded word makes the corpus *less*
compliant. A 500M-token corpus at 12% manual fails a stated requirement; a
300M-token corpus at 20% manual does not.

So by default this script reads the existing manifests, computes how much manual
data has actually been collected, and caps itself at 4x that. The cap is
computed from measurement, not assumed, and it is printed before any work
starts. `--max-words` overrides it if you need to.

CROSS-CORPUS FILTERING STILL APPLIES
------------------------------------
Konkani-labelled text is rejected here exactly as it is in the manual
collectors. A public Marathi web crawl will contain some Konkani - the two
languages are closely related and share Devanagari - and the specification
forbids the two corpora sharing documents.

USAGE
-----
    python3 marathi/scripts/ingest_indiccorp.py --dry-run --pilot 50000
    python3 marathi/scripts/ingest_indiccorp.py                    # auto-capped
    python3 marathi/scripts/ingest_indiccorp.py --max-words 220000000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.checkpoint import Checkpoint                             # noqa: E402
from common.dedup import Deduplicator                                 # noqa: E402
from common.manifest import (                                         # noqa: E402
    CollectionType,
    ManifestWriter,
    make_record,
    summarize,
)
from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import NORMALIZATION_STEPS, normalize            # noqa: E402

DATASET_NAME = "ai4bharat/IndicCorpV2"
# The dataset exposes ONE builder config containing every language as a separate
# split. `mar_Deva` is the split name, not the config - passing it as the config
# fails with "BuilderConfig 'mar_Deva' not found. Available: ['indiccorp_v2']".
# Caught by a --dry-run before the real ingest.
DATASET_CONFIG = "indiccorp_v2"
DATASET_SPLIT = "mar_Deva"

# Free disk required before starting. 205M downloaded words is roughly 3.5-4 GB
# of text, plus the HuggingFace cache while streaming.
MIN_FREE_GB = 12.0
SOURCE_NAME = "ai4bharat_indiccorp_v2_mar"
JOB_NAME = "marathi_indiccorp"
LANGUAGE = "mr"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET_NAME}"
LICENSE_NOTE = "CC-0 (public domain)"

DATA_DIR = REPO_ROOT / "marathi" / "data"
OUT_DIR = DATA_DIR / "processed" / SOURCE_NAME
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / f"{JOB_NAME}.json"
MANIFEST_PATH = DATA_DIR / "manifests" / f"{JOB_NAME}.jsonl"

MIN_WORDS = 40
MIN_DEVANAGARI_RATIO = 0.70
SHARD_SIZE = 20000
MANUAL_RATIO = 0.20          # the specification's floor


def current_manual_words(language_dir: Path) -> tuple[int, int]:
    """Total manual and downloaded words already recorded for this language."""
    manifests = sorted((language_dir / "manifests").glob("*.jsonl"))
    manual = downloaded = 0
    for path in manifests:
        acc = summarize(path)
        manual += acc.manual_words
        downloaded += acc.downloaded_words
    return manual, downloaded


def evaluate(raw_text: str):
    text = normalize(raw_text, keep_paragraphs=False)

    if len(text.split()) < MIN_WORDS:
        return text, None, None, "too_short"

    profile = profile_script(text)
    if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
        return text, profile, None, "not_enough_devanagari"

    langid = identify_marathi_konkani(text)
    if langid.label == "kok":
        return text, profile, langid, "langid_konkani_rejected"
    if langid.label == "undecided":
        # Short crawl fragments frequently lack function-word evidence. Kept,
        # but flagged in the manifest so the decision remains auditable.
        return text, profile, langid, None
    return text, profile, langid, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest IndicCorp v2 Marathi.")
    parser.add_argument("--pilot", type=int, default=0,
                        help="stop after N source rows (0 = no row limit)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-words", type=int, default=0,
                        help="override the automatic 4x-manual cap")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 1

    manual_words, downloaded_words = current_manual_words(DATA_DIR)

    if args.max_words:
        cap = args.max_words
        cap_reason = "set explicitly with --max-words"
    else:
        allowed_total = manual_words / MANUAL_RATIO if manual_words else 0
        cap = max(int(allowed_total - manual_words - downloaded_words), 0)
        cap_reason = f"4 x manual ({manual_words:,} manual words already collected)"

    print("=" * 70)
    print(f"INGESTING: {SOURCE_NAME}")
    print(f"Language: {LANGUAGE}   Collection type: DOWNLOADED (not manual)")
    print(f"License:  {LICENSE_NOTE}")
    print("=" * 70)
    print("\nMANUAL-RATIO BUDGET")
    print("-" * 70)
    print(f"  manual words collected so far    {manual_words:>15,}")
    print(f"  downloaded words already held    {downloaded_words:>15,}")
    print(f"  total corpus permitted (5x)      {manual_words * 5:>15,}")
    print(f"  downloaded words still allowed   {cap:>15,}   <- hard stop")
    print(f"  cap basis: {cap_reason}")

    if cap <= 0:
        print("\n  STOP: no downloaded budget remains at the current manual total.")
        print("  Collect more manual data first - every manual word earns four")
        print("  downloaded words. Running this now would break the 20% floor.")
        return 1

    if manual_words < 1_000_000:
        print(f"\n  WARNING: only {manual_words:,} manual words so far. The cap")
        print("  will rise as manual collection continues, so consider waiting")
        print("  until the manual crawls finish before doing the full ingest.")

    # DISK CHECK BEFORE, NOT DURING.
    # Running out of space part-way through leaves a truncated shard and a
    # manifest that disagrees with it. Cheaper to refuse up front.
    import shutil
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    est_gb = cap * 6 * 3 / 1e9      # ~6 chars/word, ~3 bytes/char in Devanagari
    print(f"\n  disk free                        {free_gb:>15,.1f} GB")
    print(f"  estimated output for this cap    {est_gb:>15,.1f} GB")
    if free_gb < MIN_FREE_GB and not args.dry_run:
        print(f"\n  STOP: only {free_gb:.1f} GB free, below the {MIN_FREE_GB:.0f} GB "
              f"minimum.")
        print("  Free space first, or lower the cap with --max-words.")
        return 1

    print("=" * 70)

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # A dry run must not persist checkpoint state - see the note in
    # konkani/scripts/collect_archive_books.py. A preview that consumes the work
    # it is previewing is worse than no preview.
    if args.dry_run:
        import tempfile
        checkpoint_path = Path(tempfile.mkdtemp(prefix="dryrun_")) / f"{JOB_NAME}.json"
    else:
        checkpoint_path = CHECKPOINT_PATH

    checkpoint = Checkpoint(checkpoint_path, JOB_NAME)
    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = None if args.dry_run else ManifestWriter(MANIFEST_PATH)

    print(f"\nStreaming {DATASET_NAME} [{DATASET_CONFIG}] split={DATASET_SPLIT} ...")
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG,
                               split=DATASET_SPLIT, streaming=True)
    except ValueError as exc:
        # Report what IS available rather than just failing - the config/split
        # distinction is the exact thing that went wrong here once already.
        print(f"\nERROR loading dataset: {exc}", file=sys.stderr)
        try:
            from datasets import get_dataset_config_names, get_dataset_split_names
            configs = get_dataset_config_names(DATASET_NAME)
            print(f"\n  available configs: {configs}", file=sys.stderr)
            for cfg in configs:
                splits = get_dataset_split_names(DATASET_NAME, cfg)
                marathi = [s for s in splits if "mar" in s.lower()]
                print(f"  config '{cfg}': {len(splits)} splits; "
                      f"Marathi-looking: {marathi}", file=sys.stderr)
        except Exception:
            pass
        return 1

    rejected: dict[str, int] = {}
    accepted = words_total = rows = 0
    undecided = 0
    shard_index = 0
    shard = None
    if not args.dry_run:
        shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a", encoding="utf-8")
    start = time.time()

    try:
        for example in dataset:
            rows += 1
            raw = (example.get("text") or example.get("content") or "").strip()
            if not raw:
                continue

            text, profile, langid, reason = evaluate(raw)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            if deduper.is_duplicate(text):
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                continue

            if langid.label == "undecided":
                undecided += 1

            accepted += 1
            words_total += len(text.split())

            if manifest:
                manifest.write(make_record(
                    text=text, raw_text=raw,
                    source_name=SOURCE_NAME, source_url=SOURCE_URL,
                    collection_type=CollectionType.DOWNLOADED_DATASET,
                    language=LANGUAGE,
                    preprocessing_applied=NORMALIZATION_STEPS,
                    script=profile.script,
                    langid_score=langid.score,
                    langid_label=langid.label,
                    devanagari_ratio=profile.devanagari_ratio,
                    doc_id=f"indiccorp_{rows:09d}",
                    notes=f"license={LICENSE_NOTE}; config={DATASET_CONFIG}",
                ))

            if shard:
                shard.write(text + "\n")
                if accepted % SHARD_SIZE == 0:
                    shard.close()
                    shard_index += 1
                    shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt",
                                 "a", encoding="utf-8")

            if accepted % 20000 == 0:
                checkpoint.save(collected=accepted, page=rows)
                elapsed = max(time.time() - start, 1e-6)
                print(f"  rows={rows:,} accepted={accepted:,} "
                      f"words={words_total:,}/{cap:,} "
                      f"({words_total / cap:.1%} of budget) "
                      f"dup={deduper.stats.duplicate_rate:.1%} "
                      f"{rows / elapsed:,.0f} rows/s", flush=True)

            if words_total >= cap:
                print(f"\nBudget reached: {words_total:,} downloaded words.")
                print("Stopping deliberately to preserve the 20% manual ratio.")
                break
            if args.pilot and rows >= args.pilot:
                print("\nPilot row limit reached.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved; re-run to resume.")
    finally:
        if shard:
            shard.close()
        if manifest:
            manifest.close()
        checkpoint.save(collected=accepted, page=rows,
                        dedup_stats=deduper.stats.to_dict(),
                        rejection_reasons=rejected)
        checkpoint.close()

    elapsed = max(time.time() - start, 1e-6)
    new_total = manual_words + downloaded_words + words_total

    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"Source rows read:      {rows:,}")
    print(f"Documents accepted:    {accepted:,}")
    print(f"  langid undecided:    {undecided:,} "
          f"({undecided / max(accepted, 1):.1%})")
    print(f"Words accepted:        {words_total:,}")
    print(f"Elapsed:               {elapsed / 60:.1f} min")
    print(f"\nDeduplication:         {deduper.stats.to_dict()}")
    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:28s} {count:,}")

    print("\n" + "-" * 70)
    print("MANUAL RATIO AFTER THIS RUN")
    print("-" * 70)
    print(f"  manual        {manual_words:>15,}")
    print(f"  downloaded    {downloaded_words + words_total:>15,}")
    print(f"  total         {new_total:>15,}")
    if new_total:
        ratio = manual_words / new_total
        verdict = "OK" if ratio >= MANUAL_RATIO else "BELOW 20% - FIX BEFORE SUBMITTING"
        print(f"  manual share  {ratio:>14.1%}   {verdict}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
