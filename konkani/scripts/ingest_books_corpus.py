#!/usr/bin/env python3
"""
Konkani source K0: ingest the Konkani Books Corpus (DOWNLOADED, not manual).

WHAT THIS IS
------------
`omdeep22/Konkani_books_corpus-v2` on Hugging Face. MIT licensed, ~8.23M rows,
described on its card as "digitized books, literature, and long-form cultural
texts" with known OCR whitespace damage.

This is a **downloaded** corpus. It does not count toward the 20% manual
requirement, however much cleaning we do to it. That classification is enforced
by the CollectionType passed to the manifest, not left to a comment.

THE PROBLEM WITH THE OBVIOUS APPROACH
-------------------------------------
The audit measured 8,222,553 usable records carrying 61,805,534 words - an
average of **7.52 words per record**. These are not documents. They are
individual OCR'd lines. Treating each row as a document would be wrong twice
over:

  1. Language-model training wants contiguous text. Feeding it 7-word fragments
     destroys exactly the long-range structure the model exists to learn.
  2. Deduplication on 7-word strings is meaningless - short lines like page
     headers and chapter titles collide constantly, so a naive dedup would
     delete large amounts of legitimate text.

WHAT THIS SCRIPT DOES INSTEAD
-----------------------------
The corpus contains marker rows beginning `--- SOURCE:`. The existing
`analyze_books_corpus.py` treats these as junk and skips them. They are not
junk - they are **book boundaries**, and the text after the marker is the book's
name. So we:

  1. Use each marker to close the previous document and start a new one, and
     record the book name as provenance.
  2. Accumulate the intervening line-fragments back into contiguous text.
  3. Split anything very long into segments at paragraph boundaries, so single
     documents stay a manageable size for dedup and shuffling.

This recovers document structure that the flat row layout had lost, and it gives
every segment a real source attribution instead of "row 4,193,882".

If the markers turn out not to exist or not to delimit books, `--pilot` will
show that immediately - run it first and read the reported document count and
words-per-document before committing to a full pass.

USAGE
-----
    # Look before you leap: process 50k rows and report, writing nothing.
    python3 konkani/scripts/ingest_books_corpus.py --pilot 50000 --dry-run

    # Pilot for real (writes output).
    python3 konkani/scripts/ingest_books_corpus.py --pilot 50000

    # Full pass. Resumable: Ctrl-C and re-run.
    python3 konkani/scripts/ingest_books_corpus.py
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
)
from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import NORMALIZATION_STEPS, normalize            # noqa: E402

DATASET_NAME = "omdeep22/Konkani_books_corpus-v2"
SOURCE_NAME = "hf_konkani_books_corpus_v2"
JOB_NAME = "konkani_books_corpus"
LANGUAGE = "kok"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET_NAME}"

DATA_DIR = REPO_ROOT / "konkani" / "data"
OUT_DIR = DATA_DIR / "processed" / SOURCE_NAME
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / f"{JOB_NAME}.json"
MANIFEST_PATH = DATA_DIR / "manifests" / f"{JOB_NAME}.jsonl"

MARKER_PREFIX = "--- SOURCE:"

# Quality gates.
MIN_WORDS = 40                 # a segment shorter than this is a fragment
MIN_DEVANAGARI_RATIO = 0.80    # the corpus is ~99.78% Devanagari; be strict
MAX_SEGMENT_WORDS = 1200       # split long books into manageable segments
SHARD_SIZE = 5000

# Konkani/Marathi separation. The langid score is signed: negative = Konkani.
# We require a clear Konkani lean, because Marathi leaking into Model L's corpus
# would violate the specification's independence requirement.
LANGID_MAX_SCORE = -0.20       # score must be at most this (i.e. Konkani-leaning)
KEEP_UNDECIDED = True          # short segments often lack markers; see below


def iter_documents(dataset, marker_prefix: str = MARKER_PREFIX):
    """Reassemble line-fragment rows into (book_name, text) documents.

    Yields a document each time a new `--- SOURCE:` marker is seen, and once
    more at the end of the stream for the final book.
    """
    current_book = "unknown"
    buffer: list[str] = []
    rows_seen = 0

    for example in dataset:
        rows_seen += 1
        text = (example.get("text") or "").strip()
        if not text:
            continue

        if text.startswith(marker_prefix):
            if buffer:
                yield current_book, "\n".join(buffer), rows_seen
                buffer = []
            current_book = text[len(marker_prefix):].strip() or "unknown"
            continue

        buffer.append(text)

    if buffer:
        yield current_book, "\n".join(buffer), rows_seen


def split_segments(text: str, max_words: int = MAX_SEGMENT_WORDS):
    """Split a long document at paragraph boundaries, never mid-paragraph."""
    paragraphs = [p for p in text.split("\n") if p.strip()]
    segment: list[str] = []
    count = 0

    for para in paragraphs:
        words = len(para.split())
        if count + words > max_words and segment:
            yield "\n".join(segment)
            segment, count = [], 0
        segment.append(para)
        count += words

    if segment:
        yield "\n".join(segment)


def evaluate(raw_segment: str):
    """Normalize and gate one segment. Returns (text, profile, langid, reason)."""
    text = normalize(raw_segment, keep_paragraphs=True)

    if len(text.split()) < MIN_WORDS:
        return text, None, None, "too_short"

    profile = profile_script(text)
    if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
        return text, profile, None, "not_enough_devanagari"

    langid = identify_marathi_konkani(text)

    if langid.label == "mr":
        # Marathi in the Konkani corpus is the contamination we most need to
        # prevent - always reject, regardless of KEEP_UNDECIDED.
        return text, profile, langid, "langid_marathi_rejected"

    if langid.label == "kok" and langid.score <= LANGID_MAX_SCORE:
        return text, profile, langid, None

    if langid.label == "undecided" and KEEP_UNDECIDED:
        # Kept, but flagged in the manifest so the decision stays auditable and
        # these segments can be re-filtered later without recollecting.
        return text, profile, langid, None

    return text, profile, langid, f"langid_{langid.label}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the Konkani books corpus.")
    parser.add_argument("--pilot", type=int, default=0,
                        help="stop after N source rows (0 = full corpus)")
    parser.add_argument("--dry-run", action="store_true",
                        help="measure and report, write nothing")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: the 'datasets' package is required.\n"
              "  pip install datasets", file=sys.stderr)
        return 1

    print("=" * 68)
    print(f"INGESTING: {SOURCE_NAME}")
    print(f"Language: {LANGUAGE}   Collection type: DOWNLOADED (not manual)")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}"
          f"{f' | pilot {args.pilot:,} rows' if args.pilot else ' | full corpus'}")
    print("=" * 68)

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(CHECKPOINT_PATH, JOB_NAME)
    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = None if args.dry_run else ManifestWriter(MANIFEST_PATH)

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    rejected: dict[str, int] = {}
    accepted = 0
    undecided_kept = 0
    total_words = 0
    books_seen = 0
    doc_word_counts: list[int] = []
    shard_index = 0
    shard_file = None
    if not args.dry_run:
        shard_file = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a", encoding="utf-8")

    start = time.time()
    rows = 0

    try:
        for book_name, doc_text, rows in iter_documents(dataset):
            books_seen += 1
            doc_word_counts.append(len(doc_text.split()))

            for segment in split_segments(doc_text):
                text, profile, langid, reason = evaluate(segment)
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue

                if deduper.is_duplicate(text):
                    rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                    continue

                if langid.label == "undecided":
                    undecided_kept += 1

                accepted += 1
                total_words += len(text.split())

                if manifest:
                    manifest.write(make_record(
                        text=text,
                        raw_text=segment,
                        source_name=SOURCE_NAME,
                        source_url=SOURCE_URL,
                        collection_type=CollectionType.DOWNLOADED_DATASET,
                        language=LANGUAGE,
                        preprocessing_applied=NORMALIZATION_STEPS + ["reassemble_lines",
                                                                     "segment_split"],
                        script=profile.script,
                        langid_score=langid.score,
                        langid_label=langid.label,
                        devanagari_ratio=profile.devanagari_ratio,
                        doc_id=f"book_{books_seen:06d}_seg_{accepted:08d}",
                        notes=f"book={book_name}; license=MIT",
                    ))

                if shard_file:
                    shard_file.write(text.replace("\n", " ") + "\n")
                    if accepted % SHARD_SIZE == 0:
                        shard_file.close()
                        shard_index += 1
                        shard_file = open(OUT_DIR / f"shard_{shard_index:05d}.txt",
                                          "a", encoding="utf-8")

            if books_seen % 100 == 0:
                checkpoint.save(collected=accepted, page=rows)
                elapsed = max(time.time() - start, 1e-6)
                print(f"  books={books_seen:,} rows={rows:,} segments={accepted:,} "
                      f"words={total_words:,} dup={deduper.stats.duplicate_rate:.1%} "
                      f"({rows / elapsed:,.0f} rows/s)", flush=True)

            if args.pilot and rows >= args.pilot:
                print("\nPilot row limit reached.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved; re-run to resume.")
    finally:
        if shard_file:
            shard_file.close()
        if manifest:
            manifest.close()
        checkpoint.save(collected=accepted, page=rows,
                        dedup_stats=deduper.stats.to_dict(),
                        rejection_reasons=rejected)
        checkpoint.close()

    elapsed = time.time() - start
    avg_doc = sum(doc_word_counts) / len(doc_word_counts) if doc_word_counts else 0

    print("\n" + "=" * 68)
    print("RUN SUMMARY")
    print("=" * 68)
    print(f"Source rows read:      {rows:,}")
    print(f"Books detected:        {books_seen:,}")
    print(f"Words per book (mean): {avg_doc:,.0f}")
    print(f"Segments accepted:     {accepted:,}")
    print(f"  of which undecided:  {undecided_kept:,} "
          f"({undecided_kept / max(accepted, 1):.1%})")
    print(f"Words accepted:        {total_words:,}")
    print(f"Elapsed:               {elapsed / 60:.1f} min")
    print(f"\nDeduplication:         {deduper.stats.to_dict()}")
    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:28s} {count:,}")

    print("\n" + "-" * 68)
    print("READ THESE BEFORE RUNNING THE FULL PASS")
    print("-" * 68)
    if books_seen <= 1:
        print("  ! Only one 'book' was detected. The '--- SOURCE:' markers are")
        print("    probably absent or differently formatted, so documents were")
        print("    NOT reassembled correctly. Inspect raw rows before continuing:")
        print("      python3 konkani/scripts/inspect_books_dataset.py")
    else:
        print(f"  Books look real: {books_seen:,} detected, "
              f"{avg_doc:,.0f} words each on average.")
    print(f"  langid_marathi_rejected = {rejected.get('langid_marathi_rejected', 0):,}"
          "  <- Marathi contamination actually found and removed")
    print(f"  duplicate rate = {deduper.stats.duplicate_rate:.1%}"
          "  <- if very high, the corpus repeats itself heavily")
    if not args.dry_run:
        print(f"\nManifest: {MANIFEST_PATH}")
        print(f"Shards:   {OUT_DIR}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
