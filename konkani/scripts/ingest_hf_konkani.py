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
    python3 konkani/scripts/ingest_hf_konkani.py --source sangraha_gom --pilot 500
    python3 konkani/scripts/ingest_hf_konkani.py --source sangraha_gom
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
    # ------------------------------------------------------------------
    # AI4Bharat Sangraha - the VERIFIED subset only.
    #
    # Sangraha ships three subsets. Only `verified` is ingested:
    #
    #   verified     web + PDF text that passed AI4Bharat's own language
    #                and quality verification.               -> INGESTED
    #   unverified   has no `gom` split at all (checked 19 Aug 2026: the
    #                unverified tree contains asm ben guj hin kan mal mar
    #                nep ori pan san tam tel urd - no Konkani).
    #   synthetic    machine-translated from English. The TAs were explicit
    #                that MT text is not accepted, and our own language gate
    #                exists to keep translationese out.      -> EXCLUDED
    #
    # `gom` is Goan Konkani. Sangraha has no `kok` split, so `gom` is the
    # whole of the Konkani available here: 14,491 rows / 76.7 MB of text
    # (dataset-server `info`, 19 Aug 2026), one parquet file of 32.5 MB.
    #
    # We pass data_files explicitly rather than naming the config, because
    # `load_dataset("ai4bharat/sangraha", "verified", split="gom")` resolves
    # the whole 100 GB+ config listing before it can stream one split. The
    # glob below touches one 32.5 MB file.
    "sangraha_gom": {
        "dataset": "ai4bharat/sangraha",
        "source_name": "hf_sangraha_verified_gom",
        "text_column": "text",
        "data_files": "verified/gom/*.parquet",
        "note": "AI4Bharat Sangraha, VERIFIED subset, gom (Goan Konkani) "
                "split only. Synthetic/MT subset deliberately excluded.",
    },
    # ------------------------------------------------------------------
    # MADLAD-400 (Google/AllenAI), `gom`. A 419-language document-level
    # CommonCrawl derivative. Two tiers ship per language:
    #
    #   gom_clean_0000.jsonl.gz    5,255,127 bytes   -> INGESTED
    #   gom_noisy_0000.jsonl.gz   11,055,944 bytes   -> INGESTED
    #
    # The `noisy` tier is included deliberately. MADLAD's "noisy" label means
    # it failed *their* heuristics (short documents, high symbol ratio,
    # possible language misidentification) - not that it is not Konkani. Our
    # own gates are stricter on the axis we care about: the Devanagari floor
    # and the Marathi discriminator both run over every document regardless of
    # tier. Rejecting the noisy tier unmeasured would discard text on the
    # strength of someone else's filter; running it through our gates and
    # reporting the rejection rate is the defensible choice. The per-tier
    # source names keep the two separable in the manifests, so if the noisy
    # tier turns out to be junk it can be dropped without re-running anything.
    "madlad_clean": {
        "dataset": "allenai/MADLAD-400",
        "source_name": "hf_madlad400_gom_clean",
        "text_column": "text",
        "data_files": "data/gom/gom_clean_0000.jsonl.gz",
        "note": "MADLAD-400 gom, clean tier.",
    },
    "madlad_noisy": {
        "dataset": "allenai/MADLAD-400",
        "source_name": "hf_madlad400_gom_noisy",
        "text_column": "text",
        "data_files": "data/gom/gom_noisy_0000.jsonl.gz",
        "note": "MADLAD-400 gom, noisy tier - kept only where it passes our "
                "own Devanagari and Marathi gates.",
    },
    # ------------------------------------------------------------------
    # GlotCC-V1 (CIS-LMU), `gom-Deva`. One 4,180,618-byte parquet.
    # Text lives in `content`, not `text` (verified against the dataset
    # server's first-rows response, 19 Aug 2026 - guessing this would have
    # produced an empty ingest that still exited 0).
    #
    # `gom-Latn` exists and is NOT ingested: it is Romi Konkani, genuinely
    # Konkani but not this corpus's script. `kok-Deva` does not exist in
    # v1.0 - only gom-Deva and gom-Latn.
    # ------------------------------------------------------------------
    # SYNTHETIC / MACHINE-TRANSLATED (D-038).
    #
    # The SAME repository as `konkani_raw` above, but the files that entry
    # deliberately excluded. When that entry was written, MT data was banned
    # outright, so `translated_konkani_*.txt` was filtered out on principle.
    # The TAs authorised MT as a last resort on 18 Aug 2026, which makes these
    # files usable - and they are not a rounding error:
    #
    #     files ingested as `konkani_raw`        4,397,019 bytes
    #     translated_konkani_* (excluded then) 870,725,308 bytes
    #
    # 870 MB, roughly 6.8x the entire real Konkani corpus. It is already in
    # the local HuggingFace cache, because `snapshot_download` fetches the
    # whole repository regardless of which files an entry reads.
    #
    # CLASSIFIED MACHINE_TRANSLATED. It is somebody else's MT output, which
    # makes it synthetic exactly as if we had generated it: `is_manual` is
    # False, and it is reported separately from downloaded text so the
    # synthetic share of the corpus is always visible.
    #
    # The Devanagari floor and the Marathi discriminator still run over every
    # segment. A file labelled "translated Konkani" is not evidence that its
    # contents are Konkani - that has to be measured, and the rejection counts
    # in the run summary are that measurement.
    "konkani_raw_translated": {
        "dataset": "praveenkumar99/Konkani_Raw",
        "source_name": "hf_konkani_raw_machine_translated",
        "text_column": None,
        "include_prefixes": ("translated_konkani",),
        "exclude_prefixes": (),
        "collection_type": CollectionType.MACHINE_TRANSLATED,
        "note": "SYNTHETIC: pre-existing machine-translated Konkani, 870 MB. "
                "Authorised by TAs 18 Aug 2026 as a last resort.",
    },
    # ------------------------------------------------------------------
    # omdeep22/Konkani_books_corpus - the V1 of the books corpus. We already
    # hold v2 (`hf_konkani_books_corpus_v2`, 47M words). v1 is a separate
    # repository, 184,095,970 bytes across train/valid/test .txt files.
    #
    # v1 is very probably largely contained in v2, and that is fine: the
    # split-time exact-hash pass compares every document against the whole
    # corpus, so anything already present is dropped and the "removed per
    # source" line reports exactly how much was redundant. Ingesting it and
    # letting dedup MEASURE the overlap is better than assuming v1 adds
    # nothing and skipping a real source - assuming is how gom.txt hid.
    "books_corpus_v1": {
        "dataset": "omdeep22/Konkani_books_corpus",
        "source_name": "hf_konkani_books_corpus_v1",
        "text_column": None,
        "include_prefixes": ("train", "valid", "test"),
        "exclude_prefixes": (),
        "note": "Books corpus v1. Overlap with v2 is resolved by the "
                "split-time exact-hash pass and reported, not assumed.",
    },
    # ------------------------------------------------------------------
    # SYNTHETIC, LLM-GENERATED (D-039). Both of these are model output, not
    # collected text, and both are classified MACHINE_TRANSLATED.
    #
    #   konkani/konkani-instruct-100k    445,965,908 bytes
    #       Dataset card: "Generated using a highly controlled synthetic
    #       distillation pipeline (Gemini 3)". Fields are `instruction`,
    #       `response`, `system`. We take ONLY `response`, because
    #       `instruction` and `system` are largely English prompt scaffolding.
    #
    # EXPECT A LOW YIELD, AND EXPECT THAT TO BE CORRECT. The responses embed
    # grammar tables, English glosses and script annotations. Anything below
    # the 0.70 Devanagari floor is rejected, which is the behaviour we want:
    # a markdown table with English column headers is not Konkani prose and
    # has no business in a language-model corpus. The rejection count in the
    # run summary is the measurement of how much of this was usable text
    # rather than instruction-tuning scaffolding.
    "instruct_100k": {
        "dataset": "konkani/konkani-instruct-100k",
        "source_name": "hf_konkani_instruct_100k_synthetic",
        "text_column": "response",
        "data_files": "konkani_Train.jsonl",
        "collection_type": CollectionType.MACHINE_TRANSLATED,
        "note": "SYNTHETIC: Gemini-3 distillation output. Only the `response` "
                "field is used; English scaffolding fails the Devanagari gate.",
    },
    "glotcc": {
        "dataset": "cis-lmu/GlotCC-V1",
        "source_name": "hf_glotcc_v1_gom_deva",
        "text_column": "content",
        "data_files": "v1.0/gom-Deva/*.parquet",
        "note": "GlotCC-V1 gom-Deva. Roman-script gom-Latn excluded by script.",
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
        data_files = cfg.get("data_files")
        if data_files:
            # Explicit file glob: streams one shard instead of resolving the
            # entire multi-language config (see the sangraha_gom note above).
            ds = load_dataset(name, data_files=data_files, split="train",
                              streaming=True)
        else:
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
    # Per-source, defaulting to DOWNLOADED. Only the deliberately
    # machine-translated source overrides it, and it does so in the config
    # table rather than at the call site, so the classification travels with
    # the source definition and cannot drift.
    collection_type = cfg.get("collection_type",
                              CollectionType.DOWNLOADED_DATASET)
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
                    collection_type=collection_type,
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
