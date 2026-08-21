#!/usr/bin/env python3
"""
Konkani source K-10: BPCC `gom_Deva` (DOWNLOADED, human-translated - not MT).

WHAT THIS IS, AND WHY IT IS *NOT* SYNTHETIC
-------------------------------------------
`ai4bharat/BPCC` (Bharat Parallel Corpus Collection) is a parallel corpus. Its
Konkani side matters here for a reason worth being precise about:

    BPCC-Human   professional HUMAN translators produced the Konkani text.
                 It is human-written Konkani prose. Real data.
    BPCC-Mined   sentences MINED from existing web text and aligned. The
                 Konkani side is human-written text that already existed.
                 Also real data.

Neither is machine-translated, so this source is classified DOWNLOADED_DATASET,
not MACHINE_TRANSLATED. That distinction is load-bearing given the TAs' rule
that MT is penalised where real data existed: this IS the real data, and
skipping it while running MT would be exactly the criticised case.

WHERE THE KONKANI ACTUALLY IS (checked 19 Aug 2026)
---------------------------------------------------
The two largest BPCC subsets have NO Konkani at all:

    nllb_filtered/            16 language files, no gom
    samanantar_v0.3_filtered/ 11 language files, no gom

It lives in the human-curated subsets instead:

    bpcc-seed-latest/gom_Deva.tsv    32,756,783 bytes
    wiki/gom_Deva.tsv                 7,345,863 bytes
    ... and any other subset carrying a gom_Deva file

Rather than hard-coding that list - which would silently miss a subset, the
same failure that hid IndicCorp's gom.txt from us for weeks - this script asks
the Hub for every file in the repository whose path contains `gom_Deva` and
ingests all of them. What it found is printed before any work starts.

COLUMN DETECTION
----------------
These are TSV files pairing English with Konkani, and the column order is not
guaranteed to be identical across subsets. Rather than assume, each row's
fields are scored by Devanagari character ratio and the most-Devanagari field
is taken as the Konkani side. An English column scores ~0 and can never win;
a header row fails the length gate and is dropped.

USAGE
-----
    python3 konkani/scripts/ingest_bpcc_konkani.py --pilot 20000
    python3 konkani/scripts/ingest_bpcc_konkani.py
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
from common.scriptid import (                                         # noqa: E402
    DEVANAGARI_RE,
    identify_marathi_konkani,
    profile_script,
)
from common.textnorm import NORMALIZATION_STEPS, normalize            # noqa: E402

DATASET_NAME = "ai4bharat/BPCC"
SOURCE_NAME = "bpcc_gom_deva"
JOB_NAME = "konkani_bpcc"
LANGUAGE = "kok"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET_NAME}"

DATA_DIR = REPO_ROOT / "konkani" / "data"
OUT_DIR = DATA_DIR / "processed" / SOURCE_NAME
MANIFEST_PATH = DATA_DIR / "manifests" / f"{JOB_NAME}.jsonl"
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / f"{JOB_NAME}.json"

MIN_WORDS = 25
MIN_UNIT_WORDS = 3
MIN_DEVANAGARI_RATIO = 0.70
TARGET_DOC_WORDS = 300
SHARD_SIZE = 5000


def devanagari_ratio(text: str) -> float:
    non_ws = sum(1 for c in text if not c.isspace())
    if not non_ws:
        return 0.0
    return len(DEVANAGARI_RE.findall(text)) / non_ws


def konkani_field(line: str) -> str:
    """Pick the most-Devanagari tab-separated field.

    The column order is not guaranteed across BPCC subsets, so it is measured
    per row instead of assumed. An English field scores ~0.0 and cannot win.
    """
    fields = line.rstrip("\n").split("\t")
    best, best_ratio = "", -1.0
    for field in fields:
        ratio = devanagari_ratio(field)
        if ratio > best_ratio:
            best, best_ratio = field, ratio
    return best.strip() if best_ratio >= 0.50 else ""


def discover_files() -> list[str]:
    """Every path in the repo containing `gom_Deva`, asked of the Hub."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(DATASET_NAME, repo_type="dataset")
    return sorted(f for f in files
                  if "gom_Deva" in f and f.endswith((".tsv", ".txt", ".csv")))


def iter_packed(paths: list[Path], target: int, stats: dict,
                seen_exact: set, rejected: dict):
    """Sentence-level dedup and gating, then pack into documents.

    Same order as the IndicCorp ingest and for the same reason: BPCC repeats
    sentences across subsets, and once packed into 300-word documents no two
    documents are byte-identical, so repeats would survive inside unique-looking
    documents.
    """
    buf: list[str] = []
    count = 0
    for path in paths:
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"    [skip] {path.name}: {exc}")
            continue
        with handle:
            for line in handle:
                stats["rows_read"] = stats.get("rows_read", 0) + 1
                unit = konkani_field(line)
                if not unit:
                    rejected["no_devanagari_column"] = \
                        rejected.get("no_devanagari_column", 0) + 1
                    continue
                if len(unit.split()) < MIN_UNIT_WORDS:
                    rejected["unit_too_short"] = \
                        rejected.get("unit_too_short", 0) + 1
                    continue
                h = exact_hash(unit)
                if h in seen_exact:
                    rejected["unit_exact_duplicate"] = \
                        rejected.get("unit_exact_duplicate", 0) + 1
                    continue
                seen_exact.add(h)
                langid = identify_marathi_konkani(unit)
                if langid.label == "mr":
                    rejected["unit_langid_marathi"] = \
                        rejected.get("unit_langid_marathi", 0) + 1
                    continue

                stats["units_kept"] = stats.get("units_kept", 0) + 1
                buf.append(unit)
                count += len(unit.split())
                if count >= target:
                    stats["docs_packed"] = stats.get("docs_packed", 0) + 1
                    yield "\n".join(buf)
                    buf, count = [], 0
    if buf:
        stats["docs_packed"] = stats.get("docs_packed", 0) + 1
        yield "\n".join(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest BPCC Konkani (gom_Deva).")
    ap.add_argument("--pilot", type=int, default=0,
                    help="stop after N packed documents; writes nothing")
    ap.add_argument("--dedup-threshold", type=float, default=0.85)
    ap.add_argument("--doc-words", type=int, default=TARGET_DOC_WORDS)
    args = ap.parse_args()

    dry = args.pilot > 0

    print("=" * 74)
    print(f"INGESTING (DOWNLOADED, human-translated): {SOURCE_NAME}")
    print(f"  dataset : {DATASET_NAME}")
    print(f"  mode    : {'PILOT (writes nothing)' if dry else 'WRITE'}")
    print("  NOTE: BPCC's Konkani is HUMAN-translated / human-written mined")
    print("        text. It is real data, NOT machine translation.")
    print("=" * 74)

    from huggingface_hub import hf_hub_download

    print("\nAsking the Hub for every gom_Deva file in the repository...")
    remote = discover_files()
    if not remote:
        print("  none found - nothing to do.")
        return 1
    for name in remote:
        print(f"  found  {name}")

    local: list[Path] = []
    for name in remote:
        print(f"  downloading {name} ...", flush=True)
        try:
            local.append(Path(hf_hub_download(repo_id=DATASET_NAME,
                                              filename=name,
                                              repo_type="dataset")))
        except Exception as exc:                                # noqa: BLE001
            print(f"    [failed] {exc.__class__.__name__}: {exc}")
    if not local:
        print("  every download failed - nothing to do.")
        return 1

    total_bytes = sum(p.stat().st_size for p in local)
    print(f"\n  {len(local)} file(s), {total_bytes:,} bytes on disk")

    if dry:
        import tempfile
        checkpoint_path = Path(tempfile.mkdtemp(prefix="pilot_")) / "cp.json"
    else:
        stale = sorted(OUT_DIR.glob("shard_*.txt"))
        if stale or MANIFEST_PATH.exists():
            print(f"\n  Clearing previous output ({len(stale)} shard(s)); "
                  f"append mode would double-count on a re-run.")
            for path in stale:
                path.unlink()
            MANIFEST_PATH.unlink(missing_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_PATH

    checkpoint = Checkpoint(checkpoint_path, JOB_NAME)
    deduper = Deduplicator(threshold=args.dedup_threshold)
    manifest = None if dry else ManifestWriter(MANIFEST_PATH)

    seen_exact: set[str] = set()
    rejected: dict[str, int] = {}
    stats: dict[str, int] = {}
    accepted = words_total = docs_read = 0
    shard_index, shard = 0, None
    if not dry:
        shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a",
                     encoding="utf-8")
    start = time.time()

    try:
        for raw in iter_packed(local, args.doc_words, stats, seen_exact,
                               rejected):
            docs_read += 1
            if args.pilot and docs_read > args.pilot:
                break

            text = normalize(raw, keep_paragraphs=False)
            if len(text.split()) < MIN_WORDS:
                rejected["doc_too_short"] = rejected.get("doc_too_short", 0) + 1
                continue
            profile = profile_script(text)
            if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
                rejected["doc_not_devanagari"] = \
                    rejected.get("doc_not_devanagari", 0) + 1
                continue
            langid = identify_marathi_konkani(text)
            if langid.label == "mr":
                rejected["doc_langid_marathi"] = \
                    rejected.get("doc_langid_marathi", 0) + 1
                continue
            if deduper.is_duplicate(text):
                rejected["doc_near_duplicate"] = \
                    rejected.get("doc_near_duplicate", 0) + 1
                continue

            accepted += 1
            words_total += len(text.split())

            if manifest:
                manifest.write(make_record(
                    text=text, raw_text=raw,
                    source_name=SOURCE_NAME,
                    source_url=SOURCE_URL,
                    collection_type=CollectionType.DOWNLOADED_DATASET,
                    language=LANGUAGE,
                    preprocessing_applied=NORMALIZATION_STEPS + [
                        "tsv_devanagari_column_detection", "sentence_dedup",
                        "packing", "devanagari_only_D001", "near_dedup"],
                    script=profile.script,
                    langid_score=langid.score,
                    langid_label=langid.label,
                    devanagari_ratio=profile.devanagari_ratio,
                    doc_id=f"{SOURCE_NAME}_{accepted:08d}",
                    notes="BPCC gom_Deva: human-translated / human-written "
                          "mined Konkani. Real data, not MT.",
                ))
            if shard:
                shard.write(text.replace("\n", " ") + "\n")
                if accepted % SHARD_SIZE == 0:
                    shard.close()
                    shard_index += 1
                    shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a",
                                 encoding="utf-8")

    except KeyboardInterrupt:
        print("\nInterrupted; flushing what was collected.")
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
    print(f"Rows read:            {stats.get('rows_read', 0):,}")
    print(f"Documents packed:     {stats.get('docs_packed', 0):,}")
    print(f"Documents accepted:   {accepted:,}")
    print(f"Words accepted:       {words_total:,}")
    print(f"Elapsed:              {el:.1f} min")
    if stats.get("rows_read"):
        print(f"Sentence survival:    "
              f"{stats.get('units_kept', 0) / stats['rows_read']:.1%}")

    print("\nRejection reasons:")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<34}{count:>12,}")

    print("\n" + "-" * 74)
    print("EFFECT ON THE KONKANI CORPUS")
    print("-" * 74)
    print(f"  downloaded words added   {words_total:>14,}")
    print("  (real data - human-translated, NOT machine translation)")
    if dry:
        print("\nPILOT: nothing was written. Re-run without --pilot.")
    else:
        print(f"\nManifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
