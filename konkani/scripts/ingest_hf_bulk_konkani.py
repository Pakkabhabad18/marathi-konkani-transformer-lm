#!/usr/bin/env python3
"""
Konkani K-11: bulk ingest of the remaining Hugging Face Konkani datasets.

WHY A SEPARATE, GENERIC SCRIPT
------------------------------
A HuggingFace search for "konkani" returns ~40 datasets. Most are small
instruction / Alpaca-translation sets, individually worth 0.1-10M words, and
collectively worth pursuing. Adding each to a hand-written config table would
mean looking up every schema by hand - and the dataset-server endpoint that
reports schemas has been returning cached responses for the WRONG dataset when
several are queried in a row, so a hand-copied column name is not trustworthy.

So this script does not need to be told the column name. It reads the first
`--probe` rows of each dataset and picks the column with the highest Devanagari
character ratio, exactly as the BPCC ingest picks the Konkani side of a TSV.
An English instruction column scores ~0.0 and can never win; a column of ids or
floats scores 0.0 as well. The chosen column is printed per dataset, so the
decision is visible and auditable rather than assumed.

CLASSIFICATION
--------------
Every dataset here is instruction data, an Alpaca translation, or otherwise
model-generated, so all are `MACHINE_TRANSLATED`. That keeps them out of the
manual figure and separable from genuinely downloaded human text in every
report. Where a dataset is genuinely human-written it should be moved to the
DOWNLOADED table in `ingest_hf_konkani.py` instead - do not relabel here.

EXPECT LOW YIELDS, AND EXPECT THAT TO BE RIGHT
----------------------------------------------
Instruction datasets embed English prompts, grammar tables and JSON
scaffolding. The 0.70 Devanagari floor rejects those, which is the behaviour we
want: a markdown table with English headers is not Konkani prose. The per
-dataset survival rate in the summary is the measurement of how much of each
was usable text.

USAGE
-----
    python3 konkani/scripts/ingest_hf_bulk_konkani.py --list
    python3 konkani/scripts/ingest_hf_bulk_konkani.py --pilot 2000
    python3 konkani/scripts/ingest_hf_bulk_konkani.py
    python3 konkani/scripts/ingest_hf_bulk_konkani.py --only anag007_alpaca
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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

DATA_DIR = REPO_ROOT / "konkani" / "data"
LANGUAGE = "kok"

MIN_WORDS = 25
MIN_DEVANAGARI_RATIO = 0.70
TARGET_DOC_WORDS = 300
SHARD_SIZE = 5000

# key -> (dataset id, short note). Ordered largest-first so a partial run
# still captures most of the available text.
DATASETS = {
    "anag007_alpaca":      ("anag007/asmitai_konkani_gemma-3-12b_noisified_alpaca_instruction_data",
                            "Gemma-3-12b noisified Alpaca, 230 MB"),
    "telugu_labs_alpaca":  ("Telugu-LLM-Labs/konkani_alpaca_yahma_cleaned_filtered",
                            "Alpaca yahma cleaned, 45 MB"),
    "anag007_instruct":    ("anag007/asmitai_konkani_gemma-3-12b_noisified_instruction_data",
                            "Gemma-3-12b noisified instructions"),
    "saillab_taco":        ("saillab/alpaca_konkani_taco", "Alpaca TACO"),
    "saillab_cleaned":     ("saillab/alpaca-konkani-cleaned", "Alpaca cleaned"),
    "gpteacher":           ("Tensoic/GPTeacher-Konkani", "GPTeacher"),
    "anag007_wiki":        ("anag007/asmitai_wiki_konkani_dataset", "wiki-derived"),
    "konkani_instructions": ("konkani/konkani_instructions", "instructions"),
    "goan_data":           ("konkani/Goan_Data", "Goan data"),
    "english_konkani":     ("konkani/english-konkani", "en-kok pairs"),
    "reubencf_1":          ("Reubencf/konkani-instruct-20k-1", "instruct 20k #1"),
    "reubencf_2":          ("Reubencf/konkani-instruct-20k-2", "instruct 20k #2"),
    "reubencf_3":          ("Reubencf/konkani-instruct-20k-3", "instruct 20k #3"),
    "reubencf_4":          ("Reubencf/konkani-instruct-20k-4", "instruct 20k #4"),
    "reubencf_5":          ("Reubencf/konkani-instruct-20k-5", "instruct 20k #5"),
    "reubencf_6":          ("Reubencf/konkani-instruct-20k-6", "instruct 20k #6"),
    "devarshee_v1":        ("devarsheegaunekar/Konkani-Instruct-v1", "instruct v1"),
    "devarshee_v2":        ("devarsheegaunekar/Konkani-Instruct-v2", "instruct v2"),
    "shrusti_translation": ("shrusti333/konkani_translation", "translation pairs"),
    "predictionguard":     ("predictionguard/english-hindi-marathi-konkani-corpus",
                            "en/hi/mr/kok corpus"),
}


def devanagari_ratio(text: str) -> float:
    non_ws = sum(1 for c in text if not c.isspace())
    if not non_ws:
        return 0.0
    return len(DEVANAGARI_RE.findall(text)) / non_ws


def detect_columns(rows: list[dict]) -> list[str]:
    """Rank columns by mean Devanagari ratio; keep those clearly Devanagari.

    Returns every column above 0.50, not just the best one: instruction
    datasets often carry Konkani in BOTH an instruction and a response field,
    and taking only the longest would silently discard half the text.
    """
    if not rows:
        return []
    scores: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and value.strip():
                scores.setdefault(key, []).append(devanagari_ratio(value))
    ranked = []
    for key, vals in scores.items():
        mean = sum(vals) / len(vals)
        if mean >= 0.50:
            ranked.append((mean, key))
    ranked.sort(reverse=True)
    return [k for _, k in ranked]


def iter_texts(dataset_id: str, probe: int, limit: int):
    """Yield Konkani text from a dataset, detecting its columns first."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train", streaming=True)
    head = []
    for i, row in enumerate(ds):
        head.append(row)
        if i + 1 >= probe:
            break
    cols = detect_columns(head)
    if not cols:
        print(f"    no Devanagari column found in {probe} probed rows - skipping")
        return
    print(f"    columns chosen: {', '.join(cols)}")

    seen = 0
    for row in head:
        for col in cols:
            value = row.get(col)
            if isinstance(value, str) and value.strip():
                yield value
        seen += 1
        if limit and seen >= limit:
            return
    for row in ds:
        for col in cols:
            value = row.get(col)
            if isinstance(value, str) and value.strip():
                yield value
        seen += 1
        if limit and seen >= limit:
            return


def run_one(key: str, dataset_id: str, note: str, args) -> tuple[int, int]:
    """Ingest one dataset. Returns (documents, words)."""
    source_name = f"hf_bulk_{key}"
    out_dir = DATA_DIR / "synthetic" / source_name
    manifest_path = DATA_DIR / "manifests" / f"{source_name}.jsonl"
    dry = args.pilot > 0

    print(f"\n  {dataset_id}")
    print(f"    note: {note}")

    if not dry:
        stale = sorted(out_dir.glob("shard_*.txt"))
        if stale or manifest_path.exists():
            for path in stale:
                path.unlink()
            manifest_path.unlink(missing_ok=True)
            print(f"    cleared {len(stale)} stale shard(s)")
        out_dir.mkdir(parents=True, exist_ok=True)

    deduper = Deduplicator(threshold=0.85)
    manifest = None if dry else ManifestWriter(manifest_path)
    seen_exact: set[str] = set()
    rejected: dict[str, int] = {}
    accepted = words_total = rows = 0
    shard_index, shard = 0, None
    if not dry:
        shard = open(out_dir / f"shard_{shard_index:05d}.txt", "a",
                     encoding="utf-8")

    buf: list[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words, accepted, words_total, shard, shard_index
        if not buf:
            return
        raw = "\n".join(buf)
        buf, buf_words = [], 0
        text = normalize(raw, keep_paragraphs=False)
        if len(text.split()) < MIN_WORDS:
            rejected["doc_too_short"] = rejected.get("doc_too_short", 0) + 1
            return
        profile = profile_script(text)
        if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
            rejected["doc_not_devanagari"] = rejected.get("doc_not_devanagari", 0) + 1
            return
        langid = identify_marathi_konkani(text)
        if langid.label == "mr":
            rejected["doc_langid_marathi"] = rejected.get("doc_langid_marathi", 0) + 1
            return
        if deduper.is_duplicate(text):
            rejected["doc_near_duplicate"] = rejected.get("doc_near_duplicate", 0) + 1
            return
        accepted += 1
        words_total += len(text.split())
        if manifest:
            manifest.write(make_record(
                text=text, raw_text=raw,
                source_name=source_name,
                source_url=f"https://huggingface.co/datasets/{dataset_id}",
                collection_type=CollectionType.MACHINE_TRANSLATED,
                language=LANGUAGE,
                preprocessing_applied=NORMALIZATION_STEPS + [
                    "auto_column_detection", "packing",
                    "devanagari_only_D001", "exact_dedup", "near_dedup"],
                script=profile.script,
                langid_score=langid.score,
                langid_label=langid.label,
                devanagari_ratio=profile.devanagari_ratio,
                doc_id=f"{source_name}_{accepted:08d}",
                notes=f"SYNTHETIC: {note}",
            ))
        if shard:
            shard.write(text.replace("\n", " ") + "\n")
            if accepted % SHARD_SIZE == 0:
                shard.close()
                shard_index += 1
                shard = open(out_dir / f"shard_{shard_index:05d}.txt", "a",
                             encoding="utf-8")

    try:
        for value in iter_texts(dataset_id, args.probe, args.pilot or args.limit):
            rows += 1
            h = exact_hash(value)
            if h in seen_exact:
                rejected["exact_duplicate"] = rejected.get("exact_duplicate", 0) + 1
                continue
            seen_exact.add(h)
            buf.append(value)
            buf_words += len(value.split())
            if buf_words >= TARGET_DOC_WORDS:
                flush()
        flush()
    except KeyboardInterrupt:
        raise
    except Exception as exc:                                    # noqa: BLE001
        print(f"    [failed] {exc.__class__.__name__}: {exc}")
    finally:
        if shard:
            shard.close()
        if manifest:
            manifest.close()

    top = ", ".join(f"{k}={v:,}" for k, v in
                    sorted(rejected.items(), key=lambda kv: -kv[1])[:3])
    print(f"    rows {rows:,} -> docs {accepted:,}, words {words_total:,}"
          + (f"   ({top})" if top else ""))
    return accepted, words_total


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bulk-ingest remaining HF Konkani datasets (synthetic).")
    ap.add_argument("--pilot", type=int, default=0,
                    help="rows per dataset; writes nothing")
    ap.add_argument("--limit", type=int, default=0, help="rows per dataset")
    ap.add_argument("--probe", type=int, default=50,
                    help="rows inspected to detect the Devanagari column(s)")
    ap.add_argument("--only", default="", help="comma-separated keys to run")
    ap.add_argument("--list", action="store_true", help="list keys and exit")
    args = ap.parse_args()

    if args.list:
        for key, (ds, note) in DATASETS.items():
            print(f"  {key:22s} {ds:70s} {note}")
        return 0

    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(DATASETS)
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown key(s): {', '.join(unknown)}")

    print("=" * 78)
    print("BULK INGEST (MACHINE_TRANSLATED / SYNTHETIC)")
    print(f"  datasets: {len(keys)}")
    print(f"  mode    : {'PILOT (writes nothing)' if args.pilot else 'WRITE'}")
    print("  Columns are DETECTED per dataset by Devanagari ratio, not assumed.")
    print("=" * 78)

    start = time.time()
    results = []
    for key in keys:
        dataset_id, note = DATASETS[key]
        try:
            docs, words = run_one(key, dataset_id, note, args)
        except KeyboardInterrupt:
            print("\n  Interrupted - stopping after the current dataset.")
            break
        results.append((key, docs, words))

    print("\n" + "=" * 78)
    print("BULK RUN SUMMARY")
    print("=" * 78)
    print(f"  {'dataset':24s}{'documents':>14s}{'words':>16s}")
    print("  " + "-" * 54)
    total_docs = total_words = 0
    for key, docs, words in sorted(results, key=lambda r: -r[2]):
        print(f"  {key:24s}{docs:>14,}{words:>16,}")
        total_docs += docs
        total_words += words
    print("  " + "-" * 54)
    print(f"  {'TOTAL':24s}{total_docs:>14,}{total_words:>16,}")
    print(f"\n  elapsed {(time.time() - start) / 60:.1f} min")
    print("\n  All classified MACHINE_TRANSLATED: never counted as manual, and")
    print("  reported separately from downloaded text in every stats table.")
    if args.pilot:
        print("\n  PILOT: nothing was written.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
