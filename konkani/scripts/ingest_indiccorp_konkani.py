#!/usr/bin/env python3
"""
Konkani source K-8: IndicCorp v2, `gom` (DOWNLOADED, not manual).

WHY THIS SOURCE WAS MISSED UNTIL 19 AUG 2026
--------------------------------------------
We already use `ai4bharat/IndicCorpV2` for Marathi
(`marathi/scripts/ingest_indiccorp.py`), so the repository was not an unknown
one. What was missed is that it contains Konkani at all.

The reason is worth recording, because it is a general lesson about how these
repositories are searched. The Hugging Face *root* tree of `IndicCorpV2` lists
only three entries - `.gitattributes`, `README.md`, and a `data` directory. The
dataset card's language tags do not enumerate `gom`. Every earlier search of
this repository stopped at the root listing and concluded "Marathi yes, Konkani
no". Listing one level deeper, into `data/`, shows:

    data/gom.txt      533,108,246 bytes

That is roughly seven times the size of every other Konkani source in this
project combined. A negative result from a search is only as trustworthy as the
depth the search actually reached, and ours had not reached far enough.

WHY A SEPARATE SCRIPT RATHER THAN `--language konkani` ON THE MARATHI ONE
--------------------------------------------------------------------------
The language gate runs in the opposite direction. The Marathi ingest REJECTS
documents the discriminator labels `kok`; this one rejects documents labelled
`mr`. Parameterising a single script by a flag that inverts a correctness check
is how the two corpora would eventually end up sharing documents, which the
specification forbids. Two scripts, two explicit directions.

WHY `hf_hub_download` AND NOT `load_dataset`
---------------------------------------------
`load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="gom_Deva")` has to
resolve the builder config across all 26 language files - hundreds of gigabytes
of listing - before it can stream the one we want. `hf_hub_download` fetches
exactly `data/gom.txt`, resumes if interrupted, and caches, so a re-run after a
failure does not re-download 533 MB.

DOCUMENT ASSEMBLY - AND A CORRECTION (see D-034)
------------------------------------------------
The first version of this script treated a blank line as a DOCUMENT boundary.
That was wrong, and the run proved it:

    documents assembled   1,361,209        533,108,246 B / 1,361,209 = 392 B
    flush_blank_line      1,361,209        i.e. 100% of flushes
    rejected too_short      821,054        60% of everything assembled
    words accepted        8,112,757        of roughly 30M in the file

392 bytes is about 22 Devanagari words. No real document is 22 words long. The
blank lines separate SENTENCES, so the script was chopping the corpus into
fragments and then discarding 60% of them for failing a 25-word document floor
that its own chopping had made unreachable.

The corrected order is unit-level filtering first, then packing:

    1. UNIT   exact-hash dedup + Marathi discriminator. Dedup belongs here:
              IndicCorp repeats individual sentences across crawled pages, and
              once packed into 300-word documents no two documents are
              byte-identical, so those repeats would survive invisibly inside
              unique-looking documents.
    2. PACK   accumulate surviving units to TARGET_DOC_WORDS. IndicCorp
              preserves crawl order, so consecutive units generally come from
              the same source page; packing reconstructs page-level context
              rather than inventing it.
    3. DOC    length, Devanagari ratio, language and near-duplicate gates - all
              markedly more reliable on 300 words than on 22.

`--inspect N` measures the file's real line/unit structure and exits, so this
assumption is never made silently again.

CLASSIFICATION
--------------
`DOWNLOADED_DATASET`, without exception. Per TA guidance, a prepared corpus
pulled from Hugging Face is not manual no matter how much cleaning we apply.

USAGE
-----
    python3 konkani/scripts/ingest_indiccorp_konkani.py --pilot 200000
    python3 konkani/scripts/ingest_indiccorp_konkani.py
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
    summarize,
)
from common.scriptid import identify_marathi_konkani, profile_script  # noqa: E402
from common.textnorm import NORMALIZATION_STEPS, normalize            # noqa: E402

DATASET_NAME = "ai4bharat/IndicCorpV2"
REMOTE_FILE = "data/gom.txt"
REMOTE_BYTES = 533_108_246          # verified from the HF tree API, 19 Aug 2026

SOURCE_NAME = "ai4bharat_indiccorp_v2_gom"
JOB_NAME = "konkani_indiccorp"
LANGUAGE = "kok"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET_NAME}"
LICENSE_NOTE = "CC-0 (public domain)"

DATA_DIR = REPO_ROOT / "konkani" / "data"
OUT_DIR = DATA_DIR / "processed" / SOURCE_NAME
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / f"{JOB_NAME}.json"
MANIFEST_PATH = DATA_DIR / "manifests" / f"{JOB_NAME}.jsonl"

MIN_WORDS = 25                      # same floor as every other Konkani source
MIN_UNIT_WORDS = 3                  # unit floor: drops navigation/menu scraps
MIN_DEVANAGARI_RATIO = 0.70         # D-001
TARGET_DOC_WORDS = 300              # units are packed up to this before gating
SHARD_SIZE = 20000
MANUAL_RATIO = 0.20
MIN_FREE_GB = 6.0


def current_words(language_dir: Path) -> tuple[int, int]:
    """Manual and downloaded words already recorded for this language."""
    manual = downloaded = 0
    for path in sorted((language_dir / "manifests").glob("*.jsonl")):
        acc = summarize(path)
        manual += acc.manual_words
        downloaded += acc.downloaded_words
    return manual, downloaded


def evaluate(raw_text: str):
    """The identical gate stack every Konkani source passes through."""
    text = normalize(raw_text, keep_paragraphs=False)
    if len(text.split()) < MIN_WORDS:
        return text, None, None, "too_short"
    profile = profile_script(text)
    if profile.devanagari_ratio < MIN_DEVANAGARI_RATIO:
        return text, profile, None, "not_devanagari_excluded_by_D001"
    langid = identify_marathi_konkani(text)
    if langid.label == "mr":
        return text, profile, langid, "langid_marathi_rejected"
    return text, profile, langid, None


def iter_units(path: Path):
    """Yield the file's natural text units (blank-line separated)."""
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if buf:
                    yield " ".join(buf)
                    buf = []
                continue
            buf.append(stripped)
    if buf:
        yield " ".join(buf)


def inspect(path: Path, sample_lines: int) -> None:
    """Measure the file's actual structure instead of assuming it.

    The first version of this script assumed a blank line was a *document*
    boundary. It is not - it separates sentences. That assumption silently
    discarded 821,054 units as `too_short`, because a 22-word sentence cannot
    clear a 25-word document floor. This mode exists so the structure is
    measured and printed before any collection decision depends on it.
    """
    lines = blanks = words = nonblank = 0
    unit_words: list[int] = []
    current = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines += 1
            stripped = line.strip()
            if not stripped:
                blanks += 1
                if current:
                    unit_words.append(current)
                    current = 0
            else:
                nonblank += 1
                n = len(stripped.split())
                words += n
                current += n
            if lines >= sample_lines:
                break
    if current:
        unit_words.append(current)

    unit_words.sort()
    print("\n" + "=" * 74)
    print(f"STRUCTURE INSPECTION - first {lines:,} lines")
    print("=" * 74)
    print(f"  blank lines                {blanks:>12,}  ({blanks / max(lines,1):.1%})")
    print(f"  non-blank lines            {nonblank:>12,}")
    print(f"  words                      {words:>12,}")
    print(f"  words per non-blank line   {words / max(nonblank,1):>12.1f}")
    print(f"  units (blank-separated)    {len(unit_words):>12,}")
    if unit_words:
        med = unit_words[len(unit_words) // 2]
        p90 = unit_words[int(len(unit_words) * 0.9)]
        print(f"  words per unit  median     {med:>12,}")
        print(f"  words per unit  p90        {p90:>12,}")
        print(f"  units below the {MIN_WORDS}-word document floor: "
              f"{sum(1 for u in unit_words if u < MIN_WORDS) / len(unit_words):.1%}")
    print("\n  If most units sit below the floor, blank lines are SENTENCE")
    print("  separators and units must be packed into documents before the")
    print("  document gates run. That is what this script now does.")
    print("=" * 74)


def iter_packed_documents(path: Path, target_words: int, stats: dict,
                          seen_exact: set, rejected: dict):
    """Dedup and language-filter at the UNIT level, then pack into documents.

    Order matters, and it is the whole point of this function:

    1. UNIT level - exact-hash dedup and the Marathi discriminator. Duplicate
       detection belongs here: IndicCorp repeats individual sentences across
       crawled pages, and once sentences are packed into 300-word documents no
       two documents are byte-identical, so the repeats would survive
       invisibly *inside* otherwise-unique documents. Filtering after packing
       would silently readmit every duplicate this catches.

    2. PACK - accumulate surviving units to `target_words`. Blank lines are
       unit separators, not document boundaries; IndicCorp preserves crawl
       order, so consecutive units generally come from the same source page
       and packing reconstructs page-level context rather than inventing it.

    3. DOCUMENT level (in the caller) - length, Devanagari ratio, language and
       near-duplicate gates, all of which are far more reliable on a 300-word
       document than on a 22-word fragment.
    """
    buf: list[str] = []
    count = 0
    for unit in iter_units(path):
        stats["units_read"] = stats.get("units_read", 0) + 1

        h = exact_hash(unit)
        if h in seen_exact:
            rejected["unit_exact_duplicate"] = \
                rejected.get("unit_exact_duplicate", 0) + 1
            continue
        seen_exact.add(h)

        if len(unit.split()) < MIN_UNIT_WORDS:
            rejected["unit_too_short"] = rejected.get("unit_too_short", 0) + 1
            continue

        # Marathi rejection at unit level. The discriminator keys on
        # closed-class function words (आणि/आनी, आहे/आसा, मी/हांव) which are
        # frequent enough to fire on a single sentence. Doing this before
        # packing stops a Marathi sentence being laundered into a document
        # that then passes as Konkani overall.
        langid = identify_marathi_konkani(unit)
        if langid.label == "mr":
            rejected["unit_langid_marathi"] = \
                rejected.get("unit_langid_marathi", 0) + 1
            continue

        stats["units_kept"] = stats.get("units_kept", 0) + 1
        words = len(unit.split())
        buf.append(unit)
        count += words
        if count >= target_words:
            stats["docs_packed"] = stats.get("docs_packed", 0) + 1
            yield "\n".join(buf)
            buf, count = [], 0

    if buf:
        stats["docs_packed"] = stats.get("docs_packed", 0) + 1
        yield "\n".join(buf)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest IndicCorp v2 Konkani (gom).")
    parser.add_argument("--pilot", type=int, default=0,
                        help="stop after N assembled documents; writes nothing")
    parser.add_argument("--max-words", type=int, default=0,
                        help="override the automatic manual-ratio cap")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--doc-words", type=int, default=TARGET_DOC_WORDS,
                        help="target words per packed document")
    parser.add_argument("--inspect", type=int, default=0, metavar="N",
                        help="measure the file's line/unit structure over the "
                             "first N lines, print it, and exit")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 1

    dry = args.pilot > 0
    manual_words, downloaded_words = current_words(DATA_DIR)

    if args.max_words:
        cap = args.max_words
        cap_reason = "set explicitly with --max-words"
    else:
        allowed_total = manual_words / MANUAL_RATIO if manual_words else 0
        cap = max(int(allowed_total - manual_words - downloaded_words), 0)
        cap_reason = (f"manual/total >= {MANUAL_RATIO:.0%} on "
                      f"{manual_words:,} manual words")

    print("=" * 74)
    print(f"INGESTING (DOWNLOADED): {SOURCE_NAME}")
    print(f"  dataset : {DATASET_NAME}  ->  {REMOTE_FILE}")
    print(f"  size    : {REMOTE_BYTES:,} bytes")
    print(f"  license : {LICENSE_NOTE}")
    print(f"  mode    : {'PILOT (writes nothing)' if dry else 'WRITE'}")
    print("=" * 74)
    print("\nMANUAL-RATIO BUDGET")
    print("-" * 74)
    print(f"  manual words already collected   {manual_words:>15,}")
    print(f"  downloaded words already held    {downloaded_words:>15,}")
    print(f"  total corpus permitted           {manual_words * 5:>15,}")
    print(f"  downloaded words still allowed   {cap:>15,}   <- hard stop")
    print(f"  cap basis: {cap_reason}")

    if cap <= 0 and not dry:
        print("\n  STOP: no downloaded budget remains at the current manual total.")
        return 1

    import shutil
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
    print(f"\n  disk free                        {free_gb:>15,.1f} GB")
    print(f"  (needs ~{REMOTE_BYTES / 1e9:.1f} GB for the download plus "
          f"about the same again for shards)")
    if free_gb < MIN_FREE_GB and not dry:
        print(f"\n  STOP: only {free_gb:.1f} GB free, below the "
              f"{MIN_FREE_GB:.0f} GB minimum.")
        return 1
    print("=" * 74)

    print(f"\nDownloading {REMOTE_FILE} (resumes if interrupted, then cached)...")
    local_path = Path(hf_hub_download(repo_id=DATASET_NAME,
                                      filename=REMOTE_FILE,
                                      repo_type="dataset"))
    actual = local_path.stat().st_size
    print(f"  local file : {local_path}")
    print(f"  bytes      : {actual:,}")
    if actual != REMOTE_BYTES:
        print(f"  NOTE: size differs from the {REMOTE_BYTES:,} bytes recorded "
              f"when this script was written; the dataset may have been "
              f"revised. Proceeding, but the figure above is the one to quote.")

    if args.inspect:
        inspect(local_path, args.inspect)
        return 0

    if dry:
        import tempfile
        checkpoint_path = Path(tempfile.mkdtemp(prefix="pilot_")) / "cp.json"
    else:
        # Shards are opened in append mode, and the manifest is appended to as
        # well. A re-run over an existing output directory would therefore
        # DOUBLE-COUNT every document rather than replace it, and the resulting
        # word totals would be wrong in a way nothing downstream could detect.
        # Clear prior output explicitly and say so.
        stale = sorted(OUT_DIR.glob("shard_*.txt"))
        if stale or MANIFEST_PATH.exists():
            print(f"\n  Clearing previous output for this source "
                  f"({len(stale)} shard(s), manifest "
                  f"{'present' if MANIFEST_PATH.exists() else 'absent'}).")
            print("  Append mode would otherwise double-count on a re-run.")
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
    structure: dict[str, int] = {}
    accepted = words_total = docs_read = 0
    shard_index, shard = 0, None
    if not dry:
        shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a",
                     encoding="utf-8")
    start = time.time()

    try:
        for raw in iter_packed_documents(local_path, args.doc_words,
                                         structure, seen_exact, rejected):
            docs_read += 1
            if args.pilot and docs_read > args.pilot:
                break
            if cap and words_total >= cap:
                print(f"\n  CAP REACHED at {words_total:,} words - stopping "
                      f"cleanly. This is the ratio guard, not a failure.")
                break

            # No exact-hash check here: it already ran at the unit level, where
            # IndicCorp's repetition actually lives. See iter_packed_documents.
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
                    source_name=SOURCE_NAME,
                    source_url=SOURCE_URL,
                    collection_type=CollectionType.DOWNLOADED_DATASET,
                    language=LANGUAGE,
                    preprocessing_applied=NORMALIZATION_STEPS + [
                        "line_grouping", "devanagari_only_D001",
                        "marathi_rejected", "exact_dedup", "near_dedup"],
                    script=profile.script,
                    langid_score=langid.score,
                    langid_label=langid.label,
                    devanagari_ratio=profile.devanagari_ratio,
                    doc_id=f"{SOURCE_NAME}_{accepted:08d}",
                    notes=f"IndicCorp v2 gom.txt, {LICENSE_NOTE}",
                ))
            if shard:
                shard.write(text.replace("\n", " ") + "\n")
                if accepted % SHARD_SIZE == 0:
                    shard.close()
                    shard_index += 1
                    shard = open(OUT_DIR / f"shard_{shard_index:05d}.txt", "a",
                                 encoding="utf-8")

            if docs_read % 200_000 == 0:
                el = time.time() - start
                print(f"  docs={docs_read:,} accepted={accepted:,} "
                      f"words={words_total:,} "
                      f"({docs_read / max(el, 1e-9):,.0f} docs/s)", flush=True)

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
    print(f"Documents assembled:  {docs_read:,}")
    print(f"Documents accepted:   {accepted:,}")
    print(f"Words accepted:       {words_total:,}")
    print(f"Elapsed:              {el:.1f} min")
    if accepted:
        print(f"Words per document:   {words_total / accepted:,.0f}  "
              f"(target {args.doc_words})")

    print("\nUnit -> document assembly:")
    for kind, count in sorted(structure.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<34}{count:>12,}")
    if structure.get("units_read"):
        keep = structure.get("units_kept", 0) / structure["units_read"]
        print(f"  {'unit survival rate':<34}{keep:>11.1%}")

    print("\nRejection reasons (unit_* fire before packing, the rest after):")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<34}{count:>12,}")

    print("\n" + "-" * 74)
    print("EFFECT ON THE KONKANI CORPUS")
    print("-" * 74)
    new_downloaded = downloaded_words + words_total
    new_total = manual_words + new_downloaded
    print(f"  manual words       {manual_words:>15,}  (unchanged)")
    print(f"  downloaded words   {new_downloaded:>15,}  "
          f"(+{words_total:,})")
    print(f"  total words        {new_total:>15,}")
    if new_total:
        print(f"  manual share       {manual_words / new_total:>14.1%}  "
              f"(requirement: >= 20%)")
    if dry:
        print("\nPILOT: nothing was written. Re-run without --pilot to collect.")
    else:
        print(f"\nManifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
