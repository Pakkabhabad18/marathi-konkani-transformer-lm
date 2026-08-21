#!/usr/bin/env python3
"""
Full token-level accounting: raw collected -> cleaned -> deduplicated -> final.

WHY THIS IS SEPARATE FROM corpus_stats.py
-----------------------------------------
`corpus_stats.py` describes the corpus as it stands. This traces how it got
there, stage by stage, and attributes every word lost to a named cause. Those
are different questions, and the second is the one that has to be defended:

    "You collected X. You report Y. Where did X-Y go, and why?"

Every stage below is measured from artifacts on disk - collector checkpoints,
per-document manifests, the split files - never estimated.

THE STAGES
----------
    1. RAW COLLECTED    what the collectors fetched, before any filter.
                        From each checkpoint's `collected` + `skipped`.
    2. ACCEPTED         survived the per-source quality gates: minimum length,
                        Devanagari ratio, language identification.
    3. DEDUPLICATED     survived cross-source exact + near-duplicate removal.
    4. RATIO-CAPPED     downloaded data discarded to hold `total <= 5 x manual`.
    5. FINAL SPLITS     what actually reaches train / validation / test.
    6. TOKENS           counted once, with that language's ONE final tokenizer.

WORDS vs TOKENS - the distinction that has to stay straight
-----------------------------------------------------------
Stages 1-5 are counted in WORDS, because words are tokenizer-independent and can
be tracked while collection is still running. Tokens appear only at stage 6,
after the corpus is frozen and one tokenizer exists.

Mixing the two units is a real error, not a pedantic one: a manual/total ratio
computed as (manual WORDS)/(total TOKENS) is meaningless and will read far below
the true value, because fertility multiplies the denominator by ~1.6-1.8 and not
the numerator. The 20% requirement holds in either unit as long as both sides
use the SAME unit.

USAGE
-----
    python3 tools/pipeline_accounting.py
    python3 tools/pipeline_accounting.py --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.manifest import (atomic_write_json, read_manifest,       # noqa: E402
                             CollectionType)

LANGUAGES = ("marathi", "konkani")
MANUAL_FLOOR = 0.20
TOKEN_TARGET = 500_000_000


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def stage1_raw(language: str) -> dict:
    """What the collectors fetched, from the checkpoints."""
    total = {"documents_fetched": 0, "documents_rejected": 0, "by_job": {}}
    cp_dir = REPO_ROOT / language / "data" / "checkpoints"
    if not cp_dir.exists():
        return total

    for path in sorted(cp_dir.glob("*.json")):
        state = load_json(path)
        if not state:
            continue
        collected = int(state.get("collected") or 0)
        skipped = int(state.get("skipped") or 0)
        reasons = (state.get("extra", {}) or {}).get("rejection_reasons", {}) or {}
        total["documents_fetched"] += collected + skipped
        total["documents_rejected"] += skipped
        total["by_job"][path.stem] = {
            "accepted": collected, "rejected": skipped,
            "rejection_reasons": reasons,
        }
    return total


def stage2_accepted(language: str) -> dict:
    """What survived the per-source gates, from the manifests."""
    # Three buckets. `synthetic_words` is tracked separately from
    # `downloaded_words` so the accounting table can never imply that
    # machine-generated text is downloaded human text (D-036).
    out = {"documents": 0, "words": 0, "raw_chars": 0, "clean_chars": 0,
           "manual_words": 0, "downloaded_words": 0, "synthetic_words": 0,
           "by_source": {}}

    mf_dir = REPO_ROOT / language / "data" / "manifests"
    if not mf_dir.exists():
        return out

    for path in sorted(mf_dir.glob("*.jsonl")):
        for row in read_manifest(path):
            words = int(row.get("words") or 0)
            is_manual = bool(row.get("is_manual"))
            is_synthetic = (row.get("collection_type")
                            == CollectionType.MACHINE_TRANSLATED.value)
            source = row.get("source_name", "unknown")

            out["documents"] += 1
            out["words"] += words
            out["raw_chars"] += int(row.get("raw_chars") or 0)
            out["clean_chars"] += int(row.get("clean_chars") or 0)
            # Synthetic first: a machine-translated row is never manual, and
            # falling through to the else branch would hide it as downloaded.
            if is_synthetic:
                out["synthetic_words"] += words
            elif is_manual:
                out["manual_words"] += words
            else:
                out["downloaded_words"] += words

            entry = out["by_source"].setdefault(
                source, {"documents": 0, "words": 0, "is_manual": is_manual,
                         "is_synthetic": is_synthetic})
            entry["documents"] += 1
            entry["words"] += words
    return out


def stage345_splits(language: str) -> dict:
    """Dedup, ratio-cap and final split counts, from the splits report."""
    report = load_json(REPO_ROOT / "report" / f"phase1_splits_{language}.json")
    if not report:
        return {}

    splits = report.get("splits", {})
    out = {
        "cross_source_dedup": report.get("cross_source_dedup"),
        "ratio_cap_applied": report.get("ratio_cap_applied"),
        "ratio_cap_dropped": report.get("ratio_cap_dropped", {}),
        "leaked_documents": report.get("leaked_documents"),
        "train_manual_share": report.get("train_manual_share"),
        "splits": {},
    }
    for name in ("train", "val", "test"):
        s = splits.get(name)
        if s:
            out["splits"][name] = {
                "documents": s["docs"], "words": s["words"],
                "manual_words": s["manual_words"],
                "downloaded_words": s["downloaded_words"],
            }
    return out


def stage6_tokens(language: str) -> dict:
    """Token counts from the ONE final tokenizer, per split."""
    stats = load_json(REPO_ROOT / "report" / f"phase1_corpus_stats_{language}.json")
    if not stats or not stats.get("tokens"):
        return {}
    return stats["tokens"]


def report_language(language: str) -> dict:
    raw = stage1_raw(language)
    accepted = stage2_accepted(language)
    splits = stage345_splits(language)
    tokens = stage6_tokens(language)

    print("\n" + "=" * 76)
    print(f"PIPELINE ACCOUNTING - {language.upper()}")
    print("=" * 76)

    # ---- stage 1 ----
    print("\n[1] RAW COLLECTED  (documents fetched before any quality filter)")
    print("-" * 76)
    print(f"  documents fetched     {raw['documents_fetched']:>14,}")
    print(f"  documents rejected    {raw['documents_rejected']:>14,}")
    if raw["documents_fetched"]:
        rate = raw["documents_rejected"] / raw["documents_fetched"]
        print(f"  rejection rate        {rate:>14.1%}")

    reasons: dict = defaultdict(int)
    for job in raw["by_job"].values():
        for reason, count in (job.get("rejection_reasons") or {}).items():
            reasons[reason] += count
    if reasons:
        print("\n  why documents were rejected:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<34} {count:>12,}")

    # ---- stage 2 ----
    print("\n[2] ACCEPTED  (passed length, script and language gates)")
    print("-" * 76)
    print(f"  documents             {accepted['documents']:>14,}")
    print(f"  words                 {accepted['words']:>14,}")
    print(f"    manual              {accepted['manual_words']:>14,}")
    print(f"    downloaded (real)   {accepted['downloaded_words']:>14,}")
    print(f"    synthetic (MT/LLM)  {accepted['synthetic_words']:>14,}")
    print(f"    real (manual + dl)  "
          f"{accepted['manual_words'] + accepted['downloaded_words']:>14,}")
    if accepted["raw_chars"]:
        removed = accepted["raw_chars"] - accepted["clean_chars"]
        print(f"\n  characters before cleaning  {accepted['raw_chars']:>14,}")
        print(f"  characters after cleaning   {accepted['clean_chars']:>14,}")
        print(f"  removed by cleaning         {removed:>14,}  "
              f"({removed / accepted['raw_chars']:.1%})")
        print("    NOTE: for scraped pages `raw_chars` is the whole HTML document,")
        print("    so a large figure here is markup stripped, not text lost.")

    # ---- stages 3-5 ----
    if splits:
        print("\n[3-5] DEDUPLICATED, RATIO-CAPPED, SPLIT")
        print("-" * 76)
        print(f"  cross-source dedup run    {splits.get('cross_source_dedup')}")
        print(f"  ratio cap applied         {splits.get('ratio_cap_applied')}")
        dropped = splits.get("ratio_cap_dropped") or {}
        if dropped:
            print("\n  discarded to hold the 20% manual floor:")
            for source, info in dropped.items():
                print(f"    {source:<34} {info['words_dropped']:>12,} words  "
                      f"({info['documents_kept']:,} of "
                      f"{info['documents_before']:,} docs kept)")
        print(f"\n  documents leaked between splits  {splits.get('leaked_documents')}"
              "   (must be 0)")

        print(f"\n  {'split':<8}{'documents':>12}{'words':>16}"
              f"{'manual':>16}{'manual %':>11}")
        for name, s in splits["splits"].items():
            share = s["manual_words"] / s["words"] if s["words"] else 0
            print(f"  {name:<8}{s['documents']:>12,}{s['words']:>16,}"
                  f"{s['manual_words']:>16,}{share:>10.1%}")
    else:
        print("\n[3-5] not yet run - execute tools/make_splits.py")

    # ---- stage 6 ----
    print("\n[6] FINAL TOKENS  (one tokenizer, counted once)")
    print("-" * 76)
    train_tokens = manual_tokens = None
    if tokens:
        print(f"  tokenizer             {tokens['tokenizer']}")
        print(f"  vocabulary size       {tokens['vocab_size']:,}")
        print(f"  byte-fallback pieces  {tokens['byte_pieces']}"
              f"{'  (byte_fallback ON)' if tokens['byte_pieces'] == 256 else '  !! EXPECTED 256'}")
        for name, s in tokens["splits"].items():
            unk = s["unk_tokens"] / s["tokens"] if s["tokens"] else 0
            print(f"  {name:<8}{s['tokens']:>16,} tokens   unk {unk:.6%}")

        train_tokens = tokens["splits"].get("train", {}).get("tokens")
        if train_tokens and splits:
            share = splits.get("train_manual_share") or 0
            manual_tokens = int(train_tokens * share)
            print(f"\n  FINAL TRAINING TOKENS      {train_tokens:>16,}")
            print(f"    manual                   {manual_tokens:>16,}")
            print(f"    downloaded               {train_tokens - manual_tokens:>16,}")
            print(f"    manual percentage        {share:>15.1%}")
    else:
        print("  not yet counted - execute tools/build_tokenizer.py then corpus_stats.py")

    # ---- verdict ----
    print("\n" + "-" * 76)
    print("REQUIREMENT CHECK")
    print("-" * 76)
    if train_tokens and splits:
        share = splits.get("train_manual_share") or 0
        ok = share >= MANUAL_FLOOR
        print(f"  manual >= 20% of training tokens   {share:>8.1%}   "
              f"{'PASS' if ok else 'FAIL'}")
        print(f"  ~500M training tokens              "
              f"{train_tokens / TOKEN_TARGET:>8.1%}   "
              f"{'PASS' if train_tokens >= TOKEN_TARGET else 'SHORTFALL - justify'}")
    else:
        print("  pending - run the full pipeline first")

    return {
        "language": language,
        "stage1_raw": raw,
        "stage2_accepted": accepted,
        "stage345_splits": splits,
        "stage6_tokens": tokens,
        "final_train_tokens": train_tokens,
        "final_manual_tokens": manual_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage-by-stage corpus accounting.")
    parser.add_argument("--markdown", action="store_true",
                        help="also write report/phase1_pipeline_accounting.md")
    args = parser.parse_args()

    print("=" * 76)
    print("PHASE 1 - CORPUS PIPELINE ACCOUNTING")
    print("Every figure below is read from artifacts on disk, never estimated.")
    print("Stages 1-5 are in WORDS (tokenizer-independent); stage 6 in TOKENS.")
    print("=" * 76)

    results = {lang: report_language(lang) for lang in LANGUAGES}

    print("\n" + "=" * 76)
    print("SIDE BY SIDE")
    print("=" * 76)
    print(f"  {'':<30}{'Marathi':>20}{'Konkani':>20}")
    rows = [
        ("documents fetched", lambda r: r["stage1_raw"]["documents_fetched"]),
        ("documents accepted", lambda r: r["stage2_accepted"]["documents"]),
        ("words accepted", lambda r: r["stage2_accepted"]["words"]),
        ("  manual words", lambda r: r["stage2_accepted"]["manual_words"]),
        ("  downloaded words (real)",
         lambda r: r["stage2_accepted"]["downloaded_words"]),
        ("  synthetic words (MT/LLM)",
         lambda r: r["stage2_accepted"]["synthetic_words"]),
        ("  real words (manual+dl)",
         lambda r: (r["stage2_accepted"]["manual_words"]
                    + r["stage2_accepted"]["downloaded_words"])),
        ("final training tokens", lambda r: r["final_train_tokens"]),
        ("  manual tokens", lambda r: r["final_manual_tokens"]),
    ]
    for label, getter in rows:
        vals = []
        for lang in LANGUAGES:
            v = getter(results[lang])
            vals.append(f"{v:,}" if isinstance(v, int) else "pending")
        print(f"  {label:<30}{vals[0]:>20}{vals[1]:>20}")

    out = REPO_ROOT / "report" / "phase1_pipeline_accounting.json"
    atomic_write_json(out, results)
    print(f"\n  {out.relative_to(REPO_ROOT)}")

    if args.markdown:
        md = ["# Phase 1 — Pipeline Accounting", "",
              "Raw collected → cleaned → deduplicated → final training tokens.",
              "Stages 1–5 in **words** (tokenizer-independent); stage 6 in **tokens**.",
              "", "| Stage | Marathi | Konkani |", "|---|---:|---:|"]
        for label, getter in rows:
            cells = []
            for lang in LANGUAGES:
                v = getter(results[lang])
                cells.append(f"{v:,}" if isinstance(v, int) else "pending")
            md.append(f"| {label.strip()} | {cells[0]} | {cells[1]} |")
        path = REPO_ROOT / "report" / "phase1_pipeline_accounting.md"
        path.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"  {path.relative_to(REPO_ROOT)}")

    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
