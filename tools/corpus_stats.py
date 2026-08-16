#!/usr/bin/env python3
"""
Phase 1 deliverable #3: per-language dataset statistics report.

WHAT THE SPECIFICATION ASKS FOR
-------------------------------
"report corpus statistics (size, sources, cleaning steps, manual fraction)" and
"Report the manual vs. downloaded token split in your dataset statistics."

So this is not a generic summary - it has to answer specific questions, and it
has to answer them with measured numbers rather than estimates.

THE TOKEN COUNT RULE
--------------------
Token counts come from ONE tokenizer per language, applied to the whole corpus
in one pass, after the corpus is frozen. This is why the manifests store
`tokens: null` during collection.

The failure this prevents is real and already happened once in this project: the
earlier progress report summed 86.75M tokens (measured with a Devanagari
tokenizer, on its own training data) and 5.16M tokens (measured with a different
mixed-script tokenizer) into a single "~91.91M tokens" figure. Two tokenizers,
two measurement conditions, one meaningless total.

If the language's tokenizer does not exist yet, this script reports words and
characters - which are tokenizer-independent - and says plainly that tokens are
pending, rather than inventing a conversion factor.

USAGE
-----
    python3 tools/corpus_stats.py --language marathi
    python3 tools/corpus_stats.py --language konkani
    python3 tools/corpus_stats.py --language marathi --markdown
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.manifest import atomic_write_json, read_manifest           # noqa: E402

MANUAL_FLOOR = 0.20
TOKEN_TARGET = 500_000_000


def load_manifest_rows(language: str) -> list[dict]:
    rows = []
    for path in sorted((REPO_ROOT / language / "data" / "manifests").glob("*.jsonl")):
        rows.extend(read_manifest(path))
    return rows


def load_split_counts(language: str) -> dict:
    splits = {}
    split_dir = REPO_ROOT / language / "data" / "splits"
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.txt"
        if not path.exists():
            continue
        docs = words = chars = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    docs += 1
                    words += len(line.split())
                    chars += len(line)
        splits[name] = {"documents": docs, "words": words, "characters": chars}
    return splits


def count_tokens(language: str, splits: dict) -> dict | None:
    """Tokenize each split with the language's ONE final tokenizer."""
    model = REPO_ROOT / language / "tokenizer" / f"{language}_bpe.model"
    if not model.exists():
        return None

    try:
        import sentencepiece as spm
    except ImportError:
        return None

    sp = spm.SentencePieceProcessor(model_file=str(model))
    out = {"tokenizer": str(model.relative_to(REPO_ROOT)),
           "vocab_size": sp.get_piece_size(),
           "byte_pieces": sum(1 for i in range(sp.get_piece_size())
                              if sp.id_to_piece(i).startswith("<0x")),
           "splits": {}}

    for name in splits:
        path = REPO_ROOT / language / "data" / "splits" / f"{name}.txt"
        tokens = unk = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ids = sp.encode(line)
                tokens += len(ids)
                unk += sum(1 for i in ids if i == sp.unk_id())
        out["splits"][name] = {"tokens": tokens, "unk_tokens": unk}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Corpus statistics report.")
    parser.add_argument("--language", required=True, choices=["marathi", "konkani"])
    parser.add_argument("--markdown", action="store_true",
                        help="also write a Markdown report into report/")
    args = parser.parse_args()

    lang = args.language
    rows = load_manifest_rows(lang)
    if not rows:
        raise SystemExit(f"No manifests for {lang}. Run the collectors first.")

    print("=" * 74)
    print(f"CORPUS STATISTICS - {lang}")
    print("=" * 74)

    # ---- provenance and accounting ---------------------------------------
    by_source: dict = defaultdict(lambda: {
        "documents": 0, "words": 0, "characters": 0, "raw_characters": 0,
        "is_manual": False, "collection_method": "", "urls": set(),
        "langid_scores": [], "doc_words": []})
    scripts: Counter = Counter()
    langid_labels: Counter = Counter()
    preprocessing: Counter = Counter()

    manual_words = downloaded_words = 0
    manual_docs = downloaded_docs = 0

    for row in rows:
        source = row.get("source_name", "unknown")
        entry = by_source[source]
        words = int(row.get("words") or 0)
        entry["documents"] += 1
        entry["words"] += words
        entry["characters"] += int(row.get("clean_chars") or 0)
        entry["raw_characters"] += int(row.get("raw_chars") or 0)
        entry["is_manual"] = bool(row.get("is_manual"))
        entry["collection_method"] = row.get("collection_method", "")
        entry["doc_words"].append(words)
        score = row.get("langid_score")
        if isinstance(score, (int, float)):
            entry["langid_scores"].append(score)

        scripts[row.get("script", "unknown")] += 1
        langid_labels[row.get("langid_label") or "unlabelled"] += 1
        for step in row.get("preprocessing_applied", []) or []:
            preprocessing[step] += 1

        if row.get("is_manual"):
            manual_words += words
            manual_docs += 1
        else:
            downloaded_words += words
            downloaded_docs += 1

    total_words = manual_words + downloaded_words
    total_docs = manual_docs + downloaded_docs
    manual_share = manual_words / total_words if total_words else 0.0

    print(f"\nDocuments      {total_docs:>15,}")
    print(f"Words          {total_words:>15,}")
    print(f"  manual       {manual_words:>15,}  ({manual_share:.1%})")
    print(f"  downloaded   {downloaded_words:>15,}  ({1 - manual_share:.1%})")

    # ---- sources ----------------------------------------------------------
    print("\n" + "-" * 74)
    print("SOURCES")
    print("-" * 74)
    print(f"{'source':<34}{'docs':>9}{'words':>14}{'w/doc':>8}{'type':>12}")
    print("-" * 74)
    for source, e in sorted(by_source.items(), key=lambda kv: -kv[1]["words"]):
        per_doc = e["words"] / e["documents"] if e["documents"] else 0
        tag = "manual" if e["is_manual"] else "downloaded"
        print(f"{source[:33]:<34}{e['documents']:>9,}{e['words']:>14,}"
              f"{per_doc:>8,.0f}{tag:>12}")

    # ---- cleaning ---------------------------------------------------------
    print("\n" + "-" * 74)
    print("CLEANING STEPS APPLIED  (documents each step touched)")
    print("-" * 74)
    for step, count in preprocessing.most_common():
        print(f"  {step:<40} {count:>12,}")

    total_raw = sum(e["raw_characters"] for e in by_source.values())
    total_clean = sum(e["characters"] for e in by_source.values())
    if total_raw:
        print(f"\n  raw characters      {total_raw:>15,}")
        print(f"  cleaned characters  {total_clean:>15,}")
        print(f"  removed by cleaning {total_raw - total_clean:>15,}  "
              f"({1 - total_clean / total_raw:.1%})")

    # ---- composition ------------------------------------------------------
    print("\n" + "-" * 74)
    print("SCRIPT AND LANGUAGE COMPOSITION")
    print("-" * 74)
    for script, count in scripts.most_common():
        print(f"  script   {script:<16} {count:>12,}  ({count / total_docs:.1%})")
    for label, count in langid_labels.most_common():
        print(f"  langid   {label:<16} {count:>12,}  ({count / total_docs:.1%})")

    all_lengths = [w for e in by_source.values() for w in e["doc_words"] if w]
    if all_lengths:
        ordered = sorted(all_lengths)
        print(f"\n  document length (words):")
        print(f"    min {ordered[0]:,}  median {statistics.median(ordered):,.0f}  "
              f"mean {statistics.fmean(ordered):,.0f}  "
              f"p90 {ordered[int(len(ordered) * 0.9)]:,}  max {ordered[-1]:,}")

    # ---- splits -----------------------------------------------------------
    splits = load_split_counts(lang)
    if splits:
        print("\n" + "-" * 74)
        print("TRAIN / VALIDATION / TEST SPLITS")
        print("-" * 74)
        for name, s in splits.items():
            print(f"  {name:<6} {s['documents']:>10,} docs  "
                  f"{s['words']:>14,} words  {s['characters']:>15,} chars")
    else:
        print("\n  splits: not built yet (run tools/make_splits.py)")

    # ---- tokens -----------------------------------------------------------
    print("\n" + "-" * 74)
    print("TOKENS")
    print("-" * 74)
    token_info = count_tokens(lang, splits) if splits else None

    if token_info is None:
        print("  Not yet counted. Tokens are measured once, with the language's")
        print("  final tokenizer, after the corpus is frozen - never estimated")
        print("  from a preliminary tokenizer or a conversion factor.")
        print("  Run: tools/build_tokenizer.py then tools/make_splits.py")
        train_tokens = None
    else:
        print(f"  tokenizer     {token_info['tokenizer']}")
        print(f"  vocab size    {token_info['vocab_size']:,}")
        print(f"  byte pieces   {token_info['byte_pieces']}  "
              f"{'(byte_fallback ON)' if token_info['byte_pieces'] == 256 else '!! EXPECTED 256'}")
        for name, s in token_info["splits"].items():
            unk_rate = s["unk_tokens"] / s["tokens"] if s["tokens"] else 0
            print(f"  {name:<6} {s['tokens']:>15,} tokens   unk {unk_rate:.6%}")
        train_tokens = token_info["splits"].get("train", {}).get("tokens")

        if train_tokens:
            print(f"\n  training tokens vs ~{TOKEN_TARGET / 1e6:.0f}M target: "
                  f"{train_tokens / TOKEN_TARGET:.1%}")
            print(f"  manual training tokens (by word share): "
                  f"~{int(train_tokens * manual_share):,}")

    # ---- requirements -----------------------------------------------------
    print("\n" + "=" * 74)
    print("REQUIREMENT STATUS")
    print("=" * 74)
    ok_manual = manual_share >= MANUAL_FLOOR
    print(f"  manual >= 20%        {manual_share:>8.1%}   "
          f"{'PASS' if ok_manual else 'FAIL'}")
    if not ok_manual:
        need = int((MANUAL_FLOOR * total_words - manual_words) / (1 - MANUAL_FLOOR))
        print(f"       -> collect {need:,} more manual words, OR drop "
              f"{total_words - manual_words * 5:,} downloaded words")
    if train_tokens:
        print(f"  ~500M train tokens   {train_tokens / TOKEN_TARGET:>8.1%}   "
              f"{'PASS' if train_tokens >= TOKEN_TARGET else 'SHORTFALL - justify in report'}")
    else:
        print("  ~500M train tokens   pending tokenizer")
    print(f"  splits built         {'yes' if splits else 'no':>8}")

    payload = {
        "language": lang,
        "documents": total_docs,
        "words": {"total": total_words, "manual": manual_words,
                  "downloaded": downloaded_words, "manual_share": manual_share},
        "sources": {
            s: {k: (v if k not in ("urls", "langid_scores", "doc_words") else None)
                for k, v in e.items() if k not in ("urls", "langid_scores", "doc_words")}
            for s, e in by_source.items()},
        "scripts": dict(scripts),
        "langid_labels": dict(langid_labels),
        "cleaning_steps": dict(preprocessing),
        "raw_characters": total_raw,
        "clean_characters": total_clean,
        "splits": splits,
        "tokens": token_info,
        "meets_manual_requirement": ok_manual,
    }
    out_json = REPO_ROOT / "report" / f"phase1_corpus_stats_{lang}.json"
    atomic_write_json(out_json, payload)
    print(f"\n  {out_json.relative_to(REPO_ROOT)}")

    if args.markdown:
        md = [f"# Phase 1 — Corpus Statistics: {lang}", "",
              f"- Documents: **{total_docs:,}**",
              f"- Words: **{total_words:,}**",
              f"- Manual: **{manual_words:,}** ({manual_share:.1%})",
              f"- Downloaded: **{downloaded_words:,}**", "",
              "## Sources", "",
              "| Source | Documents | Words | Words/doc | Type |",
              "|---|---:|---:|---:|---|"]
        for source, e in sorted(by_source.items(), key=lambda kv: -kv[1]["words"]):
            per_doc = e["words"] / e["documents"] if e["documents"] else 0
            tag = "manual" if e["is_manual"] else "downloaded"
            md.append(f"| {source} | {e['documents']:,} | {e['words']:,} | "
                      f"{per_doc:,.0f} | {tag} |")
        if splits:
            md += ["", "## Splits", "",
                   "| Split | Documents | Words |", "|---|---:|---:|"]
            for name, s in splits.items():
                md.append(f"| {name} | {s['documents']:,} | {s['words']:,} |")
        out_md = REPO_ROOT / "report" / f"phase1_corpus_stats_{lang}.md"
        out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"  {out_md.relative_to(REPO_ROOT)}")

    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
