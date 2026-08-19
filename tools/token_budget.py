#!/usr/bin/env python3
"""
How many raw words do we still need to collect to hit the token targets?

WHY THIS TOOL EXISTS
--------------------
Every target in the specification is stated in TOKENS:

    >= 100M manual tokens        ~500M total tokens        manual/total >= 20%

Every number our collectors produce is in WORDS. The bridge between them is
**fertility** - tokens per word - and it is a measured property of a specific
tokenizer applied to specific text, not a constant.

Getting this wrong has already cost this project twice. Once when a manual
figure in words was divided by a total in tokens and produced "11.2% manual",
which nearly triggered a redesign. Once when 96.1M manual words was read as
"nearly at the 100M target" when it is in fact ~171M tokens, 71% past it.
This tool exists so nobody has to do that conversion in their head again.

WHY PER-SOURCE FERTILITY, NOT ONE GLOBAL NUMBER
-----------------------------------------------
Fertility is not uniform across a corpus. A subword tokenizer splits clean
newspaper prose into fewer pieces per word than OCR text carrying scanning
artefacts, and an 80-word web-crawl fragment behaves differently again. Applying
one average to a corpus whose composition is about to change - which is exactly
our situation, since the downloaded share is growing - produces an estimate that
drifts precisely when it matters.

So fertility is measured **per source**, and corpus totals are computed as

    tokens = SUM over sources of  words_s * fertility_s

which stays correct as the mix shifts.

WHAT IT REPORTS
---------------
1. Reduction rate at each stage: raw chars -> clean chars, and rows -> accepted
   documents, read from the manifests rather than assumed.
2. Fertility per source, on a random sample of real documents.
3. Current manual and downloaded tokens, computed source by source.
4. The gap to each target, expressed in the unit the collectors actually
   consume: RAW WORDS still to collect, and for downloaded data, raw ROWS.

USAGE
-----
    python3 tools/token_budget.py --language marathi
    python3 tools/token_budget.py --language konkani --total-token-target 200000000
    python3 tools/token_budget.py --language marathi --sample-per-source 400 --json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.manifest import content_hash, read_manifest   # noqa: E402

MANUAL_TOKEN_TARGET = 100_000_000
TOTAL_TOKEN_TARGET = 500_000_000
RATIO_FLOOR = 0.20


def load_tokenizer(language: str):
    model = REPO_ROOT / language / "tokenizer" / f"{language}_bpe.model"
    if not model.exists():
        return None, model
    try:
        import sentencepiece as spm
    except ImportError:
        print("  [error] sentencepiece not installed: "
              "pip3 install sentencepiece")
        return None, model
    sp = spm.SentencePieceProcessor()
    sp.load(str(model))
    return sp, model


def manifest_stats(language: str) -> dict:
    """Per-source words, chars and document counts, straight from the manifests."""
    out: dict[str, dict] = {}
    mdir = REPO_ROOT / language / "data" / "manifests"
    for path in sorted(mdir.glob("*.jsonl")):
        for row in read_manifest(path):
            name = row.get("source_name") or path.stem
            e = out.setdefault(name, {"documents": 0, "words": 0,
                                      "raw_chars": 0, "clean_chars": 0,
                                      "manual": bool(row.get("is_manual"))})
            e["documents"] += 1
            e["words"] += int(row.get("words") or 0)
            e["raw_chars"] += int(row.get("raw_chars") or 0)
            e["clean_chars"] += int(row.get("clean_chars") or 0)
    return out


def hash_to_source(language: str) -> dict[str, str]:
    """content_hash -> source_name, straight from the manifests.

    CORRECTED (D-022). The first version of this tool guessed shard paths as
    `<lang>/data/<manual|downloaded>/<source_name>/shard_*.txt`. Measured against
    the real repository, that layout is wrong for almost everything:

        manual/archive_org_maharashtra_gr/   matches source_name   (worked)
        manual/news/                         ALL news sources share one dir
        processed/ai4bharat_indiccorp_v2_mar downloaded lives in processed/

    So 8 of 9 sources sampled zero documents, silently fell back to the mean of
    the one source that did work, and the tool applied a single global fertility
    - precisely the failure its own docstring says it exists to prevent. A tool
    that degrades quietly into the error it was built to catch is worse than no
    tool.

    The fix is not three more path guesses. Documents carry a `content_hash` in
    the manifest, so attribution can be exact and layout-independent: walk every
    shard under `<lang>/data/`, hash each line, look the source up. If the
    layout changes again, this keeps working.
    """
    out: dict[str, str] = {}
    mdir = REPO_ROOT / language / "data" / "manifests"
    for path in sorted(mdir.glob("*.jsonl")):
        for row in read_manifest(path):
            h = row.get("content_hash")
            if h:
                out[h] = row.get("source_name") or path.stem
    return out


def sample_by_source(language: str, n_per_source: int, rng: random.Random,
                     max_scan_lines: int = 4_000_000
                     ) -> tuple[dict[str, list[str]], dict]:
    """Reservoir-sample per source: directory attribution first, hash second.

    CORRECTED AGAIN (D-023). Pure hash attribution cannot work, and the reason
    is in the collectors. Both of them do:

        manifest.write(make_record(text=text, ...))   # hashes text WITH newlines
        shard.write(text.replace("\\n", " ") + "\\n")   # writes it WITHOUT them

    So a shard line only hashes back to its manifest row when the document had
    no internal newlines. Single-paragraph news articles usually do; the
    multi-paragraph government resolutions never do. That is why
    `archive_org_maharashtra_gr` - 48.6% of all Marathi words - sampled fine
    under the old path guess and then vanished under hashing. The transform is
    lossy in one direction and the original text is not in the manifest, so it
    cannot be inverted.

    The fix is to use the signal that is actually reliable for each directory:

      * A shard directory whose NAME equals a manifest source name is
        unambiguous - every document in it belongs to that source. Attribute
        directly. This covers `manual/archive_org_maharashtra_gr/` and
        `processed/ai4bharat_indiccorp_v2_mar/`.
      * A directory shared by several sources - `manual/news/`, which holds all
        eight news sources - carries no per-source signal in its path, so fall
        back to hashing there. Empirically that works for news.

    Neither path silently substitutes an average, which is the property that
    matters. Whatever cannot be attributed is counted and reported.
    """
    stats = manifest_stats(language)
    source_names = set(stats)
    h2s = hash_to_source(language)

    buckets: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    by_dir = by_hash = unattributed = 0
    scanned = 0

    data_root = REPO_ROOT / language / "data"
    shards = sorted(p for p in data_root.rglob("shard_*.txt") if p.is_file())
    rng.shuffle(shards)          # spread the sample across the whole collection

    def offer(src: str, line: str) -> None:
        counts[src] += 1
        bucket = buckets[src]
        if len(bucket) < n_per_source:
            bucket.append(line)
        else:
            j = rng.randrange(counts[src])
            if j < n_per_source:
                bucket[j] = line

    for shard in shards:
        dir_source = shard.parent.name if shard.parent.name in source_names else None
        try:
            fh = shard.open(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if len(line.split()) < 20:
                    continue
                scanned += 1
                if dir_source is not None:
                    offer(dir_source, line)
                    by_dir += 1
                else:
                    src = h2s.get(content_hash(line))
                    if src is None:
                        unattributed += 1
                        continue
                    offer(src, line)
                    by_hash += 1
        if scanned >= max_scan_lines:
            break
        # Stop once every source that has any words is adequately sampled.
        need = {n for n, s in stats.items() if s["words"] > 0}
        if need and all(len(buckets.get(n, ())) >= n_per_source for n in need):
            break

    total = by_dir + by_hash + unattributed
    return buckets, {
        "shards_available": len(shards),
        "lines_scanned": scanned,
        "attributed_by_directory": by_dir,
        "attributed_by_hash": by_hash,
        "unattributed": unattributed,
        "attribution_rate": ((by_dir + by_hash) / total) if total else 0.0,
        "scan_capped": scanned >= max_scan_lines,
    }


def measure(language: str, sample_per_source: int, seed: int) -> dict:
    sp, model_path = load_tokenizer(language)
    sources = manifest_stats(language)
    rng = random.Random(seed)

    buckets, diag = ({}, {}) if sp is None else sample_by_source(
        language, sample_per_source, rng)

    per_source = {}
    for name, s in sources.items():
        entry = dict(s)
        entry["clean_ratio"] = (s["clean_chars"] / s["raw_chars"]
                                if s["raw_chars"] else None)
        entry["words_per_doc"] = (s["words"] / s["documents"]
                                  if s["documents"] else 0)
        entry["fertility"] = None
        entry["fertility_sample_docs"] = 0

        docs = buckets.get(name) or []
        if sp is not None and docs:
            words = sum(len(d.split()) for d in docs)
            toks = sum(len(sp.encode(d)) for d in docs)
            if words:
                entry["fertility"] = toks / words
                entry["fertility_sample_docs"] = len(docs)
        per_source[name] = entry

    measured = [e["fertility"] for e in per_source.values() if e["fertility"]]
    fallback = sum(measured) / len(measured) if measured else None

    manual_tokens = downloaded_tokens = 0.0
    manual_words = downloaded_words = 0
    for e in per_source.values():
        f = e["fertility"] or fallback
        if f is None:
            continue
        if e["manual"]:
            manual_words += e["words"]
            manual_tokens += e["words"] * f
        else:
            downloaded_words += e["words"]
            downloaded_tokens += e["words"] * f

    unmeasured = [n for n, e in per_source.items()
                  if e["fertility"] is None and e["words"] > 0]
    unmeasured_words = sum(per_source[n]["words"] for n in unmeasured)
    total_words_all = sum(e["words"] for e in per_source.values()) or 1

    return {
        "language": language,
        "tokenizer": str(model_path),
        "tokenizer_found": sp is not None,
        "sampling_diagnostics": diag,
        "unmeasured_sources": unmeasured,
        "unmeasured_word_share": unmeasured_words / total_words_all,
        "per_source": per_source,
        "mean_measured_fertility": fallback,
        "manual_words": manual_words,
        "downloaded_words": downloaded_words,
        "manual_tokens": int(manual_tokens),
        "downloaded_tokens": int(downloaded_tokens),
        "total_tokens": int(manual_tokens + downloaded_tokens),
        "manual_share_tokens": (manual_tokens / (manual_tokens + downloaded_tokens)
                                if (manual_tokens + downloaded_tokens) else 0.0),
    }


def plan(m: dict, manual_target: int, total_target: int) -> dict:
    """Convert token gaps back into the raw words a collector must produce."""
    man_f = None
    dl_f = None
    man_w = dl_w = 0
    for e in m["per_source"].values():
        f = e["fertility"] or m["mean_measured_fertility"]
        if not f:
            continue
        if e["manual"]:
            man_f = (man_f or 0) + f * e["words"]; man_w += e["words"]
        else:
            dl_f = (dl_f or 0) + f * e["words"]; dl_w += e["words"]
    man_f = (man_f / man_w) if man_w else m["mean_measured_fertility"]
    dl_f = (dl_f / dl_w) if dl_w else m["mean_measured_fertility"]

    manual_gap_tokens = max(manual_target - m["manual_tokens"], 0)
    downloaded_target_tokens = max(total_target - max(m["manual_tokens"],
                                                      manual_target), 0)
    downloaded_gap_tokens = max(downloaded_target_tokens - m["downloaded_tokens"], 0)

    # The ratio also caps how much downloaded data may be admitted at all.
    max_downloaded_tokens = (m["manual_tokens"] / RATIO_FLOOR) - m["manual_tokens"]
    admissible = min(downloaded_gap_tokens,
                     max(max_downloaded_tokens - m["downloaded_tokens"], 0))

    return {
        "manual_fertility": man_f,
        "downloaded_fertility": dl_f,
        "manual_target_tokens": manual_target,
        "manual_gap_tokens": int(manual_gap_tokens),
        "manual_gap_words": int(manual_gap_tokens / man_f) if man_f else None,
        "downloaded_target_tokens": int(downloaded_target_tokens),
        "downloaded_gap_tokens": int(downloaded_gap_tokens),
        "downloaded_gap_words": int(downloaded_gap_tokens / dl_f) if dl_f else None,
        "max_downloaded_tokens_at_20pct": int(max_downloaded_tokens),
        "downloaded_admissible_tokens": int(admissible),
        "downloaded_admissible_words": int(admissible / dl_f) if dl_f else None,
        "projected_total_tokens": int(m["manual_tokens"] +
                                      m["downloaded_tokens"] + admissible),
    }


def render(m: dict, p: dict) -> None:
    print()
    print("=" * 86)
    print(f"  TOKEN BUDGET — {m['language'].upper()}")
    print("=" * 86)
    if not m["tokenizer_found"]:
        print(f"  [!] tokenizer not found at {m['tokenizer']}")
        print("      Fertility cannot be measured. Build it first:")
        print(f"      python3 tools/build_tokenizer.py --language {m['language']}")
        return

    print(f"  tokenizer: {Path(m['tokenizer']).name}")
    print()
    print(f"  {'source':<34}{'kind':<11}{'words':>15}{'clean/raw':>11}"
          f"{'fert':>7}{'n':>6}")
    print("  " + "-" * 82)
    for name, e in sorted(m["per_source"].items(), key=lambda kv: -kv[1]["words"]):
        cr = f"{e['clean_ratio']:.3f}" if e["clean_ratio"] else "-"
        fe = f"{e['fertility']:.3f}" if e["fertility"] else "(est)"
        print(f"  {name[:33]:<34}{'MANUAL' if e['manual'] else 'downloaded':<11}"
              f"{e['words']:>15,}{cr:>11}{fe:>7}{e['fertility_sample_docs']:>6}")
    print("  " + "-" * 82)
    print(f"  {'manual':<34}{m['manual_words']:>26,} words"
          f"  ->{m['manual_tokens']:>15,} tokens")
    print(f"  {'downloaded':<34}{m['downloaded_words']:>26,} words"
          f"  ->{m['downloaded_tokens']:>15,} tokens")
    print(f"  {'TOTAL':<34}{m['manual_words']+m['downloaded_words']:>26,} words"
          f"  ->{m['total_tokens']:>15,} tokens")
    print()
    print(f"  manual share (TOKENS)   {m['manual_share_tokens']:>7.2%}"
          f"   [{'OK' if m['manual_share_tokens'] >= RATIO_FLOOR else 'BELOW 20%'}]")
    print(f"  fertility  manual {p['manual_fertility']:.3f}"
          f"   downloaded {p['downloaded_fertility']:.3f}")

    d = m.get("sampling_diagnostics") or {}
    if d:
        print(f"  sampling   {d.get('lines_scanned', 0):,} lines from "
              f"{d.get('shards_available', 0)} shards | "
              f"by-dir {d.get('attributed_by_directory', 0):,} "
              f"by-hash {d.get('attributed_by_hash', 0):,} "
              f"unattributed {d.get('unattributed', 0):,} "
              f"({d.get('attribution_rate', 0):.1%} attributed)"
              + ("  [SCAN CAPPED]" if d.get("scan_capped") else ""))

    # Fail loud. A silently-estimated source is how the first version of this
    # tool applied one global fertility while claiming to measure per source.
    unmeasured = m.get("unmeasured_sources") or []
    share = m.get("unmeasured_word_share", 0.0)
    if unmeasured and share < 0.01:
        # Tiny sources estimated from the measured mean move nothing. Say so
        # plainly rather than raising an alarm that trains people to ignore it.
        print(f"  note       {len(unmeasured)} tiny source(s) estimated, "
              f"{share:.2%} of words — immaterial to every figure below")
        for name in unmeasured[:6]:
            print(f"                - {name}")
    elif unmeasured:
        print()
        print("  " + "!" * 82)
        print(f"  WARNING: {len(unmeasured)} source(s) could not be sampled — "
              f"{share:.1%} of all words.")
        print("  Their tokens are ESTIMATED from the mean of the measured "
              "sources, which is")
        print("  exactly the global-multiplier error this tool exists to avoid. "
              "Treat every")
        print("  token figure below as provisional until this is zero.")
        for name in unmeasured[:10]:
            print(f"      - {name}")
        if d.get("hash_hit_rate", 1.0) < 0.5:
            print("  Low hash hit rate suggests shard text no longer matches the "
                  "manifest")
            print("  content_hash (re-normalized after collection?). Investigate "
                  "before trusting.")
        print("  " + "!" * 82)

    print()
    print("  " + "=" * 82)
    print("  WHAT IS STILL NEEDED")
    print("  " + "-" * 82)
    if p["manual_gap_tokens"] == 0:
        over = m["manual_tokens"] - p["manual_target_tokens"]
        print(f"  manual {p['manual_target_tokens']:,} tokens: "
              f"MET, with {over:,} tokens to spare "
              f"({over/p['manual_target_tokens']:.0%} buffer)")
        print("     -> stop manual collection; more adds nothing to this target")
    else:
        print(f"  manual gap        {p['manual_gap_tokens']:>15,} tokens "
              f"= {p['manual_gap_words']:,} more raw words to collect")

    print(f"  downloaded target {p['downloaded_target_tokens']:>15,} tokens")
    print(f"  downloaded gap    {p['downloaded_gap_tokens']:>15,} tokens "
          f"= {p['downloaded_gap_words']:,} words to ingest"
          if p["downloaded_gap_tokens"] else
          "  downloaded target: MET")
    print(f"  ratio cap allows  {p['max_downloaded_tokens_at_20pct']:>15,} "
          f"downloaded tokens in total")
    print(f"  -> admissible now {p['downloaded_admissible_tokens']:>15,} tokens "
          f"= {p['downloaded_admissible_words']:,} words")
    print(f"  -> projected total{p['projected_total_tokens']:>15,} tokens")
    print("  " + "=" * 82)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert word counts to token budgets using measured fertility.")
    ap.add_argument("--language", required=True, choices=("marathi", "konkani"))
    ap.add_argument("--manual-token-target", type=int, default=MANUAL_TOKEN_TARGET)
    ap.add_argument("--total-token-target", type=int, default=TOTAL_TOKEN_TARGET)
    ap.add_argument("--sample-per-source", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    m = measure(args.language, args.sample_per_source, args.seed)
    p = plan(m, args.manual_token_target, args.total_token_target)

    if args.json:
        print(json.dumps({"measured": m, "plan": p}, ensure_ascii=False, indent=2))
        return 0

    render(m, p)
    out = REPO_ROOT / "report" / f"phase1_token_budget_{args.language}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"measured": m, "plan": p},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  written: {out.relative_to(REPO_ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
