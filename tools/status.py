#!/usr/bin/env python3
"""
One-screen answer to "how much have we actually collected, for both languages?"

WHY A SEPARATE TOOL
-------------------
`pipeline_accounting.py` reports the full stage-by-stage journey and
`corpus_stats.py` reports one language in depth. Neither answers the question
you actually ask ten times a day, which is "where do we stand right now". This
does, in about a second, by streaming the manifests.

WHAT IT REPORTS, AND WHY THOSE COLUMNS
--------------------------------------
Per source: documents, words, and whether the source is MANUAL or DOWNLOADED.
Then per language: the manual share and the headroom.

**Headroom** is the number that matters and the one people forget. The rule is

    manual / total >= 0.20      =>      total <= 5 x manual

so headroom = `5 x manual_words - total_words`. It is how many more downloaded
words the corpus is still allowed to take. When it goes negative the corpus has
already broken the rule and `make_splits.py` will subsample to fix it - which
means data you spent hours collecting gets thrown away. Watching headroom is how
you notice that *before* it happens.

Everything is reported in WORDS, not tokens. Words are tokenizer-independent, so
they are comparable across runs; token counts only become meaningful once the
final tokenizer exists, and mixing the two units is what produced the bogus
"11.2% manual" scare earlier in this project.

USAGE
-----
    python3 tools/status.py
    python3 tools/status.py --language konkani
    python3 tools/status.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.manifest import read_manifest   # noqa: E402

LANGUAGES = ("marathi", "konkani")
RATIO = 0.20


def collect(language: str) -> dict:
    manifest_dir = REPO_ROOT / language / "data" / "manifests"
    sources: dict[str, dict] = {}
    for path in sorted(manifest_dir.glob("*.jsonl")):
        for row in read_manifest(path):
            name = row.get("source_name") or path.stem
            entry = sources.setdefault(
                name, {"documents": 0, "words": 0, "chars": 0, "manual": None})
            entry["documents"] += 1
            entry["words"] += int(row.get("words") or 0)
            entry["chars"] += int(row.get("clean_chars") or 0)
            if entry["manual"] is None:
                entry["manual"] = bool(row.get("is_manual"))

    manual_words = sum(s["words"] for s in sources.values() if s["manual"])
    downloaded_words = sum(s["words"] for s in sources.values() if not s["manual"])
    total = manual_words + downloaded_words
    return {
        "language": language,
        "manifest_dir": str(manifest_dir),
        "sources": sources,
        "documents": sum(s["documents"] for s in sources.values()),
        "manual_words": manual_words,
        "downloaded_words": downloaded_words,
        "total_words": total,
        "manual_share": (manual_words / total) if total else 0.0,
        "headroom_words": int(manual_words / RATIO) - total,
    }


def render(stats: dict) -> None:
    lang = stats["language"]
    print()
    print("=" * 78)
    print(f"  {lang.upper()}")
    print("=" * 78)

    if not stats["sources"]:
        print("  no manifests yet — nothing collected for this language")
        print(f"  (looked in {stats['manifest_dir']})")
        return

    print(f"  {'source':<38}{'kind':<12}{'docs':>9}{'words':>16}")
    print("  " + "-" * 74)
    for name, s in sorted(stats["sources"].items(),
                          key=lambda kv: -kv[1]["words"]):
        kind = "MANUAL" if s["manual"] else "downloaded"
        print(f"  {name[:37]:<38}{kind:<12}{s['documents']:>9,}{s['words']:>16,}")

    print("  " + "-" * 74)
    print(f"  {'manual':<50}{stats['manual_words']:>24,}")
    print(f"  {'downloaded':<50}{stats['downloaded_words']:>24,}")
    print(f"  {'TOTAL':<50}{stats['total_words']:>24,}")
    print()

    share = stats["manual_share"]
    flag = "OK" if share >= RATIO else "BELOW 20% — splits will subsample"
    print(f"  manual share      {share:>8.2%}   [{flag}]")

    head = stats["headroom_words"]
    if head >= 0:
        print(f"  headroom          {head:>12,} more downloaded words allowed")
    else:
        print(f"  headroom          {head:>12,}  OVER THE CAP — "
              f"{abs(head):,} words must be dropped")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="How much data is collected, right now, per language.")
    parser.add_argument("--language", choices=LANGUAGES,
                        help="limit to one language (default: both)")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output only")
    args = parser.parse_args()

    langs = [args.language] if args.language else list(LANGUAGES)
    results = [collect(l) for l in langs]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for stats in results:
        render(stats)

    if len(results) > 1:
        print()
        print("=" * 78)
        print(f"  {'':<14}{'manual words':>18}{'total words':>18}{'manual %':>12}")
        for stats in results:
            print(f"  {stats['language']:<14}{stats['manual_words']:>18,}"
                  f"{stats['total_words']:>18,}{stats['manual_share']:>11.2%}")
        print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
