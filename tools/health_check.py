#!/usr/bin/env python3
"""
Health monitor for long-running collection jobs.

WHY THIS EXISTS
---------------
"Is the process still running?" is the wrong question. A collector can sit in a
retry loop for hours, or page through a source that returns nothing acceptable,
and still look perfectly healthy to `ps`. What matters is whether the *output*
is growing and whether it is growing at a sane rate.

This tool answers that by reading the checkpoint and the manifest - the two
things the job writes as it works - and comparing them against the previous
health snapshot. It never touches the job itself, so it is safe to run from a
second terminal at any time while a collection is in progress.

It reports, as required:
  documents collected, words and manual/downloaded token split, last successful
  update, processing rate, errors and HTTP 429s, checkpoint freshness, disk
  usage, language and script distribution, duplicate rate, and an explicit
  stalled/abnormal verdict.

STALL DETECTION
---------------
Three independent signals, because any one alone gives false alarms:

  1. Checkpoint freshness - time since the job last wrote its state. A healthy
     job writes at least every few minutes.
  2. Document delta - documents added since the previous health check. Zero
     across two consecutive checks with a fresh checkpoint means the job is
     spinning: fetching, but rejecting everything.
  3. Error and rate-limit ratio - a rising share of failures relative to
     accepted documents.

The verdict is deliberately loud. A silent degradation that is discovered on the
morning of the deadline is the failure mode this is built to prevent.

USAGE
-----
    python3 tools/health_check.py --job marathi_archive_gr
    python3 tools/health_check.py --all
    watch -n 3600 python3 tools/health_check.py --all      # hourly, unattended
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.manifest import read_manifest, summarize, atomic_write_json  # noqa: E402

LANGUAGES = ("marathi", "konkani")
SNAPSHOT_DIR = REPO_ROOT / ".health"

STALE_CHECKPOINT_SECONDS = 15 * 60      # no state write in 15 min = suspicious
HIGH_ERROR_RATIO = 0.10                 # >10% errors relative to accepted
LOW_DISK_GB = 5.0


def human_time(ts: float) -> str:
    if not ts:
        return "never"
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def human_delta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def discover_jobs() -> list[tuple[str, Path, Path]]:
    """Find (job_name, checkpoint_path, manifest_path) for every language."""
    jobs = []
    for lang in LANGUAGES:
        cp_dir = REPO_ROOT / lang / "data" / "checkpoints"
        mf_dir = REPO_ROOT / lang / "data" / "manifests"
        if not cp_dir.exists():
            continue
        for cp in sorted(cp_dir.glob("*.json")):
            jobs.append((cp.stem, cp, mf_dir / f"{cp.stem}.jsonl"))
    return jobs


def load_checkpoint(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def analyse_manifest(path: Path, sample_limit: int = 0) -> dict:
    """Aggregate the manifest: accounting plus script and language distributions."""
    acc = summarize(path)
    scripts: Counter = Counter()
    langs: Counter = Counter()
    langid_scores = []

    for i, row in enumerate(read_manifest(path)):
        if sample_limit and i >= sample_limit:
            break
        scripts[row.get("script", "unknown")] += 1
        langs[row.get("langid_label") or row.get("language", "unknown")] += 1
        score = row.get("langid_score")
        if isinstance(score, (int, float)):
            langid_scores.append(score)

    return {
        "accounting": acc,
        "scripts": scripts,
        "languages": langs,
        "mean_langid_score": (sum(langid_scores) / len(langid_scores)) if langid_scores else None,
    }


def check_job(job_name: str, cp_path: Path, mf_path: Path, quiet: bool = False) -> dict:
    state = load_checkpoint(cp_path)
    analysis = analyse_manifest(mf_path)
    acc = analysis["accounting"]

    now = time.time()
    updated_at = state.get("updated_at", 0)
    checkpoint_age = now - updated_at if updated_at else float("inf")

    snapshot_path = SNAPSHOT_DIR / f"{job_name}.json"
    previous = {}
    if snapshot_path.exists():
        try:
            previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}

    prev_docs = previous.get("documents", 0)
    prev_time = previous.get("checked_at", 0)
    delta_docs = acc.documents - prev_docs
    delta_time = now - prev_time if prev_time else 0

    rate_per_min = (delta_docs / (delta_time / 60)) if delta_time > 60 else None

    collected = int(state.get("collected", 0) or 0)
    errors = int(state.get("errors", 0) or 0)
    rate_limits = int(state.get("rate_limit_hits", 0) or 0)
    error_ratio = errors / collected if collected else 0.0

    dedup_stats = (state.get("extra", {}) or {}).get("dedup_stats", {})
    duplicate_rate = dedup_stats.get("duplicate_rate")

    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / 1e9

    # ---- verdict -------------------------------------------------------
    problems, warnings = [], []

    if state.get("finished"):
        status = "FINISHED"
    elif not state:
        status = "NOT STARTED"
    else:
        if checkpoint_age > STALE_CHECKPOINT_SECONDS:
            problems.append(
                f"checkpoint not written for {human_delta(checkpoint_age)} "
                f"(threshold {human_delta(STALE_CHECKPOINT_SECONDS)})")
        if prev_time and delta_docs == 0 and checkpoint_age < STALE_CHECKPOINT_SECONDS:
            problems.append(
                "checkpoint is fresh but zero documents added since last check - "
                "job is fetching and rejecting everything")
        if error_ratio > HIGH_ERROR_RATIO:
            problems.append(f"error ratio {error_ratio:.1%} exceeds {HIGH_ERROR_RATIO:.0%}")
        if free_gb < LOW_DISK_GB:
            problems.append(f"only {free_gb:.1f} GB disk free")
        if rate_limits > 0 and collected and rate_limits / collected > 0.05:
            warnings.append(f"{rate_limits} rate-limit hits - consider raising POLITE_DELAY")
        if duplicate_rate is not None and duplicate_rate > 0.5:
            warnings.append(
                f"duplicate rate {duplicate_rate:.1%} - source may be exhausted "
                f"or heavily boilerplate")
        if acc.words and acc.headroom() < 0:
            warnings.append(
                f"manual ratio below 20%: corpus is {-acc.headroom():,} words over "
                f"what the manual total supports")
        status = "STALLED" if problems else ("WARNING" if warnings else "HEALTHY")

    result = {
        "job": job_name,
        "status": status,
        "documents": acc.documents,
        "checked_at": now,
        "problems": problems,
        "warnings": warnings,
    }

    if not quiet:
        badge = {"HEALTHY": "[ OK ]", "WARNING": "[WARN]", "STALLED": "[STALL]",
                 "FINISHED": "[DONE]", "NOT STARTED": "[----]"}[status]
        print(f"\n{'=' * 68}")
        print(f"{badge}  {job_name}")
        print("=" * 68)

        print(f"  documents collected   {acc.documents:,}"
              + (f"  (+{delta_docs:,} since last check)" if prev_time else ""))
        print(f"  words                 {acc.words:,}")
        print(f"    manual              {acc.manual_words:,} ({acc.manual_word_fraction:.1%})")
        print(f"    downloaded          {acc.downloaded_words:,}")
        print(f"  tokens                {acc.tokens:,}"
              + ("   (filled by final tokenizer pass)" if acc.tokens == 0 else ""))
        print(f"    manual              {acc.manual_tokens:,}")
        print(f"    downloaded          {acc.downloaded_tokens:,}")
        print(f"  headroom at 20% rule  {acc.headroom():,} more words allowed")

        print(f"\n  last checkpoint       {human_time(updated_at)} "
              f"({human_delta(checkpoint_age)} ago)")
        if rate_per_min is not None:
            print(f"  processing rate       {rate_per_min:.1f} docs/min "
                  f"(over last {human_delta(delta_time)})")
        else:
            print("  processing rate       n/a (first check, or interval too short)")
        print(f"  errors / 429s         {errors:,} / {rate_limits:,}")
        if duplicate_rate is not None:
            print(f"  duplicate rate        {duplicate_rate:.1%}")
        print(f"  disk free             {free_gb:.1f} GB")

        if analysis["scripts"]:
            dist = "  ".join(f"{k}={v:,}" for k, v in analysis["scripts"].most_common(5))
            print(f"\n  script distribution   {dist}")
        if analysis["languages"]:
            dist = "  ".join(f"{k}={v:,}" for k, v in analysis["languages"].most_common(5))
            print(f"  langid distribution   {dist}")
        if analysis["mean_langid_score"] is not None:
            print(f"  mean langid score     {analysis['mean_langid_score']:+.3f}")

        if acc.by_source:
            print("\n  by source:")
            for src, s in sorted(acc.by_source.items(), key=lambda kv: -kv[1]["words"]):
                tag = "manual" if s["is_manual"] else "downloaded"
                print(f"    {src:38s} {s['documents']:>8,} docs  "
                      f"{s['words']:>12,} words  [{tag}]")

        for p in problems:
            print(f"\n  !! PROBLEM: {p}")
        for w in warnings:
            print(f"\n  ~  warning: {w}")
        if status == "HEALTHY":
            print("\n  No problems detected.")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(snapshot_path, {"documents": acc.documents, "checked_at": now,
                                      "status": status})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Health monitor for collection jobs.")
    parser.add_argument("--job", help="job name (checkpoint stem), e.g. marathi_archive_gr")
    parser.add_argument("--all", action="store_true", help="check every discovered job")
    parser.add_argument("--json", action="store_true", help="machine-readable output only")
    args = parser.parse_args()

    jobs = discover_jobs()
    if not jobs:
        print("No collection jobs found. Nothing has been started yet.")
        return 0

    if args.job:
        jobs = [j for j in jobs if j[0] == args.job]
        if not jobs:
            print(f"No job named '{args.job}'. Known jobs: "
                  f"{', '.join(j[0] for j in discover_jobs())}")
            return 1

    results = [check_job(name, cp, mf, quiet=args.json) for name, cp, mf in jobs]

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return 0

    bad = [r for r in results if r["status"] == "STALLED"]
    print(f"\n{'=' * 68}")
    print(f"{len(results)} job(s) checked; {len(bad)} stalled.")
    print("=" * 68)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
