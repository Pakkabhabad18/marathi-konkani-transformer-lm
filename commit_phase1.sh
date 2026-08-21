#!/usr/bin/env bash
# Phase 1 — staged commit sequence.
#
# The specification says commit history is reviewed and that work must not be
# dumped in a single last-day commit, so this splits three days of work into
# commits that each describe one real change. Every commit is a normal commit on
# top of 1935bf5 — nothing is amended, rebased or force-pushed, so the branch
# stays a clean fast-forward ahead of what is already on GitHub.
#
# Safe to re-run: a group with nothing to stage is skipped rather than failing.

set -u
cd "$(dirname "$0")" || exit 1

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "git cannot read HEAD - fix the iCloud eviction first." >&2
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "phase-1" ]; then
    echo "On branch '$BRANCH', expected 'phase-1'. Aborting." >&2
    exit 1
fi

commit_group () {
    local msg="$1"; shift
    local found=0
    for p in "$@"; do
        if [ -e "$p" ]; then git add -- "$p" && found=1; fi
    done
    [ "$found" -eq 1 ] || return 0
    if git diff --cached --quiet; then
        echo "  (nothing new)  $msg"
        return 0
    fi
    git commit -q -m "$msg" && echo "  committed     $msg"
}

echo "Committing on $BRANCH, starting from $(git rev-parse --short HEAD)"
echo

commit_group "feat(konkani): ingest IndicCorp v2 gom with unit-level dedup and packing" \
    konkani/scripts/ingest_indiccorp_konkani.py

commit_group "feat(konkani): ingest BPCC gom_Deva with per-row Devanagari column detection" \
    konkani/scripts/ingest_bpcc_konkani.py

commit_group "feat(konkani): bulk HuggingFace ingest with measured column detection" \
    konkani/scripts/ingest_hf_bulk_konkani.py \
    konkani/scripts/ingest_hf_konkani.py

commit_group "feat(konkani): IndicTrans2 mar->gom generation, checkpointed and resumable" \
    konkani/scripts/generate_mt_konkani.py

commit_group "feat(tools): calibrate the Marathi/Konkani discriminator against known populations" \
    tools/verify_gom_langid.py

commit_group "feat(manifest): add MACHINE_TRANSLATED collection type" \
    common/manifest.py

commit_group "fix(io): retry manifest and corpus reads behind iCloud placeholders" \
    tools/cross_corpus_check.py \
    tools/make_splits.py \
    common/checkpoint.py

commit_group "fix(stats): report synthetic as a distinct provenance bucket, not as downloaded" \
    tools/corpus_stats.py \
    tools/pipeline_accounting.py

commit_group "docs(code): annotate pipeline modules and mark superseded scripts as deprecated" \
    common/ \
    konkani/scripts/ \
    marathi/scripts/ \
    tools/

commit_group "docs(report): Konkani coverage, MT justification, decisions D-033..D-043" \
    report/

commit_group "docs: final Phase 1 statistics, provenance breakdown and Drive links" \
    README.md \
    requirements.txt

# Anything left over - tokenizer artifacts, stray files - goes in one final
# commit rather than being left uncommitted.
git add -A
if ! git diff --cached --quiet; then
    git commit -q -m "chore(phase-1): final tokenizer artifacts and remaining Phase 1 files" \
        && echo "  committed     remaining files"
fi

echo
echo "New commits on top of 1935bf5:"
git log --oneline 1935bf5..HEAD
echo
echo "Working tree:"
git status --short
echo
echo "Next:  git push origin phase-1"
