#!/usr/bin/env bash
# Phase 1 — tidy report/ so the graded surface is results, not planning notes.
#
# Nothing is deleted destructively: planning documents move to report/archive/
# and stay in the repository and in git history. Only two files leave the repo
# entirely, and both are explained below.
#
# Safe to re-run: every step checks whether it has already been done.

set -u
cd "$(dirname "$0")" || exit 1

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "git cannot read HEAD here. Fix that before running this." >&2
    exit 1
fi

mkdir -p report/archive

move_to_archive () {
    for f in "$@"; do
        if [ -f "report/$f" ]; then
            git mv -f "report/$f" "report/archive/$f" 2>/dev/null \
                || mv -f "report/$f" "report/archive/$f"
            echo "  archived   report/$f"
        fi
    done
}

echo "1. Archiving planning and scratch documents"
# These describe what we intended to do on 14-15 August, against a repository
# state and a deadline that no longer exist. Their conclusions are superseded by
# the measured reports; their reasoning survives in phase1_decisions.md. Kept in
# the repo rather than deleted, because they are part of how the work happened.
move_to_archive \
    phase1_execution_plan.md \
    phase1_gap_analysis.md \
    phase1_schedule.md \
    phase1_runbook.md \
    phase1_konkani_progress.md \
    phase1_token_budget_marathi.json \
    phase1_token_budget_konkani.json

echo
echo "2. Renaming the work log"
# "viva log" describes when it gets read, not what it is. It is a chronological
# engineering record, and the name should say so.
if [ -f report/phase1_viva_log.md ]; then
    git mv -f report/phase1_viva_log.md report/phase1_work_log.md 2>/dev/null \
        || mv -f report/phase1_viva_log.md report/phase1_work_log.md
    echo "  renamed    phase1_viva_log.md -> phase1_work_log.md"
fi

echo
echo "3. Removing files that are not project artifacts"
# phase1_viva_guide.md is a personal question-and-answer prep sheet. It is not a
# deliverable, and a prep sheet sitting in a graded repository invites exactly
# the question it is meant to prepare for. A copy is kept on the Desktop.
if [ -f report/phase1_viva_guide.md ]; then
    cp report/phase1_viva_guide.md "$HOME/Desktop/phase1_viva_guide.md"
    git rm -q --cached report/phase1_viva_guide.md 2>/dev/null
    mv report/phase1_viva_guide.md "$HOME/Desktop/.phase1_viva_guide_removed.md"
    echo "  removed    report/phase1_viva_guide.md"
    echo "             copy kept at ~/Desktop/phase1_viva_guide.md"
fi

# commit_phase1.sh is git housekeeping for one afternoon, not project code.
if [ -f commit_phase1.sh ]; then
    git rm -q --cached commit_phase1.sh 2>/dev/null
    mv commit_phase1.sh "$HOME/Desktop/commit_phase1.sh"
    echo "  removed    commit_phase1.sh  (moved to ~/Desktop)"
fi

echo
echo "4. Writing report/archive/README.md"
cat > report/archive/README.md <<'ARCHIVE_EOF'
# Archived Phase 1 documents

These are planning and progress documents from 14–19 August 2026. They are kept
because they record how the work was actually sequenced, but their figures and
conclusions are superseded and should not be read as current.

| file | written | why it is superseded |
|---|---|---|
| `phase1_gap_analysis.md` | 14 Aug | audits a repository state that no longer exists |
| `phase1_execution_plan.md` | 14 Aug | the plan that followed from that audit |
| `phase1_konkani_progress.md` | 14 Aug | Konkani figures from before the archive.org search was corrected (D-018) |
| `phase1_schedule.md` | 15 Aug | schedule against the original 19 August deadline |
| `phase1_runbook.md` | 15 Aug | replaced by the Reproduction section of the top-level README |
| `phase1_token_budget_*.json` | 18 Aug | target-setting artifacts, measured at vocabulary 10,000 |

Current figures are in `report/phase1_report.md` and the statistics files beside
it. Design decisions and corrections are in `report/phase1_decisions.md`.
ARCHIVE_EOF
echo "  written    report/archive/README.md"

echo
echo "5. Staging and committing"
git add -A report/ README.md tools/build_tokenizer.py 2>/dev/null
git add -A 2>/dev/null
if git diff --cached --quiet; then
    echo "  nothing to commit"
else
    git commit -q -m "docs(report): add the Phase 1 report, archive planning notes, correct tokenizer reproduction steps"
    echo "  committed"
fi

echo
echo "report/ now contains:"
ls report/
echo
echo "Next:  git push origin phase-1"
