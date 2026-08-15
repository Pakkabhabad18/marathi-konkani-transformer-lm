# Phase 1 — Runbook

Every command, in order, with what it does and what you should see. Run these
from the repository root:

```bash
cd ~/Desktop/individual-project-Pakkabhabad18
```

Run **one step at a time**. Read the output before moving on. If a step's output
does not look like what is described here, stop and report it rather than
continuing.

---

## Step 0 · Confirm every file is present

```bash
ls -1 common/ tools/ marathi/scripts/ report/
```

**Expect to see:**

```
common/:        __init__.py  checkpoint.py  dedup.py  manifest.py  scriptid.py  textnorm.py
tools/:         diagnose_source.py  health_check.py
marathi/scripts/: collect_archive_gr.py
report/:        phase1_decisions.md  phase1_execution_plan.md  phase1_gap_analysis.md
                phase1_konkani_progress.md  phase1_runbook.md  phase1_source_inventory.md
                phase1_viva_log.md
```

If anything is missing, say so before continuing. Nothing below will work
without these.

---

## Step 1 · Check git state — do this first, it is the one with a deadline

```bash
git status -sb
```

**What it tells you.** The first line looks like
`## phase-1...origin/phase-1 [ahead 3]`. The `[ahead N]` part is what matters:
it means N commits exist locally that are **not on GitHub**. Phase 1 is graded
from the branch as it stands on GitHub on 19 August, so local commits score
nothing.

If it says `[ahead N]`, push now:

```bash
git push origin phase-1
```

If it shows no `[ahead]` marker, you are already in sync.

---

## Step 2 · Verify the library works on your machine

Each module tests itself. Run them one at a time.

```bash
python3 common/textnorm.py
python3 common/scriptid.py
python3 common/manifest.py
python3 common/checkpoint.py
python3 common/dedup.py
```

**Expect** each to end with `... self-test: all assertions passed`.
`scriptid.py` additionally prints its language separation:

```
Marathi  script=Devanagari  deva=0.983 -> label=mr   score=+1.000 ...
Konkani  script=Devanagari  deva=0.969 -> label=kok  score=-1.000 ...
```

**Why this matters.** These five modules are the foundation everything else sits
on. If `dedup.py` is broken you will silently keep duplicate documents and
inflate your token count — which the specification specifically forbids. Testing
them takes ten seconds and removes an entire class of doubt.

*You have already run these successfully.*

---

## Step 3 · Diagnose the archive.org failure

```bash
python3 tools/diagnose_source.py
```

**What it does.** The first pilot attempt failed with HTTP 500. The same query
succeeds from elsewhere, so the problem is specific to your request. This script
changes one variable at a time — five User-Agent strings, four page sizes, the
fallback endpoint, and one real document download — and prints a verdict.

It writes nothing and modifies nothing. Safe to run repeatedly.

**Expect** four probe sections and then a `VERDICT` block telling you which of
three situations you are in:

| Verdict | Meaning | Action |
|---|---|---|
| Some User-Agents work | The UA string was the problem | Verdict names which one to use |
| All work | The 500s were transient load shedding | Just re-run the pilot |
| None work | Network/ISP/outage, not our code | Run the `curl` line the verdict prints |

**Send me the whole output.** The next step depends on which verdict you get.

---

## Step 4 · Run the pilot (300 documents)

Only after Step 3 gives a verdict.

```bash
python3 marathi/scripts/collect_archive_gr.py --limit 300 --page-size 200
```

**What it does.** Enumerates Maharashtra Government Resolutions on the Internet
Archive, downloads each one's OCR text layer, normalizes it, checks it is really
Marathi, removes duplicates, and writes both the text and a provenance row per
document.

**Expect** progress lines every 25 accepted documents:

```
accepted=25 skipped=112 dup_rate=31.2% rate=18.4/min
```

then a `RUN SUMMARY` block with a rejection-reason breakdown.

**It is safe to stop with Ctrl-C at any time.** The checkpoint means re-running
the same command resumes where it stopped; it does not start over and it does
not duplicate anything.

**Roughly how long.** 300 documents at ~15–25 per minute is about 15–20 minutes.
Leave it running.

### What we are looking for in the output

This is the whole point of the pilot. Three numbers decide what happens next:

1. **`dup_rate`** — the near-duplicate rate. My threshold of 0.85 rejected 11 of
   12 synthetic test documents, but those differed by only one word. If the real
   rate comes back above ~60%, the threshold is too aggressive and is throwing
   away good data; it needs loosening before the full run.
2. **Rejection reasons** — especially `not_enough_devanagari` and `langid_mr`.
   These tell us how bilingual the documents really are and whether the gates
   are set sensibly.
3. **Words per accepted document** — this is what turns 170,725 items into an
   honest estimate of how many manual Marathi tokens this source can supply.

---

## Step 5 · Check job health

```bash
python3 tools/health_check.py --all
```

**What it shows.** Documents, words, manual vs downloaded split, headroom under
the 20% rule, last checkpoint time, processing rate, errors and rate limits,
duplicate rate, disk free, script and language distribution, and a
HEALTHY / WARNING / STALLED verdict.

**When to run it.** Any time, from a second Terminal window, while a collection
job is running. It reads files only and never touches the job.

During the long full run, run it about **once an hour**. The important field is
`documents collected  N (+M since last check)`. If `M` is zero while the
checkpoint is fresh, the job is fetching and rejecting everything — which no
"is the process alive?" check would ever catch.

---

## Step 6 · Commit

```bash
git add common/ tools/ marathi/scripts/ report/ .gitignore
git status
```

Check the list. **`marathi/data/` and `konkani/data/` must NOT appear** — they
are gitignored because corpus data does not belong in git. If you see them,
stop and tell me.

Then:

```bash
git commit -m "Add shared pipeline foundation, M1 pilot collector, health monitor, viva log"
git push origin phase-1
```

---

## Where things run — the short version

| What | Where | Why |
|---|---|---|
| All collection, cleaning, statistics, tokenizers | **Your Mac, in Terminal** | Network-bound work; a GPU does nothing for an HTTP request |
| Writing and testing these scripts | Assistant's cloud container | It cannot reach archive.org or huggingface.co, and cannot run commands on your Mac |
| Model pretraining (Phase 2 only) | Colab / Kaggle GPU | The only genuinely GPU-bound stage |

The assistant writes files directly into this repository. **You do not need to
save or copy anything by hand** — when a file is mentioned, it is already on
disk. Verify with `ls` if unsure.

---

## If something goes wrong

| Symptom | What it means | Do this |
|---|---|---|
| `ModuleNotFoundError: No module named 'common'` | Run from the wrong directory | `cd ~/Desktop/individual-project-Pakkabhabad18` first |
| `ModuleNotFoundError: No module named 'requests'` | Dependency missing | `pip3 install requests` |
| Repeated `[500 n/8]` messages | archive.org rejecting us | Step 3 |
| `[429 n/8] rate limited` | We are fetching too fast | Normal; it backs off automatically. If frequent, raise `POLITE_DELAY` in the collector |
| Collector exits immediately, `accepted=0` | Checkpoint thinks it is finished | Check `marathi/data/checkpoints/marathi_archive_gr.json` |
| Want to start a source completely over | Checkpoint is holding old state | Delete `marathi/data/checkpoints/marathi_archive_gr.*` and the matching manifest |

Nothing here deletes data. The collector only ever appends.
