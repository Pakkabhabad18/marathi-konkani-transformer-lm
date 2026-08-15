# Phase 1 — Schedule and Verification Gates

**Now:** Saturday 15 August 2026 · **Deadline:** Wednesday 19 August, 23:59
**Time remaining:** ~4 days

Every stage has a **verification gate**: a specific thing to check before moving
on. If a gate does not pass, stop and report it. Do not proceed on the
assumption that it probably worked — twice already in this project a green
summary line has hidden a real failure.

---

## The one thing to understand before starting

Collection takes **wall-clock time** and cannot be compressed. Everything else
is bounded work that can happen while it runs. So the order is not "finish each
piece then start the next" — it is **get the crawler running first, then build
everything else alongside it.**

The second thing, which changes what "success" means:

> `manual / total ≥ 20%` is a **hard requirement**.
> `~500M tokens` is a **target**.

If manual Marathi collection reaches only 60M tokens, the correct response is a
**300M-token corpus at 20% manual**, not a 500M-token corpus at 12%. A corpus
that misses the target is a reported shortfall. A corpus that misses the ratio
fails a stated requirement. When they conflict, the ratio wins.

---

## STAGE 0 · Secure what exists — 5 minutes — do this first

```bash
cd ~/Desktop/individual-project-Pakkabhabad18
git add -A
git status
```

**Gate 0a.** The staged list must **not** contain `marathi/data/` or
`konkani/data/`. If it does, `.gitignore` is not being applied — stop.

```bash
git commit -m "Add pipeline foundation, M1 collector, health monitor, README, reports"
git push origin phase-1
git status -sb
```

**Gate 0b.** Final line reads `## phase-1...origin/phase-1` with **no**
`[ahead N]`. That means GitHub has everything. Phase 1 is graded from the
remote branch, so until this passes, none of the work counts.

---

## STAGE 1 · Verify the collector actually works — 20 minutes — today

```bash
python3 tools/diagnose_source.py
```

**Gate 1a.** In PROBE 4, `resolved name` shows **HTTP 200** and prints a
Devanagari preview. (`guessed name` will show 404 — that is expected and is the
bug we fixed; the probe keeps both to show the difference.)

Then the pilot:

```bash
python3 marathi/scripts/collect_archive_gr.py --limit 300 --page-size 200
```

**Gate 1b.** `RUN SUMMARY` shows `Accepted this run: 300` (or close to it) and
`Errors: 0`.

```bash
python3 tools/health_check.py --all
```

**Gate 1c.** Status is `HEALTHY`, `manual` words > 0, `downloaded` words = 0.

### Read these three numbers before going further

They decide whether Stage 2 is worth starting at all:

| Number | Where | What it means |
|---|---|---|
| `dup_rate` | RUN SUMMARY | Above ~60% → threshold too aggressive, loosen before the full run |
| Rejection reasons | RUN SUMMARY | If `not_enough_devanagari` dominates, the documents are more English than expected |
| words ÷ accepted | health check | Multiply by 170,796 × acceptance rate → honest estimate of what M1 can supply |

**Send me these numbers.** If words-per-document times a realistic acceptance
rate does not project to roughly 50M+ words, M1 is not the right primary source
and we change plan rather than spend two days crawling it.

---

## STAGE 2 · Start the full Marathi crawl — today, and leave it running

This is **the** critical-path item. Everything else fits around it.

```bash
mkdir -p logs
nohup python3 marathi/scripts/collect_archive_gr.py \
      > logs/marathi_gr.log 2>&1 &
echo $!            # note this process id
```

`nohup ... &` runs it in the background so it survives closing the terminal.

**Gate 2a.** After 5 minutes:

```bash
tail -20 logs/marathi_gr.log
python3 tools/health_check.py --job marathi_archive_gr
```

Document count is rising and status is `HEALTHY`.

**Gate 2b — every 1–2 hours while it runs:**

```bash
python3 tools/health_check.py --all
```

Look at `documents collected  N (+M since last check)`. **If `M` is zero while
the checkpoint is fresh, the job is fetching and rejecting everything** — stop
and report. That is the failure mode a "is the process running?" check would
never catch.

**Keep your Mac awake:** `caffeinate -i -w <pid>` in another terminal, or set
Energy Saver to never sleep. A sleeping laptop pauses the crawl.

**Expected throughput.** ~2 requests per document plus a 0.7s delay ≈ 25–35
docs/min ≈ 1,800/hour. Over 48 hours that is roughly 85,000 documents examined.
After filtering, plausibly 40,000–50,000 accepted.

**If the pilot shows the numeric-suffix filename pattern is consistent across
all GR items,** we can skip the metadata request and nearly double throughput.
Do not apply this before the pilot confirms it.

---

## STAGE 3 · Konkani downloaded corpus — today/Sunday, runs alongside

Independent of Stage 2. Different network target, so they do not contend.

Scripts needed (not yet written — I will write them next):

- `konkani/scripts/ingest_books_corpus.py` — stream `omdeep22/Konkani_books_corpus-v2`,
  apply NFC normalization, script and language gates, dedup, write manifest
- `konkani/scripts/ingest_sangraha.py` — the `gom` split, ~10.1M tokens

**Gate 3.** Manifest exists, `is_manual=false` on every row, duplicate rate
recorded, and the recomputed Devanagari ratio comes out near **99.78%** — which
also confirms the corrected measurement from decision D-004 on real data rather
than by arithmetic.

---

## STAGE 4 · Konkani manual collection — Sunday/Monday

The hardest constraint in the project. Archive.org has only ~13–15 real Konkani
books, so this is many small sources rather than one large one.

- `konkani/scripts/collect_archive_books.py` — the ~15 books (~1–2M words)
- K2 investigation: Goa Konkani Akademi, Goa government publications
- Existing Wikipedia collection, Devanagari subset only (~2.6M tokens),
  as a documented secondary source

**Gate 4.** `health_check.py` reports Konkani `manual` words, and `headroom`
tells you directly how large the Konkani corpus is allowed to be. Whatever
manual total we reach, the corpus is capped at 5× it. That capped number **is**
the answer, and the shortfall is justified by the measured evidence: Sangraha
contains only 10.1M Konkani tokens in total.

---

## STAGE 5 · Tokenizers — Monday/Tuesday

Scripts needed:

- `marathi/scripts/train_tokenizer.py` and `konkani/scripts/train_tokenizer.py`
  — SentencePiece BPE, **`byte_fallback=True`**, separate vocabularies
- `<lang>/scripts/eval_tokenizer.py` — fertility and chars/token on **held-out**
  text, plus token-frequency statistics and tokenization examples

**Gate 5a.** Byte pieces present in each vocabulary:

```bash
python3 -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='marathi/tokenizer/marathi_bpe.model')
print(sum(1 for i in range(sp.get_piece_size()) if sp.id_to_piece(i).startswith('<0x')))
"
```

Must print **256**, not 0. Zero means `byte_fallback` is off again and every
reported UNK figure will be an artefact — the exact bug found in the audit.

**Gate 5b.** UNK rate on held-out text is ~0%, and vocabulary size is chosen by
comparing fertility across candidates, not by picking 32,000 out of habit.

---

## STAGE 6 · Splits, statistics, contamination check — Tuesday

- `tools/make_splits.py` — **document-level** train/val/test. Splitting at line
  level would leak content between train and test, since consecutive lines come
  from the same document.
- `tools/corpus_stats.py` — per-language statistics with the manual vs
  downloaded split
- `tools/cross_corpus_check.py` — hash overlap between the two corpora

**Gate 6a.** No document hash appears in both the Marathi and Konkani manifests.
Required result: **0 shared documents**, printed as a number.

**Gate 6b.** Final token counts computed by **one** tokenizer per language, in
one pass, after the corpus is frozen. Never mix counts from different
tokenizers — that is how the earlier "~91.91M tokens" figure went wrong.

---

## STAGE 7 · Ship — Wednesday, with hours to spare

1. Upload corpora and tokenizers to Google Drive.
2. Fill the link table in `README.md`; set permissions so TAs need not request
   access.
3. Final `report/phase1_report.md` with: corpus statistics, sources, cleaning
   steps, manual fraction, tokenizer comparison, splits.
4. Verify all 7 Phase 1 deliverables are present in the branch.
5. `git push origin phase-1`.

**Gate 7 — final check.** Open
`https://github.com/CL3-410/individual-project-Pakkabhabad18/tree/phase-1`
in a browser and confirm every file is visible there. Not locally — **on
GitHub**. That is what gets graded.

---

## Deliverables tracker

| # | Deliverable | Status | Stage |
|---|---|---|---|
| 1 | Dataset collection scripts (both languages) | partial — M1 done, others pending | 2, 3, 4 |
| 2 | Dataset preprocessing pipelines | library done, not yet wired end to end | 3 |
| 3 | Per-language dataset statistics reports | not started | 6 |
| 4 | Per-language train/val/test splits | not started | 6 |
| 5 | Tokenizer training code (both languages) | Konkani only, and needs the byte-fallback fix | 5 |
| 6 | Vocabulary files (one per language) | Konkani preliminary only | 5 |
| 7 | Tokenizer model files (one per language) | Konkani preliminary only | 5 |

---

## If time runs short

Cut in this order. Do **not** cut from the top.

1. Konkani corpus size — a smaller corpus with a documented, evidence-backed
   shortfall is explicitly permitted by the specification.
2. Marathi corpus size — reduce total to preserve the 20% ratio.
3. Number of manual sources — two well-documented sources beat five undocumented
   ones.

**Never cut:** the 20% manual ratio, `byte_fallback=True`, document-level
splits, the cross-corpus contamination check, or the README Drive links. Each of
those is a stated requirement, and each is cheap relative to what it costs to
lose.
