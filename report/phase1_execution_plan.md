# Phase 1 — Execution Plan

**Written:** 14 August 2026 · **Phase 1 deadline:** 19 August 2026, 23:59

Read alongside `phase1_gap_analysis.md` (what is wrong) and
`phase1_source_inventory.md` (what we can collect). This document says what we
do about it, in what order.

---

## 1. The plan in one paragraph

The manual ratio, not the token target, decides the size of both corpora. We
first build a shared accounting and provenance layer so that every document is
attributed to a source and classified manual/downloaded *as it arrives*, then
pilot two or three sources per language at ~300 documents each to measure real
yield and duplicate rates, then scale only the sources the pilots justify.
Marathi runs a large manual OCR collection because a verified 170,725-item
source exists. Konkani runs a small, careful, many-source collection because no
large source exists — and that finding is itself the evidence for the documented
shortfall the specification permits.

---

## 2. Files: retain, modify, replace

Nothing is deleted. Existing statistics are preserved as experimental results
and superseded rather than overwritten.

### Retain unchanged

| File | Why |
|---|---|
| `konkani/scripts/collect_wikipedia_sample.py` | Documented self-collection experiment. Its wikitext cleaner is reusable. |
| `konkani/scripts/resume_wikipedia_collection.py` | Its 429/timeout/backoff handling is the pattern the new collectors copy. |
| `konkani/scripts/analyze_wikipedia_metadata.py` | Correct as written. |
| `konkani/scripts/inspect_books_dataset.py` | Correct; useful for provenance spot-checks. |
| `konkani/tokenizer/preliminary_*.model/.vocab` | Evidence for the byte-fallback finding. Kept as `preliminary_*`, never used for a final count. |
| `report/phase1_konkani_progress.md` | Preserved verbatim as the experimental record. Corrections go in a new section, not over the old numbers. |

### Modify

| File | Change | Reason |
|---|---|---|
| `konkani/scripts/corpus_audit.py` | Delegate to `common.scriptid.profile_script` | Its ratio logic is already correct; centralising prevents the two scripts drifting apart again. |
| `konkani/scripts/filter_wikipedia_corpus.py` | Add NFC normalization + script gate + langid | Currently filters on length only. |
| `.gitignore` | Add `marathi/data/`, `.health/`, `*.seen` | Only `konkani/data/` is ignored today; Marathi data would be committed by accident. |
| `requirements.txt` | Pin what the new scripts need | Reproducibility. |

### Replace (old kept, superseded)

| Old | New | Reason |
|---|---|---|
| `analyze_books_corpus.py` Devanagari-percentage logic | `common/scriptid.py` | The old code divides by total characters including whitespace, producing 83.54% where the true figure is 99.78%. Old script kept; a header comment will point at the correction. |
| `train_preliminary_tokenizer.py` / `train_mixed_tokenizer.py` | new `train_tokenizer.py` per language | Both omit `byte_fallback=True`, which is the root cause of every UNK figure reported so far. |
| `tokenizer_stats.py` evaluation on training data | new held-out evaluation | Fertility measured on the tokenizer's own training file is optimistic and not what the spec asks for. |

### New (built this session, tested)

| File | Purpose | Test status |
|---|---|---|
| `common/textnorm.py` | NFC, whitespace, danda spacing; preserves ZWJ/ZWNJ | self-test passes |
| `common/scriptid.py` | Script profiling + Marathi/Konkani discriminator | self-test passes; clean ±1.000 separation on control sentences |
| `common/manifest.py` | Provenance schema, JSONL manifest, manual/downloaded accounting | self-test passes |
| `common/checkpoint.py` | Atomic, resumable checkpoints with persisted cursor | self-test passes, including corrupt-file recovery |
| `common/dedup.py` | Exact SHA-256 + MinHash/LSH near-duplicate detection | self-test passes; 0.875 similarity on boilerplate variant, 0.0 on unrelated text |
| `marathi/scripts/collect_archive_gr.py` | Pilot source M1 | logic verified offline against mock documents; network run pending on your Mac |
| `tools/health_check.py` | Job health monitoring | runs; reports "no jobs" correctly on empty state |

---

## 3. First sources to pilot

### Marathi — M1: Maharashtra Government Resolutions (archive.org) — **ready to run**

Verified 14 Aug 2026 via the archive.org APIs:

- `identifier:in.gov.maharashtra.gr.*` → **170,725 items** (scrape API `total`)
- Sample item `in.gov.maharashtra.gr.202607071620477816`: has `_djvu.txt` (34.5 KB),
  metadata language "Marathi, English", OCR by Tesseract 5.3.0
- Bulk enumeration via the cursor-paginated scrape API — resumable by design

Counts as **manual**: there is no ready-made "Marathi GR corpus" to download. We
enumerate, fetch each document individually, extract the OCR text layer, and do
all filtering, normalization and deduplication ourselves.

What the pilot must measure, because these are the risks: how much survives the
Devanagari-ratio gate (documents are bilingual), how much survives the langid
gate, and above all the **near-duplicate rate** — these documents share
departmental headers and closing formulae by design.

### Marathi — M2: news / literary scraping — **after M1 pilot**

Deferred deliberately until M1 reports, for one reason: IndicCorpV2 and Sangraha
are themselves built from Marathi news crawls. Scraping the same sites and
calling the result manual would be self-deception, and cross-dedup would delete
most of it. The overlap check must come first.

### Konkani — K1: archive.org Konkani texts — **small, verified**

Verified 14 Aug 2026: `language:Konkani` returns **44 items total**, of which
~25 are Wikipedia ZIM dumps, 4 are Wikipedia-derived PDFs, and only **~13–15 are
actual books**. One was fetched and confirmed: `konkanibhashaman0000jbmo` is a
Konkani Bhasha Mandal golden-jubilee publication, full text downloadable,
predominantly Devanagari but mixed with English and Kannada script.

**This corrects `phase1_source_inventory.md`**, which called Internet Archive
"the strongest manual-token evidence available" for Konkani. That was written
before the holdings were counted. It is a real source but a *small* one —
likely 1–2M words, not tens of millions. The correction is recorded in
`phase1_decisions.md` rather than edited silently over the original claim.

### Konkani — K2: Goa Konkani Akademi / Goa government — **needs investigation**

`konkaniakademi.goa.gov.in` exists and publishes a Konkani magazines list and a
publishers list. Access method, volume and licensing are all unverified. This is
the next investigation, not yet a pilot.

---

## 4. Expected token accounting

The rule is `manual / total ≥ 0.20`, i.e. **`total ≤ 5 × manual`**. Adding
downloaded data without adding manual data makes us *less* compliant, so the
downloaded contribution is capped by the manual total, not maximised.

### Marathi

| Bucket | Source | Expected | Type |
|---|---|---|---|
| Manual | M1 Maharashtra GRs | 60–100M (pilot will refine) | manual_ocr |
| Manual | M2 news/literary/gov PDFs | 20–40M | manual_scrape |
| **Manual subtotal** | | **~100M** | |
| Downloaded | IndicCorpV2 `mar_Deva` (CC-0) | capped at 4 × manual | downloaded |
| **Total** | | **~500M at 20% manual** | |

Marathi can reach the 500M target. The binding constraint is purely the ~100M
manual, and M1 alone may supply most of it.

### Konkani

| Bucket | Source | Expected | Type |
|---|---|---|---|
| Manual | K1 archive.org books | 1–2M | manual_ocr |
| Manual | K2+ Akademi / government / publications | unknown — the open question | manual_* |
| Manual | Wikipedia (Devanagari subset, secondary) | ~2.6M | manual_scrape |
| Downloaded | Konkani books corpus (MIT) | ~87M available, **capped by the rule** | downloaded |
| Downloaded | Sangraha `gom` | 10.1M available | downloaded |

The honest position: **if Konkani manual collection reaches 10M tokens, the
total corpus is capped at 50M; if it reaches 20M, the cap is 100M.** The books
corpus alone exceeds either cap, so the question is not "can we find more
Konkani data" but "how much manual data can we legitimately collect, and
therefore how much of the books corpus may we keep".

That is a genuinely defensible Phase 1 result. Sangraha — the largest systematic
Indic corpus effort available — contains only 10.1M Konkani tokens in total,
which is the strongest possible evidence for the shortfall the specification
explicitly permits.

---

## 5. Order of work

**Step 1 — now.** Commit the foundation layer. Run the M1 pilot at 300
documents. *Nothing scales until this reports.*

**Step 2 — from the pilot numbers.** Tune the Devanagari-ratio gate, the langid
threshold and the dedup threshold from measured distributions rather than
guesses. The dedup threshold especially: on synthetic GR-like documents a 0.85
threshold rejected 11 of 12, which is correct for near-identical boilerplate but
would be too aggressive if real GRs differ more than the mock ones did.

**Step 3.** Launch M1 full run in the background with hourly health checks.
Start the K1 Konkani pilot and the K2 investigation in parallel.

**Step 4.** Ingest downloaded corpora, capped by the manual totals then in hand.
Cross-dedup against everything already collected.

**Step 5.** Cross-corpus contamination check between the two languages. Required
result: 0 shared documents, proven by hash.

**Step 6.** One final tokenizer per language, `byte_fallback=True`, vocabulary
chosen on held-out fertility. One recount of everything. Document-level splits.

**Step 7.** Corpus statistics, tokenization examples, token-frequency tables,
README with Drive links, final Phase 1 report.

---

## 6. Where each thing runs

| Work | Where | Why |
|---|---|---|
| Collection, preprocessing, dedup, statistics, tokenizer training | **Your Mac (Terminal)** | Network-bound, not compute-bound. A GPU does nothing for an HTTP fetch. |
| Script development and offline testing | Assistant's cloud container | Cannot reach archive.org or huggingface.co through its proxy; used for logic verification only. |
| Model pretraining (Phase 2) | Colab / Kaggle GPU | The only genuinely GPU-bound stage. |

All scripts take paths relative to the repository root and use only
`requests` plus the standard library, so the same file runs unchanged on macOS
locally or in a Colab cell.
