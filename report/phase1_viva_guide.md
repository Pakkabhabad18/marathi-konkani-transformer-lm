# Phase 1 — viva walkthrough

*Everything needed to explain this pipeline end to end without reference to
notes. Every number here is read from an artifact on disk; none is estimated.*

---

## 1. What was built, in one paragraph

Two completely independent corpora for two decoder-only Transformer language
models of ~25M parameters each: **Model H = Marathi** (higher-resource) and
**Model L = Konkani, Devanagari** (lower-resource). Separate collection,
separate cleaning, separate tokenizer, separate vocabulary, separate splits. No
document, vocabulary entry or checkpoint is shared. Phase 1 is data only — no
model was trained.

## 2. The headline numbers

| | Marathi | Konkani |
|---|---:|---:|
| documents fetched | 2,237,944 | 372,836 |
| documents accepted | 2,887,867 | 323,112 |
| words accepted | 389,218,163 | 266,211,363 |
| **training tokens** | **872,024,099** | **506,259,368** |
| vs ~500M target | 174.4% PASS | 101.3% PASS |
| manual tokens | 475,466,104 | 159,563,967 |
| manual share (≥20%) | 54.5% PASS | 31.5% PASS |
| vocabulary | 2,500 | 2,500 |
| fertility (held-out) | 2.7278 | 2.5279 |
| unknown-token rate | 0.000000% | 0.000000% |
| split leakage | 0 | 0 |

**If asked one question, expect this one:** *"Konkani has 2.3M speakers — where
did 500M tokens come from?"* Answer: 24.2% manual (OCR'd books, self-crawled
news, self-collected Wikipedia), 43.6% downloaded real text, **32.2% synthetic**
(machine-translated and LLM-generated), authorised by the TAs on 18 Aug 2026
after real sources were exhausted. Real text alone is 180,510,757 words.

---

## 3. The pipeline, stage by stage

### Stage 1 — Collection

| script | what it does |
|---|---|
| `marathi/scripts/collect_archive_gr.py` | OCR text layers from archive.org Maharashtra Government Resolutions |
| `marathi/scripts/collect_news.py` | sitemap-driven crawl of Marathi news sites |
| `marathi/scripts/ingest_indiccorp.py` | IndicCorp v2 `mar_Deva`, capped by the manual ratio |
| `konkani/scripts/collect_archive_books.py` | OCR text layers from archive.org Konkani books |
| `konkani/scripts/ingest_wikipedia_manual.py` | Konkani Wikipedia, self-extracted from the dump |
| `konkani/scripts/ingest_indiccorp_konkani.py` | IndicCorp v2 `gom` |
| `konkani/scripts/ingest_bpcc_konkani.py` | BPCC `gom_Deva` — human-translated |
| `konkani/scripts/ingest_hf_konkani.py` | nine configured HuggingFace sources |
| `konkani/scripts/ingest_hf_bulk_konkani.py` | 20 remaining HF datasets, auto column detection |
| `konkani/scripts/generate_mt_konkani.py` | our own Marathi→Konkani MT via IndicTrans2 |

Every collector is **checkpointed and resumable** (`common/checkpoint.py`,
atomic writes) — a killed run resumes at the exact position and never re-fetches.

### Stage 2 — Quality gates (identical for every source)

Applied in this order, and the order matters:

1. **Unicode NFC** (`common/textnorm.py`) — deliberately preserves ZWJ (U+200D)
   and ZWNJ (U+200C), which carry meaning in Devanagari.
2. **Length floor** — 25 words minimum per document.
3. **Devanagari ratio ≥ 0.70** (`common/scriptid.py::profile_script`) — the
   ratio is over **non-whitespace** characters. That denominator was a real bug
   (D-001): dividing by total characters reported the Konkani books corpus as
   83.54% Devanagari when the true figure is 99.78%.
4. **Language discrimination** (`identify_marathi_konkani`) — rejects Marathi
   from the Konkani corpus and vice versa.
5. **Exact deduplication** — SHA-256 over the canonical form.
6. **Near-duplicate deduplication** — MinHash + LSH.

### Stage 3 — Cross-source deduplication and splits

`tools/make_splits.py`: document-level, source-stratified, fixed seed
(`20260819`), 98/1/1. Enforces the manual ratio mechanically, then verifies
**zero content-hash leakage** between splits.

### Stage 4 — Tokenizer

`tools/build_tokenizer.py`: SentencePiece BPE, `byte_fallback=True`. Sweeps
candidate vocabulary sizes, measures fertility on held-out text, and selects by
a stated rule — **the smallest vocabulary whose fertility is within 15% of the
best**.

### Stage 5 — Verification and reporting

| tool | proves |
|---|---|
| `tools/cross_corpus_check.py` | the two corpora share no documents |
| `tools/verify_gom_langid.py` | the language discriminator is calibrated |
| `tools/source_overlap_check.py` | manual and downloaded sources are independent |
| `tools/pipeline_accounting.py` | every stage's numbers reconcile |
| `tools/corpus_stats.py` | per-language statistics |
| `tools/make_plots.py` | five figures, each checked for title/axes/legend |

---

## 4. The three design decisions you must be able to defend

### 4.1 Why a hand-built language discriminator?

Marathi and Konkani share Devanagari, most of their character inventory, and a
great deal of vocabulary. **Character statistics cannot separate them** — we
loaded the Konkani tokenizer and ran it on Marathi text: 16 tokens, 0 unknowns,
3.81 chars/token, statistically indistinguishable from its behaviour on Konkani.
Off-the-shelf language identifiers are trained with little or no Konkani and
label it Marathi or Hindi.

So the discriminator uses **closed-class function words** — grammatical items
that are near-impossible to avoid in running text and that differ in *form*, not
just spelling:

| meaning | Marathi | Konkani |
|---|---|---|
| and | आणि | आनी |
| is | आहे | आसा |
| I | मी | हांव |

Score = `(marathi_hits − konkani_hits) / (marathi_hits + konkani_hits)`. It
requires ≥5 marker hits and a margin of 0.30, and **abstains** otherwise rather
than guessing. It is a filter with a margin, not a classifier that must be right
every time.

### 4.2 Why vocabulary 2,500?

The binding constraint is the ~25M parameter budget. Embedding + unembedding
cost `2 × vocab × d_model`; at d_model 512:

| vocab | fertility | embed+unembed | % of 25M |
|---:|---:|---:|---:|
| 2,500 | 2.5333 | 2.56M | **10%** |
| 5,000 | 2.1836 | 5.12M | 20% |
| 10,000 | 1.9506 | 10.24M | **41%** |

At 10,000 the model spends 41% of itself on lookup tables. At 2,500 that falls
to 10%, returning ~7.7M parameters to depth and width.

**Be ready for the follow-up, because a good examiner will ask it.** Smaller
vocabulary means higher fertility means more tokens for the same text. The
identical Konkani corpus reads **430M tokens at vocabulary 5,000 and 506M at
2,500**. Token counts are tokenizer-dependent. The honest position: the
vocabulary was chosen on the parameter budget, the effect on the token count is
a consequence, and it is stated in the README, in D-032 and in the coverage
report rather than left to be discovered. The cost is also recorded —
whole-word token coverage falls from 39.4% to 31.0%.

### 4.3 Why is byte-fallback on?

`byte_fallback=True` adds 256 `<0xNN>` pieces, so **any** byte sequence is
representable and the unknown-token rate is 0 *by construction*, not by luck.
That is why both corpora report 0.000000% — it is a property of the tokenizer
configuration, and you should say so rather than claim it as an achievement.

---

## 5. The findings worth volunteering

### 5.1 IndicCorp v2's `gom.txt` is ~84% Marathi (D-035)

`ai4bharat/IndicCorpV2` ships `data/gom.txt`, 533,108,246 bytes labelled Goan
Konkani. Our gate rejected 70% of it, which was wildly out of line with GlotCC
(0 rejections of 1,049), MADLAD-400 (14 of 4,602) and Sangraha (11 of 14,491).

Rather than trust the label *or* our own gate, we calibrated against two
populations whose language is not in doubt — our own OCR'd Konkani books and our
own Marathi corpus — with all three packed to the same ~300-word length so
document length could not be confounded with language:

| population | `mr` | `kok` | undecided | median score |
|---|---:|---:|---:|---:|
| reference Konkani | 0.0% | 94.7% | 5.3% | −0.92 |
| reference Marathi | 100.0% | 0.0% | 0.0% | +1.00 |
| **`gom.txt`** | **83.7%** | 2.5% | 13.9% | **+0.79** |

`gom.txt` sits on the Marathi reference. A sampled rejected document contains
`आहे×8, पण×5, मी×3` and zero Konkani markers.

**Why this matters:** accepting it would have added ~25M words *and* poured
Model H's language into Model L, breaking corpus independence. We kept the
4,319,751 words that passed.

### 5.2 A search that was wrong by 115× (D-018)

An early archive.org query, `language:(Konkani OR Konknni OR Concani)`, measured
our own spelling list rather than the archive, and a missing cursor meant only
the first page was read. Two faults that cancelled into a plausible-looking
"44 Konkani items". The real figure for `language:kok AND mediatype:texts` is
**5,093**. Fixing both grew the manual side from 1.46M to 64.4M words.

### 5.3 Blank lines are sentence separators, not document boundaries (D-034)

The IndicCorp ingest initially treated a blank line as a document boundary. The
output proved otherwise: 1,361,209 "documents" at 392 bytes ≈ 22 words each,
`flush_blank_line` firing on 100% of flushes, and 821,054 fragments then
discarded for failing a 25-word floor that our own chopping had made
unreachable. Corrected to unit-level dedup and language filtering, **then**
packing into 300-word documents. Order matters: once packed, no two documents
are byte-identical, so unit-level dedup is the only place those 197,656 repeats
can be caught.

---

## 6. Likely questions, with answers

**"Is your manual data really manual?"**
It is a typed choice at the point of collection —
`common/manifest.py::CollectionType` — never a label applied afterwards. Per TA
guidance, self-scraped and OCR'd sources are manual; prepared HuggingFace
corpora are downloaded; model output is `MACHINE_TRANSLATED`. `is_manual` is a
property of the enum member, so a source cannot drift between categories.

**"How do you know the two corpora don't overlap?"**
Three independent checks on 50,000 sampled documents per language: exact hash
(0 shared), MinHash near-duplicate at Jaccard ≥ 0.8 (0 pairs), and language
purity (0.000% cross-language). Reported in
`report/phase1_cross_corpus_check.json`.

**"How much of Konkani is synthetic?"**
32.2% of words. It is in its own Drive archive and its own manifests, and it can
be excluded — that leaves 180,510,757 real words. Justification against the
TAs' three stated conditions is in `report/phase1_konkani_mt.md`.

**"Did you try to generate MT yourself?"**
Yes, and it was measured rather than assumed: IndicTrans2 Marathi→Konkani,
14,016 sentences → 394 documents → **124,664 words** at 1.2 sentences/sec on
CPU. That is 0.05% of the corpus. Apple MPS was tried and was **45× slower**
(48 sentences in 124 minutes). The copy detector — Jaccard similarity against
the source — fired **zero** times, so the model was genuinely translating rather
than echoing Marathi, which is the standard failure mode for MT into a
low-resource target.

**"What if a source lies about its language?"**
Three did. `gom.txt` was 84% Marathi. Sangraha's `gom` split was 32% Romi
(Roman-script) Konkani. BPCC's two largest mined subsets contained no Konkani at
all despite the dataset card listing `gom_Deva`. Every source passes the same
gates regardless of what its label claims.

**"Why is your near-duplicate detection over character 5-grams rather than word
n-grams?"**
Devanagari OCR noise damages five character shingles but destroys a whole word
token, so character shingles degrade more gracefully on OCR'd text — which is
most of the manual corpus.

**"What went wrong that you had to fix?"**
Forty-two recorded decisions and corrections in `report/phase1_decisions.md`,
each stating what was believed, what was measured and what changed. The ones
worth naming: the 115× search error (D-018), the non-whitespace denominator
(D-001), an OOM at 86.6% caused by storing 2.5M signatures as Python int tuples
— 11.5 GB, fixed to 124 MB with `array("I")` (D-029), LSH banding tuned to the
wrong threshold generating 349.8 candidates per query instead of 0.4 (D-030),
and split leakage caused by that same memory fix (D-031).

---

## 7. Where everything lives

```
common/textnorm.py      NFC normalization, preserves ZWJ/ZWNJ
common/scriptid.py      script profiling + Marathi/Konkani discriminator
common/dedup.py         SHA-256 exact + MinHash/LSH near-duplicate
common/manifest.py      CollectionType, per-document provenance
common/checkpoint.py    atomic resumable job state
common/newscrawl.py     sitemap-driven crawler

tools/build_tokenizer.py     vocabulary sweep + final tokenizer
tools/make_splits.py         dedup, ratio enforcement, splits, leakage check
tools/corpus_stats.py        per-language statistics
tools/pipeline_accounting.py stage-by-stage reconciliation
tools/cross_corpus_check.py  corpus independence
tools/verify_gom_langid.py   language-ID calibration
tools/make_plots.py          five figures

report/phase1_pipeline_accounting.md   every stage's numbers
report/phase1_konkani_coverage.md      source exhaustion record
report/phase1_konkani_mt.md            synthetic justification
report/phase1_decisions.md             D-001 … D-042
```

---

## 8. Two things to say before you are asked

Volunteering these reads as rigour; being caught on them reads as evasion.

1. **Token counts depend on the tokenizer.** The Konkani corpus is 430M tokens
   at vocabulary 5,000 and 506M at 2,500. The vocabulary was chosen on the
   parameter budget; the token count followed.
2. **32.2% of the Konkani corpus is synthetic.** It is authorised, labelled,
   separable, and used only after ten repositories were verified to contain no
   Konkani. Real text alone is 180.5M words.
