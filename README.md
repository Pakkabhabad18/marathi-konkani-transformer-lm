# Language Models and Agents — Individual Project, Phase 1

**Model H (higher-resource):** Marathi  ·  **Model L (lower-resource):** Konkani (Devanagari)

Two completely independent decoder-only Transformer language models built from
scratch. Separate corpus, separate tokenizer, separate vocabulary, separate
weights per language — no data, vocabulary or checkpoint is shared between them.

**Branch:** `phase-1`

---

## Final Phase 1 statistics

| | Marathi (Model H) | Konkani (Model L) |
|---|---:|---:|
| documents fetched | 2,237,944 | 372,836 |
| documents accepted | 2,887,867 | 323,112 |
| words accepted | 389,218,163 | 266,211,363 |
| **final training tokens** | **872,024,099** | **506,259,368** |
| **vs ~500M target** | **174.4%  PASS** | **101.3%  PASS** |
| manual training tokens | 475,466,104 | 159,563,967 |
| **manual share (≥20% required)** | **54.5%  PASS** | **31.5%  PASS** |
| tokenizer vocabulary | 2,500 | 2,500 |
| fertility (tokens/word, held-out) | 2.7278 | 2.5279 |
| unknown-token rate | **0.000000%** | **0.000000%** |
| documents leaked between splits | **0** | **0** |

Both languages meet both requirements.

### Konkani corpus composition by provenance

Konkani is the low-resource language, so where its text came from matters:

| | words | share |
|---|---:|---:|
| **manual** (OCR'd books, self-crawled news, self-collected Wikipedia) | 64,435,242 | 24.2% |
| **downloaded** (real, human-written or human-translated) | 116,071,660 | 43.6% |
| **synthetic** (machine-translated / LLM-generated) | 85,704,461 | 32.2% |

**Real text (manual + downloaded) is 67.8% of the Konkani corpus.** Synthetic
data was authorised by the TAs on 18 Aug 2026 as a last resort and is reported
separately everywhere — `CollectionType.MACHINE_TRANSLATED` is a distinct
member whose `is_manual` is `False` and which is never folded into the
downloaded figure. Full justification, including the ten repositories that were
searched and found to contain no Konkani, is in
[`report/phase1_konkani_mt.md`](report/phase1_konkani_mt.md).

### Splits

| language | split | documents | words |
|---|---|---:|---:|
| Marathi | train | 2,058,209 | 319,677,665 |
| Marathi | val | 21,001 | 3,212,666 |
| Marathi | test | 21,001 | 3,202,861 |
| Konkani | train | 261,186 | 200,293,343 |
| Konkani | val | 2,655 | 2,042,816 |
| Konkani | test | 2,655 | 2,031,644 |

Document-level, source-stratified, fixed seed (`20260819`), verified for zero
content-hash leakage between splits.

### Corpus independence

`tools/cross_corpus_check.py` on 50,000 sampled documents per language:

| check | result |
|---|---|
| exact overlap | **0 shared documents** |
| near-duplicate overlap (Jaccard ≥ 0.8) | **0 pairs** |
| language purity | **0.000%** cross-language contamination in each corpus |

---

## Google Drive — datasets and artifacts

**https://drive.google.com/drive/folders/1lUSriyp7_tltkmHCgINon175xnFp2lp-?usp=sharing**

Sharing is *Anyone with the link → Viewer*; no access request is needed.

| archive | contents |
|---|---|
| `marathi_raw.tar.gz` | Marathi **raw** collected shards (manual + processed) |
| `marathi_cleaned_splits.tar.gz` | Marathi **cleaned** train / val / test |
| `konkani_manual.tar.gz` | Konkani **raw** manually-collected shards |
| `konkani_processed.tar.gz` | Konkani **raw** downloaded-corpus shards |
| `konkani_synthetic.tar.gz` | Konkani **raw** machine-translated / LLM-generated shards |
| `konkani_cleaned_splits.tar.gz` | Konkani **cleaned** train / val / test |
| `manifests.tar.gz` | per-document provenance manifests, both languages |
| `tokenizers.tar.gz` | final SentencePiece models + vocabularies |

Both **raw and cleaned** data are provided, as required. Synthetic data is in
its own archive so it can be inspected — or excluded — independently.

---

## Design notes

**Manual vs downloaded vs synthetic** is a typed choice at the point of
collection (`common/manifest.py::CollectionType`), never a label applied
afterwards. Following TA guidance — *"anything that involves you getting data,
processing, cleaning and organizing is considered Manual; anything already
organized on HuggingFace which is then used is not"* — self-scraped and OCR'd
sources are manual, prepared corpora are downloaded, and model-generated text
is `MACHINE_TRANSLATED`.

**Vocabulary size 2,500.** The specification recommends a vocabulary "in the
tens of thousands per model", so this is a deliberate deviation and it is argued
in full in D-043. Vocabularies of 2,000 / 2,500 / 3,000 / 4,000 / 5,000 / 6,000 /
8,000 / 10,000 were trained per language and fertility and unknown-token rate
measured on held-out text, which is the selection procedure the specification
asks for. Unknown-token rate turned out to be uninformative: SentencePiece is
trained with `byte_fallback=True`, so it is 0.000000% at every vocabulary size by
construction.

The deciding criterion was the ~25M parameter budget. Embedding and unembedding
cost `2 x vocab x d_model`, so at d_model 512 a 10,000 vocabulary spends 41% of
the model on two lookup tables against 10% at 2,500, a difference of roughly 7.7M
parameters or about two transformer layers.

Two qualifications, both in D-043. Weight tying would halve that cost and make a
10,000 vocabulary affordable at 20% of the budget, so the parameter argument
rules out an untied large vocabulary rather than a large vocabulary as such.
And because token count is fertility times word count, a smaller vocabulary
raises the reported token total without changing the data: Konkani measures
506M tokens at vocabulary 2,500 and 437M at 5,000 on an identical corpus of
266,211,363 words. Word counts are reported alongside token counts throughout
for that reason.

The measured cost of the choice: fertility rises from 2.1836 at 5,000 to 2.5279
at 2,500, whole-word token coverage falls from 39.4% to 31.0%, and every training
sequence is about 16% longer for the same text.

**Deduplication** is SHA-256 exact over every document, plus MinHash/LSH
near-duplicate detection with banding tuned to the 0.85 decision threshold.

**Language identification** between Marathi and Konkani uses a closed-class
function-word discriminator (आणि/आनी, आहे/आसा, मी/हांव), calibrated against
known-language reference populations. This is load-bearing: it established that
`ai4bharat/IndicCorpV2`'s 533 MB `gom.txt`, labelled Goan Konkani, is **~84%
Marathi** (D-035).

**Unicode normalization** is NFC and deliberately preserves ZWJ (U+200D) and
ZWNJ (U+200C), which are meaningful in Devanagari.

**AI tool usage.** Claude (Anthropic) was used as a coding assistant for
drafting collector and tooling scripts, for reviewing them, and for drafting
prose in the reports. It was not used to produce any corpus text: every word in
both corpora comes from the sources listed in
`report/phase1_source_inventory.md`, and the machine-translated portion of the
Konkani corpus was generated by IndicTrans2, which is documented separately and
labelled `MACHINE_TRANSLATED` in every manifest.

Verification of assisted code was by measurement rather than by reading. Each
collector was run with `--inspect` or a small `--limit` first and its output
counted before a full run; every statistic in this README is produced by a script
in `tools/` from artifacts on disk and can be regenerated with the commands in
the Reproduction section. Several assisted suggestions were wrong and were caught
this way, and the diagnoses are recorded rather than removed: D-034 (blank lines
treated as document boundaries, which silently discarded 821,054 fragments),
D-037 (a proposed NLLB-200 fallback that has no Konkani token and would have
generated a different language without erroring), and D-041 and D-042 (a library
version incompatibility and a device choice that ran 45x slower than the
alternative). `report/phase1_decisions.md` records all 43 decisions and
corrections in the same form: what was believed, what was measured, what
changed.

---

## Repository layout

```
README.md                     this file
common/                       shared pipeline modules
  textnorm.py                 Unicode NFC normalization (preserves ZWJ/ZWNJ)
  scriptid.py                 script profiling + Marathi/Konkani language ID
  dedup.py                    SHA-256 exact + MinHash/LSH near-duplicate detection
  manifest.py                 per-document provenance; manual/downloaded/synthetic typing
  checkpoint.py               atomic, resumable job state
  newscrawl.py                shared sitemap-driven crawler
marathi/
  scripts/                    collection + ingest scripts
  tokenizer/                  final SentencePiece model + vocabulary
  data/                       corpora (gitignored — see Drive)
konkani/
  scripts/
  tokenizer/
  data/                       manual/ processed/ synthetic/ splits/ manifests/
tools/                        pipeline stages and verification
report/                       all Phase 1 tables, figures and analysis
```

---

## Reproduction

```bash
pip install -r requirements.txt
```

### 1. Collect

```bash
# Marathi
python3 marathi/scripts/collect_archive_gr.py --workers 12
python3 marathi/scripts/collect_news.py --workers 8
python3 marathi/scripts/ingest_indiccorp.py --max-words <N>

# Konkani — manual
python3 konkani/scripts/discover_sources.py --sample 120     # probe only
python3 konkani/scripts/collect_archive_books.py
python3 konkani/scripts/ingest_wikipedia_manual.py

# Konkani — downloaded (real)
python3 konkani/scripts/ingest_books_corpus.py
python3 konkani/scripts/ingest_hf_konkani.py --source books_corpus_v1
python3 konkani/scripts/ingest_hf_konkani.py --source sangraha_gom
python3 konkani/scripts/ingest_hf_konkani.py --source madlad_clean
python3 konkani/scripts/ingest_hf_konkani.py --source madlad_noisy
python3 konkani/scripts/ingest_hf_konkani.py --source glotcc
python3 konkani/scripts/ingest_hf_konkani.py --source roundtripocr
python3 konkani/scripts/ingest_hf_konkani.py --source konkani_raw
python3 konkani/scripts/ingest_indiccorp_konkani.py
python3 konkani/scripts/ingest_bpcc_konkani.py

# Konkani — synthetic (TA-authorised 18 Aug 2026; see report/phase1_konkani_mt.md)
python3 konkani/scripts/ingest_hf_konkani.py --source konkani_raw_translated
python3 konkani/scripts/ingest_hf_konkani.py --source instruct_100k
python3 konkani/scripts/ingest_hf_bulk_konkani.py
python3 konkani/scripts/generate_mt_konkani.py --device cpu --max-hours 10
```

Every collector is checkpointed and resumable; re-running skips completed work.
`tools/status.py` reports progress at any time.

The gated HuggingFace repositories (`ai4bharat/BPCC`,
`ai4bharat/indictrans2-indic-indic-dist-320M`) require `hf auth login` with a
**Read** token and acceptance of each repository's terms.

`generate_mt_konkani.py` needs `transformers==4.46.3` in an isolated
environment; newer releases pass a `Cache` object that IndicTrans2's vendored
modeling code cannot index (D-041).

### 2. Verify sources

```bash
python3 tools/source_overlap_check.py --sample-archive 120
python3 tools/verify_gom_langid.py --sample 1500
```

### 3. Tokenizers

```bash
python3 tools/build_tokenizer.py --language marathi \
  --vocab-sizes 2000,2500,3000,4000,5000 --fertility-tolerance 0.15
python3 tools/build_tokenizer.py --language konkani \
  --vocab-sizes 2000,2500,3000,4000,5000 --fertility-tolerance 0.15
```

### 4. Splits

```bash
python3 tools/make_splits.py --language marathi --exact-only
python3 tools/make_splits.py --language konkani --exact-only
```

### 5. Statistics and figures

```bash
python3 tools/corpus_stats.py --language marathi --markdown
python3 tools/corpus_stats.py --language konkani --markdown
python3 tools/pipeline_accounting.py --markdown
python3 tools/cross_corpus_check.py --sample 50000
python3 tools/make_plots.py
```

---

## Reports

| file | contents |
|---|---|
| `report/phase1_corpus_stats_marathi.md` | Marathi dataset statistics |
| `report/phase1_corpus_stats_konkani.md` | Konkani dataset statistics |
| `report/phase1_pipeline_accounting.md` | stage-by-stage accounting, raw → cleaned → tokens |
| `report/phase1_source_inventory.md` | every source, accepted and rejected, with reasons |
| `report/phase1_konkani_source_discovery.md` | systematic Konkani source discovery + probe |
| `report/phase1_konkani_coverage.md` | Konkani source exhaustion: what exists, what was empty |
| `report/phase1_konkani_mt.md` | synthetic/MT justification against the TAs' three conditions |
| `report/phase1_konkani_overlap_check.md` | manual vs downloaded source independence |
| `report/phase1_decisions.md` | every design decision and correction (D-001…D-042) |
| `report/phase1_viva_log.md` | full pipeline walkthrough |
| `report/figures/` | all figures, each with title, axis labels and legend |
