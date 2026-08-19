# Language Models and Agents — Individual Project

**Model H (higher-resource):** Marathi  ·  **Model L (lower-resource):** Konkani (Devanagari)

Two completely independent decoder-only Transformer language models built from
scratch. Separate corpus, separate tokenizer, separate vocabulary, separate
weights per language — no data, vocabulary or checkpoint is shared between them.

**Phase 1 branch:** `phase-1`

---

## Google Drive — datasets and artifacts

**https://drive.google.com/drive/folders/1lUSriyp7_tltkmHCgINon175xnFp2lp-?usp=sharing**

Sharing is set to *Anyone with the link → Viewer*, so no access request is needed.

| archive | contents | size |
|---|---|---:|
| `marathi_raw.tar.gz` | Marathi **raw** collected shards (manual + processed) | 1.24 GB |
| `marathi_cleaned_splits.tar.gz` | Marathi **cleaned** train / val / test | 1.11 GB |
| `konkani_manual.tar.gz` | Konkani **raw** manually-collected shards | 217.7 MB |
| `konkani_processed.tar.gz` | Konkani **raw** downloaded-corpus shards | 159.4 MB |
| `konkani_cleaned_splits.tar.gz` | Konkani **cleaned** train / val / test | 316.2 MB |
| `manifests.tar.gz` | per-document provenance manifests, both languages | 184.7 MB |
| `tokenizers.tar.gz` | final SentencePiece models + vocabularies | ~2.5 MB |

Both **raw and cleaned** data are provided, as required.

---

## Final Phase 1 statistics

| | Marathi (Model H) | Konkani (Model L) |
|---|---:|---:|
| documents accepted | 2,887,867 | 96,932 |
| words accepted | 389,218,163 | 111,451,424 |
| **final training tokens** | **647,434,614** | **165,815,092** |
| manual training tokens | 353,009,984 | 115,064,540 |
| **manual share of training tokens** | **54.5%** | **69.4%** |
| vs ~500M target | **129.5%** | 33.2% (justified) |
| tokenizer vocabulary | 10,000 | 10,000 |
| fertility (tokens/word, held-out) | 2.02 | 1.82 |
| unknown-token rate | 0.000000% | 0.000000% |
| documents leaked between splits | **0** | **0** |

Both languages exceed the **≥20% manual** requirement by a wide margin. Marathi
exceeds the ~500M token target; Konkani's shortfall is measured and justified in
[`report/phase1_konkani_shortfall.md`](report/phase1_konkani_shortfall.md).

### Splits

| language | split | documents | words |
|---|---|---:|---:|
| Marathi | train | 2,058,209 | 319,677,665 |
| Marathi | val | 21,001 | 3,212,666 |
| Marathi | test | 21,001 | 3,202,861 |
| Konkani | train | 78,423 | 90,021,485 |
| Konkani | val | 799 | 926,734 |
| Konkani | test | 799 | 921,239 |

Splits are **document-level** and **source-stratified**, with a fixed seed
(`20260819`), and verified for zero content-hash leakage between splits.

---

## Repository layout

```
README.md                     reproduction steps + Google Drive links
common/                       shared pipeline modules
  textnorm.py                 Unicode NFC normalization (preserves ZWJ/ZWNJ)
  scriptid.py                 script profiling + Marathi/Konkani language ID
  dedup.py                    SHA-256 exact + MinHash/LSH near-duplicate detection
  manifest.py                 per-document provenance, manual/downloaded typing
  checkpoint.py               atomic, resumable job state
  newscrawl.py                shared sitemap-driven crawler
marathi/
  scripts/                    collection + ingest scripts
  tokenizer/                  final SentencePiece model + vocabulary
  data/                       corpora (gitignored — see Drive)
konkani/
  scripts/
  tokenizer/
  data/
tools/                        pipeline stages and verification
  build_tokenizer.py          vocabulary sweep + final tokenizer
  make_splits.py              dedup, ratio enforcement, splits, leakage check
  corpus_stats.py             per-language dataset statistics
  pipeline_accounting.py      stage-by-stage accounting (words -> tokens)
  token_budget.py             per-source fertility and token budgeting
  cross_corpus_check.py       proves the two corpora share no documents
  source_overlap_check.py     manual vs downloaded source independence
  status.py                   live collection status
  make_plots.py               all figures
report/                       all Phase 1 tables, figures and analysis
```

---

## Reproduction

```bash
pip install -r requirements.txt
```

### 1. Collect

```bash
python3 marathi/scripts/collect_archive_gr.py --workers 12
python3 marathi/scripts/collect_news.py --workers 8
python3 marathi/scripts/ingest_indiccorp.py --max-words <N>

python3 konkani/scripts/discover_sources.py --sample 120     # probe only
python3 konkani/scripts/collect_archive_books.py
python3 konkani/scripts/ingest_wikipedia_manual.py
python3 konkani/scripts/ingest_books_corpus.py
```

Every collector is checkpointed and resumable; re-running skips work already
done. `tools/status.py` reports progress at any time.

### 2. Verify sources

```bash
python3 tools/source_overlap_check.py --sample-archive 120
```

### 3. Tokenizers

```bash
python3 tools/build_tokenizer.py --language marathi --vocab-sizes 6000,8000,10000
python3 tools/build_tokenizer.py --language konkani --vocab-sizes 6000,8000,10000
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
| `report/phase1_pipeline_accounting.md` | stage-by-stage accounting, raw -> cleaned -> tokens |
| `report/phase1_source_inventory.md` | every source, accepted and rejected, with reasons |
| `report/phase1_konkani_source_discovery.md` | systematic Konkani source discovery + probe |
| `report/phase1_konkani_shortfall.md` | why Konkani cannot reach ~500M |
| `report/phase1_konkani_overlap_check.md` | manual vs downloaded independence |
| `report/phase1_decisions.md` | every design decision and correction (D-001…D-032) |
| `report/phase1_viva_log.md` | full pipeline walkthrough (A-001…A-021) |
| `report/figures/` | all figures, each with title, axis labels and legend |

---

## Design notes

**Manual vs downloaded** is a typed choice at the point of collection
(`common/manifest.py::CollectionType`), never a label applied afterwards.
Following TA guidance — *"anything that involves you getting data, processing,
cleaning and organizing is considered Manual; anything already organized on
HuggingFace which is then used is not"* — self-scraped and OCR'd sources are
manual; prepared HuggingFace corpora are downloaded.

**Vocabulary size 10,000** was chosen by sweeping 6k/8k/10k and measuring
fertility, unknown-token rate, vocabulary utilisation and hapax count on held-out
text. An earlier 48,000 vocabulary was rejected: its embedding and unembedding
matrices alone would consume 147% of the ~25M parameter budget, and 2,779
vocabulary slots held pieces occurring exactly once. Full reasoning in D-032.

**Deduplication** is SHA-256 exact over every document, plus MinHash/LSH
near-duplicate detection with banding tuned to the 0.85 decision threshold.

**Unicode normalization** is NFC and deliberately preserves ZWJ (U+200D) and
ZWNJ (U+200C), which are meaningful in Devanagari.

**AI tool usage.** Claude was used as an assistant for code drafting and review.
Every design decision, correction and measurement is recorded in
`report/phase1_decisions.md` and `report/phase1_viva_log.md`, including the
mistakes found and how they were diagnosed.
