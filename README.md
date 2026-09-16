# Marathi and Konkani language models

**Model H (higher-resource):** Marathi  ·  **Model L (lower-resource):** Konkani (Devanagari)

Two completely independent decoder-only Transformer language models built from
scratch. Separate corpus, separate tokenizer, separate vocabulary, separate
weights per language — no data, vocabulary or checkpoint is shared between them.

**Branch:** `phase-3`

---

## Phase 3 results

Each model was finetuned on its own synthetic comparative-reasoning set — 8,000
training items, 1,000 test — starting from its own pretrained checkpoint, with
the tokenizer and vocabulary held fixed.

| | Marathi (H) | Konkani (L) |
|---|---:|---:|
| reasoning accuracy, pretrained | 0.00% | 0.00% |
| reasoning accuracy, finetuned | 19.60% | 28.20% |
| format compliance, pretrained → finetuned | 0% → **99.60%** | 0% → **99.90%** |
| perplexity cost of finetuning | 8.55 → 9.72 (×1.14) | 26.14 → 30.85 (×1.18) |
| accuracy excluding the degenerate `equality` family | 6.41% | 16.90% |
| the same, scored without penalising misspelt names | 7.69% | **28.21%** |
| uniform chance floor, that subset | 24.72% | 24.72% |

**Finetuning taught both models the output format completely and the task
partially at best.** Every test item uses entities held out of training, so these
are generalisation numbers, not in-distribution ones.

The two models fail differently, which is the main Phase 3 finding. Marathi
answers with names that appear **only in training** — राम 273 times out of 996 —
and those can never be correct on a test item. Konkani almost never does that;
instead 30.1% of its answers are near-misses of an entity that really is in the
prompt (मीर for मीरा, 185 times). Scored without punishing the spelling, Konkani
clears the chance floor at p = 0.009 while Marathi stays far below it.

Finetuning at 5e-6 also left attention essentially untouched — mean entropy over
all 56 heads moved +0.050 bits for Marathi and +0.001 for Konkani — so the change
is in the output distribution rather than in the computation.

Full analysis, including the learning-rate calibration that this phase turned on,
is in [`report/phase3_report.md`](report/phase3_report.md).

---

## Phase 2 results

Both models: 24,892,356 parameters, trained on 499,908,608 tokens each — equal to
the token, so differences below come from the data rather than from training
budget.

| | Marathi (H) | Konkani (L) |
|---|---:|---:|
| test perplexity | **8.62** | **26.56** |
| test bits per byte | **0.4728** | **0.7086** |
| BLEU-4 (best setting) | 8.03 | 0.00 |
| chrF (best setting) | 27.82 | 21.02 |
| ROUGE-L (best setting) | 13.93 | 5.53 |
| greedy 4-gram repetition | 0.629 | 0.875 |
| training time (T4) | 3.57 h | 3.41 h |

Perplexity says the gap is 3.08×; bits per byte says 1.50×. Both are correct —
perplexity is exponential in the loss and BPB is linear — and BPB is the figure
that survives the tokenizers differing. Konkani's BLEU is exactly zero because it
produced no trigram matching any reference continuation, which is reported rather
than smoothed away.

Architecture: `d_model` 512, 7 layers, 8 heads, FFN 2048, context 512, pre-norm,
untied output head, learned absolute positions.

Full analysis, including the attention study and the resource-level comparison,
is in [`report/phase2_report.md`](report/phase2_report.md).

---

## Phase 1 statistics

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
| fertility, held-out (tokens/word) | 2.6301 | 2.5279 |
| fertility, corpus-wide (train tokens / train words) | 2.7278 | 2.5276 |
| average characters per token (held-out) | 2.6416 | 2.5583 |
| unknown-token rate | **0.000000%** | **0.000000%** |
| documents leaked between splits | **0** | **0** |

Both languages meet both requirements.

"Fetched" and "accepted" are different units and are not a funnel: fetched counts
work items a collector requested (a book, a URL, a shard) and includes items later
rejected, while accepted counts documents in the manifests after cleaning has split
multi-document items — 2,576 OCR'd Konkani volumes become 53,843 documents. The
word and token rows are the funnel. Full derivation in
[`report/phase1_pipeline_accounting.md`](report/phase1_pipeline_accounting.md).

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

### Phase 2 — pretrained checkpoints

In the same folder, under `phase2_checkpoints/`:

| file | contents |
|---|---|
| `marathi_pretrain_best.pt` | Model H weights, optimizer state, GradScaler state, step, best validation loss, model config, training config |
| `konkani_pretrain_best.pt` | Model L, same format |

Each checkpoint is 298,851,087 bytes (~285 MB) and carries everything needed to
resume training or to reload the model for evaluation: `model`, `optimizer`,
`scaler`, `step`, `best_val`, `model_config` and `train_config`. `tools/evaluate.py` rebuilds the
architecture from the `model_config` stored inside the checkpoint rather than
from a separate file, so a checkpoint cannot be loaded into a mismatched model.

### Phase 3 — finetuned checkpoints and reasoning data

In the same folder, under `phase3_checkpoints/`:

**https://drive.google.com/drive/folders/1En76luPAHhj75O9OKBdsWCAWfkQd7erO?usp=drive_link**

| file | contents |
|---|---|
| `marathi_finetune_final_best.pt` | Model H after reasoning finetuning — answer-only target, lr 5e-6, 6 epochs, 8,000 items |
| `konkani_finetune_final_best.pt` | Model L, same recipe |
| `reasoning_data.tar.gz` | the generated reasoning sets, `train`/`val`/`test` for both languages |

The finetuned checkpoints use the same payload as the pretrained ones — `model`,
`optimizer`, `scaler`, `step`, `best_val`, `model_config`, `train_config` — plus
`pretrained_from` and `phase: "finetune"`, so which pretrained model produced
them is recoverable from the file alone.

The reasoning data is on Drive rather than in git because `*/data/` is
gitignored. It is fully reproducible without it: `tools/make_reasoning_data.py`
regenerates both sets from a fixed seed, and the entity pools, pattern splits and
leakage counts are committed in
`report/phase3_reasoning_data_{marathi,konkani}.json`.

Checkpoints are not committed to git — the specification requires large binary
artifacts to go to Drive.

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
in full in D-043. Six candidate vocabularies (2,000 / 2,500 / 3,000 / 4,000 / 5,000
/ 10,000) were trained on the Konkani corpus and fertility, characters per token
and unknown-token rate measured on 5,000 held-out documents, which is the
selection procedure the specification asks for. The measured table is in
`report/phase1_tokenizer_sweep_konkani.json`. Unknown-token rate turned out to be
uninformative: SentencePiece is trained with `byte_fallback=True`, so it is
0.000000% at every vocabulary size by construction.

The deciding criterion was the ~25M parameter budget. Embedding and unembedding
cost `2 x vocab x d_model`, so at d_model 512 a 10,000 vocabulary spends 41% of
the model on two lookup tables against 10% at 2,500, a difference of roughly 7.7M
parameters or about two transformer layers.

Two qualifications, both in D-043. Weight tying would halve that cost and make a
10,000 vocabulary affordable at 20% of the budget, so the parameter argument
rules out an untied large vocabulary rather than a large vocabulary as such.
And because token count is fertility times word count, a smaller vocabulary
raises the reported token total without changing the data: on an identical
corpus of 266,211,363 words, the Konkani train split measures 506M tokens at
vocabulary 2,500 and projects to 435M at 5,000 and 387M at 10,000. Word counts are reported alongside token counts throughout
for that reason.

The measured cost of the choice: fertility rises from 2.1703 at vocabulary 5,000
to 2.5148 at 2,500, whole-word coverage falls from 40.6% to 30.6%, average
characters per token falls from 2.9814 to 2.5730, and every training sequence is
about 16% longer for the same text.

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
common/                       shared pipeline and model code
  textnorm.py                 Unicode NFC normalization (preserves ZWJ/ZWNJ)
  scriptid.py                 script profiling + Marathi/Konkani language ID
  dedup.py                    SHA-256 exact + MinHash/LSH near-duplicate detection
  manifest.py                 per-document provenance; manual/downloaded/synthetic typing
  checkpoint.py               atomic, resumable job state
  newscrawl.py                shared sitemap-driven crawler
  data.py                     memory-mapped packed-token reader and samplers
  metrics.py                  BLEU-4, chrF, ROUGE-L, diversity, bits-per-byte
  model/
    config.py                 ModelConfig + analytic parameter count
    attention.py              multi-head causal self-attention
    lm.py                     feed-forward, transformer block, DecoderLM, generate()
marathi/                      (konkani/ has the same shape)
  scripts/                    collection + ingest scripts
  tokenizer/                  final SentencePiece model + vocabulary
  configs/model_config.json   architecture actually trained, written by the run
  model/                      pointer to common/model — see marathi/model/README.md
  train/                      how this model was pretrained, and where its logs are
  eval/                       how it was evaluated, and where its results are
  data/                       corpora (gitignored — see Drive)
tools/                        pipeline, training, evaluation and verification scripts
report/                       every table, figure and analysis, both phases
  figures/                    all figures, each with title, axis labels and legend
  training_logs/              per-step training CSVs for both models
  archive/                    superseded planning documents, kept for history
```

The Transformer is implemented once in `common/model/` rather than duplicated
under each language. The two models are architecturally identical and differ only
in vocabulary size and weights; two copies of the file could drift, and a drift
between them would invalidate the claim that the only difference between Model H
and Model L is the data. Each language's `model/`, `train/` and `eval/` directory
carries a README mapping that language's artifacts to where they live.

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
# 1. Sweep for evidence. --sweep-only does NOT replace the final tokenizer.
python3 tools/build_tokenizer.py --language marathi \
  --vocab-sizes 2000,2500,3000,4000,5000,10000 --sweep-only
python3 tools/build_tokenizer.py --language konkani \
  --vocab-sizes 2000,2500,3000,4000,5000,10000 --sweep-only

# 2. Build the deliverable at the vocabulary chosen in D-043. This is a separate
# command because the script's fertility-tolerance rule selects a LARGER
# vocabulary; 2,500 was chosen on the ~25M parameter budget, which the script
# does not model.
python3 tools/build_tokenizer.py --language marathi --vocab-sizes 2500
python3 tools/build_tokenizer.py --language konkani --vocab-sizes 2500
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

### 6. Reasoning finetuning and evaluation (Phase 3)

```bash
# Generate the synthetic reasoning set. Fixed seed; entity pools and the
# held-out relation pattern are disjoint by construction.
python3 tools/make_reasoning_data.py --language marathi
python3 tools/make_reasoning_data.py --language konkani

# Finetune from that language's own pretrained checkpoint. The learning rate is
# the one the Phase 3 grid selected; 1e-4 destroys the model (see D-054).
python3 tools/finetune.py --language marathi \
  --pretrained marathi/model/marathi_pretrain_best.pt \
  --n-train 8000 --epochs 6 --lr 5e-6 --tag final

# Evaluate against the pretrained checkpoint, with the forgetting check
python3 tools/evaluate_reasoning.py --language marathi \
  --checkpoint marathi/model/finetune_final_best.pt \
  --compare-to marathi/model/marathi_pretrain_best.pt --lm-windows 256

# Classify the errors and compute the lenient score (D-060)
python3 tools/analyse_errors.py --language marathi

# Attention, pretrained against finetuned
python3 tools/attention_analysis.py --language marathi --split test \
  --checkpoint marathi/model/finetune_final_best.pt \
  --tokenizer marathi/tokenizer/marathi_bpe.model
```

Every Phase 3 experiment also runs unattended from
[`report/phase3_final.ipynb`](report/phase3_final.ipynb) as a Kaggle batch
commit. It takes no input while running.

## Reports

| file | contents |
|---|---|
| `report/phase3_report.md` | **the Phase 3 report — start here** |
| `report/phase3_decisions.md` | Phase 3 decisions and corrections (D-050…D-060) |
| `report/phase3_final.ipynb` | the run that produced every Phase 3 number |
| `report/phase2_report.md` | the Phase 2 report |
| `report/phase2_decisions.md` | Phase 2 architecture decisions (D-044…D-049) |
| `report/phase2_plan.md` | Phase 2 architecture, compute and schedule plan |
| `report/phase2_kaggle_runbook.md` | how the pretraining runs were executed |
| `report/phase1_report.md` | the Phase 1 report |
| `report/phase1_corpus_stats_marathi.md` | Marathi dataset statistics |
| `report/phase1_corpus_stats_konkani.md` | Konkani dataset statistics |
| `report/phase1_pipeline_accounting.md` | stage-by-stage accounting, raw → cleaned → tokens |
| `report/phase1_source_inventory.md` | every source, accepted and rejected, with reasons |
| `report/phase1_konkani_source_discovery.md` | systematic Konkani source discovery + probe |
| `report/phase1_konkani_coverage.md` | Konkani source exhaustion: what exists, what was empty |
| `report/phase1_konkani_mt.md` | synthetic/MT justification against the TAs' three conditions |
| `report/phase1_konkani_overlap_check.md` | manual vs downloaded source independence |
| `report/phase1_decisions.md` | Phase 1 decisions and corrections (D-001…D-043) |
| `report/phase1_work_log.md` | chronological record of the build |
| `report/archive/` | superseded planning documents, kept for history |
| `report/figures/` | all figures, each with title, axis labels and legend |
