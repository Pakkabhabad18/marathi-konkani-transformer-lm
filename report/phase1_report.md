# Phase 1 — Data Collection and Tokenizer Construction

Model H: Marathi. Model L: Konkani (Devanagari). Submitted on branch `phase-1`,
21 August 2026.

Every figure here comes from a script in `tools/` reading artifacts on disk, and
can be regenerated with the commands in section 12.

## 1. Results

| | Marathi | Konkani |
|---|---:|---:|
| documents accepted | 2,887,867 | 323,112 |
| words accepted | 389,218,163 | 266,211,363 |
| training tokens | 872,024,099 | 506,259,368 |
| manual training tokens | 475,466,104 | 159,563,967 |
| manual share of training tokens | 54.5% | 31.5% |
| vocabulary size | 2,500 | 2,500 |
| unknown-token rate | 0.000000% | 0.000000% |
| documents leaked between splits | 0 | 0 |
| documents shared with the other corpus | 0 | 0 |

Both corpora clear the ~500M token target and the 20% manual floor, and share no
document, no near-duplicate and no vocabulary.

Konkani needed machine-translated and LLM-generated text to reach the target. The
TAs authorised it on 18 August 2026. It is 32.2% of the Konkani corpus; section 9
covers what it is, how it is labelled and how to exclude it.

## 2. Language selection

We took Marathi as Model H. IndicCorp v2 alone ships about 27.8M Marathi
documents under CC-0, and Marathi news publishers keep deep sitemap-indexed
archives that can be crawled and cleaned, so both the manual and the downloaded
side of the requirement were reachable.

Konkani (Devanagari) is Model L, from the permitted list. It is genuinely scarce:
Sangraha, the largest systematic Indic corpus effort, holds 14,491 Konkani rows
against millions for Marathi.

The two languages are closely related, both written in Devanagari, and share a
lot of vocabulary. That made the independence requirement harder, not easier.
Published corpora mislabel one as the other, and a script check cannot tell them
apart, so we had to build a discriminator (section 6). The measured independence
result is in section 11.

## 3. Data collection

Per-source tables are in `phase1_source_inventory.md`. In words:

| | Marathi | Konkani |
|---|---:|---:|
| manual | 177,781,779 (45.7%) | 64,435,242 (24.2%) |
| downloaded, human-authored | 211,436,384 (54.3%) | 116,071,660 (43.6%) |
| synthetic | 0 | 85,704,461 (32.2%) |
| total | 389,218,163 | 266,211,363 |

### 3.1 Manual collection

Manual means text we gathered and cleaned ourselves, following the TA guidance of
14 August: anything involving getting, processing, cleaning and organizing data is
manual; anything already organized on HuggingFace and then used is not. The
collector assigns the category at the point of collection, in
`common/manifest.py::CollectionType`, not as a label afterwards.

Most of it is OCR text from archive.org — 152,312,784 Marathi words and
62,997,759 Konkani words. We fetched items that already had an OCR text layer and
cleaned those. Image-only scans we skipped rather than running a local OCR engine
over them, because we had no way to measure the error rate of uncontrolled OCR on
Devanagari.

The rest is sitemap-driven news crawling (`common/newscrawl.py`). We fetched and
parsed `robots.txt` before any request, honoured the declared crawl-delay, and
sent a descriptive User-Agent. Per-site selectors strip navigation,
related-article blocks and comments before the text is counted.

Before committing to a full crawl we hashed a pilot of about 1,000 pages per site
against the downloaded corpus with `tools/source_overlap_check.py`. IndicCorp v2
is itself built from Marathi news crawls, so scraping the same pages and calling
the result manual collection would have been wrong, and cross-source dedup would
have removed most of it anyway.

### 3.2 Downloaded corpora

Marathi uses one downloaded source, IndicCorp v2 (`mar_Deva`). It is CC-0, the
least restrictive licence among the candidates, and a single source avoids the
heavy overlap between IndicCorp v2 and Sangraha, which come from the same group
and overlapping crawls.

Konkani uses ten, listed in `phase1_source_inventory.md` §3.3. We measured each
one before using it rather than accepting its label, and three did not match what
they claimed. The worst case is section 8.2.

## 4. Cleaning pipeline

Applied per document, in this order, by the collector that produced it:

1. Unicode NFC normalization (`common/textnorm.py`). We preserve ZWJ (U+200D) and
   ZWNJ (U+200C) deliberately — they are meaningful in Devanagari and stripping
   them alters words silently.
2. Whitespace normalization: collapse runs of spaces, normalize line endings,
   repair the line fragmentation typical of OCR output.
3. Script gate: at least 70% Devanagari by non-whitespace character. This is what
   excludes Roman-script (Romi) Konkani — genuinely Konkani, but out of scope
   here (section 5).
4. Length gate: minimum 25 words. Below that there is too little signal for the
   language discriminator to reach a verdict.
5. Language gate: Marathi/Konkani discrimination (section 6).
6. Deduplication, exact then near-duplicate (section 7).

Rejection counts by reason and by source are in the manifests and summarised by
`tools/pipeline_accounting.py`.

Every accepted document carries a manifest row: source name, source URL,
collection method and type, access date, raw and clean character counts, word
count, preprocessing steps applied, script profile, language-ID score, content
hash. So the manual fraction can be audited per document.

## 5. Script decision for Konkani

The corpus is Devanagari only. Konkani is also written in Roman script (Romi) and
in Kannada script; we excluded both.

Devanagari is the official script of Konkani in Goa and is what the largest
available Konkani corpora use, so restricting to it costs little volume. A 25M
parameter model also has limited capacity to spend on two orthographies of one
language competing for the same embedding budget.

The cost shows up in the source table: 4,651 rows, 32% of Sangraha's `gom` split,
were rejected. They are correctly labelled Konkani in a script this corpus does
not cover.

## 6. Language identification

Script cannot separate Marathi from Konkani, and general-purpose language
identifiers treat Konkani inconsistently. Our discriminator,
`common/scriptid.py::identify_marathi_konkani`, uses closed-class function words,
which a text can neither avoid nor borrow:

| function | Marathi | Konkani |
|---|---|---|
| "and" | आणि | आनी |
| "is" | आहे | आसा |
| "I" | मी | हांव |

Below `min_markers = 5` total marker hits the document comes back undecided
rather than assigned. A 22-word fragment usually carries fewer than five markers,
and forcing a verdict on that evidence produces confident errors. The score must
also clear `margin = 0.30` in one direction; between the two thresholds the
discriminator abstains. Of 50,000 sampled Konkani documents, 2,505 are undecided
and none is misassigned.

We calibrated the discriminator against populations of known language rather than
trusting its raw output. That table is in section 8.2.

## 7. Deduplication

Exact duplicates go by SHA-256 over the NFC-normalized, whitespace-collapsed
text, applied within each source and again across sources.

Near-duplicates use MinHash over character 5-gram shingles with 128 permutations,
with LSH banding tuned so the probability curve `P = 1 − (1 − s^r)^b` has its
half-way point near the 0.85 decision threshold.

We chose character 5-grams over word n-grams because both corpora are OCR-heavy.
One OCR error damages five character shingles out of thousands, but destroys a
whole word token and every word n-gram containing it. Word n-grams would have
under-detected duplicates in exactly the material that duplicates most.

We got the order of operations wrong at first. IndicCorp v2 repeats individual
sentences across crawled pages, and deduplicating after packing sentences into
documents catches nothing, because no two packed documents are byte-identical.
Moving dedup to the unit level, before packing, removed 197,656 repeats that
document-level dedup could not see (D-034).

## 8. Two measurements that changed the corpus

### 8.1 A search wrong by a factor of 115

Our first archive.org query for Konkani books returned 44 items, which supported
an argument that manual Konkani collection was not viable. The query was matching
against a locally held spelling list instead of querying the archive index. The
corrected query, `language:kok AND mediatype:texts`, returns 5,093 items. Manual
Konkani went from 1.46M words to 64.4M, and the 20% floor became reachable
(D-018).

### 8.2 A 533 MB file labelled Konkani that is 84% Marathi

`ai4bharat/IndicCorpV2` ships `data/gom.txt` at 533,108,246 bytes, labelled Goan
Konkani. By file size it is larger than every other Konkani source combined.

Our discriminator rejected 70% of it as Marathi, far out of line with every other
source we measured that week. Rather than trust either the label or the gate, we
packed three populations to the same ~300-word document length, so length could
not be confounded with language, and scored them with the same discriminator on
the same day:

| population | labelled `mr` | labelled `kok` | undecided | median score |
|---|---:|---:|---:|---:|
| reference Konkani (our OCR'd books) | 0.0% | 94.7% | 5.3% | −0.92 |
| reference Marathi (our Marathi corpus) | 100.0% | 0.0% | 0.0% | +1.00 |
| IndicCorp v2 `gom.txt` | 83.7% | 2.5% | 13.9% | +0.79 |

`gom.txt` sits on the Marathi reference rather than between the two. A sampled
rejected document contains आहे×8, पण×5, मी×3 and no Konkani markers. The gate was
right: 4,319,751 words kept out of roughly 30M.

Taking the file unfiltered would have added about 25M words to the Konkani total
and put Marathi — the language Model H trains on — into Model L's corpus. Tool:
`tools/verify_gom_langid.py --sample 1500`. Detail in D-035.

## 9. Synthetic data in the Konkani corpus

After searching the repositories listed in `phase1_source_inventory.md` §4 and
finding them empty, real Konkani text reached about 180.5M words, short of the
target. The TAs authorised synthetic and machine-translated data on 18 August 2026
under three conditions; compliance with each is in `phase1_konkani_mt.md`.

Synthetic text is 85,704,461 words, 32.2% of the corpus. Real text — manual plus
downloaded, human-written or human-translated — is 180,506,902 words, 67.8%.

It stays separable at three levels. `CollectionType.MACHINE_TRANSLATED` is its own
enum member rather than a flag on `DOWNLOADED_DATASET`, so `is_manual` is `False`
and it can never count toward the 20% floor, and being distinct it cannot be
folded into the downloaded figure either (D-036). It lives under
`konkani/data/synthetic/` and ships in its own Drive archive. And
`tools/corpus_stats.py` and `tools/pipeline_accounting.py` report it as a third
bucket in every table.

The last 1.46M tokens to the target came from BPCC `gom_Deva`, real
human-translated Konkani, rather than from more synthetic text. That is why the
corpus crosses the target on 67.8% real data.

Our own IndicTrans2 generation run contributed 14,016 sentences, 394 documents,
124,664 words — 0.05% of the corpus. Two failures kept it small: transformers
≥ 4.49 passes a `Cache` object that IndicTrans2's vendored decoder cannot index
(D-041), and Apple MPS ran the model at 0.006 sentences per second against
0.9–1.2 on CPU, a 45× slowdown (D-042).

## 10. Splits and tokenizers

### 10.1 Splits

Document-level, source-stratified, fixed seed `20260819`. Splitting at document
level rather than line level is what keeps two halves of one article out of train
and test.

| language | split | documents | words |
|---|---|---:|---:|
| Marathi | train | 2,058,209 | 319,677,665 |
| Marathi | val | 21,001 | 3,212,666 |
| Marathi | test | 21,001 | 3,202,861 |
| Konkani | train | 261,186 | 200,293,343 |
| Konkani | val | 2,655 | 2,042,816 |
| Konkani | test | 2,655 | 2,031,644 |

We verified leakage by content hash across all three splits in both languages:
0 documents leaked. `tools/make_splits.py` enforces the 20% manual floor at
split-construction time rather than checking it afterwards.

### 10.2 Tokenizers

SentencePiece BPE, trained separately per language on that language's train split
only. No pretrained tokenizer, no shared vocabulary. `byte_fallback` is on, which
adds 256 `<0xNN>` byte pieces and makes any Unicode string representable — hence
the 0.000000% unknown-token rate.

Measured on 5,000 held-out documents per language, unseen during training:

| | Marathi | Konkani |
|---|---:|---:|
| vocabulary size | 2,500 | 2,500 |
| byte-fallback pieces | 256 | 256 |
| held-out words | 689,205 | 4,227,069 |
| tokens produced | 1,812,670 | 10,685,444 |
| fertility (tokens per word) | 2.6301 | 2.5279 |
| average characters per token | 2.6416 | 2.5583 |
| whole-word token rate | 34.4% | 31.0% |
| unknown tokens | 0 | 0 |

Token frequency on the same held-out text:

| | Marathi | Konkani |
|---|---:|---:|
| distinct tokens observed | 2,315 of 2,500 | 2,354 of 2,500 |
| vocabulary utilisation | 92.6% | 94.2% |
| tokens covered by the top 100 pieces | 40.8% | 39.2% |
| tokens covered by the top 1,000 pieces | 87.5% | 87.0% |
| pieces seen exactly once | 10 | 2 |

High utilisation with almost no hapax pieces means the vocabulary is sized to the
data rather than padded with pieces the corpus never uses.

Tokenization examples, full lists in `phase1_tokenizer_{language}.json`:

```
Marathi  महाराष्ट्रातील शेतकऱ्यांनी सरकारकडे तातडीने मदतीची मागणी केली
         ▁महाराष्ट्र ातील ▁शेतकऱ्यां नी ▁सरकार कडे ▁त ात डी ने ▁मद ती ची ▁मागणी ...

Konkani  गोंयची राजभास कोंकणी आसा आनी तिचो इतिहास खूब पोरनो आसा.
         ▁गोंय ची ▁राज भ ास ▁कोंकणी ▁आसा ▁आनी ▁ति चो ▁इतिहास ▁खूब ▁पोर न ...
```

The segmentation is morphologically plausible: `▁महाराष्ट्र + ातील` splits stem
from locative suffix, `▁गोंय + ची` splits stem from genitive. Frequent Konkani
function words like `▁आसा` and `▁आनी` come out as single pieces.

### 10.3 Vocabulary size

2,500 is an order of magnitude below the recommended range, so D-043 argues it in
full, including the strongest arguments against. In outline:

Unknown-token rate is uninformative under byte fallback — 0.000000% at every
candidate size — so fertility and parameter cost had to decide. At
`d_model = 512`, embedding plus unembedding cost `2 × V × d_model`, so a 10,000
vocabulary spends 41% of a 25M-parameter budget on two lookup tables against 10%
at 2,500. That difference is roughly two transformer layers.

Against that: weight tying halves the cost and makes 10,000 affordable at 20% of
the budget, so the parameter argument rules out an untied large vocabulary rather
than a large vocabulary as such. Whether to tie is a Phase 2 decision. And since
token count is fertility × words, a smaller vocabulary raises the reported total
without changing the data — on an identical corpus the Konkani train split
measures 506M tokens at 2,500 and projects to 435M at 5,000 and 387M at 10,000.
That is why word counts are reported alongside token counts throughout.

The measured cost: fertility rises from 2.1703 at 5,000 to 2.5148 at 2,500,
whole-word coverage falls from 40.6% to 30.6%, characters per token fall from
2.9814 to 2.5730, and every training sequence carries about 16% more tokens for
the same text. The six-candidate sweep is in
`phase1_tokenizer_sweep_konkani.json`.

Two fertility figures appear for Marathi and neither is an error. Held-out
fertility is 2.6301. The corpus-wide ratio, 872,024,099 tokens over 319,677,665
train words, is 2.7278. The held-out sample is not distributionally identical to
the full corpus.

### 10.4 Figures

In `report/figures/`, each with title, axis labels and legend:

| figure | shows |
|---|---|
| `phase1_sources_marathi.png` | word count by source, Marathi |
| `phase1_sources_konkani.png` | word count by source and provenance, Konkani |
| `phase1_manual_share.png` | manual, downloaded and synthetic share, both languages |
| `phase1_tokenizer_fertility.png` | fertility against vocabulary size |
| `phase1_split_composition.png` | train/val/test composition by source |

## 11. Corpus independence

`tools/cross_corpus_check.py --sample 50000`, run 20 August 2026, over 50,000
Marathi documents (46,965,136 words) and 50,000 Konkani documents (58,470,261
words):

| check | method | result |
|---|---|---|
| exact overlap | SHA-256 over normalized text | 0 shared documents |
| near-duplicate overlap | MinHash, 128 permutations, reported at Jaccard ≥ 0.8 | 0 pairs |
| language purity | function-word discriminator | 0.000% contamination each way |

Of the Konkani sample, 47,495 came back Konkani, 2,505 undecided, none Marathi.
The undecided ones are short documents below the five-marker gate.

## 12. Reproduction

```bash
pip install -r requirements.txt
```

Collection commands are in the README. Every collector is checkpointed and
resumable, so re-running skips completed work.

```bash
python3 tools/source_overlap_check.py --sample-archive 120
python3 tools/verify_gom_langid.py --sample 1500

# Vocabulary sweep. --sweep-only measures candidates and writes the comparison
# table; it does not train or replace the deliverable tokenizer.
python3 tools/build_tokenizer.py --language konkani \
  --vocab-sizes 2000,2500,3000,4000,5000,10000 --sweep-only

# Final tokenizers at the vocabulary chosen in D-043.
python3 tools/build_tokenizer.py --language marathi --vocab-sizes 2500
python3 tools/build_tokenizer.py --language konkani --vocab-sizes 2500

python3 tools/make_splits.py --language marathi --exact-only
python3 tools/make_splits.py --language konkani --exact-only

python3 tools/corpus_stats.py --language marathi --markdown
python3 tools/corpus_stats.py --language konkani --markdown
python3 tools/pipeline_accounting.py --markdown
python3 tools/cross_corpus_check.py --sample 50000
python3 tools/make_plots.py
```

The sweep and the final build are separate commands because they answer separate
questions. The sweep measures fertility across candidate vocabularies; the final
build installs the vocabulary we chose on the parameter-budget argument, which is
not the one the script's fertility-tolerance rule selects.

## 13. Limitations

Synthetic text is 32.2% of the Konkani corpus. It is labelled and separable, but a
model trained on it partly learns the output distribution of a translation system
rather than of Konkani writers. Excluding it leaves 180,506,902 real words, about
36% of the token target.

Token counts depend on the tokenizer. The Konkani train split measures 506M
tokens at vocabulary 2,500 and projects to 435M at 5,000. Word counts, which do
not, are reported alongside throughout.

The deliverable tokenizer was trained on 97% of the final corpus. We built it on
20 August from 312,293 lines, and about 10,800 further documents — chiefly BPCC —
were ingested afterwards. A tokenizer's training set is a sample by design, and
byte fallback guarantees the remainder is representable, which the 0.000000%
unknown rate over the full corpus confirms. It is also why the sweep reports
2.5148 at vocabulary 2,500 while the deliverable reports 2.5279: two held-out
samples drawn from slightly different corpora.

OCR quality is unmeasured. Both manual corpora rest on archive.org OCR text
layers, and we computed no character error rate because no ground-truth
transcription was available. We repaired whitespace and line-fragmentation
artifacts; character-level substitution errors we did not detect.

Konkani register is skewed toward books. Digitized books and literature dominate
and news and contemporary web text are a small fraction, so the corpus
under-represents contemporary written Konkani.

Sangraha's Romi rows were discarded rather than converted. We considered
transliteration to Devanagari and rejected it as a source of systematic error
indistinguishable from real orthographic variation.

The Marathi vocabulary sweep was not re-run. We set Marathi's vocabulary to 2,500
to match Konkani, since the parameter budget is identical for both models and
Marathi clears the token target at every vocabulary tested. Only the Konkani
sweep has a recorded artifact.

## 14. Deliverables

| deliverable | location |
|---|---|
| collection scripts, both languages | `marathi/scripts/`, `konkani/scripts/` |
| preprocessing pipeline | `common/`, `tools/` |
| per-language dataset statistics | `phase1_corpus_stats_{marathi,konkani}.{md,json}` |
| train/validation/test splits | Google Drive, `*_cleaned_splits.tar.gz` |
| tokenizer training code | `tools/build_tokenizer.py` |
| vocabulary files | `marathi/tokenizer/marathi_bpe.vocab`, `konkani/tokenizer/konkani_bpe.vocab` |
| tokenizer model files | `marathi/tokenizer/marathi_bpe.model`, `konkani/tokenizer/konkani_bpe.model` |

Large artifacts are on Google Drive, linked from the README and shared so that no
access request is needed.

Supporting documents:

| file | contents |
|---|---|
| `phase1_source_inventory.md` | every source used and rejected, with measured word counts |
| `phase1_konkani_coverage.md` | Konkani source exhaustion: what exists, what was empty |
| `phase1_konkani_mt.md` | synthetic data against the three authorised conditions |
| `phase1_konkani_source_discovery.md` | systematic Konkani source discovery and probe |
| `phase1_konkani_overlap_check.md` | manual against downloaded source independence |
| `phase1_pipeline_accounting.md` | stage-by-stage accounting, raw to cleaned to tokens |
| `phase1_decisions.md` | design decisions and corrections, D-001 to D-043 |
| `phase1_work_log.md` | chronological record of the build |
| `archive/` | superseded planning documents |
