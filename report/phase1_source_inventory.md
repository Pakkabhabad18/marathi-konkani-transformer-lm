# Phase 1 — Source Inventory

Model H: Marathi. Model L: Konkani (Devanagari).
Regenerated 21 August 2026 against the frozen corpus.

Word counts in this document are post-pipeline: measured after Unicode NFC
normalization, the Devanagari-ratio floor, the Marathi/Konkani discriminator,
the minimum-length gate, and exact plus near-duplicate removal. They are what
each source contributed to the corpus, not what the source advertises. The two
figures often differ by an order of magnitude, and section 3.2 gives the largest
example.

All figures here can be re-derived by running:

```
python3 tools/corpus_stats.py --language marathi --markdown
python3 tools/corpus_stats.py --language konkani --markdown
python3 tools/pipeline_accounting.py --markdown
```

and reconcile exactly to `report/phase1_corpus_stats_{marathi,konkani}.json`.

## 1. Accounting rules

### 1.1 The 20% manual floor

The specification requires `manual_tokens / total_tokens >= 0.20`. This is
enforced in `tools/make_splits.py` at split-construction time rather than
checked by hand afterwards: if admitting a source would push the ratio below
0.20, downloaded documents are dropped until it holds. Every non-manual word
admitted raises the manual target by a factor of four, so sources were admitted
in priority order and the non-manual contribution was capped at what the manual
total supported.

### 1.2 Provenance categories

`common/manifest.py::CollectionType` defines three categories. They are assigned
at the point of collection, in the collector script, not applied as a label
afterwards.

| category | definition | `is_manual` |
|---|---|---|
| manual | text gathered and cleaned by us: OCR from books, sites we crawled, pages we fetched and parsed | true |
| downloaded | prepared corpora obtained ready-made, however much cleaning we applied afterwards, written or translated by humans | false |
| synthetic | machine-translated or LLM-generated text | false |

`MACHINE_TRANSLATED` is a distinct enum member rather than a flag on
`DOWNLOADED_DATASET` (D-036). This is what allows `tools/corpus_stats.py` to
report the synthetic share separately in every table, and what prevents it being
counted toward the manual floor.

This follows TA guidance issued 14 August: anything involving getting data,
processing, cleaning and organizing it is manual; anything already organized on
HuggingFace and then used is not.

### 1.3 Units

Sections 2 and 3 are in words. Words are tokenizer-independent, so they remain
comparable across the vocabulary changes this project went through. Token counts
appear only in section 5, measured once per language with the final tokenizer
over the frozen train split. An earlier progress report summed token counts from
two different tokenizers into one total; that error is why the units are
separated here (D-007).

## 2. Model H — Marathi

Total: 2,887,867 documents, 389,218,163 words. 45.7% manual, no synthetic data.

### 2.1 Manual sources — 177,781,779 words

| source | documents | words | words/doc | collection method |
|---|---:|---:|---:|---|
| `archive_org_maharashtra_gr` | 154,001 | 152,312,784 | 989 | per-item OCR text layers from archive.org, fetched and cleaned by `marathi/scripts/collect_archive_gr.py` |
| `news_esakal` | 90,644 | 22,828,494 | 252 | sitemap-driven crawl via `common/newscrawl.py` |
| `news_loksatta` | 2,684 | 1,030,716 | 384 | sitemap-driven crawl |
| `news_abplive_marathi` | 2,290 | 777,748 | 340 | sitemap-driven crawl |
| `news_lokmat` | 1,158 | 319,054 | 276 | sitemap-driven crawl |
| `news_divyamarathi` | 730 | 270,062 | 370 | sitemap-driven crawl |
| `news_tv9marathi` | 654 | 208,695 | 319 | sitemap-driven crawl |
| `news_maharashtratimes` | 76 | 34,226 | 450 | sitemap-driven crawl |

Crawl policy: `robots.txt` was fetched and parsed before any request, the
declared crawl-delay was honoured, and a descriptive User-Agent was sent.
Boilerplate (navigation, related-articles blocks, comment sections) was removed
by per-site selectors before the text was counted.

Before committing to a full crawl of any site, a pilot of approximately 1,000
pages was hashed and compared against the IndicCorp v2 corpus using
`tools/source_overlap_check.py`. This matters because IndicCorp v2 is itself
built from Marathi news crawls. Scraping the same pages and reporting the result
as manually collected would be inaccurate, and cross-source deduplication would
have removed most of it in any case. The measured overlap is reported in
`report/phase1_konkani_overlap_check.md` and the equivalent Marathi run.

### 2.2 Downloaded sources — 211,436,384 words

| source | documents | words | words/doc | licence |
|---|---:|---:|---:|---|
| `ai4bharat_indiccorp_v2_mar` | 2,635,630 | 211,436,384 | 80 | CC-0 |

One downloaded source was used rather than several, for two reasons. IndicCorp
v2 is CC-0, the least restrictive licence among the candidates. And IndicCorp v2
and Sangraha `mar` are produced by the same group from overlapping crawls, so
combining them would have required a large cross-corpus deduplication pass for
tokens that were not needed once the target was met.

### 2.3 Sources examined and not used

| source | reason |
|---|---|
| `ai4bharat/sangraha` — `mar` verified and unverified | high expected overlap with IndicCorp v2; not required once the downloaded quota was met |
| `ai4bharat/sangraha` — `synthetic` | machine-translated and romanized WikiMedia content; Marathi reached its target on real text alone |

## 3. Model L — Konkani (Devanagari)

Total: 323,112 documents, 266,211,363 words. 24.2% manual, 32.2% synthetic.

### 3.1 Script decision

The training corpus is Devanagari only. Konkani is also written in Roman script
(Romi) and in Kannada script; both are excluded and the exclusion is recorded
rather than silently applied.

Three reasons. Devanagari is the official script of Konkani in Goa. It is the
script used by the largest available Konkani corpora, so restricting to it costs
little volume. And a model of approximately 25M parameters has limited capacity
to spend learning two orthographies of the same language, which would compete
for the same embedding budget without adding linguistic coverage.

The decision is load-bearing rather than cosmetic: it is why 4,651 rows (32%) of
Sangraha's `gom` split were rejected (D-033). Those rows are Konkani, correctly
labelled, in the wrong script for this corpus.

### 3.2 Manual sources — 64,435,242 words

| source | documents | words | words/doc | collection method |
|---|---:|---:|---:|---|
| `archive_org_konkani_books` | 53,843 | 62,997,759 | 1,170 | OCR text layers from archive.org, `konkani/scripts/collect_archive_books.py` |
| `konkani_wikipedia_selfcollected` | 2,459 | 1,395,235 | 567 | MediaWiki API crawler and wikitext cleaner written for this project |
| `news_goanews` | 31 | 41,728 | 1,346 | sitemap-driven crawl |
| `news_vishwakonkani` | 2 | 520 | 260 | sitemap-driven crawl |

The archive.org books carry the manual requirement almost entirely, and they
exist because an earlier search was found to be wrong by a factor of 115. The
original query returned 44 items. Inspection showed it was matching against a
locally held spelling list rather than querying the archive index. The corrected
query, `language:kok AND mediatype:texts`, returns 5,093 items (D-018). That
correction moved the manual total from 1.46M words to 64.4M.

Only items with an existing OCR text layer were used; image-only scans were
skipped rather than run through a local OCR engine, because the error rate of
uncontrolled OCR on Devanagari would have been unmeasurable.

### 3.3 Downloaded sources, human-authored — 116,071,660 words

| source | documents | words | words/doc | note |
|---|---:|---:|---:|---|
| `hf_konkani_books_corpus_v1` | 41,764 | 49,682,672 | 1,190 | digitized books |
| `hf_konkani_books_corpus_v2` | 40,597 | 47,016,182 | 1,158 | digitized books; OCR line-fragmentation repaired before counting |
| `ai4bharat_indiccorp_v2_gom` | 12,696 | 4,319,751 | 340 | see below; approximately 84% of the source file was rejected as Marathi |
| `hf_madlad400_gom_noisy` | 7,222 | 4,254,586 | 589 | web crawl |
| `bpcc_gom_deva` | 10,807 | 3,319,633 | 307 | human-translated parallel data |
| `hf_sangraha_verified_gom` | 9,827 | 3,266,816 | 332 | 32% of rows rejected as Romi script |
| `hf_madlad400_gom_clean` | 4,188 | 2,787,400 | 666 | web crawl |
| `hf_glotcc_v1_gom_deva` | 2,020 | 1,325,434 | 656 | web crawl |
| `hf_cfilt_roundtripocr_konkani` | 2,966 | 87,497 | 30 | OCR correction pairs |
| `hf_konkani_raw_scrape` | 13 | 11,689 | 899 | scraped pages bundled inside `praveenkumar99/Konkani_Raw` |

`ai4bharat/IndicCorpV2` ships `data/gom.txt` at 533,108,246 bytes, labelled Goan
Konkani. On file size alone it is larger than every other Konkani source
combined. Measurement showed it is approximately 84% Marathi.

This was established by calibration rather than by trusting the discriminator's
raw output. Three populations were packed to the same approximately 300-word
document length so that length could not be confounded with language, and scored
with the same discriminator on the same day:

| population | labelled `mr` | labelled `kok` | undecided | median score |
|---|---:|---:|---:|---:|
| reference Konkani (our OCR'd books) | 0.0% | 94.7% | 5.3% | −0.92 |
| reference Marathi (our Marathi corpus) | 100.0% | 0.0% | 0.0% | +1.00 |
| IndicCorp v2 `gom.txt` | 83.7% | 2.5% | 13.9% | +0.79 |

`gom.txt` sits on the Marathi reference, not between the two. A sampled rejected
document contains आहे×8, पण×5, मी×3 and no Konkani markers. Tool:
`tools/verify_gom_langid.py --sample 1500`. Full analysis in D-035.

Accepting the file unfiltered would have added roughly 25M words and would have
put Marathi text, the language Model H trains on, into Model L's corpus. That
would breach the specification's requirement that the two models share no data.
4,319,751 words passed the gate and were kept.

### 3.4 Synthetic sources — 85,704,461 words

Authorised by the TAs on 18 August 2026 after real sources were exhausted, under
three stated conditions. Compliance with each is documented in
`report/phase1_konkani_mt.md`.

Every row below carries `collection_type: machine_translated` in its manifest,
is stored under `konkani/data/synthetic/`, and ships in a separate Drive archive
so it can be excluded without touching the rest of the corpus.

| source | documents | words | words/doc |
|---|---:|---:|---:|
| `hf_konkani_raw_machine_translated` | 60,843 | 59,295,762 | 975 |
| `hf_bulk_anag007_instruct` | 19,154 | 7,016,222 | 366 |
| `hf_bulk_anag007_alpaca` | 13,150 | 5,038,553 | 383 |
| `hf_bulk_saillab_cleaned` | 11,235 | 4,311,915 | 384 |
| `hf_bulk_telugu_labs_alpaca` | 10,689 | 4,179,067 | 391 |
| `hf_bulk_devarshee_v2` | 7,716 | 2,470,036 | 320 |
| `hf_bulk_anag007_wiki` | 1,221 | 1,013,811 | 830 |
| `hf_bulk_gpteacher` | 2,563 | 867,359 | 338 |
| `hf_konkani_instruct_100k_synthetic` | 5,439 | 631,270 | 116 |
| `hf_bulk_saillab_taco` | 1,344 | 409,166 | 304 |
| `hf_bulk_devarshee_v1` | 767 | 296,198 | 386 |
| `mt_marathi_to_konkani_indictrans2` | 407 | 128,519 | 316 |
| `hf_bulk_predictionguard` | 149 | 46,583 | 313 |

The largest entry, `hf_konkani_raw_machine_translated`, is 870,725,308 bytes of
machine-translated text inside `praveenkumar99/Konkani_Raw`. It was deliberately
excluded during the earlier ingest of that repository, when synthetic data was
not permitted, and only the 4,397,019 bytes of genuinely scraped pages were
taken. After authorisation it was ingested in full: 73,717 rows to 60,843
documents. Two rows were rejected as Marathi and cross-source deduplication
removed none of its documents (D-038).

The ten `hf_bulk_*` sources were ingested by `ingest_hf_bulk_konkani.py`, which
probes the first 50 rows of each dataset and selects every column whose mean
Devanagari ratio exceeds 0.50. Column names were not copied from dataset cards,
because the HuggingFace datasets-server `first-rows` endpoint was observed
returning cached responses for the wrong dataset when several were queried in
sequence (D-039). The selected columns for each dataset are printed in
`logs/bulk.log`.

`mt_marathi_to_konkani_indictrans2` is our own generation run using
IndicTrans2 `indic-indic-dist-320M`: 14,016 sentences, 394 documents,
`copied_not_translated` = 0 across the run. It is 0.05% of the corpus. It is
reported because it was attempted and measured, not because it was material.
The reasons it stayed small are recorded in D-041 and D-042.

### 3.5 Sources examined and not used

| source | reason |
|---|---|
| `ai4bharat/IndicCorpV2` `gom.txt`, the rejected 84% | approximately 25M words of Marathi in a file labelled Goan Konkani |
| Romi (Roman-script) Konkani, including 32% of Sangraha `gom` | script decision, section 3.1 |
| Kannada-script Konkani | script decision, section 3.1 |
| `Reubencf/konkani-instruct-20k-1` to `-6`, `konkani/Goan_Data`, `konkani/english-konkani`, `konkani/konkani_instructions`, `shrusti333/konkani_translation` | probed with 50 rows each; no column cleared the 0.70 Devanagari floor. Zero words extracted. |

## 4. Repositories searched and found to contain no usable Konkani

These negative results are the evidence that the search was exhaustive. Each was
checked by listing the repository's actual files through the HuggingFace Hub API,
not by reading its dataset card, because three of the sources in section 3 did
not match their own cards.

| repository | finding | date checked |
|---|---|---|
| `HuggingFaceFW/fineweb-2` | `gom_Latn` only, Roman script | 19 Aug 2026 |
| `oscar-corpus/OSCAR-2301` | no `gom`, no `kok` | 19 Aug 2026 |
| `ai4bharat/sangraha` — `unverified` | 15 language directories, no Konkani | 19 Aug 2026 |
| `ai4bharat/sangraha` — `synthetic` | no `gom` split | 19 Aug 2026 |
| `ai4bharat/BPCC` — `nllb_filtered` | 16 language files, no `gom` | 19 Aug 2026 |
| `ai4bharat/BPCC` — `samanantar_v0.3_filtered` | 11 language files, no `gom` | 19 Aug 2026 |
| HPLT v2 | 191 languages, no Konkani | 20 Aug 2026 |
| `uonlp/CulturaX` — `gom` | present, 1,756,012 bytes | 20 Aug 2026 |
| `cis-lmu/GlotCC-V1` — `kok-Deva` | referenced on the dataset card, path returns 404; only `gom-Deva` exists | 19 Aug 2026 |
| CC-100 | no Konkani | 19 Aug 2026 |

Three sources did not match their published labels: IndicCorp v2's `gom.txt` is
approximately 84% Marathi, Sangraha's `gom` split is 32% Roman-script, and BPCC's
two largest mined subsets contain no Konkani despite `gom_Deva` appearing on the
dataset card. Every source in this project was therefore measured before use
rather than accepted on its label.

## 5. Totals

Words, tokenizer-independent:

| | Marathi | Konkani |
|---|---:|---:|
| manual | 177,781,779 (45.7%) | 64,435,242 (24.2%) |
| downloaded, human-authored | 211,436,384 (54.3%) | 116,071,660 (43.6%) |
| synthetic | 0 (0.0%) | 85,704,461 (32.2%) |
| real text (manual + downloaded) | 389,218,163 (100.0%) | 180,506,902 (67.8%) |
| total | 389,218,163 | 266,211,363 |

Tokens, measured once per language with the final vocabulary-2,500 SentencePiece
BPE model over the frozen train split:

| | Marathi | Konkani |
|---|---:|---:|
| training tokens | 872,024,099 | 506,259,368 |
| against the ~500M target | 174.4%, pass | 101.3%, pass |
| manual training tokens | 475,466,104 | 159,563,967 |
| manual share, floor is 20% | 54.5%, pass | 31.5%, pass |
| unknown-token rate | 0.000000% | 0.000000% |

Token counts depend on the tokenizer. The same Konkani corpus measures 437M
tokens at vocabulary 5,000 and 506M at vocabulary 2,500. The measured fertility
curve and the reasoning behind the vocabulary choice are in D-043.

## 6. Corpus independence

The specification requires that Model H and Model L share no data.
`tools/cross_corpus_check.py --sample 50000`, run 20 August 2026:

| check | method | result |
|---|---|---|
| exact overlap | SHA-256 over NFC-normalized, whitespace-collapsed text | 0 shared documents |
| near-duplicate overlap | MinHash over character 5-gram shingles, 128 permutations, LSH banding tuned to a 0.85 threshold, reported at Jaccard ≥ 0.8 | 0 pairs |
| language purity | closed-class function-word discriminator, `min_markers=5`, `margin=0.30` | 0.000% cross-language contamination in each corpus |

Sample sizes: 50,000 Marathi documents (46,965,136 words) and 50,000 Konkani
documents (58,470,261 words), drawn with seed `20260819`. Of the Konkani sample,
47,495 were labelled Konkani and 2,505 were undecided; none were labelled
Marathi. Undecided documents are short documents that fall below the five-marker
evidence gate, which abstains rather than guessing.

Output: `report/phase1_cross_corpus_check.json`.
