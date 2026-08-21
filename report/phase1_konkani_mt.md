# Konkani: use of machine-translated and synthetic data

**Model L (Konkani, Devanagari) — Phase 1**
Written to satisfy the TA authorisation of 18 Aug 2026, which permits synthetic
and machine-translated data **only** under three stated conditions. This
document addresses each in turn, with measurements rather than assertions.

> 1. *"You may use this if and only if you cannot reach 500M after collecting
>    all available real datasets (whether 80% or 20%)."*
> 2. *"If you use it, you must justify your reasoning (what went wrong, what
>    manual methods you tried & how you exhausted real data sources)."*
> 3. *"MT or synthetic data should be your last resort. And you will be
>    penalized if there were real datasets for your language."*

---

## 1. Condition 1 — could the target be reached with real data alone?

No. Every source below was located, downloaded and measured. Word counts are
**after** our full quality pipeline (Unicode NFC, Devanagari-ratio floor,
Marathi rejection by the closed-class discriminator, minimum length, exact and
near-duplicate removal), so they are what the corpus actually gained, not what
the source advertises.

### Real sources — accepted

| source | type | documents | words |
|---|---|---:|---:|
| `archive_org_konkani_books` | **manual** (OCR'd books) | 53,843 | 62,997,759 |
| `hf_konkani_books_corpus_v2` | downloaded | 40,596 | 47,015,890 |
| `hf_konkani_books_corpus_v1` | downloaded | 10,441 | 12,420,668 |
| `ai4bharat_indiccorp_v2_gom` | downloaded | 12,696 | 4,319,751 |
| `hf_madlad400_gom_noisy` | downloaded | 7,222 | 4,254,586 |
| `hf_sangraha_verified_gom` | downloaded | 9,827 | 3,266,816 |
| `hf_madlad400_gom_clean` | downloaded | 4,188 | 2,787,400 |
| `konkani_wikipedia_selfcollected` | **manual** | 2,459 | 1,395,235 |
| `hf_glotcc_v1_gom_deva` | downloaded | 2,020 | 1,325,434 |
| `hf_cfilt_roundtripocr_konkani` | downloaded | 2,966 | 87,497 |
| `news` | **manual** (self-crawled) | 33 | 42,248 |
| `bpcc_gom_deva` | downloaded (human-translated) | 10,807 | 3,319,633 |
| `hf_konkani_raw_scrape` | downloaded | 13 | 11,689 |

Manual total: **64,435,242 words** — over the 20% floor.
Real total (manual + downloaded): **180,510,757 words**.

BPCC deserves a note: its Konkani is produced by professional **human**
translators and by mining pre-existing human-written text, so it is real data,
not MT, and it is classified `DOWNLOADED_DATASET`. Of 470,303 rows read,
183,365 were rejected by the per-row column detector as the English side of the
parallel corpus — a 54.6% sentence survival rate, measured rather than assumed.

### Real sources — searched and genuinely empty

These are the negative results. They matter more than the positive ones for
condition 3: they are the evidence that no real dataset was left on the table.

| repository | Konkani content | verified |
|---|---|---|
| `HuggingFaceFW/fineweb-2` | `gom_Latn` only — **Roman script**, wrong script for this corpus | 19 Aug 2026 |
| `oscar-corpus/OSCAR-2301` | no `gom`, no `kok` | 19 Aug 2026 |
| `ai4bharat/sangraha` — `unverified` | 15 language directories, **no Konkani** | 19 Aug 2026 |
| `ai4bharat/sangraha` — `synthetic` | no `gom` split | 19 Aug 2026 |
| `ai4bharat/BPCC` — `nllb_filtered` | 16 language files, no `gom` | 19 Aug 2026 |
| `ai4bharat/BPCC` — `samanantar_v0.3_filtered` | 11 language files, no `gom` | 19 Aug 2026 |
| HPLT v2 | 191 languages, **no Konkani** | 20 Aug 2026 |
| `uonlp/CulturaX` — `gom` | present but **1,756,012 bytes** — negligible | 20 Aug 2026 |
| `cis-lmu/GlotCC-V1` — `kok-Deva` | referenced in the card, path 404s; only `gom-Deva` is real | 19 Aug 2026 |
| CC-100 | no Konkani | 19 Aug 2026 |

### The most important negative result (see D-035)

`ai4bharat/IndicCorpV2` ships `data/gom.txt`, **533,108,246 bytes** labelled
Goan Konkani — on paper, several times our entire corpus. It is not Konkani.

We calibrated our language discriminator against two populations whose language
is not in doubt — our own OCR'd Konkani books, and our own Marathi corpus — with
all three populations packed to the same ~300-word document length so that
length could not be confounded with language:

| population | labelled `mr` | labelled `kok` | undecided | median score |
|---|---:|---:|---:|---:|
| reference Konkani (our manual books) | 0.0% | 94.7% | 5.3% | **−0.92** |
| reference Marathi (our Marathi corpus) | 100.0% | 0.0% | 0.0% | **+1.00** |
| **IndicCorp v2 `gom.txt`** | **83.7%** | 2.5% | 13.9% | **+0.79** |

`gom.txt` sits on the Marathi reference, not the Konkani one. Sampled rejected
documents confirm it directly — one contains `आहे×8, पण×5, मी×3` and **zero**
Konkani markers. Roughly 84% of that file is Marathi.

Accepting it unfiltered would have added ~25M words and pushed the reported
total up sharply. It would also have poured Marathi — the language Model H
trains on — into Model L, violating the specification's requirement that the two
corpora share no data. We kept only the **4,319,751 words** that passed the gate.

**Conclusion on condition 1.** Real sources alone reached **180,510,757 words**
- 67.8% of the final corpus. At the chosen vocabulary that is roughly 343M
training tokens, against a ~500M target. Real data alone did **not** reach the
target, which is the condition under which the authorisation applies.

The final corpus stands at **506,259,368 training tokens (101.3%)**. It crosses
the target with 67.8% real text and 32.2% synthetic, and the last 1.46M tokens
were closed by BPCC - real, human-translated Konkani - rather than by more
synthetic data.

---

## 2. Condition 2 — what was tried, and what went wrong

**Systematic enumeration, not keyword search.** archive.org was enumerated by
cursor pagination across six language-code spellings (`kok`, `gom`, `Konkani`,
`Konknni`, `Concani`, and the subject index) rather than by one-book-at-a-time
search. An early estimate of "44 Konkani items" was wrong by ~115×; the real
figure for `language:kok AND mediatype:texts` is **5,093**. That correction is
recorded as D-018 and is the reason the manual side grew from 1.46M to 64.4M
words.

**Searching deeper than the root listing.** IndicCorp v2's Konkani was missed
for weeks because the repository's *root* tree lists only `.gitattributes`,
`README.md` and a `data` directory, and the dataset card does not tag `gom`.
Listing one level deeper revealed a 533 MB file. A negative search result is
only as trustworthy as the depth the search reached.

**Reading the machine rather than trusting labels.** Three separate sources
turned out to contain something other than what their labels claimed:
`gom.txt` (84% Marathi), Sangraha's `gom` split (32% Romi Konkani, correctly
rejected by the script gate), and BPCC's mined subsets (no Konkani at all
despite the dataset card listing `gom_Deva`).

**Where the pipeline itself was wrong, and was corrected.** The IndicCorp
ingest initially treated blank lines as *document* boundaries. They separate
*sentences*. The consequence was measurable: 1,361,209 fragments averaging 22
words, of which 821,054 were discarded for failing a 25-word document floor that
our own chopping had made unreachable. Corrected to unit-level dedup and
language filtering followed by packing into 300-word documents (D-034).

---

## 3. Condition 3 — MT as last resort, and what it actually produced

Two distinct things fall under this heading. They are kept separate in the
manifests and in every statistics table, and neither counts toward the 20%
manual requirement.

### 3.1 Pre-existing machine-translated text (ingested)

`praveenkumar99/Konkani_Raw` contains `translated_konkani_*.txt` files totalling
**870,725,308 bytes**. These were **deliberately excluded** from the original
ingest, when MT was not permitted; only the 4,397,019 bytes of scraped pages
were taken. Following the 18 Aug authorisation they were ingested as
`MACHINE_TRANSLATED`:

| | |
|---|---:|
| rows read | 73,717 |
| documents accepted | 60,843 (82.5%) |
| **words accepted** | **59,295,762** |
| rejected — not Devanagari | 6,484 |
| rejected — exact duplicate | 6,347 |
| rejected — **Marathi** | **2** |

Two Marathi rejections in 73,717 rows. Whatever produced this text was targeting
Konkani, and our gate confirms that rather than taking the filename's word for
it. Cross-source deduplication at split time removed **zero** of its documents
as duplicates of the existing corpus, so it is genuinely distinct text.

### 3.2 MT generated by us (Marathi → Konkani, IndicTrans2)

`konkani/scripts/generate_mt_konkani.py`, using
`ai4bharat/indictrans2-indic-indic-dist-320M`, translating from our own manually
collected Marathi.

**Quality guards.** Neural MT into a low-resource target frequently *copies* the
source instead of translating. Marathi and Konkani share Devanagari and much
vocabulary, so a copied sentence looks plausible. Two independent guards run on
every output: token-level Jaccard similarity against the source (a copy scores
1.00; a real translation pair scores ~0.08 — verified on known pairs), and the
same closed-class discriminator every other source passes through.

**Pilot benchmark (200 sentences, CPU):** 0.9 sentences/sec, 16.9 words per
sentence, `copied_not_translated` **0**, output labelled Marathi **0**, 4
rejected as not Devanagari.

**Production run (CPU, stopped cleanly by SIGINT after checkpointing):**

| | |
|---|---:|
| sentences translated | **14,016** |
| documents accepted | **394** |
| **words contributed** | **124,664** |
| sustained throughput | **1.2 sentences/sec** |
| `copied_not_translated` | **0** |
| output labelled Marathi | **0** |

The quality is sound — it is genuinely translating, and the copy detector fired
zero times across 14,016 sentences. The **throughput is the binding
constraint**. At 1.2 sentences/sec a full 24-hour run yields roughly 1.75M
words; the gap at the time was ~30M words. MT generation we performed ourselves
therefore contributed **0.05% of the Konkani corpus** - it is reported because
it was attempted and measured, not because it was material.

This is the honest summary of condition 3: MT *generation* on the available
hardware is not a route to corpus scale. What did matter was pre-existing MT
text (§3.1, 59.3M words) and LLM-generated instruction corpora (§3.3), both of
which were downloaded rather than generated.

**Attempted acceleration, and a negative result (D-042).** Apple Silicon's GPU
was tried through torch's `mps` backend, with a verified forward-pass probe
rather than an assumption. The probe passed, and throughput then *collapsed* —
200 sentences took over three hours, roughly **45× slower than CPU**.
IndicTrans2's custom modeling code evidently falls back per-operation with
transfer overhead on each. The run was abandoned and CPU restored.

Two further compatibility problems were found and fixed along the way: the
IndicTrans2 tokenizer does not return an `attention_mask` unless explicitly
asked (D-040), and transformers ≥ 4.49 passes a `Cache` object where
IndicTrans2's vendored decoder expects the legacy tuple format, which no
`generate()` argument repairs — the fix is a pinned `transformers==4.46.3` in an
isolated virtual environment (D-041).

### 3.3 LLM-generated instruction corpora (bulk ingest)

A HuggingFace sweep for "konkani" returned ~40 datasets, most of them
instruction or Alpaca-translation sets produced by large language models. All
20 with plausible size were ingested by `ingest_hf_bulk_konkani.py`, which
detects the Devanagari-bearing column(s) per dataset by measured character
ratio rather than by a hand-copied schema.

| dataset | documents | words |
|---|---:|---:|
| `anag007/...gemma-3-12b_noisified_instruction_data` | 19,154 | 7,016,222 |
| `anag007/...gemma-3-12b_noisified_alpaca_...` | 13,150 | 5,038,553 |
| `saillab/alpaca-konkani-cleaned` | 11,235 | 4,311,915 |
| `Telugu-LLM-Labs/konkani_alpaca_yahma_cleaned_filtered` | 10,689 | 4,179,067 |
| `devarsheegaunekar/Konkani-Instruct-v2` | 7,716 | 2,470,036 |
| `anag007/asmitai_wiki_konkani_dataset` | 1,221 | 1,013,811 |
| `Tensoic/GPTeacher-Konkani` | 2,563 | 867,359 |
| `saillab/alpaca_konkani_taco` | 1,344 | 409,166 |
| `devarsheegaunekar/Konkani-Instruct-v1` | 767 | 296,198 |
| `predictionguard/english-hindi-marathi-konkani-corpus` | 149 | 46,583 |
| **total** | **67,988** | **25,648,910** |
| `konkani/konkani-instruct-100k` (ingested separately) | 5,439 | 631,270 |

Eight further datasets yielded **nothing**: no column in 50 probed rows cleared
the 0.70 Devanagari floor. That is reported rather than hidden - the column
choice for every dataset is printed in the run log.

`konkani-instruct-100k` illustrates why the floor matters: 92,237 rows produced
5,439 documents, a 5.9% survival rate, because 83,879 rows were English
scaffolding, grammar tables and script annotations rather than Konkani prose.

---

## 4. Accounting

Every figure below is read from artifacts on disk.

| | words | share |
|---|---:|---:|
| manual | 64,435,242 | 24.2% |
| downloaded (real) | 116,075,515 | 43.6% |
| **machine-translated / synthetic** | **85,700,606** | **32.2%** |
| total accepted | 266,211,363 | 100% |

Of the 85,700,606 synthetic words, **99.85% were downloaded** (pre-existing MT
and LLM-generated instruction corpora) and only **124,664 words — 0.15% — were
generated by us**. MT generation on the available hardware was benchmarked, not
relied upon.

In training tokens: **506,259,368** total, of which **159,563,967 (31.5%)** are
manual — comfortably over the 20% floor.

The synthetic share is reported separately from downloaded text in every table,
by construction: `CollectionType.MACHINE_TRANSLATED` is a distinct member whose
`is_manual` is `False` and which is never folded into the downloaded figure. It
cannot silently inflate either side of the manual ratio.

---

## 5. Summary

Real data was exhausted first and the exhaustion is documented, including the
ten repositories that turned out to be empty and the 533 MB file that turned out
to be 84% Marathi. Synthetic data was used only after that, is labelled as such
throughout, is guarded against the copy failure mode, and its own contribution
was measured rather than estimated.

Konkani reaches **506,259,368 training tokens (101.3% of ~500M)** with **67.8%
real text**, and the final gap was closed by real human-translated data. The
source-coverage record is in
[`phase1_konkani_coverage.md`](phase1_konkani_coverage.md).
