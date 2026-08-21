# Konkani source coverage — what exists, and what was found empty

**Model L (Konkani, Devanagari) — Phase 1**

This document replaces `phase1_konkani_shortfall.md`. That file was written when
the Konkani corpus stood at 33% of the ~500M token target and argued that the
target could not be reached. **The corpus now stands at 506,259,368 training
tokens — 101.3% of target — so the shortfall argument no longer applies.** What
remains useful, and is preserved here, is the record of *where Konkani text
actually is*, because that record is what justifies the composition of the
corpus.

---

## 1. Final position

| | |
|---|---:|
| training tokens | **506,259,368** |
| vs ~500M target | **101.3%  PASS** |
| manual training tokens | 159,563,967 |
| manual share (≥20% required) | **31.5%  PASS** |
| documents leaked between splits | **0** |
| unknown-token rate | **0.000000%** |

### Composition by provenance

| | words | share |
|---|---:|---:|
| manual | 64,435,242 | 24.2% |
| downloaded (real) | 116,071,660 | 43.6% |
| synthetic (MT / LLM-generated) | 85,704,461 | 32.2% |
| **real text (manual + downloaded)** | **180,506,902** | **67.8%** |

---

## 2. Every real source, measured

Word counts are **after** the full pipeline — Unicode NFC, Devanagari-ratio
floor, Marathi rejection, minimum length, exact and near-duplicate removal — so
they are what the corpus gained, not what the source advertises.

| source | type | documents | words |
|---|---|---:|---:|
| `archive_org_konkani_books` | **manual** (OCR'd books) | 53,843 | 62,997,759 |
| `hf_konkani_books_corpus_v2` | downloaded | 40,597 | 47,016,182 |
| `hf_konkani_books_corpus_v1` | downloaded | 41,764 | 49,682,672 |
| `bpcc_gom_deva` | downloaded (human-translated) | 10,807 | 3,319,633 |
| `ai4bharat_indiccorp_v2_gom` | downloaded | 12,696 | 4,319,751 |
| `hf_madlad400_gom_noisy` | downloaded | 7,222 | 4,254,586 |
| `hf_sangraha_verified_gom` | downloaded | 9,827 | 3,266,816 |
| `hf_madlad400_gom_clean` | downloaded | 4,188 | 2,787,400 |
| `konkani_wikipedia_selfcollected` | **manual** | 2,459 | 1,395,235 |
| `hf_glotcc_v1_gom_deva` | downloaded | 2,020 | 1,325,434 |
| `hf_cfilt_roundtripocr_konkani` | downloaded | 2,966 | 87,497 |
| `news` | **manual** (self-crawled) | 33 | 42,248 |
| `hf_konkani_raw_scrape` | downloaded | 13 | 11,689 |

---

## 3. Repositories searched that contain no usable Konkani

These negative results are the substance of this document. They are what
establishes that no real dataset was left unexamined.

| repository | Konkani content | verified |
|---|---|---|
| `HuggingFaceFW/fineweb-2` | `gom_Latn` only — **Roman script**, wrong script | 19 Aug 2026 |
| `oscar-corpus/OSCAR-2301` | no `gom`, no `kok` | 19 Aug 2026 |
| `ai4bharat/sangraha` — `unverified` | 15 language directories, **no Konkani** | 19 Aug 2026 |
| `ai4bharat/sangraha` — `synthetic` | no `gom` split | 19 Aug 2026 |
| `ai4bharat/BPCC` — `nllb_filtered` | 16 language files, no `gom` | 19 Aug 2026 |
| `ai4bharat/BPCC` — `samanantar_v0.3_filtered` | 11 language files, no `gom` | 19 Aug 2026 |
| HPLT v2 | 191 languages, **no Konkani** | 20 Aug 2026 |
| `uonlp/CulturaX` — `gom` | present but **1,756,012 bytes** — negligible | 20 Aug 2026 |
| `cis-lmu/GlotCC-V1` — `kok-Deva` | referenced in the card, path 404s; only `gom-Deva` is real | 19 Aug 2026 |
| CC-100 | no Konkani | 19 Aug 2026 |

Eight further HuggingFace datasets returned by a `search=konkani` query yielded
**zero** words: no column in 50 probed rows cleared the 0.70 Devanagari floor
(`Reubencf/konkani-instruct-20k-1` … `-6`, `konkani/Goan_Data`,
`konkani/english-konkani`, `konkani/konkani_instructions`,
`shrusti333/konkani_translation`). The column choice for every dataset is
printed in `logs/bulk.log`.

---

## 4. The label that was wrong: IndicCorp v2 `gom.txt` (D-035)

`ai4bharat/IndicCorpV2` ships `data/gom.txt`, **533,108,246 bytes** labelled
Goan Konkani. On paper that is larger than everything else here combined.

It is ~84% Marathi. We established this by calibrating the discriminator
against two populations whose language is not in doubt, with all three packed to
the same ~300-word document length so length could not be confounded with
language:

| population | labelled `mr` | labelled `kok` | undecided | median score |
|---|---:|---:|---:|---:|
| reference Konkani (our OCR'd books) | 0.0% | 94.7% | 5.3% | **−0.92** |
| reference Marathi (our Marathi corpus) | 100.0% | 0.0% | 0.0% | **+1.00** |
| **IndicCorp v2 `gom.txt`** | **83.7%** | 2.5% | 13.9% | **+0.79** |

`gom.txt` sits on the Marathi reference. A sampled rejected document contains
`आहे×8, पण×5, मी×3` and **zero** Konkani markers.

Accepting it unfiltered would have added roughly 25M words. It would also have
poured Marathi — the language Model H trains on — into Model L, breaking the
specification's requirement that the two corpora share no data. **4,319,751
words** passed the gate and were kept.

Two further sources also failed to match their labels: Sangraha's `gom` split
is 32% Romi (Roman-script) Konkani, correctly rejected by the script gate; and
BPCC's two largest mined subsets contain no Konkani at all despite the dataset
card listing `gom_Deva`.

---

## 5. How the corpus reached the target

Three things closed the gap, in order of contribution:

1. **Correcting a search that was wrong by 115×.** An early archive.org query
   measured our own spelling list rather than the archive. `language:kok AND
   mediatype:texts` returns **5,093** items, not 44. Fixing that grew the manual
   side from 1.46M to 64.4M words (D-018).
2. **Searching one directory deeper.** IndicCorp v2's Konkani was invisible from
   the repository root listing and untagged on the dataset card.
3. **Synthetic data, after real sources were exhausted** — authorised by the TAs
   on 18 Aug 2026, contributing 32.2% of the corpus, documented in full in
   [`phase1_konkani_mt.md`](phase1_konkani_mt.md).

The final 1.46M tokens were closed by **BPCC** — real, human-translated Konkani
— rather than by more synthetic text, which is the better outcome and is why the
corpus crosses the target on 67.8% real data.

---

## 6. Honest caveats

**Token counts are tokenizer-dependent.** The identical corpus reads 430M tokens
at vocabulary 5,000 and 506M at 2,500. The vocabulary was chosen on the ~25M
parameter budget (embeddings fall from 41% to 10% of the budget), and the full
measured fertility curve is recorded in D-032. Anyone re-tokenising this corpus
at a different vocabulary will get a different token count, and that is a
property of the metric, not of the data.

**Synthetic share is 32.2%.** It is labelled throughout, kept in its own Drive
archive, and can be excluded. Excluding it leaves 180,506,902 real words.
