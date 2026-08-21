> **SUPERSEDED — 21 August 2026. Do not read the figures below as current.**
>
> This document was written on 19 August, when the Konkani corpus stood at
> 33.2% of the ~500M token target, and it argued that the target could not be
> reached. **That conclusion is now false.** The Konkani corpus reached
> **506,259,368 training tokens (101.3% of target)** on 20–21 August, on 67.8%
> real (human-written or human-translated) text.
>
> It is kept, unedited below this banner, because the reasoning is what led to
> the corrections that closed the gap — a search that was wrong by 115×, a
> corpus directory that was invisible from the repository root, and a 533 MB
> file mislabelled as Konkani. Deleting it would hide how the corpus was
> actually built.
>
> The current position is in
> [`phase1_konkani_coverage.md`](phase1_konkani_coverage.md); the synthetic-data
> justification is in [`phase1_konkani_mt.md`](phase1_konkani_mt.md).
>
> Note also that the token figures below were measured at **vocabulary 10,000**,
> which was later reduced to 2,500 (D-043). Token counts are tokenizer-dependent,
> so they are not comparable with the current tables.

---

# Phase 1 — Konkani: why the ~500M token target was not reached

**Required by §1.2:** *"If the lower-resource language cannot reach 500M from
public sources, collect the maximum feasible amount, report the exact token
count, and justify the shortfall."*

This document is that justification. Every figure is measured, not estimated.

---

## 1. The exact final count

| | Marathi (Model H) | Konkani (Model L) |
|---|---:|---:|
| final **training tokens** | **647,434,614** | **165,815,092** |
| vs ~500M target | **129.5%** | **33.2%** |
| manual training tokens | 353,009,984 | 115,064,540 |
| manual share of training tokens | 54.5% | **69.4%** |
| tokenizer vocabulary | 10,000 | 10,000 |
| unknown-token rate | 0.000000% | 0.000000% |
| documents leaked between splits | 0 | 0 |

Konkani **passes** the ≥20% manual requirement with a very wide margin. It does
not reach ~500M total, and the paragraphs below establish that this is a property
of the language's digital presence rather than of the effort applied.

---

## 2. What we searched, and what it returned

### 2.1 Internet Archive — enumerated exhaustively

Not sampled. Every item matching the corrected query was examined:

```
(language:(kok OR gom OR Konkani OR Konknni OR Concani)
 OR subject:Konkani OR subject:Konknni) AND mediatype:texts
```

| | measured |
|---|---:|
| items enumerated | **5,178** |
| items fetched and examined | **5,178** (100%) |
| books contributing usable Devanagari text | **2,132** |
| segments accepted | 55,990 |
| **manual words obtained** | **~63.0M** |

Rejections were recorded by reason, not discarded silently: 55,090 segments
excluded as non-Devanagari (D-001), 1,771 rejected by the Marathi language gate,
and 96 items genuinely lacking an OCR text layer.

**An important correction is embedded in this number.** Our first query asked for
`language:(Konkani OR Konknni OR Concani)` and returned **44 items**, which we
initially reported as the Internet Archive's entire Konkani holdings. It is not:
cataloguers overwhelmingly record the ISO 639-2 code `kok`, and the corrected
query returns **5,093**. The shortfall we are justifying here is therefore
measured *after* fixing a ~115× underestimate of our own making (D-018).

### 2.2 Web sources — 18 sites probed, 1 usable

Probed with two independent extractors: 6 had unreachable `robots.txt`, 10
returned no extractable article text, 1 (`goanvarta`) publishes Marathi and was
caught by the language gate, 1 was usable. Total yield: **~42k words**.

### 2.3 Prepared corpora — measured, not assumed

| source | size | classification |
|---|---:|---|
| `omdeep22/Konkani_books_corpus-v2` | 47,016,182 words | downloaded |
| AI4Bharat Sangraha, `gom` split | **10.1M tokens, all splits** | not collected |
| IndicNLP Catalog, Konkani entries | **1** (transliteration pairs, not prose) | not usable |

Sangraha — the largest systematic Indic corpus effort — holds **10.1M Konkani
tokens in total**. That single figure is the strongest external evidence
available: a well-funded, dedicated effort at national scale found roughly one
fiftieth of what our target would require.

### 2.4 Institutional sources — investigated, documented, rejected

| source | outcome |
|---|---|
| Vishwa Konkani Kendra eBooks | 9 children's picture books, <5,000 words total |
| Goa Konkani Akademi | catalogue of printed books; no full text exposed |
| Konkani Bhasha Mandal | no independent full-text site; grants page only |
| Goa University institutional repository | TLS chain fails verification from our environment; deferred, not rejected |

Per the quality rules, a catalogue entry is not corpus text and was not counted.

---

## 3. The comparison that frames the result

| | Marathi | Konkani |
|---|---:|---:|
| Internet Archive text items | 170,796 (GR alone) | 5,178 |
| web sites usable | 7 of 8 | 1 of 18 |
| largest prepared corpus | IndicCorpV2, ~27.8M rows | Sangraha, 10.1M tokens |
| manual words collected | 177,781,779 | 64,435,242 |

Both languages were worked with the same pipeline, the same filters and
comparable effort. The difference in outcome is the difference in what exists.

---

## 4. Why we did not use synthetic or machine-translated data

The 18 Aug announcement permits synthetic/MT data as a last resort where 500M
cannot be reached from real sources, with justification, and penalises its use
where real data was available.

We did not use it, for two reasons:

1. **Quality.** MT-generated Konkani would be Marathi or Hindi passed through a
   translation system - and Marathi contamination is the specific failure our
   language gate exists to prevent. It already rejected **1,771 segments** of
   real Marathi from the Konkani corpus. Injecting machine-translated Marathi
   deliberately would undo that work.
2. **The requirement is already met.** Konkani passes the ≥20% manual floor at
   **69.4%**, and §1.2 explicitly permits the total-token shortfall provided it
   is reported and justified. Synthetic data would inflate a number that the
   specification does not require us to hit, at the cost of corpus quality.

---

## 5. What the shortfall is *not*

- It is **not** a manual-collection shortfall. Manual is 69.4% of Konkani
  training tokens, more than three times the required floor.
- It is **not** an effort shortfall. 5,178 archive items enumerated and fetched
  in full, 18 sites probed with two extractors, 4 institutional sources
  investigated, 3 prepared corpora evaluated.
- It is **not** an unexamined claim. Every rejection above carries a measured
  reason recorded in `konkani/data/manifests/*.jsonl` and the run summaries.

**It is a measurement of how much Devanagari Konkani text exists in public
digital form**, and our answer is larger than any previously published figure we
could find - Sangraha's 10.1M tokens being the closest comparator.

---

## 6. The remaining gap, itemised

To reach ~500M, Konkani would need **334,184,908 more tokens**. Everything still
uncollected, measured:

| remaining source | tokens available |
|---|---:|
| AI4Bharat Sangraha, `gom` split (all splits) | 10,100,000 |
| IndicNLP Catalog Konkani prose entries | 0 |
| archive.org items not yet fetched | 0 (all 5,178 fetched) |
| web sites not yet probed | 0 (all 18 probed) |
| **total available** | **10,100,000** |

### 6.1 Two further Hugging Face datasets, ingested and measured (19 Aug)

A final systematic search of the Hugging Face dataset index - by language filter
rather than by card text, which is how the earlier search missed them - found two
Konkani datasets of substantial file size. Both were ingested through the full
pipeline rather than judged from their cards:

| dataset | size on HF | rows / files read | **unique words yielded** |
|---|---:|---:|---:|
| `cfilt/RoundTripOCR-konkani` (IIT Bombay) | 1.44 GB | 1,950,874 rows | **87,497** |
| `praveenkumar99/Konkani_Raw` | 1.37 GB | 370 segments | **11,689** |
| **total** | **2.81 GB** | | **99,186** |

**2.81 GB of published "Konkani" data yielded 99,186 usable words - 0.09% of our
corpus.** The reasons are measured, not inferred:

* RoundTripOCR is an OCR error-correction dataset that renders each source
  sentence in **hundreds of different fonts**. Of 1,950,874 rows, **1,880,882
  were exact duplicates** of text already seen. Its file size measures font
  coverage, not text.
* `Konkani_Raw` rejected **245 of 370** segments as non-Devanagari - much of what
  is published as "Konkani" is Roman-script (Romi) or English. Its
  `translated_konkani_*` files were excluded outright as machine translation.

This is the clearest available evidence for the shortfall. We did not estimate
the scarcity of Konkani text from dataset cards or search-result counts; we
ingested the largest published datasets in full and counted what survived
deduplication, script filtering and language identification.

Ingesting Sangraha in full would move Konkani from 33.2% to **35.2%** of target.
The remaining ~324M tokens do not exist in public digital Konkani. This is not a
statement about our search budget; it is the result of having exhausted the
searchable population and counted what was there.
