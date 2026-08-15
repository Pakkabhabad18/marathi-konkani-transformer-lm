# Phase 1 — Source Inventory

**Last updated:** 14 August 2026
**Purpose:** track every candidate and accepted corpus source for Model H (Marathi) and
Model L (Konkani), with the evidence needed to defend each one in the viva.

**Verification status key**

| Tag | Meaning |
|---|---|
| `VERIFIED` | Dataset card / page read during this audit; figures below come from it. |
| `CANDIDATE` | Identified as plausible, **not yet verified**. Must be checked before use. |
| `REJECTED` | Deliberately excluded, with the reason recorded. |

**Collection-type key** — the spec counts as *manual* only what we gather and clean ourselves:
OCR from books/PDFs, scraping + cleaning pages we collect, typed/transcribed text. Anything
downloaded as a ready-made corpus is *downloaded*, however much cleaning we do afterwards.

---

## 1. Accounting rule that governs this table

Required: `manual_tokens / total_tokens ≥ 0.20`, therefore `total ≤ 5 × manual`.

Every downloaded source added to a corpus **raises the manual target**. Sources are therefore
admitted in priority order, and the downloaded contribution is capped to whatever the manual
total will support — not maximised.

---

## 2. Model L — Konkani (Devanagari only)

**Script decision:** Devanagari only. Konkani is also written in Roman (Romi) and Kannada
scripts; those are excluded from the training corpus and documented as a separate sub-corpus so
the decision is visible rather than hidden. Rationale: Devanagari is the official script of
Konkani in Goa, it is what the largest available corpus uses (99.78% of non-whitespace
characters), and a ~25M-parameter model has too little capacity to spend on learning two
orthographies of the same language.

### 2.1 Accepted / in use

| Field | Value |
|---|---|
| **Source name** | Konkani Books Corpus v2 |
| **Type** | Digitized books / literature (aggregated dataset) |
| **Location** | `omdeep22/Konkani_books_corpus-v2` (Hugging Face) |
| **Status** | `VERIFIED` — dataset card read 14 Aug 2026 |
| **Script** | Devanagari (99.78% of non-whitespace chars, recomputed) |
| **Collection method** | `datasets` streaming download |
| **Manual or downloaded** | **Downloaded** |
| **License** | MIT (declared on card) |
| **Raw size** | 1.02 GB · 8,231,150 rows |
| **Clean size** | 8,222,553 usable records · 379,610,529 chars · 61,805,534 words |
| **Tokens** | ~86.75M *preliminary* (fertility measured on training data — will change) |
| **Duplicate risk** | **High.** 7.52 words/record means line-fragmented OCR, not documents. Short lines duplicate heavily. Exact + near-duplicate dedup mandatory. |
| **Already in another dataset?** | Unknown — provenance is only "digitized books, literature, and long-form cultural texts". No per-book manifest. Overlap with Sangraha `gom` must be checked by hash. |
| **Known issues** | Card warns of "large gaps between words and excessive line breaks" from OCR. Whitespace normalization required. Language purity unverified — needs a Marathi-contamination pass. |
| **Notes** | Cannot be counted as manual. Record the vague provenance honestly in the report rather than glossing it. |

| Field | Value |
|---|---|
| **Source name** | Konkani Wikipedia (self-collected) |
| **Type** | Encyclopedia |
| **Location** | `gom.wikipedia.org` MediaWiki API, `action=query&generator=allpages` |
| **Status** | `VERIFIED` — collected and filtered by us |
| **Script** | Mixed: 2,613 Devanagari / 1,727 Roman / 593 mixed pages |
| **Collection method** | Our own API crawler + wikitext cleaner (`collect_wikipedia_sample.py`) |
| **Manual or downloaded** | **Manual** (self-scraped) — but see caveat |
| **License** | CC BY-SA 4.0 — attribution required in the report |
| **Raw size** | 4,933 pages · 2,275,434 words |
| **Clean size** | 3,999 pages retained (81.07%) · 2,252,016 words |
| **Tokens** | 5.16M all-scripts; **~2.6M Devanagari-only** under the script decision |
| **Duplicate risk** | Low internally; overlaps Sangraha `gom` (which includes WikiMedia content) — must dedup against it. |
| **Status decision** | **Keep as a documented experiment and a secondary source.** Do not treat as the primary corpus (TA guidance) and do not lean on it for the manual quota — it is a well-known public dataset regardless of who scraped it. |

### 2.2 Candidate sources — to verify before use

These are the non-Wikipedia directions the TAs pointed toward. **None has been verified yet.**
For each, the checks required are: is the text actually Konkani (not Marathi), which script, how
much text, how it can be accessed, licensing, and overlap with what we already hold.

| Source | Type | Script | Access | Manual? | Why it matters | Checks needed |
|---|---|---|---|---|---|---|
| Internet Archive — Konkani collections (e.g. Goa Konkani Akademi uploads) | Digitized books | Devanagari + Romi | `archive.org` API / full-text endpoints | **Manual (OCR)** | Strongest manual-token evidence available. Confirmed to exist: Goa Konkani Akademi titles are on archive.org. | Which items have usable OCR text vs image-only; per-item rights statement; Devanagari vs Romi split; total page count |
| Sunaparant | Daily newspaper | Devanagari | Web / archive pages | **Manual (scrape)** | The Devanagari Konkani daily — highest-volume natural-language Konkani in existence | Site structure, archive depth, robots.txt, whether an article archive is publicly reachable |
| Goa Konkani Akademi publications | Literature, periodicals | Devanagari | Institutional | **Manual** | Official state literary body | What is available digitally vs print-only; permission terms |
| Goa government / Rajya Patra material in Konkani | Official documents | Devanagari | Government portals | **Manual** | Clean licensing, formal register — balances a literature-heavy corpus | Whether Konkani versions exist as text PDFs vs scans |
| Konkani educational material / textbooks (Goa Board) | Educational | Devanagari | Public PDFs | **Manual (OCR)** | Clean, well-edited prose | Availability, copyright status |
| `ai4bharat/sangraha` — `gom` split | Aggregated web/OCR | Devanagari | Hugging Face | Downloaded | **10.1M tokens total across all splits** — `VERIFIED`. Small, but it is the ceiling evidence for the shortfall justification. | Overlap with books corpus and Wikipedia |
| Vishwakonkani / Mangaluru Konkani outlets | Periodicals | Devanagari + Kannada script | Web | **Manual (scrape)** | Non-Goa register diversity | Script mix — Kannada-script Konkani is out of scope under our decision |
| Dalgado Konknni Akademi | Literature | Romi | Institutional | — | `REJECTED` for the training corpus (Romi), keep as documented excluded sub-corpus | — |

### 2.3 Rejected

| Source | Reason |
|---|---|
| Romi (Roman-script) Konkani generally | Script decision — excluded from training corpus, documented as separate sub-corpus |
| Kannada-script Konkani | Same |
| Sangraha *synthetic* split | Machine-translated / romanized content, not naturally-occurring Konkani |
| Any site scraped purely to raise token count | Explicitly against the working brief and the spirit of the spec |

---

## 3. Model H — Marathi

### 3.1 Downloaded sources — verified, ample

| Field | Value |
|---|---|
| **Source name** | IndicCorp v2 (`mar_Deva`) |
| **Type** | Large-scale web/news crawl |
| **Location** | `ai4bharat/IndicCorpV2` (Hugging Face) |
| **Status** | `VERIFIED` — dataset card read 14 Aug 2026 |
| **Script** | Devanagari |
| **Manual or downloaded** | **Downloaded** |
| **License** | CC-0 (public domain) — cleanest licensing available |
| **Size** | ~27.8M rows for Marathi |
| **Duplicate risk** | Moderate — web crawl, expect boilerplate and near-duplicates |
| **Notes** | First choice for the downloaded bucket on license grounds alone. |

| Field | Value |
|---|---|
| **Source name** | Sangraha — `mar` verified + unverified |
| **Type** | Curated web + OCR-extracted PDFs (verified); filtered existing corpora (unverified) |
| **Location** | `ai4bharat/sangraha` |
| **Status** | `VERIFIED` — dataset card read 14 Aug 2026 |
| **Script** | Devanagari |
| **Manual or downloaded** | **Downloaded** |
| **License** | CC-BY-4.0 — attribution required |
| **Size** | Verified 2,827.0M tokens · Unverified 652.1M tokens |
| **Duplicate risk** | **High overlap with IndicCorpV2** — same group, overlapping crawls. Cross-dedup mandatory before counting. |
| **Notes** | *Synthetic* split (10,816.7M tokens) is `REJECTED` — machine-translated and romanized WikiMedia content. |

Between these two, the ~400M downloaded-token requirement is comfortably met. **The downloaded
side of Marathi is not a risk. The manual side is the whole problem.**

### 3.2 Manual collection — the critical path (~100M tokens)

Target: **≥100M manually collected Marathi tokens** ≈ 50M words ≈ ~125k articles at 400 words.

Routes selected for this project, in priority order by tokens-per-hour-of-effort:

| Priority | Route | Type | Manual? | Expected contribution | Status |
|---|---|---|---|---|---|
| 1 | Marathi news & opinion sites, sitemap-driven crawl, cleaned by us | News | **Manual (scrape)** | Bulk of the 100M | `CANDIDATE` — sites to be selected and robots.txt checked individually |
| 2 | Marathi blogs, magazines, literary portals | Long-form prose | **Manual (scrape)** | Register diversity; 10–20M | `CANDIDATE` |
| 3 | Maharashtra government / educational PDFs, textbooks, gazettes | Official / educational | **Manual (OCR or text extract)** | 5–15M; clean licensing | `CANDIDATE` |
| 4 | Public-domain Marathi books via OCR (archive.org) | Literature | **Manual (OCR)** | 5–15M; strongest viva evidence per token | `CANDIDATE` |
| 5 | Books and articles (user-supplied) | Mixed | **Manual** | TBD | `CANDIDATE` |

**Selection rules for every candidate site, to be applied before crawling:**

1. `robots.txt` permits the paths we intend to fetch; we honour crawl-delay.
2. The site publishes substantial original Marathi prose, not aggregated wire copy duplicated
   across every outlet.
3. Terms of use do not prohibit automated access for research.
4. A stable sitemap or dated archive index exists, so the crawl is resumable and countable.
5. Not already represented in IndicCorpV2 / Sangraha — checked by URL and content hash after a
   pilot crawl of ~1000 pages, *before* committing to a full crawl.

Rule 5 matters more than it looks: IndicCorpV2 and Sangraha are themselves built from Marathi
news crawls. Scraping the same sites and calling the result "manual" would be self-deception,
and cross-dedup would delete most of it anyway. **The pilot-crawl overlap check is what makes
the manual claim defensible**, and it should be run and reported for every accepted site.

### 3.3 Per-document record kept for every manually collected item

Required as evidence for the manual token count:

```
source_name, source_url, collection_method, access_date,
raw_chars, clean_chars, words, tokens,
preprocessing_applied, script, langid_score, content_hash
```

Stored alongside each corpus shard as a metadata CSV, mirroring the pattern already used in
`konkani_wikipedia_sample_metadata.csv` — that pattern was a good instinct and should be
generalised to every source.

---

## 4. Running totals

| Language | Manual tokens | Downloaded tokens | Total | Manual % | Target | Gap |
|---|---|---|---|---|---|---|
| Konkani | ~2.6M (Wikipedia, Devanagari subset) | ~87M (books, preliminary) | ~90M | **~2.9%** | ≥20% | **~15M manual short** |
| Marathi | 0 | 0 | 0 | — | ≥20% | **~100M manual short** |

All figures are preliminary and measured with inconsistent tokenizers. They will be replaced by
a single recount with the final per-language tokenizer before anything is reported as official.

---

## 5. Open questions to resolve with the TAs

1. Does self-scraped Wikipedia count toward the 20% manual requirement? (We are assuming it
   does technically, but are not relying on it.)
2. Is a ~100M-token Konkani corpus with a documented 500M shortfall acceptable, given that the
   largest systematic Indic corpus effort (Sangraha) contains only 10.1M Konkani tokens?
3. For closely-related language pairs like Marathi/Konkani, what standard of evidence is
   expected for the "no shared documents" constraint?
