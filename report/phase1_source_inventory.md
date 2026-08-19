# Phase 1 — Source Inventory

**Status: measured, not projected.** Every figure below comes from a real
collection run recorded in `<lang>/data/manifests/*.jsonl`, not from an estimate.
An earlier version of this document was written before collection began and
contained claims that later measurement disproved; those corrections are kept in
§5 rather than edited away.

**Last updated:** 16 August 2026

---

## 1. The accounting rule that governs everything here

The specification requires that at least 20% of final training tokens come from
manual collection, for **both** languages:

```
manual / total >= 0.20      =>      total <= 5 x manual
```

This is enforced mechanically in `tools/make_splits.py`, which subsamples
downloaded sources until the inequality holds. So the manual total is not a
target to hit — it is the **hard cap on corpus size**. Each manual word admits
four downloaded words.

Classification is a typed choice at the point of collection
(`common/manifest.py::CollectionType`), not a label applied later:

| Type | Counts as manual? | Used for |
|---|---|---|
| `MANUAL_OCR` | yes | OCR text layers of scanned books and documents |
| `MANUAL_SCRAPE` | yes | pages we discovered, fetched and cleaned ourselves |
| `MANUAL_TRANSCRIBED` | yes | typed or transcribed text (unused) |
| `DOWNLOADED_DATASET` | **no** | ready-made public corpora |

A downloaded corpus does not become manual because we cleaned it, however much
cleaning we did. The Konkani books corpus is the case in point: its underlying
material was originally digitised from books, but *we* obtained it as a prepared
Hugging Face dataset, so it is `DOWNLOADED_DATASET`.

---

## 2. Model H — Marathi

### 2.1 Manual sources (accepted)

| Source | Type | Documents | Words | Words/doc |
|---|---|---:|---:|---:|
| `archive_org_maharashtra_gr` | `MANUAL_OCR` | 15,768+ | 48.7M+ | 939 |
| `news_esakal` | `MANUAL_SCRAPE` | 13,686 | 3,294,330 | 241 |
| `news_loksatta` | `MANUAL_SCRAPE` | 2,058 | 782,805 | 380 |
| `news_divyamarathi` | `MANUAL_SCRAPE` | 368 | 130,164 | 354 |
| `news_tv9marathi` | `MANUAL_SCRAPE` | 305 | 100,454 | 329 |
| `news_abplive_marathi` | `MANUAL_SCRAPE` | 73 | 28,233 | 387 |
| `news_lokmat` | `MANUAL_SCRAPE` | 85 | 23,236 | 273 |
| `news_maharashtratimes` | `MANUAL_SCRAPE` | 40 | 19,553 | 489 |
| **Total (still growing)** | | **32,383+** | **51.8M+** | |

**M1 — Maharashtra Government Resolutions, Internet Archive.**
Query `identifier:in.gov.maharashtra.gr.*` returns **170,796 items**, each a
scanned government resolution with an OCR text derivative. Enumerated through
the cursor-paginated scrape API; each document fetched, its text layer
extracted, then normalized, language-checked and deduplicated by us. Rights
fields are absent on these items and are recorded as `not_stated` rather than
assumed public domain.

*Why manual:* no ready-made GR corpus exists. We enumerate, fetch and clean
every document ourselves.

*Measured quality:* 86.0% Devanagari, 9.8% OCR noise, 4.1% self-repetition, most
common opening phrase covers only **0.6%** of documents — the corpus is
genuinely 170k distinct documents, not one template repeated.

**M2 — Marathi news and long-form sites.**
Sitemap-driven, `robots.txt` obeyed, publication-date floor of 2025-01-01.
**7 of 8 candidate sites verified usable** by `--probe`.

*Why the date floor matters:* IndicCorpV2 and Sangraha are themselves built from
Marathi news crawls. Both are fixed snapshots, so an article published after
their release cannot be in them. The floor makes the manual claim true by
construction; the cross-corpus hash check then proves it empirically.

*Measured quality:* 96.4% Devanagari, 5.1% OCR noise, 0.6% self-repetition, top
opening phrase 0.1%, langid `mr` 100%.

### 2.2 Downloaded source (accepted, capped)

| Field | Value |
|---|---|
| Source | `ai4bharat/IndicCorpV2`, config `indiccorp_v2`, split `mar_Deva` |
| Type | `DOWNLOADED_DATASET` |
| Licence | **CC-0 (public domain)** — cleanest available, hence preferred over Sangraha (CC-BY-4.0) |
| Available | ~27.8M rows |
| **Budget** | **4 × manual**, computed at run time from the manifests |
| Throughput | 1,515 rows/s measured; ~2.4 h for the full budget |
| Words/document | **80** — much shorter than our manual sources |
| langid `undecided` | **24.8%** |

The 24.8% undecided rate is a genuine quality contrast worth reporting: our
manual sources return `mr` 100%, because an 80-word crawl fragment often carries
too few function words to identify confidently.

### 2.3 Rejected for Marathi

| Source | Reason |
|---|---|
| `marathivishwakosh.org` | `robots.txt` unreachable on two probes |
| Sangraha *synthetic* split (10,817M tokens) | machine-translated and romanized WikiMedia content, not naturally-occurring Marathi |
| Horoscope / video / gallery / AMP web-story URLs | excluded by URL pattern — fragmentary or no body text |

---

## 3. Model L — Konkani (Devanagari only)

**Script decision (D-001):** Devanagari only. Konkani is also written in Roman
(Romi) and Kannada script; both are excluded and reported rather than silently
dropped. In the Wikipedia collection alone the excluded material was **1,203
Latin-script, 156 Kannada-script and 141 mixed pages**.

### 3.1 Manual sources (accepted — this is the complete list)

| Source | Type | Documents | Words | Notes |
|---|---|---:|---:|---|
| `konkani_wikipedia_selfcollected` | `MANUAL_SCRAPE` | 2,459 | 1,395,235 | CC BY-SA 4.0, attribution required |
| `archive_org_konkani_books` | `MANUAL_OCR` | 52 segments | 60,925 | 5 books of 14 candidates |
| `news_vishwakonkani` | `MANUAL_SCRAPE` | 2 | 520 | only usable site of 18 probed |
| **Total** | | **2,513** | **1,456,680** | |

### 3.2 Downloaded source (accepted, heavily capped)

| Field | Value |
|---|---|
| Source | `omdeep22/Konkani_books_corpus-v2` |
| Type | `DOWNLOADED_DATASET` |
| Licence | MIT |
| Collected | 40,597 documents, **47,016,182 words** |
| **Admitted to the corpus** | **~5.8M words** (4 × manual) |
| **Discarded to hold the 20% floor** | **~41M words** |

Discarding 41M words of usable text is deliberate. A large corpus that fails a
stated requirement is worth less than a smaller one that meets it.

*Structural note:* the raw dataset averages **7.52 words per row** — these are
OCR'd lines, not documents. `--- SOURCE:` marker rows delimit books, so the
ingester uses them to reassemble line fragments into contiguous documents with
real per-book attribution.

### 3.3 Sources investigated and found unusable

This section is the evidence for the shortfall the specification permits.

| Investigation | Method | Result |
|---|---|---|
| Internet Archive, `language:Konkani` | search API | **44 items**; ~25 Wikipedia ZIM dumps, 4 Wikipedia PDFs, 13–15 real books, of which **5** yielded Devanagari text |
| Konkani web publications | `--probe`, 18 sites, two extractors | **1 usable** (vishwakonkani) |
| Sangraha `gom` split | dataset card | **10.1M tokens in total**, all splits |

Breakdown of the 18-site probe: 6 unreachable `robots.txt`, 10 returned pages
with no extractable article text, 1 (`goanvarta`) publishes **Marathi** — caught
by the language gate, 6 of 8 sampled pages rejected as `langid_mr` — and 1 was
usable.

The `no_article_text` failures were retried with a second, block-level extractor
after the first probe, in case they reflected our `<p>`-only parser rather than
the sites. They did not: the result was unchanged.

**Comparison that frames the whole project:**

| | Marathi | Konkani |
|---|---:|---:|
| Internet Archive items | 170,796 | 44 |
| Web sites usable | 7 of 8 | 1 of 18 |
| Largest public corpus | IndicCorpV2, ~27.8M rows | Sangraha, 10.1M tokens |
| Manual words collected | 51.8M+ | 1,456,680 |
| Corpus ceiling at 20% | ~259M words | ~7.3M words |

### 3.4 Rejected for Konkani

| Source | Reason |
|---|---|
| Romi (Roman-script) Konkani | D-001; documented as an excluded sub-corpus |
| Kannada-script Konkani | D-001 |
| `goanvarta.net` | publishes Marathi; 6 of 8 sampled pages rejected by the language gate |
| Wikipedia-derived Archive items (`wikipedia_*.zim`, `gomwiki-*`) | already held from our own Wikipedia collection; counting twice would inflate the manual total |
| Sangraha *synthetic* split | machine-translated content |

---

## 4. Provenance recorded per document

Every document carries, in its manifest row:

```
source_name, source_url, collection_method, access_date, raw_chars,
clean_chars, words, tokens, preprocessing_applied, script, langid_score,
content_hash
```

`tokens` is deliberately `null` at collection time. Token counts depend on the
tokenizer, and Phase 1 requires one final tokenizer per language and one
internally consistent count. Words and characters are tokenizer-independent and
tracked continuously instead.

---

## 5. Corrections to earlier versions of this document

Kept visible rather than edited away, per the standing instruction not to
overwrite superseded results.

| Original claim | Measurement | Correction |
|---|---|---|
| "Internet Archive is the strongest manual route for Konkani" | 44 items, 5 usable books, 60,925 words | Demoted to a small but genuine source. It is a **major** Marathi source (170,796 items). |
| Konkani books corpus "83.54% Devanagari" | recomputed over non-whitespace characters | **99.78%** — the original divided by total characters including whitespace |
| Marathi manual "~65M words" (estimated from shard bytes) | manifest totals | **51.8M** — the byte-based estimate was 27% high; manifests are authoritative |
| Marathi GR source viable at scale | 2.1 docs/min, 393 h projected | Demoted to secondary; news scraping measured **758 docs/min** and became primary. archive.org later recovered to ~88/min and both now run. |
