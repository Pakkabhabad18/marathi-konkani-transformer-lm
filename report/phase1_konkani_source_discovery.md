# Phase 1 — Konkani source discovery and probe

**Date:** 16 August 2026
**Status:** discovery + probe complete. **No bulk collection has been started.**
Awaiting review and approval before any large-scale run.

**Scope rule applied throughout:** Wikipedia is excluded as a discovery target,
per TA guidance. The Wikipedia material already collected is discussed only
where it affects the accounting, and §7 recommends what to do with it.

---

## 0. The headline, stated plainly

The Konkani shortfall we documented was **overstated, and we caused it.**

| | previously reported | measured 16 Aug 2026 |
|---|---:|---:|
| Internet Archive Konkani text items | **44** | **5,093** |

The old figure came from one query in `collect_archive_books.py`:

```
language:(Konkani OR Konknni OR Concani) AND mediatype:texts    ->     44
language:kok AND mediatype:texts                                ->  5,093
```

Archive.org's `language` field is free text, and cataloguers overwhelmingly
write the ISO 639-2 code `kok`, not the English word. **That query measured our
spelling list, not the Internet Archive** — and its answer was then used as the
central evidence that Konkani data does not exist.

Two independent faults cancelled to hide this: the query was wrong *and* the
enumeration never followed the pagination cursor. With a 44-item result set,
one page is the whole result set, so the missing cursor produced no visible
symptom. Both are fixed (§8).

**Consequence for the project:** Konkani manual collection is not exhausted. A
conservative estimate puts an additional **~30M manual words** within reach,
which moves the Konkani corpus ceiling from ~7.3M words to well over 100M.

---

## 1. How the probe was run

Discovery is separated from collection by design, in a script that has **no
seen-set and writes no manifest** (`konkani/scripts/discover_sources.py`). This
is deliberate: an earlier `--dry-run` marked all 14 candidate books as seen, so
the real run afterwards collected zero. The architectural fix is for the preview
path to have no way to consume anything.

For each sampled item the probe:

1. reads `/metadata/<id>` to resolve the **real** OCR filename and its byte size
   (never guessing `<id>_djvu.txt` — that assumption 404'd on every Marathi GR
   item, and one Konkani book happens to use the identifier form, which is
   exactly why the wrong guess looked safe);
2. downloads the OCR text;
3. runs it through the project's own `textnorm.normalize`,
   `scriptid.profile_script` and `scriptid.identify_marathi_konkani`;
4. records Devanagari / Kannada / Latin ratios, langid label and score, word
   count, and an accept/reject verdict with a reason.

Items measured by hand during this session: **8**. Bodies actually downloaded
and script-verified: **5**. This is a small sample and every estimate below
carries that caveat explicitly.

### 1.1 The finding that changes how we must filter

**Catalogue metadata does not predict body script, and neither does the title.**

Konkani is written in Devanagari (Goa) and Kannada script (coastal Karnataka).
Both are catalogued `language: kok`. Two sampled items carried a **Devanagari
title over a Kannada body**:

| identifier | title | body script measured |
|---|---|---|
| `20veashekddeachy0000drje` | 20व्या शेक्ड्यांचे कोंकणी म्हान मनिस् | **Kannada, 96.2%** |
| `27kavitha0000step` | 27 कविता | **Kannada** |

Filtering on `language`, or on the script of the title, would have silently
admitted Kannada-script Konkani into a Devanagari-only corpus (D-001). Only the
body text decides. Our existing pipeline already does the right thing — verified
below — but the reason it is right is now on the record.

### 1.2 Existing pipeline validated against the new material, unchanged

The three verbatim excerpts were run through the current `common/` modules with
no modification:

| item | script detected | Devanagari ratio | langid | verdict |
|---|---|---:|---|---|
| `acchev0000pund` | Devanagari | 0.969 | `kok` (−0.60, confident) | **accepted** |
| `Lokdhan` | Devanagari | 0.978 | `kok` (−1.00, confident) | **accepted** |
| `20veashekddeachy0000drje` | Kannada | 0.000 | undecided | **rejected, not Devanagari** |

No change to `scriptid.py`, `textnorm.py`, `dedup.py` or `manifest.py` was
required. The existing implementation is correct for this source.

---

## 2. Source inventory — 14 fields

### S-1 · Internet Archive, Konkani texts (`language:kok`) — **HIGH**

| # | field | value |
|---|---|---|
| 1 | Source name | `archive_org_konkani_books` (existing source, re-scoped) |
| 2 | URL | `https://archive.org/services/search/v1/scrape?q=language:kok AND mediatype:texts` |
| 3 | Type | books — literature, poetry, folklore, dictionaries, medical and educational non-fiction |
| 4 | Script | **Mixed Devanagari + Kannada**; must be split by measured body ratio |
| 5 | Estimated documents | **5,093 items** (verified via `advancedsearch.php`, raw JSON) |
| 6 | Estimated words | **~30M–50M usable** (§3) |
| 7 | Full text downloadable | **Yes** — `_djvu.txt` OCR layer, no auth |
| 8 | OCR required | **No** — Internet Archive has already OCR'd; we take the text layer |
| 9 | OCR quality sampled | Good. Clean Devanagari prose, correct conjuncts and matras; noise limited to stray punctuation and page furniture. Sample excerpts in `konkani/data/probe_samples/` |
| 10 | Language-ID confidence | `kok` at −0.60 and −1.00, both `confident=True`, 0 Marathi false-accepts in sample |
| 11 | Estimated usable yield | **~1,070 items / ~29.8M words** (conservative, §3) |
| 12 | Copyright / access | `access-restricted-item` **absent** on all sampled items → publicly readable. But `licenseurl` and `rights` are **absent too**: most are in-copyright books that IA has scanned and left open. Record as `not_stated`; **use for model training only, do not redistribute raw text** |
| 13 | Qualifies as MANUAL | **Yes — `MANUAL_OCR`.** We enumerate, fetch, extract, normalize, script-filter, language-check and deduplicate each item ourselves. The brief names "OCR from books/PDFs" first among manual methods |
| 14 | Accept / reject | **ACCEPT — highest priority.** This single source changes the feasibility conclusion for Model L |

Sampled items, measured (`djvu.txt` bytes read from item metadata):

| identifier | pages | djvu bytes | body script |
|---|---:|---:|---|
| `20veashekddeachy0000drje` | 450 | 2,046,727 | Kannada ✗ |
| `1konkanishabdasa0000pand` | 316 | 1,867,844 | unverified |
| `acchev0000pund` | 212 | 623,038 | Devanagari ✓ |
| `aanganatalyokany0000amru` | 146 | 573,548 | Devanagari ✓ |
| `aamachodotor0000drbh` | 128 | 442,440 | Devanagari ✓ |
| `Lokdhan` | 52 | 160,278 | Devanagari ✓ |
| `27kavitha0000step` | 62 | 140,343 | Kannada ✗ |
| `adimayecheule0000bala` | 106 | 77,113 | Devanagari ✓ |

median 507,994 bytes · mean 741,416 bytes · measured **15.86 bytes/word** on
Devanagari Konkani, discounted 15% for OCR page furniture.

### S-2 · Konkani Vishwakosh (encyclopedia), Goa University — **HIGH**

| # | field | value |
|---|---|---|
| 1 | Source name | `archive_org_konkani_vishwakosh` (**new, separate source name**) |
| 2 | URL | `archive.org/details/konkanivishwakos0000mano` and `.../KONKANIENCYCLOPEDIAALLFOURVOLUMESINONEDEVANAGARISCRIPTAPROJECTOFGOAUNIVERSITY` |
| 3 | Type | **encyclopedia** — deliberately its own genre |
| 4 | Script | **Devanagari** (confirmed: title page, editorial matter and body all Devanagari) |
| 5 | Estimated documents | 4 volumes; Khand-1 alone is **900 pages** |
| 6 | Estimated words | Khand-1 `djvu.txt` = 12,568,207 bytes → **~690k words**; combined item 12,689,162 bytes. Realistic total **~0.7M–2.7M words** depending on how many volumes are separately scanned |
| 7 | Full text downloadable | **Yes**, `_djvu.txt` |
| 8 | OCR required | No |
| 9 | OCR quality | Good on the pages sampled |
| 10 | Language-ID | Devanagari Konkani; encyclopedic register, high proper-noun density |
| 11 | Estimated usable yield | **~0.7M–2.7M words** |
| 12 | Copyright | **Released under a free licence** (CC-BY-SA) by Goa University via CIS-A2K in 2013 — the cleanest licence position of any Konkani source we have found |
| 13 | Qualifies as MANUAL | **Yes — `MANUAL_OCR`** |
| 14 | Accept / reject | **ACCEPT**, but as a **separate source name**, so that a single reference work with a distinctive register cannot silently dominate the corpus. Cap its share explicitly at split time |

### S-3 · Goa University institutional repository (DSpace) — **MEDIUM, blocked here**

| # | field | value |
|---|---|---|
| 1 | Source name | `goa_university_ir` (proposed) |
| 2 | URL | `https://irgu.unigoa.ac.in/drs/` — Konkani community `handle/unigoa/14`, "Konkani language and literature" `handle/unigoa/132` |
| 3 | Type | theses, dissertations, journal articles, literary works |
| 4 | Script | Mixed Devanagari / English |
| 5 | Estimated documents | ~303 Konkani items (prior evidence, **not yet re-verified**) |
| 6 | Estimated words | unknown — see field 14 |
| 7 | Full text downloadable | **Unverified.** DSpace records frequently expose metadata + a contents page only |
| 8 | OCR required | Likely yes for scanned theses |
| 9 | OCR quality | not sampled |
| 10 | Language-ID | not sampled |
| 11 | Estimated usable yield | **unknown; do not budget for it** |
| 12 | Copyright | institutional repository, typically permissive for research |
| 13 | Qualifies as MANUAL | Yes if full text is retrievable (`MANUAL_OCR` / `MANUAL_SCRAPE`) |
| 14 | Accept / reject | **DEFERRED — could not be probed from this environment.** `irgu.unigoa.ac.in` presents an incomplete TLS chain (`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`). Browsers accept it via AIA fetching; Python's `requests` will not. This is *our* environment's limitation, not a property of the source, and must not be recorded as "source unavailable" |

**Next step for S-3 (do not skip):** DSpace exposes OAI-PMH at
`/drs/oai/request`. Run from the Mac:

```bash
curl -sS "https://irgu.unigoa.ac.in/drs/oai/request?verb=Identify" | head -40
curl -sS "https://irgu.unigoa.ac.in/drs/oai/request?verb=ListSets" | grep -i konkani
```

If the certificate also fails locally, fetch the intermediate certificate once
and pin it — do **not** disable verification globally.

### S-4 · Vishwa Konkani Kendra eBooks — **LOW, reject**

| # | field | value |
|---|---|---|
| 1 | Source name | — |
| 2 | URL | `https://www.vishwakonkani.org/konkani-ebooks/` |
| 3 | Type | children's picture books |
| 4 | Script | 7 Devanagari, 2 Kannada |
| 5 | Estimated documents | **9 titles**, no pagination |
| 6 | Estimated words | **< 5,000 total** — picture books, a few dozen words each |
| 7 | Full text downloadable | No direct links; delivered as ePub via iBooks / Readium |
| 8–11 | — | not worth sampling |
| 12 | Copyright | publisher-controlled |
| 13 | Qualifies as MANUAL | would, but the volume is negligible |
| 14 | Accept / reject | **REJECT.** Nine dialect picture books cannot move a corpus that needs tens of millions of words. Rejected on measured volume, not on principle |

### S-5 · Goa Konkani Akademi — **LOW**

Site loads; navigation is largely English with a Devanagari heading
*"अकादेमीन उजवाडायिल्लीं पुस्तकां"* (books published by the Academy). No PDF
links and no full-text bodies exposed on the pages retrieved. This is a
**catalogue of printed books, not a text source** — and per the quality rules, a
catalogue entry is not corpus text. **REJECT as a text source**; keep it as a
*title list* to look up on archive.org, where the scans actually live.

### S-6 · Konkani Bhasha Mandal — **LOW**

No independent full-text site found. The Directorate of Official Language page
(`dol.goa.gov.in/konkani-bhasha-mandal/`) is a **grants table**, not literature.
KBM's own material appears as printed books and as e-books released through
third parties. **REJECT as a crawl target**; its publication list is, like S-5,
useful only as a lookup list for archive.org.

### S-7 · AI4Bharat / IndicNLP catalog — **LOW as a source, useful as a pointer**

The IndicNLP Catalog names exactly **one** Konkani-specific resource: the
AI4Bharat StoryWeaver transliteration dataset — transliteration pairs, not
running prose. Sangraha's `gom` split remains **10.1M tokens total**.

Neither is manual data and neither should be counted as such. Their value here
was purely diagnostic: they confirm that *prepared* Konkani corpora are small,
which is a different claim from "Konkani text does not exist". Archive.org shows
the raw material does exist; nobody had packaged it. **REJECT as a source.**

---

## 3. Estimated yield — MEASURED, 120-item probe (supersedes §3.1)

Run 16 Aug 2026, `discover_sources.py --sample 120`. Machine output in
`report/phase1_konkani_source_discovery_measured.md`.

| quantity | measured |
|---|---:|
| unique candidates enumerated | **5,178** |
| items fetched and measured | **120** |
| **fetch failures** | **0** — every item resolved an OCR text layer |
| accepted (Devanagari body, not Marathi) | **58** |
| rejected, not Devanagari | 59 |
| rejected by the Marathi language gate | **3** |
| **Devanagari accept share** | **48.3%** (95% CI 39.4%–57.3%) |
| median words / accepted item | 16,776 |
| mean words / accepted item | 28,852 |

Extrapolation, with the estimator named honestly:

| | items | words |
|---|---:|---:|
| **expected (mean × N)** | **2,503** | **72,207,900** |
| conservative floor (median × N) | 2,503 | 41,985,295 |
| 95% CI on the share, mean-based | 2,040–2,966 | 58.9M–85.6M |

`mean × N` is the unbiased estimator of a **sum** and is the expected yield.
`median × N` is not an estimator of a sum at all — on a right-skewed
distribution it sits systematically low. Use it as a planning floor. (An earlier
version of the script called the median figure "the one to quote"; that was
wrong and is corrected in D-021.)

Three results deserve separate mention:

- **0 fetch failures in 120.** The Marathi GR source ran at 61% failure and
  needed a timeout and worker rebuild. This source does not; collection should
  be fast and near-lossless.
- **3 items rejected by the Marathi gate.** Marathi contamination in
  `language:kok` is real but small (2.5%), and the gate catches it. This is
  direct evidence for the cross-corpus independence claim.
- **`newspapers_miscellaneous` (188 items) and
  `christian-tracts-literature-society` (52)** appear among parent collections —
  additional genres beyond books, worth tagging separately in the manifest.

### 3.1 Earlier hand-sampled estimate (n=8, superseded)

Measured inputs:

- population: **5,093** items (`language:kok AND mediatype:texts`, verified)
- bytes per word on Devanagari Konkani: **15.86** (measured on probe excerpts)
- OCR page-furniture discount: **15%** → effective 18.24 bytes/word
- median `djvu.txt`: **507,994 bytes** → **~27,856 words/item**
- mean `djvu.txt`: 741,416 bytes → ~40,656 words/item
- Devanagari body share in sample: **3 of 5 body-checked = 60%**;
  5 of 7 including strong inference = 71%

Because the sample is small, the estimate is given as a band, and the headline
uses the **median** (item sizes are strongly right-skewed — an encyclopedia is
100× a poetry booklet, so the mean flatters the total) and an extra 30%
usability haircut for fetch failures, duplicates and short items:

| assumed Devanagari share | usable items | **words (median-based)** | words (mean-based) |
|---:|---:|---:|---:|
| 30% *(pessimistic)* | 1,070 | **29,792,874** | 43,482,649 |
| 40% | 1,426 | 39,723,832 | 57,976,866 |
| 50% | 1,783 | 49,654,791 | 72,471,082 |
| 60% *(sample rate)* | 2,139 | 59,585,749 | 86,965,298 |

**Quote the 30% row.** It is below the measured sample rate and it is still an
order of magnitude more than everything Konkani we have collected to date.

---

## 4. Impact on the manual percentage — and the constraint flips

Current Konkani manual: **1,456,680 words**. Downloaded pool available
(`omdeep22/Konkani_books_corpus-v2`): **47,016,182 words**, measured.

Updated to the measured 120-item probe:

| scenario | manual words | downloaded admitted | total words | **manual %** | ≈ tokens |
|---|---:|---:|---:|---:|---:|
| today | 1,456,680 | 5,826,720 | 7,283,400 | **20.0%** | 13.0M |
| S-1 floor (median × N) | 43,441,975 | 47,016,182 | 90,458,157 | **48.0%** | 161.0M |
| **S-1 expected (mean × N)** | **73,664,580** | 47,016,182 | **120,680,762** | **61.0%** | **214.8M** |

If the overlap test (§5) forces the downloaded corpus to be dropped entirely,
Konkani still lands at **77.3M tokens (floor)** to **131.1M tokens (expected)**,
100% manual — comfortably above the 20% rule with no downloaded data at all.

**The binding constraint flips.** Today, manual data is the cap and we discard
~41M words of usable downloaded text to hold the 20% floor. After S-1, `4 ×
manual` is ~125M words of downloaded capacity but only ~47M downloaded words are
known to exist — so the **downloaded pool** becomes the cap, and the manual
share rises to ~40% on its own. The 20% requirement stops being tight.

This also means the corpus grows roughly **10×**, from ~13M to ~139M tokens,
without relaxing any quality rule.

---

## 5. Risk that must be checked before the run: S-1 ↔ books-corpus overlap

`omdeep22/Konkani_books_corpus-v2` is described only as "digitized books" — the
dataset card names **no provenance at all**: no scanning source, no book count,
no title list. Its rows average 7.52 words, i.e. OCR *lines*.

That is exactly what an archive.org `_djvu.txt` dump looks like. **There is a
real possibility the downloaded corpus is derived from the same scans we are
about to collect.** If so, counting ours as MANUAL and theirs as DOWNLOADED
would be double-counting the same text on both sides of the ratio — which would
corrupt the one number the whole phase turns on.

**Test before collecting at scale** (cheap, decisive): take the `--- SOURCE:`
marker values from the books corpus, and the titles of the 5,093 archive items,
and compute overlap; then content-hash a few hundred paragraphs from each side
and intersect. `tools/cross_corpus_check.py` already does hash intersection and
needs only to be pointed at the two sets.

**If overlap is high, the correct response is to drop the downloaded corpus, not
ours.** Collecting the originals ourselves is legitimately manual, better
attributed, and better cleaned. A corpus of 31M manual words with no downloaded
component passes the 20% rule trivially.

---

## 6. Ranking and recommended order of collection

| rank | source | value | why this order |
|---|---|---|---|
| 1 | **S-2 Konkani Vishwakosh** | HIGH | 2 identifiers, ~0.7–2.7M words, **free licence**, one afternoon. Highest words-per-unit-of-risk in the project |
| 2 | **S-1 archive.org `language:kok`** | HIGH | the decisive source. Run overlap test (§5) **first**, then enumerate → probe 120 → collect |
| 3 | **S-3 Goa University IR** | MEDIUM | genuinely unknown; resolve TLS and check OAI-PMH before budgeting anything |
| 4 | S-5 / S-6 title lists | LOW | use only as lookup lists against archive.org |
| — | S-4, S-7 | REJECT | measured too small / not a text source |

Recommended sequence:

```bash
# 1. enumerate only — no downloads, nothing marked seen
python3 konkani/scripts/discover_sources.py --enumerate-only

# 2. probe a real sample and tighten the Devanagari share
python3 konkani/scripts/discover_sources.py --sample 120

# 3. review report/phase1_konkani_source_discovery_measured.md
#    -> then, and only then, approve the collection run
```

---

## 7. What to do about the existing Wikipedia material

Konkani manual is currently **95.8% self-scraped Wikipedia** (1,395,235 of
1,456,680 words). The TAs advised against relying on Wikipedia, and that
concentration is the weakest part of the Model L story.

If S-1 delivers even the pessimistic 30M words, Wikipedia falls to **~4.5%** of
Konkani manual. Recommendation: **keep it, reclassify it as a minor source, and
stop citing it as the backbone.** Deleting collected, correctly-attributed data
to make a point is worse than reporting it honestly at its new small share.

---

## 8. Files created and modified

| file | change | reason |
|---|---|---|
| `konkani/scripts/discover_sources.py` | **created** | probe-only discovery: multi-spelling enumeration with cursor pagination, metadata-resolved OCR filenames, sampled body-script and langid measurement, yield extrapolation. Has no seen-set and writes no manifest, so it cannot consume the work it previews |
| `konkani/scripts/collect_archive_books.py` | **modified — 2 changes** | (a) `QUERY` now includes `kok` and `gom`; the old query returned 44 of 5,093 items. (b) `list_items()` now follows the scrape cursor; the old version read one page only. Superseded docstring kept in place and marked, per the standing rule |
| `konkani/data/probe_samples/*.txt` | **created** (3 files) | verbatim OCR excerpts used as evidence for the script/langid measurements in §1.2 |
| `report/phase1_konkani_source_discovery.md` | **created** | this document |
| `report/phase1_konkani_source_discovery_measured.md` | generated by the script on the Mac | machine-written enumeration + probe results |
| `report/phase1_decisions.md` | appended D-018, D-019, D-020 | see below |
| `report/phase1_viva_log.md` | appended A-016 | see below |

**Not modified, deliberately:** `common/scriptid.py`, `common/textnorm.py`,
`common/dedup.py`, `common/manifest.py`, `common/newscrawl.py`,
`tools/make_splits.py`, and every Marathi script. The probe demonstrated the
existing implementation is correct on the new material (§1.2); there was no
technical justification to touch it.

---

## 9. Corrections log (this session)

| previous claim | measurement | correction |
|---|---|---|
| "The Internet Archive holds 44 Konkani items" | `language:kok AND mediatype:texts` → **5,093** | **Wrong by ~115×.** Cause: querying the English language name, never the ISO code `kok`. Fixed in D-018 |
| Enumeration reads all matching items | one page, no cursor | Fixed. Invisible before only because 44 items fit in one page |
| Konkani manual collection is complete | ~30M more words identified | **Not complete.** Reopened |
| "~11.4M tokens is the measured feasible ceiling for Konkani" | ~139M tokens at the pessimistic estimate | Ceiling raised ~10×. The old figure was a true consequence of a false premise |
| Metadata `language` identifies usable items | 2 of 5 sampled had Devanagari titles over Kannada bodies | Body script must be measured, never inferred from catalogue or title |
| Sangraha/IndicNLP scarcity proves Konkani text is scarce | archive.org holds thousands of unpackaged books | Conflated "no prepared corpus exists" with "no text exists". Only the first was true |

---

## 10. Open questions for review

1. **Approve the S-1 collection run?** Nothing large-scale starts without this.
2. **Overlap test first (§5)** — recommended, and it may retire the downloaded
   corpus entirely.
3. **Cap the encyclopedia's share?** Proposed: separate source name, and an
   explicit ceiling at split time.
4. **Human-translated works** (e.g. Amrita Pritam → Konkani) — these are genuine
   Konkani prose and not machine translation. Proposed: accept, tag
   `translated_work` in the manifest so the share is visible and reportable.
