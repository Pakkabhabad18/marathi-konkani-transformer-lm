# Phase 1 — Decisions Log

Every entry: the decision, why, and what would change our mind. Newest first
within each section. This is the short document; the long reasoning lives in
`phase1_viva_log.md`.

---

## D-001 · Konkani corpus is Devanagari only

**Decision.** The Konkani training corpus contains Devanagari script only. Romi
(Roman-script) and Kannada-script Konkani are excluded and documented as
separate sub-corpora.

**Why.** Devanagari is the official script of Konkani in Goa. The largest corpus
available to us is 99.78% Devanagari over non-whitespace characters. A ~25M
parameter model has little capacity to spare, and making it learn two
orthographies of the same language spends that capacity on transliteration
rather than on the language.

**What would change it.** If manual Devanagari collection stalls below ~5M
tokens, including Romi becomes the difference between having a corpus and not
having one, and the trade-off flips.

---

## D-002 · Tokenizers will be trained with `byte_fallback=True`

**Decision.** Every tokenizer from this point uses `byte_fallback=True`. The two
existing preliminary tokenizers are retained as evidence, never used for a
reported count.

**Why.** Both existing tokenizers were trained without it. Verified by loading
the models and counting byte pieces: **0** in each vocabulary. Consequence: any
character outside the learned inventory becomes `<unk>`, which is where the
46.36% and 7.19% UNK figures came from. With byte fallback, any UTF-8 string is
representable and UNK is ~0 by construction.

**What this changes downstream.** UNK rate stops being a useful comparison
metric, because it will be ~0 everywhere. The metric that actually decides
vocabulary size and script coverage is **fertility** — tokens per word and
characters per token on *held-out* text. Poorly-covered script then shows up as
expensive rather than as unrepresentable, which is the real signal.

---

## D-003 · The 46.36% UNK result is superseded, not deleted

**Decision.** `report/phase1_konkani_progress.md` keeps its original numbers.
Corrections are added as a clearly-labelled section; nothing is overwritten.

**Why.** The original conclusion — "Konkani tokenizer training should represent
both Devanagari and Roman-script Konkani" — was drawn from a measurement
artefact. But a documented wrong turn, with the diagnosis, is stronger viva
material than a clean number with no history. It demonstrates that we can debug
our own pipeline.

**Evidence for the supersession.** Both tokenizers were run on controlled inputs:

| Input | Devanagari tokenizer | Mixed tokenizer |
|---|---|---|
| Konkani (Devanagari) | 11 tokens, 0% UNK | 11 tokens, 0% UNK |
| Konkani (Roman) | 20 tokens, 45.0% UNK | 24 tokens, 8.3% UNK |
| English | 19 tokens, 47.4% UNK | 24 tokens, 4.2% UNK |
| Marathi | 16 tokens, 0% UNK | 16 tokens, 0% UNK |

The UNK rate tracks *Latin characters present*, not *Konkani orthography*.

---

## D-004 · The Devanagari percentage of the books corpus is 99.78%, not 83.54%

**Decision.** The corrected figure is 99.78% of non-whitespace characters. The
83.54% figure is retained in the progress report as the superseded value.

**Why.** `analyze_books_corpus.py` assigns `non_whitespace = total_characters`
and then divides by it. The variable name and its value disagree.

**Proof, using only the numbers already reported:** 379,610,529 total characters
− 317,109,283 Devanagari = 62,501,246 non-Devanagari, against 61,805,534 words.
That is **1.011 non-Devanagari characters per word** — the inter-word spaces and
essentially nothing else. Removing whitespace from the denominator gives
317,109,283 / 317,804,995 = **99.78%**.

**Consequence.** The corpus is effectively pure Devanagari. The apparent ~16%
"contamination" that partly motivated the mixed-script tokenizer never existed.

---

## D-005 · Internet Archive is a *small* Konkani source — correcting our own inventory

**Decision.** Internet Archive is demoted from "strongest manual route" for
Konkani to "small but genuine source, ~1–2M words". It remains a **major**
Marathi source.

**Why.** `phase1_source_inventory.md` was written before the holdings were
counted. Counted on 14 Aug 2026: `language:Konkani` returns **44 items total** —
~25 Wikipedia ZIM dumps, 4 Wikipedia-derived PDFs, ~13–15 actual books. One book
(`konkanibhashaman0000jbmo`) was fetched to confirm full text is downloadable
and is predominantly Devanagari, mixed with English and Kannada.

By contrast `identifier:in.gov.maharashtra.gr.*` returns **170,725 items**, each
with an OCR text derivative.

**Recorded as a correction, not an edit**, per the standing instruction not to
silently overwrite prior results. This is also the single most important number
for the Konkani shortfall justification: the low-resource language really is
low-resource, and we measured it rather than asserting it.

---

## D-006 · Maharashtra Government Resolutions are pilot source M1 for Marathi

**Decision.** M1 is the first Marathi manual source, piloted at 300 documents
before any scaling.

**Why.** Verified volume (170,725 items), verified extractable text layer,
verified Marathi metadata, clean bulk API with cursor pagination that makes the
job genuinely resumable, and public government records.

**Why it counts as manual.** No ready-made GR corpus exists to download. We
enumerate through a search API, fetch each document, extract the OCR text layer,
and perform all segmentation, normalization, language filtering and dedup
ourselves. Per the brief, downloading a ready-made corpus and cleaning it does
*not* qualify; this is not that.

**Known limitation, stated up front.** The register is narrow — bureaucratic
prose only. M1 can supply volume but must not be the only manual source, or
Model H learns government circulars and nothing else. M2 (news, literary,
educational) exists to correct the register balance, not just to add tokens.

---

## D-007 · Marathi news scraping is deferred until after the M1 pilot

**Decision.** No news crawling begins until M1 reports and an overlap check has
run.

**Why.** IndicCorpV2 and Sangraha are themselves built from Marathi news crawls.
Scraping the same sites and labelling the output "manual" would be
self-deception, and cross-dedup would delete most of it anyway. A ~1000-page
pilot crawl with a content-hash overlap check against the downloaded corpora is
what makes the manual claim defensible — and it must come before the full crawl,
not after.

---

## D-008 · Language identification uses closed-class function words, not an off-the-shelf model

**Decision.** Marathi/Konkani discrimination uses a function-word marker
discriminator (`common/scriptid.py`), with abstention.

**Why.** Character statistics cannot separate the two: during the audit the
Konkani tokenizer processed Marathi at 0% UNK and 3.81 chars/token, essentially
identical to its behaviour on Konkani. Off-the-shelf language identifiers are
trained with little or no Konkani and mislabel it as Marathi or Hindi. But
closed-class grammatical words differ reliably and are unavoidable in running
text: "and" is आणि vs आनी, "is" is आहे vs आसा, "I" is मी vs हांव. Content words
are borrowed between the two languages; function words are not.

**Design choice: abstention over guessing.** A document with fewer than 5 marker
hits, or with a marker score inside ±0.30, is labelled `undecided` and routed to
review rather than into a corpus. Precision on corpus membership matters far
more than recall — we can afford to discard ambiguous documents; we cannot
afford Marathi in the Konkani corpus.

**Measured.** ±1.000 separation on control sentences; a Konkani test document
was correctly rejected by the Marathi collector's language gate during pipeline
testing.

**What would change it.** If abstention rates on real data exceed ~20%, upgrade
to a character n-gram model trained on our own verified seed corpora.

---

## D-009 · Shared utility code, strictly separate data

**Decision.** `common/` holds normalization, script ID, dedup, manifest and
checkpoint code used by both languages. Data, tokenizers, vocabularies, splits,
statistics and weights remain completely separate per language.

**Why.** The specification forbids sharing *data, tokenizer, vocabulary and
weights*; it says nothing against shared library code, and it explicitly asks
for modular design. Sharing the normalizer is positively desirable: it
guarantees both corpora are normalized identically, which is a precondition for
the cross-corpus contamination check comparing like with like. If each pipeline
had its own copy of the normalizer, the two could drift — which is exactly how
`corpus_audit.py` and `analyze_books_corpus.py` ended up disagreeing about the
same statistic.

---

## D-010 · Token counts are not recorded at collection time

**Decision.** The manifest's `tokens` field is written as `null` during
collection. Words and characters are recorded instead. A single later pass fills
in token counts for the entire corpus using the final tokenizer.

**Why.** Token counts are a property of the tokenizer, not of the text. Writing
a count during collection, using whatever preliminary tokenizer existed that
day, is precisely how the earlier "~91.91M tokens" figure came to mix two
different tokenizers. Words and characters are tokenizer-independent and can be
tracked continuously; tokens cannot.

---

## D-012 · Marathi news scraping (M2) replaces archive.org (M1) as the primary manual source

**Decision.** M2 — sitemap-driven collection from Marathi news and long-form
sites — is the primary manual Marathi source. M1 (Maharashtra Government
Resolutions on the Internet Archive) is demoted to a secondary background
source; it is **not** deleted and keeps running.

**Why.** Measured, not assumed:

| | M1 | M2 |
|---|---|---|
| Throughput | 2.1 docs/min | **758 docs/min** |
| Fetch failures | 61–68% | ~0% |
| Time for ~55M words | **393 hours** | **3.1 hours** |

Four days were available. M1 was not a slow option; it was an impossible one.

The fault was external. M1's rejection breakdown was 80 `fetch_failed` against
6 `langid_undecided` and 4 `not_enough_devanagari` — our filters worked fine,
archive.org did not, and its second run returned 0 documents in 10.7 minutes
after the scrape API went down entirely.

**What would change it.** If archive.org recovers, M1 contributes more manual
tokens at no extra cost, since it is already running and fully resumable.

---

## D-013 · Manual news collection uses a publication-date floor

**Decision.** M2 collects only articles whose sitemap `lastmod` is on or after
2025-01-01 (`--since`, default conservative).

**Why.** IndicCorpV2 and Sangraha — our *downloaded* corpora — are themselves
built from Marathi news crawls. Scraping the same sites and calling the result
"manual" would be self-deception, and cross-deduplication would delete most of
it anyway.

Both public corpora are **fixed snapshots**. An article published after their
release cannot be in them. The date floor therefore makes the manual claim true
*by construction* rather than by assertion, and a content-hash check against the
downloaded corpora then proves it empirically. The date filter makes overlap
unlikely; the hash check makes it verified.

**Supporting evidence.** The pilot's articles carried August 2026 timestamps —
comfortably after any published snapshot of either corpus.

---

## D-014 · Boilerplate stripping is a corpus-quality decision, not cosmetics

**Decision.** Byline and timestamp furniture is stripped from every scraped
paragraph, and non-prose URL patterns are excluded before fetching.

**Why.** The probe surfaced this real example from lokmat:

```
By ऑनलाइन लोकमत | Updated: August 16, 2026 00:25 IST
2026-08-16T00:25:11+5:30 2026-08-16T00:25:40+5:30 - विकास कामांचा आढावा…
```

Two distinct harms if left in. First, the model would learn that Marathi
articles begin with an English date stamp. Second, and less obvious, identical
boilerplate across thousands of articles **inflates their pairwise similarity**,
so unrelated articles start looking like near-duplicates and legitimate text
gets deleted by the deduplicator.

URL exclusion (horoscopes, videos, galleries, AMP web-stories, panchang, live
blogs) is the cheaper half of the same decision: rejecting a page by its URL
costs nothing, while fetching and then rejecting it costs a request.

---

## D-015 · The 20% manual floor is enforced by discarding downloaded data

**Decision.** `tools/make_splits.py` subsamples downloaded sources until
`total <= 5 x manual`, capping to a 22% target so that split variance cannot
push training below the 20% floor.

**Why.** Measured on the real Konkani corpus: 1.13M manual words against 29.4M
downloaded words is **3.7% manual** - a clear failure. Enforcing the floor meant
discarding ~23.8M words of a perfectly good books corpus.

That trade is deliberate and worth stating plainly: **a large corpus that fails a
stated requirement is worth less than a smaller one that meets it.** The 500M
token figure is a target; the 20% ratio is a requirement. When they conflict, the
requirement wins.

**Why 22% and not 20%.** Capping exactly at the floor lands on the boundary, and
the stratified split then moves each split's ratio by a fraction of a percent -
enough to drop training to 19.99% and fail by a few hundred words. Observed in
testing before it could happen for real.

**Implementation note.** Downloaded sources are subsampled *proportionally*, so
the source mix of the downloaded portion is preserved rather than one source
disappearing entirely.

---

## D-016 · Konkani manual collection is deliberately diversified away from Wikipedia

**Decision.** Konkani manual data comes from three sources, not one: self-scraped
Wikipedia (1,395,235 words), OCR of Internet Archive books (60,925 words), and
vishwakonkani (520 words).

**Why.** Before this, Konkani manual was **99.95% self-scraped Wikipedia**. Even
though we wrote the crawler and cleaned the wikitext ourselves - which meets the
brief's definition of manual collection - the TAs advised specifically against
relying on Wikipedia. A manual claim resting almost entirely on one well-known
public dataset is weak however legitimately it was gathered.

OCR from digitised books is the *first* example the brief gives for manual
collection. It is the strongest form of the claim available to us.

**Honest accounting of the size.** The books contribute 60,925 words - about 4%
of Konkani manual. This does not transform the corpus. It changes what the manual
claim rests on, and it raises the corpus ceiling by ~305k words.

**Why so little.** Of 14 candidate items, only 5 contributed and 267 segments
were rejected as non-Devanagari. Konkani books on the Internet Archive include
Romi (Latin script) and Kannada-script volumes, which decision D-001 excludes.
That is a fact about how Konkani is published, not a defect - and it is more
evidence for the shortfall the specification permits.

---

## D-017 · Scripts that read a fixed input must refuse to re-run silently

**Decision.** Ingest scripts whose input is a file, rather than a stream of new
URLs, refuse to run when previous output exists unless `--fresh` is passed. Dry
runs never persist checkpoint state.

**Why.** Two failures, one after the other:

1. `collect_archive_books.py --dry-run` marked all 14 books as seen and
   persisted that, so the real run collected **zero** documents. The same bug was
   found in `ingest_books_corpus.py` and `ingest_indiccorp.py` by grepping for
   the pattern rather than fixing only the script that failed.

2. Re-running `ingest_wikipedia_manual.py` appended a second copy of every
   document, doubling the manual word total - the single number that determines
   how much downloaded data the corpus may hold.

**The general principle.** The web collectors are safe from both because their
checkpoints carry a seen-set of URLs. Scripts that re-read a fixed file have no
such protection, so the guard has to be explicit. **Silent double-counting is
more dangerous than a crash**, because the corpus still looks fine afterwards.

---

## D-011 · Collection runs on the Mac, not in the cloud sandbox or on GPU

**Decision.** All collection, preprocessing, statistics and tokenizer work runs
locally on the M1 Mac. The assistant's cloud container is used only for writing
and offline-testing scripts.

**Why.** Two independent reasons. First, this work is network-bound and I/O
bound; a GPU contributes nothing to an HTTP fetch. Second, measured during this
session: the cloud container's proxy blocks both `huggingface.co` (403) and
`archive.org` (ProxyError through all 5 retries), and the desktop bridge's shell
environment failed to start, so the assistant cannot execute anything on the Mac
either. Scripts are therefore written and logic-tested in the container, then
written into the repository and run by hand in Terminal.

**Portability requirement that follows.** Every script resolves paths relative
to the repository root and depends only on `requests` plus the standard library,
so the same file runs unchanged locally or in a Colab cell.

---

## D-018 · The "44 Konkani items" figure was wrong, and why that happened

**Correction.** `collect_archive_books.py` reported that the Internet Archive
holds **44** Konkani text items. Verified 16 Aug 2026 against
`archive.org/advancedsearch.php` (raw JSON, `numFound`):

```
language:(Konkani OR Konknni OR Concani) AND mediatype:texts   ->     44
language:kok AND mediatype:texts                               ->  5,093
```

**Cause.** Archive.org's `language` field is free text, not a controlled
vocabulary. Cataloguers overwhelmingly record the ISO 639-2 code `kok`, not the
English word "Konkani". The query asked only for the English spellings, so it
measured our spelling list rather than the archive.

**Why it survived so long.** Two independent faults cancelled out. The query was
wrong *and* `list_items()` never followed the scrape cursor, reading only the
first page. With a 44-item result set one page *is* the whole result set, so the
pagination bug produced no symptom. The output looked internally consistent, and
was precise, and was confidently wrong.

**Consequence.** That 44 became the headline evidence for the Konkani shortfall
and propagated into the source inventory, the collector docstring and the viva
log. The corrected estimate raises the Konkani corpus ceiling from ~11.4M tokens
to ~139M tokens at the pessimistic end.

**Fix.** Query extended to `language:(kok OR gom OR Konkani OR Konknni OR
Concani)`; `list_items()` rewritten to follow the cursor and stop only when the
API returns none. Superseded docstring text left in place and marked.

**The transferable lesson.** A single query returning a plausible number is not
a measurement. Enumeration claims need at least one independent cross-check
before they are used as evidence of *absence* — absence claims are exactly the
ones that shape a project's conclusions.

---

## D-019 · Body script must be measured, never inferred from catalogue or title

**Decision.** Every candidate item's Devanagari share is computed from its
downloaded body text. Neither the `language` field nor the script of the title
may be used as a proxy.

**Why — measured, not hypothetical.** Konkani is written in Devanagari (Goa) and
Kannada script (coastal Karnataka), and both are catalogued `language: kok`.
Two of five sampled items carried a **Devanagari title over a Kannada body**:

```
20veashekddeachy0000drje   title "20व्या शेक्ड्यांचे कोंकणी म्हान मनिस्"
                           body   96.2% Kannada characters
27kavitha0000step          title "27 कविता"
                           body   Kannada script
```

Filtering on either signal would have admitted Kannada-script Konkani into a
Devanagari-only corpus, violating D-001 silently — the failure would have shown
up as a slightly odd tokenizer, not as an error.

**Already correct.** `common/scriptid.py` and the `MIN_DEVANAGARI_RATIO` gate in
`collect_archive_books.py` handle this properly; verified on the real excerpts
in `konkani/data/probe_samples/`. No code change was needed. This decision
records *why* that gate is load-bearing, so nobody later "optimizes" it away by
trusting the cheaper metadata signal.

---

## D-020 · Discovery is a separate script with no seen-set and no manifest

**Decision.** Source discovery and probing live in
`konkani/scripts/discover_sources.py`, which cannot write to any manifest,
checkpoint or `.seen` file.

**Why.** A `--dry-run` of `collect_archive_books.py` previously marked all 14
candidate books as seen; the real run afterwards collected zero. The flag-based
fix works but relies on every future code path respecting it. Removing the
capability entirely is stronger than remembering to guard it: the preview path
has no way to consume the work it previews.

**Consequence.** Discovery can be re-run freely, by anyone, at any point, without
risk to collection state — which is what makes probe-before-collect practical
rather than nerve-wracking.

---

## D-021 · `mean x N`, not `median x N`, estimates a total

**Correction to our own method.** The first version of `discover_sources.py`
reported the median-based extrapolation as "the figure to quote", describing the
mean-based one as "optimistic". That is backwards.

For estimating a **sum** over a population, `mean x N` is the unbiased
estimator. `median x N` is not an estimator of a sum at all. On a right-skewed
distribution — and Konkani book lengths are strongly right-skewed, from ~5k-word
poetry booklets to a ~700k-word encyclopedia — it sits systematically below the
true total.

Measured on the 120-item probe:

```
mean   x N  =  72,207,900 words     <- expected yield
median x N  =  41,985,295 words     <- conservative planning floor
```

**Why this mattered enough to write down.** Labelling a deliberate underestimate
as the better estimate is not caution, it is a mislabelled statistic, and it
would have been indefensible in a viva. Both numbers are now reported with the
role of each stated: expected yield, and planning floor. Plan against the floor;
quote the estimate.

**Uncertainty is reported separately and properly.** The dominant uncertainty is
the Devanagari accept share, 58/120 = 48.3%, 95% CI 39.4%-57.3% (normal
approximation), which propagates to 58.9M-85.6M words. That interval, not the
gap between mean and median, is the honest statement of what we do not know.

---

## D-022 · `token_budget.py` silently became the bug it was written to prevent

**What happened.** The first version of `tools/token_budget.py` located a
source's text by guessing the path
`<lang>/data/<manual|downloaded>/<source_name>/shard_*.txt`. Measured against the
real repository, that layout holds for exactly one source:

```
manual/archive_org_maharashtra_gr/      matches source_name        -> sampled
manual/news/                            ALL news sources share it  -> missed
processed/ai4bharat_indiccorp_v2_mar    downloaded is in processed/-> missed
```

So 8 of 9 Marathi sources sampled zero documents, each fell back to "the mean of
the measured sources", and since only one source was measured that mean *was*
`archive_org_maharashtra_gr`. The tool then applied 1.882 tokens/word - the
fertility of noisy OCR'd government resolutions - to 80-word web-crawl
fragments, and printed `fertility manual 1.882 downloaded 1.882`.

**Why this was serious rather than cosmetic.** The tool's own docstring says it
exists because "applying one average to a corpus whose composition is about to
change produces an estimate that drifts precisely when it matters". It then did
that, and reported the result with three decimal places and no warning. Its
output, `96,717,817 words to ingest`, was about to be passed straight to
`ingest_indiccorp.py --max-words`. A wrong number that looks measured is worse
than an obviously missing one.

**The fix, and why it is not "correct the three paths".** Three new path guesses
would break again at the next layout change. Every document already carries a
`content_hash` in its manifest, so attribution can be exact and
layout-independent: walk every `shard_*.txt` anywhere under `<lang>/data/`, hash
each line, look up its source. Verified on a fixture reproducing the real
layout - shared `news/` directory, downloaded under `processed/` - giving a 100%
hash hit rate and all four sources measured.

**The more important fix: fail loud.** Silent degradation is what made this
dangerous. The tool now reports the sampling hit rate, lists every source it
could not measure, states what share of all words they represent, and prints a
blocking warning that the token figures are provisional. Verified by deleting a
source's shards and confirming the warning fires.

**Transferable lesson.** A fallback that quietly substitutes a worse method is a
liability, not robustness. If a tool cannot do the thing it claims, it must say
so where the reader cannot miss it.

---

## D-023 · Shard text and manifest hashes do not correspond — attribute by directory first

**The defect.** Both collectors write a document to its shard with newlines
flattened, while the manifest hashes the document with newlines intact:

```python
manifest.write(make_record(text=text, ...))    # content_hash(text), newlines KEPT
shard.write(text.replace("\n", " ") + "\n")    # newlines REPLACED
```

A shard line therefore hashes back to its manifest row only when the document
contained no internal newlines. Single-paragraph news articles usually qualify.
The multi-paragraph Maharashtra GR documents never do.

**How it presented.** After D-022 replaced path-guessing with hash attribution,
`archive_org_maharashtra_gr` - **48.6% of all Marathi words** - went from being
the one source that sampled correctly to one that could not be sampled at all.
The failure mode inverted while the underlying data never changed.

**Why it cannot be fixed by better hashing.** `text.replace("\n", " ")` is lossy
in the direction we would need to invert, and the manifest stores no copy of the
original text. No amount of re-normalizing a shard line recovers where its
newlines were.

**Decision.** Attribute by the signal that is actually reliable per directory:

| directory | attribution | why |
|---|---|---|
| `manual/archive_org_maharashtra_gr/` | **directory name** | name equals a manifest source; every document in it belongs to that source |
| `processed/ai4bharat_indiccorp_v2_mar/` | **directory name** | same |
| `manual/news/` | **content hash** | eight sources share this directory, so the path carries no per-source signal; hashing works here because the documents are single-paragraph |

Anything attributable by neither is counted as `unattributed` and reported, never
folded into an average.

**Verified** on a fixture that reproduces the failure exactly - multi-paragraph
GR documents, a shared `news/` directory and a 12-document source: 100%
attribution, all four sources measured, zero warnings.

**Related fix: warning severity now tracks word share, not source count.** The
previous version raised an identical alarm for `news_lokmat` (0.1% of words) and
for `archive_org_maharashtra_gr` (48.6%). Sources under 1% of words are now
reported as an immaterial note. An alarm that fires on things that do not matter
teaches people to ignore it, which is how the alarm that does matter gets missed.

**Should the collectors be changed?** Not now. Rewriting the shard format would
invalidate 168M words of collected data for no gain in corpus quality. The
mismatch is recorded here, and any future collector should either write the
document in its hashed form or store both hashes.

---

## D-024 · The overlap checker silently skipped its own main test

**What happened.** `tools/source_overlap_check.py` hard-coded the downloaded
corpus location as `konkani/data/downloaded/`. The corpus actually lives at
`konkani/data/processed/hf_konkani_books_corpus_v2/`. So `corpus_paragraphs()`
returned an empty list, and the run ended at:

```
[2/3] Loading up to 40,000 corpus paragraphs
  loaded 0
  [error] no corpus paragraphs found
```

Both content signals - exact paragraph overlap and near-duplicate overlap - are
the reason this script exists, and neither ran. Only title matching completed.

**Why it is the same bug twice.** This is D-022 again: a path constant written
before the real repository layout was known. It was fixed in `token_budget.py`
and not carried across to this script. Fixing an instance is not fixing a class.

**Fix.** Shard discovery is now by search, keyed off the manifest's own
`source_name`, with a fallback sweep of `processed/`. Verified on a fixture:
600 paragraphs loaded where the old code loaded 0.

**A limitation the fix exposed, worth knowing before reading the output.** The
fixture also showed that an archive.org title `लोकधन Lokdhan` does not match a
corpus title `lokdhan` after normalization - Devanagari-prefixed catalogue
titles will not align with romanized corpus markers. **Title overlap therefore
under-reports and a 0% title result proves nothing on its own.** The script now
prints example titles from both sides when it finds no matches, so the reader
can see whether the two namespaces are even comparable. The exact and
near-duplicate content signals remain the decisive ones.

---

## D-025 · Marathi collection stopped; the GR source is a domain monoculture

**Final Marathi state (measured with the final 48k tokenizer):**

```
manual      163,786,439 words  ->  303,389,818 tokens   51.85%
downloaded  211,436,384 words  ->  281,766,985 tokens
TOTAL       375,222,823 words  ->  585,156,803 tokens   (+17% over the ~500M target)
```

Both requirements are met with wide margin: 303M manual tokens against a 100M
floor, and 51.85% manual against a 20% floor.

**The concern that the totals hide.** `archive_org_maharashtra_gr` is
**85.0% of all manual words and 46.1% of all corpus tokens**. Nearly half the
training data is Maharashtra government resolutions - a single, highly formulaic
administrative register. The tokenizer shows the consequence: 8 of its 10 most
frequent pieces are punctuation, and `<0x7C>` - the byte-fallback for `|`, almost
certainly a misread danda or table rule - accounts for **1.314% of all tokens**,
roughly 1 token in 76.

**Decision: document now, decide at split time.** The corpus meets every stated
Phase 1 requirement, and no requirement mentions domain balance, so nothing is
deleted on this basis today. But the 17% overshoot is headroom that can be spent
on diversity rather than volume:

| GR words kept | total tokens | manual % | GR % of all tokens |
|---:|---:|---:|---:|
| 139,257,093 (all) | 585.2M | 51.8% | 46.1% |
| 110,000,000 | 528.6M | 46.7% | 40.3% |
| **90,000,000** | **489.9M** | **42.5%** | **35.6%** |

Capping GR at ~90M words lands within 1% of the 500M target while cutting the
monoculture from 46% to 36% of tokens. This is recorded as an available option,
to be taken if Phase 2 shows the model over-fitting administrative register.

---

## D-026 · Collection order is shuffled with a fixed seed

**What prompted it.** A 30-item smoke test of the Konkani book collector
disagreed sharply with the 120-item discovery probe on the same population:

| | probe (n=120, random) | smoke test (n=30, in order) |
|---|---:|---:|
| items with no OCR text layer | **0 (0.0%)** | **13 (43.3%)** |
| items contributing text | 58 (48.3%) | 11 (36.7%) |

The probe drew a seeded random sample. The collector walked archive.org's
enumeration order and took the first 30. Same population, different sampling
method, and the head of the list is measurably unlike the whole: identifier
order correlates with uploader, parent collection and upload era, so the leading
items skew toward one kind of upload.

**Why it matters here specifically.** The full pass is ~5,110 items at ~7 s each,
about **10 hours**. With two days left before the deadline there is a real
chance this run gets truncated - by the deadline, a sleeping laptop, or a
network drop. An in-order run stopped at 60% yields "every Konkani book whose
identifier sorts first", which is a biased corpus, and nothing in the output
would reveal the bias. A shuffled run stopped at 60% yields a 60% random sample,
which is defensible and reportable.

**Decision.** Shuffle the enumerated identifiers before collection, using a
fixed seed (`--shuffle-seed`, default 20260816). `--no-shuffle` restores the old
behaviour.

The seed is fixed rather than random so the permutation is identical on every
resume; combined with the `.seen` checkpoint, an interrupted run continues
through the same order instead of re-walking a different permutation and
sampling unevenly.

**Yield, extrapolated from the smoke test:** 30 items produced 238,227 words, so
5,110 items project to **~40.6M words** - closely matching the probe's
median-based floor of 41,985,295, from an entirely independent measurement.

---

## D-027 · Transient fetch failures were recorded as permanent absences

**The defect.** `fetch_text()` in the Konkani book collector returned a bare
`None` for four different situations - metadata request failed, metadata JSON did
not parse, the item genuinely has no `djvu.txt`, the text download failed. The
caller recorded all four as `no_text_layer` and then called `mark_seen(ident)`,
so a network hiccup was written down as a permanent property of the item and the
item was never retried.

**Measured consequence.**

```
120-item random probe (16 Aug)        no text layer:     0  ( 0.0%)
5,110-item full run  (18 Aug)         no text layer: 3,734  (73.1%)
60-item metadata recheck              HAS TEXT LAYER:   59  (98.3%)
```

Over a 12-hour run archive.org rate-limits, and every throttled request became a
permanent rejection. **98.3% of the discarded items do have a text layer** -
roughly 13.4M words lost to a misclassification, on a corpus whose scarcity is
the entire subject of the Konkani report.

**Why this is the same bug we already fixed once.** `phase1_decisions.md`
already records, for the Marathi GR collector: *"Identifiers marked seen before
fetch - would have permanently discarded 61% of the collection. Now marked only
after a definitive outcome."* The principle was established, written down, and
then not applied to the Konkani collector. Fixing an instance is not fixing a
class - the same lesson as D-024.

**Fix.** `fetch_text()` now returns an explicit three-valued status:

| status | meaning | marked seen? |
|---|---|---|
| `ok` | text retrieved | yes |
| `no_text_layer` | item genuinely has no `djvu.txt` | yes - definitive |
| `transient` | request failed after 3 attempts with backoff | **no** |

Transient failures are counted under their own rejection reason so they are
visible in the run summary instead of hiding inside `no_text_layer`. Three
attempts with linear backoff are made before giving up.

**Recovery.** `konkani/scripts/recheck_failed_items.py --unmark` clears the
non-contributing identifiers from the seen-set (after a timestamped backup),
letting a normal re-run retry exactly those.

---

## D-028 · The overlap check declared "LOW OVERLAP" from 26 paragraphs

**What happened.** The first successful run of `source_overlap_check.py`
reported:

```
26 paragraphs from 59 items
shared titles 0 (0.0%) | exact overlap 0 (0.0%) | near-duplicate 0 (0.0%)
LOW OVERLAP - the two sources are substantially independent.
```

The comparison ran on **26 archive paragraphs**. That is not evidence of
independence; it is an absence of measurement wearing the costume of a result.

**Cause.** `MIN_PARAGRAPH_WORDS = 25`. Both sides of this comparison are
line-oriented OCR - the HF corpus averages **7.52 words per row**, which is the
premise of the whole suspicion, and archive.org `_djvu.txt` files are the same
shape. A 25-word floor therefore rejected nearly every line on both sides. The
threshold contradicted the documented nature of the data it was filtering.

**Fix.** Threshold lowered to 12 words - long enough that random collisions stay
negligible, short enough that line-oriented OCR passes. More importantly, a
**power gate** now runs before any verdict: if either side yields fewer than 400
paragraphs the script returns `INCONCLUSIVE` and says explicitly that this is
not evidence of independence, rather than printing a confident answer.

**Status of the overlap question: still open.** The keep-or-drop decision on the
47M-word downloaded corpus has not yet been made on evidence.

---

## D-029 · Cross-source dedup indexes the manual side only

**What happened.** `make_splits.py --language marathi` was SIGKILLed by macOS at
**86.6%** of cross-source deduplication:

```
2,500,000/2,885,232 (86.6%)   67 docs/s   eta 95.9 min
zsh: killed     python3 tools/make_splits.py --language marathi
```

This was memory, not time. The pass registered every document in the MinHash
index, so by 2.5M documents it held 2.5M signatures. Each signature was a tuple
of 128 Python ints - and a Python int is a ~28-byte object plus an 8-byte
pointer, so one signature costs ~4.6 KB, not the 512 bytes the data itself
needs. That is ~11.5 GB of signatures alone, on top of ~2.3 GB of document
strings.

**Why the obvious fix (`--no-dedup`) was the wrong one.** It would have finished
immediately and been defensible - Marathi sits at 51.85% manual against a 20%
floor - but it discards a real quality check to work around an implementation
detail. Worth avoiding if the real problem is cheap to fix. It was.

**The fix follows from what this pass is for.** Within-source duplicates were
already removed during collection. What remains to detect is a document
appearing in **both** a manual and a downloaded source. That only requires
indexing the manual side and streaming the downloaded side past it:

| | documents indexed | signature memory |
|---|---:|---:|
| before | ~2,500,000 | ~11.5 GB |
| after | ~248,000 (manual only) | **~124 MB** |

Two changes make it work:

1. `Deduplicator.is_duplicate(text, register=False)` checks without growing the
   index, so the downloaded side streams past at constant memory.
2. Signatures are stored as `array("I")` rather than tuples of Python ints -
   512 B instead of ~4.6 KB each, a ~9x reduction independent of the above.

Manual is indexed first, which was already the rule and remains the correct one:
when a document exists on both sides the manual copy survives, because dropping
it would silently reduce the manual ratio.

**Verified** on a fixture with 300 manual and 3,040 downloaded documents, 40 of
them planted duplicates: all 40 caught, and the index size after streaming was
identical to its size after indexing manual - confirming it does not grow.

**Cost of the delay:** roughly 8 hours of wall-clock. The signature
representation was the underlying inefficiency; the unbounded index was the
fault that turned it fatal.

---

## D-030 · LSH bands retuned 32x4 -> 16x8 to match the 0.85 decision threshold

**Symptom.** Cross-source deduplication of the Marathi corpus ran at ~18 docs/s
with an ETA of ~44 hours for 2.6M downloaded documents.

**Diagnosis.** LSH banding decides which pairs become candidates:
`P(candidate) = 1 - (1 - s^r)^b`, with the 50% point near `(1/b)^(1/r)`.

```
bands=32 rows=4  ->  candidates from s ~ 0.42
bands=16 rows=8  ->  candidates from s ~ 0.71
```

The duplicate threshold is **0.85**, so with 32x4 every pair between 0.42 and
0.85 was generated, fetched, scored by a 128-element Python loop, and thrown
away. The Maharashtra GR corpus is formulaic, so a large fraction of its pairs
live in exactly that dead band.

**Measured on 5,000 formulaic documents:**

| bands x rows | indexing docs/s | candidates per query |
|---|---:|---:|
| 32 x 4 | 249 | 349.8 |
| 16 x 8 | 1,237 | 0.4 |

Candidates scale with index size; at the real 251,880-document index this puts
per-query candidates in the tens of thousands, consistent with the observed
18 docs/s.

**Change.** `Deduplicator(bands=...)` default 32 -> 16.

**Equivalence, measured rather than argued:** identical / ~97% / ~90% / ~70%
similar probes were caught 100/100, 100/100, 70/100, 0/100 by *both*
configurations. Detection behaviour is unchanged; only the volume of
never-going-to-match candidates differs. Theoretical recall at s=0.85 falls from
1.000 to 0.994.

**Not done, deliberately.** Streaming the corpus from disk and checkpoint/resume
were both requested. With the projected runtime now in minutes rather than days,
resume adds failure modes without buying anything, and the memory fix (D-029)
already removed the OOM. If the benchmark shows otherwise, they go in.

**`--benchmark N`** added to `make_splits.py`: indexes the manual side, streams
N downloaded documents, reports measured docs/s with a projected full-run time,
and exits without writing splits. No full run should be started on the strength
of an estimate when a measurement costs two minutes.

---

## D-031 · Manual-only indexing caused split leakage; exact-hash dedup restores it

**Symptom.** The Marathi split completed and its own leakage check failed:

```
train vs val   !! 15,372 SHARED
train vs test  !! 15,309 SHARED
val   vs test  !!    165 SHARED
```

Document-level splits must share zero content hashes. This is the guarantee that
test loss measures generalisation rather than memorisation.

**Cause - a regression I introduced in D-029.** Bounding the MinHash index to the
manual side means downloaded documents are compared against manual documents but
**never against each other**. Exact duplicates inside IndicCorp therefore
survived, and the shuffle assigned the two copies to different splits.

They existed because the IndicCorp ingest ran in **two sessions** (63.1M words,
then a resumed run adding 148.3M). Each session built its own in-memory
deduplicator, so a row appearing in both runs was never seen twice by the same
index.

**Fix.** An exact-hash gate over **every** document, manual and downloaded, in
front of the near-duplicate check. A SHA-256 hex digest costs ~64 bytes against
~512 bytes for a MinHash signature, so covering all 2.9M documents is cheap
where covering them with MinHash was not. The expensive near-duplicate index
stays manual-only.

This removes 100% of hash-level leakage by construction, because the leakage
check *is* a content-hash comparison.

**Verified** by reproducing the exact scenario - 5,500 downloaded documents with
500 duplicated across two simulated ingest runs:

| | train/val | train/test | val/test |
|---|---:|---:|---:|
| without exact dedup | 17 | 9 | 0 |
| with exact dedup | **0** | **0** | **0** |

**`--exact-only` added.** The full run costs ~74 minutes, of which ~58 is
building the MinHash index. That index found **23 near-duplicates in 2,887,866
documents (0.0008%)** while exact hashing found 53. Under deadline, `--exact-only`
skips the MinHash pass and completes in minutes, removing all leakage. The
trade is explicit: near-duplicate cross-source detection is dropped, its measured
yield was 23 documents, and `source_overlap_check.py` already established source
independence separately.

**Lesson.** D-029 optimised memory by narrowing what the index covered, and in
doing so silently narrowed what the *guarantee* covered. The leakage check caught
it - which is the argument for having verification that runs on the real output
rather than on the reasoning about it.

---

## D-032 · Vocabulary size 48,000 -> 10,000 for both languages

**Why the change.** TA guidance on 18 Aug:

> **[AS]** "Please try to keep a vocab size of **5 to 10K**."
> **Kallind Soni:** "too large sizes like 24k 32k is not recommended since you
> might end up having **every word as a token**."

Our first tokenizers were **48,000**. Two independent arguments made that
untenable, and both are measurable rather than stylistic.

**1. It does not fit the model.** The specification allows ~25M parameters per
model. At `d_model = 384`:

| vocab | embedding + unembedding | share of the 25M budget |
|---:|---:|---:|
| 48,000 | 36,864,000 | **147%** |
| 10,000 | 7,680,000 | **31%** |

At 48k the embedding matrices alone exceed the entire parameter budget before a
single transformer block exists. At 10k they take under a third, leaving the
rest for the layers that actually do the modelling.

**2. The 48k vocabulary was mostly dead weight.** Measured on our own corpora:

| metric | Marathi 48k | Konkani 48k | Konkani 10k |
|---|---:|---:|---:|
| whole-word tokens | 75.5% | 61.4% | **50.1%** |
| vocabulary utilisation | 85.5% | 82.8% | **97.5%** |
| hapax tokens | 7,359 | 2,779 | **41** |
| tokens per word (held-out) | 1.6839 | 1.5729 | **1.8245** |
| unknown-token rate | 0.000000% | 0.000000% | **0.000000%** |

The hapax count is the clearest signal: at 48k, **2,779 Konkani vocabulary slots
held a piece that occurs exactly once** in held-out text. Those are parameters
that can never be learned usefully. At 10k that falls to **41**, and utilisation
rises to 97.5% - nearly every slot earns its place.

75.5% whole-word tokens on Marathi is precisely the failure Kallind described.
At 10k, Konkani drops to 50.1% and genuine subword structure appears:
`पोर|नो`, `उठ|ून`, `सांज|वेळार`.

**3. It increased the corpus, it did not shrink it.** Finer segmentation means
more tokens from the same text:

| | 48k vocab | 10k vocab | change |
|---|---:|---:|---:|
| Marathi training tokens | 525,255,454 | **647,434,614** | +23.3% |
| Marathi vs ~500M target | 105.1% | **129.5%** | |
| Konkani training tokens | 142,544,039 | **~165M** | +16.0% |

So the change satisfies the TA guidance, fits the parameter budget, produces a
healthier vocabulary, **and** moves both languages further above target. There
is no trade-off being made here, which is worth stating plainly because a
"constraint" that improves every metric usually means the original choice was
simply wrong.

**Fertility check.** **[AS]** "at the very least, try to achieve a fertility > 1
i.e. sub-word tokenization." Konkani 1.8245, Marathi comparable - both above 1
with more margin than at 48k, since a larger vocabulary pushes fertility *down*
toward whole-word tokenization.

**Both 48k models are retained** in the sweep record as the documented rejected
alternative, so the comparison above is reproducible rather than asserted.

**Selection method.** `tools/build_tokenizer.py --vocab-sizes 6000,8000,10000`
sweeps candidates and reports fertility, UNK rate, utilisation and hapax count
for each on held-out text, choosing the smallest vocabulary within a tolerance
of the best fertility. This follows §1.3 ("choose using fertility /
unknown-token rate on held-out text"), which **[AS]** later clarified is a
recommendation rather than a strict requirement - we followed it anyway because
it turns a design choice into a measurement.

---

## D-033 — Sangraha `gom` verified, `kok` does not exist

**Believed:** `ai4bharat/sangraha` might hold a large Konkani split.
**Measured:** `verified/gom/*.parquet` — one file, 32,496,312 bytes, 14,491 rows.
No `kok` split exists. `unverified` has no Konkani directory at all; `synthetic`
has no `gom`.
**Result:** ingested `verified/gom` only — 9,827 documents, 3,266,816 words.
4,651 rows (32%) were rejected as **not Devanagari**: they are Romi Konkani,
genuinely Konkani but in Roman script. 11 rows were rejected as Marathi. A
split labelled `gom` by its publisher is not evidence that its rows are
Devanagari Konkani.

---

## D-034 — Blank lines are sentence separators, not document boundaries

**Believed:** IndicCorp v2's `gom.txt` used blank lines to separate documents.
**Measured:** 1,361,209 "documents" from 533,108,246 bytes = **392 bytes ≈ 22
words each**, and `flush_blank_line` fired on **100%** of flushes. 821,054 of
those fragments (60%) were then discarded for failing the 25-word document
floor — a floor our own chopping had made unreachable.
**Result:** rewritten to dedup and language-filter at the **unit** level, then
pack surviving units into 300-word documents, then apply the document gates.
Order matters: IndicCorp repeats individual sentences across crawled pages, and
once packed no two documents are byte-identical, so unit-level dedup is the only
place those 197,656 repeats can be caught. `--inspect N` was added so the file's
structure is measured, never assumed again.

---

## D-035 — IndicCorp v2 `gom.txt` is ~84% Marathi

**Believed:** a 533 MB file labelled Goan Konkani is Goan Konkani.
**Measured:** the discriminator rejected 70% of packed documents as Marathi —
wildly out of line with GlotCC (0 of 1,049), MADLAD-400 noisy (14 of 4,602) and
Sangraha (11 of 14,491) on the same day. Rather than trust either the label or
our own gate, we calibrated against two reference populations whose language is
not in doubt, all packed to the same ~300-word length:

| population | `mr` | `kok` | undecided | median score |
|---|---:|---:|---:|---:|
| reference Konkani (our OCR'd books) | 0.0% | 94.7% | 5.3% | −0.92 |
| reference Marathi (our Marathi corpus) | 100.0% | 0.0% | 0.0% | +1.00 |
| **IndicCorp v2 `gom.txt`** | **83.7%** | 2.5% | 13.9% | **+0.79** |

A sampled rejected document contains `आहे×8, पण×5, मी×3` and zero Konkani markers.
**Result:** the gate was correct. 4,319,751 words kept out of ~30M. Accepting
the file unfiltered would have inflated the corpus by ~25M words *and* injected
Model H's language into Model L, breaking corpus independence.
Tool: `tools/verify_gom_langid.py`.

---

## D-036 — `MACHINE_TRANSLATED` as a distinct collection type

**Context:** the TAs authorised synthetic/MT data on 18 Aug 2026 as a last
resort.
**Decision:** added `CollectionType.MACHINE_TRANSLATED` rather than reusing
`DOWNLOADED_DATASET`. `is_manual` is `False`, so it can never count toward the
20% floor; and because it is a *distinct* member it can never be silently folded
into the downloaded figure either. Every statistics table can therefore state
the synthetic share separately, which is what condition 2 requires.

---

## D-037 — NLLB-200 cannot produce Konkani, and would have failed silently

**Believed:** `facebook/nllb-200-distilled-600M` was a safe ungated fallback for
Marathi→Konkani.
**Measured:** its `special_tokens_map.json` contains no `gom_Deva` and no `kok_*`.
**Why this mattered:** `convert_tokens_to_ids` returns the *unknown* id for an
unseen language code, `generate` accepts that as `forced_bos_token_id` without
complaint, and the model emits fluent text in some other language. Marathi and
Hindi share Devanagari with Konkani, so the output would have passed the script
check and partially survived the language gate — counterfeit Konkani.
**Result:** the engine now verifies the target-language token at load time and
exits with instructions rather than running.

---

## D-038 — 870 MB of MT text that policy, not availability, had excluded

**Context:** `praveenkumar99/Konkani_Raw` was ingested selectively when MT was
banned; only 4,397,019 bytes of scraped pages were taken and
`translated_konkani_*.txt` — **870,725,308 bytes** — was filtered out on
principle.
**Result after authorisation:** 73,717 rows → 60,843 documents → **59,295,762
words**, the single largest source in the Konkani corpus. Only **2** rows were
rejected as Marathi, and cross-source dedup removed **zero** of its documents,
so it is genuinely Konkani and genuinely distinct. Classified
`MACHINE_TRANSLATED`.

---

## D-039 — Column detection by measurement, not by dataset card

**Problem:** ~20 remaining HuggingFace Konkani datasets, each with a different
schema. The dataset-server `first-rows` endpoint was returning **cached
responses for the wrong dataset** when several were queried in sequence, so any
hand-copied column name was untrustworthy.
**Result:** `ingest_hf_bulk_konkani.py` probes the first 50 rows and selects
*every* column whose mean Devanagari ratio clears 0.50 — not just the best one,
because instruction sets often carry Konkani in both an instruction and a
response field. Tested against Alpaca-shaped rows (picks `output`, ignores the
English `instruction` and the `id`), dual-field rows (picks both), and
English-only rows (picks nothing).
**Yield:** 67,988 documents, 25,648,910 words across 10 datasets; 8 datasets
correctly yielded zero. The chosen columns are printed per dataset.

---

## D-040 — Fail on the first batch, not the ten-thousandth

**Problem:** the MT generator's per-batch exception handler printed a one-line
summary and continued. When a *configuration* error made every batch fail, it
printed the same line indefinitely with no traceback and no way to tell whether
it would recover.
**Result:** a first-batch failure now aborts with the full traceback, because a
first-batch failure is always configuration and every later batch will fail
identically. Later failures are still tolerated — those are genuinely per-batch.
This is what surfaced the real cause of the `'NoneType' object has no attribute
'shape'` error, which was not the attention mask (fixed separately) but D-041.

---

## D-041 — transformers ≥ 4.49 breaks IndicTrans2's vendored decoder

**Measured:** IndicTrans2's custom modeling code executes
`past_key_values[0][0].shape[2] if past_key_values is not None else 0`.
transformers ≥ ~4.49 passes a `Cache` **object**, which is not `None`, so the
guard passes and indexing an empty cache yields `None`.
**Attempted fix:** `use_cache=False` at both `generate()` and model-config level.
It works, and is unusably slow — a batch of 24 needs 256 full forward passes
over a growing sequence. Measured: **0 sentences in 5.1 minutes.**
**Result:** pinned `transformers==4.46.3` in an isolated venv, which restores
the cached path. The runtime fallback is retained so the fast path is used
automatically wherever the environment permits it.

---

## D-042 — Apple MPS is 45× *slower* than CPU for this model

**Believed:** Apple Silicon's GPU would accelerate IndicTrans2 generation.
**Measured, with a verified forward-pass probe rather than an assumption:**

| device | throughput |
|---|---:|
| CPU (8 threads) | **0.9–1.2 sentences/sec** |
| MPS | 48 sentences in 124 minutes ≈ **0.006 sentences/sec** |

The probe passed, so the model genuinely ran on the GPU; IndicTrans2's custom
ops evidently fall back per-operation with transfer overhead on each.
**Result:** MPS abandoned, CPU restored, negative result recorded. Also folded
in: `max_length` 256 → 160 (source sentences are capped at 60 words ≈ 120
tokens, so the rest was paid for nothing) and `torch.set_num_threads(all cores)`.

**Final MT generation contribution: 14,016 sentences → 394 documents →
124,664 words**, with `copied_not_translated` = 0 across the whole run. That is
**0.05%** of the Konkani corpus. It is reported because it was attempted and
measured, not because it was material.

---

## D-032 (revised) — Vocabulary 2,500, and what it costs

The vocabulary was re-swept over 2,000 / 2,500 / 3,000 / 4,000 / 5,000 with
fertility measured on held-out text:

| vocab | fertility | embed+unembed (d=512) | % of 25M budget |
|---:|---:|---:|---:|
| 2,000 | 2.6714 | 2.05M | 8% |
| **2,500** | **2.5333** | **2.56M** | **10%** |
| 3,000 | 2.4262 | 3.07M | 12% |
| 4,000 | 2.2832 | 4.10M | 16% |
| 5,000 | 2.1836 | 5.12M | 20% |
| 10,000 | 1.9506 | 10.24M | **41%** |

**Chosen: 2,500 for both languages.** The governing argument is the parameter
budget — at 10,000 the embedding and unembedding matrices alone consume 41% of a
25M-parameter model, versus 10% at 2,500, returning ~7.7M parameters to depth
and width.

**The cost, stated plainly.** Fertility rises from 2.1836 to 2.5279, whole-word
token coverage falls from 39.4% to 31.0%, and every training sequence is ~16%
longer for the same text. And because tokens are what the target is measured in,
**the same corpus reads 430M tokens at vocabulary 5,000 and 506M at 2,500**.
That is a property of the metric. It is recorded here, in the README, and in the
coverage report so that no reader has to discover it for themselves.

---

## D-043 — Vocabulary size 2,500, against a recommendation of tens of thousands

Specification §1.3: "Recommended vocabulary size is in the tens of thousands per
model; choose using fertility / unknown-token rate on held-out text."

We chose 2,500 for both languages. That is roughly an order of magnitude below
the recommendation, so this entry records the measurement, the argument, the
counter-argument and the side effect.

### Measurement

Six candidate vocabularies were trained on the Konkani train split and evaluated
on 5,000 held-out documents that no candidate saw during training. Command:

```
python3 tools/build_tokenizer.py --language konkani \
  --vocab-sizes 2000,2500,3000,4000,5000,10000 --sweep-only
```

Artifact: `report/phase1_tokenizer_sweep_konkani.json`. Corpus 323,111 lines,
318,111 training, 5,000 held out.

| vocab | fertility (tok/word) | chars/token | whole-word rate | unk rate | embed+unembed untied, d=512 | share of 25M | tied | share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 2.6531 | 2.4389 | 27.2% | 0.000000% | 2.05M | 8% | 1.02M | 4% |
| 2,500 | 2.5148 | 2.5730 | 30.6% | 0.000000% | 2.56M | 10% | 1.28M | 5% |
| 3,000 | 2.4112 | 2.6836 | 33.3% | 0.000000% | 3.07M | 12% | 1.54M | 6% |
| 4,000 | 2.2632 | 2.8591 | 37.6% | 0.000000% | 4.10M | 16% | 2.05M | 8% |
| 5,000 | 2.1703 | 2.9814 | 40.6% | 0.000000% | 5.12M | 20% | 2.56M | 10% |
| 10,000 | 1.9317 | 3.3498 | 48.8% | 0.000000% | 10.24M | 41% | 5.12M | 20% |

Every candidate reports 256 byte-fallback pieces, which is the check that
`byte_fallback` is actually active.

The unknown-token rate is 0.000000% at every size, and that is by construction
rather than by luck. With `byte_fallback=True` the 256 `<0xNN>` pieces make any
Unicode string representable, so the unknown token can never be emitted. The
specification names unknown-token rate as a selection criterion, but for a
byte-fallback tokenizer it carries no information. Fertility therefore had to
carry the decision alone, and fertility alone always favours the largest
vocabulary tested. That is why a third criterion was needed.

An earlier version of this entry quoted fertility figures that were roughly 0.5%
higher across the board. They came from a sweep whose record had been overwritten
when the final tokenizer was rebuilt, and could not be reproduced. The table
above replaces them with values that have an artifact behind them.

### Why the deliverable tokenizer reports 2.5279 and this sweep reports 2.5148

The deliverable Konkani tokenizer was built on 20 August from a corpus of 312,293
lines. Approximately 10,800 further documents, chiefly BPCC, were ingested
afterwards, so the corpus this sweep measures is 323,111 lines. The two held-out
samples are therefore drawn from slightly different corpora, and the ~0.5%
difference is that, not a change in the tokenizer.

This also means the deliverable tokenizer was trained on about 97% of the final
corpus rather than all of it. That is acceptable — a tokenizer's training set is
a sample by design, and byte fallback guarantees the remaining 3% is
representable, which the measured 0.000000% unknown rate over the full corpus
confirms — but it is recorded rather than left to be discovered.

### Parameter cost

The ~25M parameter budget is the criterion that decided it. Embedding and
unembedding cost `2 × V × d_model` untied. At `d_model = 512` a transformer block
costs approximately `12 × d_model²` = 3.15M parameters, so the vocabulary choice
is worth about two layers:

| vocab, untied | lookup tables | remaining for the stack | layers at d=512 |
|---:|---:|---:|---:|
| 10,000 | 10.24M | 14.8M | ~4.7 |
| 5,000 | 5.12M | 19.9M | ~6.3 |
| 2,500 | 2.56M | 22.4M | ~7.1 |

At 10,000 the two lookup tables consume 41% of the model. At 2,500 they consume
10%, returning roughly 7.7M parameters to depth and width.

### Counter-argument

Weight tying halves that cost, and the table above does not assume it. Sharing
one matrix between the embedding and the output projection is standard and is
used in GPT-2. With tying, vocabulary 10,000 costs 5.12M rather than 10.24M —
20% of the budget rather than 41% — leaving room for about 6.3 layers. That is a
buildable model.

So the parameter-budget argument does not on its own rule out the recommended
range. It rules out an untied 10,000-vocabulary model at this parameter count.
Whether to tie is a Phase 2 architecture decision, recorded here so that Phase 2
inherits an open question rather than an assumption.

### Effect on the reported token count

Token count is fertility multiplied by word count, and the train split is fixed
at 200,293,343 words. A smaller vocabulary therefore raises the reported token
count without changing the data. Projected from the measured fertilities above:

| vocab | projected Konkani training tokens | against ~500M |
|---:|---:|---:|
| 2,000 | 531,398,268 | 106.3% |
| 2,500 | 503,697,699 | 100.7% |
| 3,000 | 482,947,308 | 96.6% |
| 4,000 | 453,303,895 | 90.7% |
| 5,000 | 434,696,642 | 86.9% |
| 10,000 | 386,906,651 | 77.4% |

The figure actually reported for the corpus, **506,259,368**, is not a projection:
it is the count produced by encoding the whole train split with the deliverable
tokenizer, and it is slightly above the 2,500 projection for the corpus-difference
reason given above.

Konkani clears the target at 2,500 and misses it at every larger size tested.
Marathi clears it at all of them, so the vocabulary choice affects only the
Konkani figure.

This is a real consequence, and presenting the parameter-budget argument as the
sole reason would be incomplete. The corpus did not grow: it is 266,211,363 words
before and after this decision, and only the unit of measurement changed. Word
counts are reported alongside token counts throughout this project so that a
reader can see the corpus size independently of the tokenizer.

### Cost of the choice

Fertility rises from 2.1703 at vocabulary 5,000 to 2.5148 at 2,500, so each
training sequence carries about 16% more tokens for the same text: a fixed
context window holds about 16% less Konkani, and a fixed token budget sees fewer
words. Whole-word coverage falls from 40.6% to 30.6%, meaning more words are
split into two or three pieces and the model must learn more composition from
subwords. Average characters per token falls from 2.9814 to 2.5730.

### Marathi

The recorded sweep is Konkani only. Marathi's vocabulary was set to 2,500 to
match, because the parameter budget is identical for both models and because
Konkani is the language where the token target is tight — Marathi clears it at
every vocabulary tested. The two vocabularies remain separately trained and
share no pieces, which is what the specification requires. Re-running the
Marathi sweep was not attempted: it requires reading approximately 2 GB of OCR
shards that are currently iCloud placeholders, and the result would not change
the decision.

### Decision

Vocabulary 2,500 for both languages, tokenizers and vocabularies trained
separately per language. The selection procedure was the one the specification
asks for, extended with a parameter-budget criterion because unknown-token rate
is uninformative under byte fallback. The deviation from the recommended range is
deliberate, and the two facts needed to argue against it — that weight tying makes
a larger vocabulary affordable, and that a larger vocabulary places Konkani below
the target — are stated above rather than left for a reader to find.
