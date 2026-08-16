# Phase 1 — Viva Log

A running record of every major action, written so that the whole pipeline can
be explained out loud without relying on memory. Each entry answers the same ten
questions.

Entries are in chronological order. **A-nnn** = action.

---

## A-001 · Audit the existing repository against the specification

**1. What we did.** Read the official project PDF end to end, then read every
file in the repository — all 15 Konkani scripts, the progress report, the git
configuration and refs — and compared what exists against every Phase 1
requirement.

**2. Why we did it.** Work had already been done with AI assistance. Before
building on it we needed to know which parts were correct. Building on an
unverified foundation is how a project fails quietly.

**3. Command/script used.** Direct file reading; `git` refs and object store
inspected by reading `.git/config`, `.git/HEAD`, `.git/logs/HEAD`,
`.git/refs/**`.

**4. Input source.** `LMA_Individual_Project_v1.pdf`; the repository at commit
`8a11864`.

**5. Output produced.** `report/phase1_gap_analysis.md`.

**6. Important numbers.** 65 project files (excluding `.venv` and `.git`); 5
commits; `.git` object store total **1.49 MB**.

**7. Why the result matters.** The 1.49 MB figure proves no corpus data was ever
committed to git — a specification requirement ("do not commit large binary
artifacts"). We can state this as verified fact rather than assumption.

**8. Problems encountered.** The 61.5 MB `tokenizer_sample.txt` could not be
transferred for inspection (the file bridge timed out).

**9. How we fixed it.** Derived the needed conclusions arithmetically from
statistics already reported, instead of guessing or dropping the question. See
A-003.

**10. Decision made.** Preserve everything; correct specific defects; do not
restart. Four defects found that change previously-written conclusions.

---

## A-002 · Verify what the existing tokenizers actually do

**1. What we did.** Loaded both trained SentencePiece models and ran them on
four controlled inputs: Devanagari Konkani, Roman Konkani, English, and Marathi.
Also counted how many byte-fallback pieces exist in each vocabulary.

**2. Why we did it.** The progress report claimed a 46.36% unknown-token rate
and drew a design conclusion from it. A 46% UNK rate is not a normal result for
a subword tokenizer, so it needed to be reproduced and explained before being
believed.

**3. Command/script used.**

```python
sp = spm.SentencePieceProcessor(model_file="preliminary_konkani_bpe.model")
sum(1 for i in range(sp.get_piece_size()) if sp.id_to_piece(i).startswith("<0x"))
```

**4. Input source.** `konkani/tokenizer/preliminary_konkani_bpe.model` and
`preliminary_konkani_mixed_bpe.model`.

**5. Output produced.** Measured table:

| Input | Devanagari tokenizer | Mixed tokenizer |
|---|---|---|
| Konkani (Devanagari) | 11 tokens, 0% UNK | 11 tokens, 0% UNK |
| Konkani (Roman) | 20 tokens, 45.0% UNK | 24 tokens, 8.3% UNK |
| English | 19 tokens, 47.4% UNK | 24 tokens, 4.2% UNK |
| Marathi | 16 tokens, 0% UNK | 16 tokens, 0% UNK |

**6. Important numbers.** Byte-fallback pieces in vocabulary: **0** in both
models. Vocabulary size 32,000 in both.

**7. Why the result matters.** This is the central technical finding of the
audit. A SentencePiece model trained with `byte_fallback=True` contains 256
special `<0xNN>` pieces and can encode *any* UTF-8 string, so its UNK rate is
~0 by construction. These models have none, so every character outside the
learned inventory becomes `<unk>`. The 46.36% figure is therefore not a fact
about Konkani orthography — it is a fact about Latin characters meeting a
tokenizer that has none. The original conclusion ("we must use a mixed-script
tokenizer") does not follow from this evidence.

**8. Problems encountered.** The original conclusion was plausible and could
easily have been carried into the final report unchallenged.

**9. How we fixed it.** Retrain with `byte_fallback=True`, and change the
decision metric from UNK rate to **fertility** (tokens/word, chars/token) on
held-out text. With byte fallback, poorly-covered script becomes *expensive*
rather than *unrepresentable* — and cost is the thing we actually care about.

**10. Decision made.** D-002 and D-003 in `phase1_decisions.md`.

**Likely viva question:** *"Your report says a 7% UNK rate. Why can your
tokenizer not represent its own training data?"* — The honest answer, which we
can now give: it could not, because byte fallback was off; we found this by
counting byte pieces in the vocabulary, fixed it, and the metric that actually
decides script coverage is fertility, not UNK.

---

## A-003 · Recompute the Devanagari percentage of the Konkani books corpus

**1. What we did.** Re-derived the script composition of the books corpus from
the statistics already reported, after noticing that two scripts in the repo
compute the same quantity differently.

**2. Why we did it.** `corpus_audit.py` divides Devanagari characters by
non-whitespace characters. `analyze_books_corpus.py` contains:

```python
non_whitespace = total_characters      # <-- total, including spaces
devanagari_percentage = (total_devanagari / non_whitespace) * 100
```

The variable name and its value disagree. Two scripts computing one statistic
two ways means at least one is wrong.

**3. Command/script used.** Arithmetic on the reported totals (the corpus file
itself could not be transferred — see A-001, problem 8).

**4. Input source.** Figures in `report/phase1_konkani_progress.md`: 379,610,529
characters, 317,109,283 Devanagari characters, 61,805,534 words.

**5. Output produced.**

```
non-Devanagari characters = 379,610,529 − 317,109,283 = 62,501,246
words                                                 = 61,805,534
non-Devanagari per word   = 62,501,246 / 61,805,534   = 1.0113
```

**6. Important numbers.** **1.0113** non-Devanagari characters per word. Removing
whitespace from the denominator: 317,109,283 / 317,804,995 = **99.78%**.

**7. Why the result matters.** There is almost exactly one non-Devanagari
character per word — that is the inter-word space and essentially nothing else.
The corpus is not 83.54% Devanagari with ~16% foreign contamination; it is
**99.78% Devanagari**. The apparent contamination was whitespace sitting in the
denominator. Since the supposed contamination was part of the motivation for a
mixed-script tokenizer, that argument loses its factual basis.

**8. Problems encountered.** The corpus file itself was unavailable, so the
result could not be measured directly.

**9. How we fixed it.** Used the already-published totals, which are sufficient
because the relationship between words and non-Devanagari characters is
diagnostic on its own. The direct measurement will be redone locally when the
corpus is reprocessed, using `common/scriptid.py`.

**10. Decision made.** D-004. Old figure preserved as superseded, not deleted.

**Likely viva question:** *"How do you know your corpus is really Konkani?"* —
Script purity is 99.78% Devanagari, measured over non-whitespace characters
(and we can explain why that denominator matters, because we got it wrong once).
Script purity alone does not prove the *language*, which is why we also built
the Marathi/Konkani discriminator in A-005.

---

## A-004 · Count what actually exists at the candidate sources

**1. What we did.** Queried the Internet Archive search APIs to count real
holdings for both languages, and fetched one Konkani book to confirm the text is
downloadable and in the expected script.

**2. Why we did it.** The source inventory had been written from plausible
reasoning, not from counts. "Internet Archive has Konkani books" is a claim, not
a number. Before committing days of collection effort we needed the number.

**3. Command/script used.** archive.org `advancedsearch.php` and
`services/search/v1/scrape` endpoints; one direct file fetch of a `_djvu.txt`.

**4. Input source.** `archive.org`.

**5. Output produced.**

| Query | Result |
|---|---|
| `language:Konkani` | **44 items** — ~25 Wikipedia ZIM dumps, 4 Wikipedia PDFs, ~13–15 real books |
| `language:Marathi` | 173,956 items |
| `identifier:in.gov.maharashtra.gr.*` | **170,725 items**, each with an OCR text derivative |

Sample Konkani book `konkanibhashaman0000jbmo`: full text downloadable,
predominantly Devanagari, mixed with English and Kannada script.
Sample GR `in.gov.maharashtra.gr.202607071620477816`: `_djvu.txt` 34.5 KB,
metadata language "Marathi, English", OCR by Tesseract 5.3.0.

**6. Important numbers.** **44** vs **170,725**. Roughly a four-thousand-fold
difference between the two languages at the same source.

**7. Why the result matters.** Two things at once. It gives Marathi a verified,
large, genuinely manual source. And it *disproves our own earlier claim* that
Internet Archive was the strongest manual route for Konkani — 13–15 books is
perhaps 1–2M words, not tens of millions. This is also the cleanest single piece
of evidence for the Konkani shortfall the specification permits: the
low-resource language is measurably low-resource.

**8. Problems encountered.** Our own source inventory contained an unverified
claim that turned out to be wrong.

**9. How we fixed it.** Recorded it as an explicit correction (D-005) rather
than editing the original document silently, per the standing instruction to
preserve superseded results.

**10. Decision made.** D-005, D-006. Marathi GRs become pilot source M1;
Konkani strategy shifts to many small sources rather than one large one.

---

## A-005 · Build the shared foundation layer

**1. What we did.** Wrote and tested five library modules: `textnorm`,
`scriptid`, `manifest`, `checkpoint`, `dedup`.

**2. Why we did it.** The audit found that normalization, dedup, provenance
tracking and manual/downloaded accounting were all missing, and that the one
statistic implemented twice was implemented inconsistently. Building these once,
tested, before any collection starts is what stops the same class of bug
recurring at scale.

**3. Command/script used.** Each module has a `__main__` self-test:
`python3 common/textnorm.py`, `python3 common/scriptid.py`, etc.

**4. Input source.** Control sentences in Marathi and Konkani; synthetic
government-resolution text.

**5. Output produced.** Five modules, all self-tests passing.

**6. Important numbers.**
- Marathi/Konkani discriminator: **+1.000** on the Marathi control sentence,
  **−1.000** on the Konkani control sentence — clean separation.
- MinHash near-duplicate similarity: **0.875** between a document and a
  boilerplate variant of it; **0.000** between unrelated documents.
- Checkpoint recovers cursor and seen-set after simulated crash, and survives a
  deliberately corrupted state file.

**7. Why the result matters, module by module.**

- **`textnorm`** applies Unicode NFC. Devanagari can write one visible letter two
  ways — क़ as a single code point, or क + combining nukta. A human sees one
  letter; a tokenizer sees two different strings and learns two sets of merges
  for the same word, wasting vocabulary. NFC collapses them.
  It deliberately does **not** strip ZWJ/ZWNJ (U+200D/U+200C), which are
  meaningful in Devanagari — they control half-forms and conjuncts. Many naive
  cleaners delete all "invisible" characters and silently corrupt Indic text.
- **`scriptid`** computes script ratios over **non-whitespace** characters,
  fixing the A-003 bug at the library level so it cannot recur, and provides the
  Marathi/Konkani discriminator.
- **`manifest`** writes one provenance row per document with the required schema,
  and makes manual vs downloaded a *typed* choice rather than a string that can
  drift.
- **`checkpoint`** persists the pagination cursor, which the existing Wikipedia
  collector does not.
- **`dedup`** does exact SHA-256 plus MinHash/LSH near-duplicate detection.

**8. Problems encountered.** The first dedup self-test appeared to prove
near-duplicate detection worked, but was actually passing through the *exact*
path: the canonicaliser folds all digits to `0`, so a document differing only in
its reference number and date became byte-identical after canonicalisation.

**9. How we fixed it.** Wrote a second test where a whole phrase differs, not
just digits, and confirmed the MinHash path itself: 0.875 similarity, correctly
flagged; 0.000 for unrelated text, correctly kept. **Lesson: a passing test is
not evidence that the intended code path ran.**

**10. Decision made.** D-008, D-009, D-010.

**Likely viva question:** *"Why is your language-ID a hand-written word list
rather than a trained model?"* — Because the two languages are too close for
character statistics (demonstrated in A-002: the Konkani tokenizer handles
Marathi at 0% UNK and identical chars/token), and off-the-shelf identifiers are
trained with little Konkani. Closed-class function words — आणि vs आनी, आहे vs
आसा, मी vs हांव — are the stable difference, because content words are borrowed
between the languages but grammar words are not. It abstains rather than guesses
when evidence is thin.

---

## A-006 · Build and test the M1 collector

**1. What we did.** Wrote `marathi/scripts/collect_archive_gr.py` and verified
its filtering logic offline against mock documents.

**2. Why we did it.** M1 is the largest verified manual source for Marathi. The
collector must be resumable, must record provenance per document, and must
filter language and duplicates — before it runs for days.

**3. Command/script used.**
`python3 marathi/scripts/collect_archive_gr.py --limit 5 --page-size 20`
(network), then an offline harness feeding 15 synthetic documents through
`evaluate()` + manifest + dedup.

**4. Input source.** Live: archive.org. Offline: 12 near-identical synthetic
GRs, one English-only document, one Konkani document, one 2-word stub.

**5. Output produced.** Network run failed as expected (see 8). Offline run:

```
accepted: 1 | manual docs: 1 | words: 87 | manual fraction: 100%
dedup: seen=12 exact=1 near=10 kept=1 duplicate_rate=91.7%
rejections: duplicate=11, not_enough_devanagari=1, langid_kok=1, too_short=1
```

**6. Important numbers.** All four rejection reasons fired exactly once each on
the documents designed to trigger them.

**7. Why the result matters.** Every quality gate is demonstrably working: the
English document was caught by the script gate, the **Konkani document was
caught by the language gate**, the stub by the length gate, and the boilerplate
variants by dedup. The Konkani rejection is the important one — it is the
cross-language contamination filter working on the Marathi pipeline, which is
the specification's "do not share documents across corpora" requirement enforced
at collection time rather than checked afterwards.

**8. Problems encountered.** The live run could not reach archive.org: the
assistant's cloud container proxies outbound HTTP and blocked it through all
five retries (it also blocks `huggingface.co` with HTTP 403). Separately, the
desktop bridge's shell environment failed to start, so the assistant cannot run
commands on the Mac either.

**9. How we fixed it.** Verified the logic offline instead, and established the
working split: scripts are written and logic-tested in the container, then
written into the repository and **run by hand in Terminal on the Mac**. The
failure also usefully exercised the retry path — five backoff attempts, then a
clean error and a saved checkpoint, with no crash and no data loss.

**10. Decision made.** D-006, D-011. The pilot is run locally as the next step.

**Open question the pilot must answer.** The 0.85 dedup threshold rejected 11 of
12 synthetic documents. That is correct for near-identical boilerplate, but the
mock documents differed by only one word. Real GRs will differ more. **The
threshold must be re-tuned from the real distribution before the full run** —
this is the main reason the pilot exists.

---

## A-007 · Build the health monitor

**1. What we did.** Wrote `tools/health_check.py`, which inspects a running
job's checkpoint and manifest without touching the job.

**2. Why we did it.** "Is the process alive?" is the wrong question. A collector
can sit in a retry loop for hours, or page through a source rejecting
everything, and still look healthy to `ps`. What matters is whether output is
growing, and at what rate.

**3. Command/script used.** `python3 tools/health_check.py --all`

**4. Input source.** `<lang>/data/checkpoints/*.json` and
`<lang>/data/manifests/*.jsonl`; previous snapshots in `.health/`.

**5. Output produced.** Per job: documents, words, manual/downloaded split,
token split, headroom under the 20% rule, last checkpoint time, processing rate,
errors and 429s, duplicate rate, disk free, script and langid distributions,
per-source breakdown, and a HEALTHY / WARNING / STALLED verdict.

**6. Important numbers.** Stall thresholds: no checkpoint write in **15
minutes**; error ratio above **10%**; free disk below **5 GB**.

**7. Why the result matters.** Stall detection uses three independent signals,
because any one alone gives false alarms: checkpoint freshness, document delta
since the previous check, and error ratio. The second is the subtle one — *a
fresh checkpoint with zero new documents means the job is fetching and rejecting
everything*, which no liveness check would catch. It also surfaces the accounting
constraint continuously: `headroom` reports how many more words the corpus may
take before the 20% manual rule is violated.

**8. Problems encountered.** None yet; it has only been run against empty state,
where it correctly reported that no jobs exist.

**9. How we fixed it.** N/A — to be re-verified against a live job during the
pilot.

**10. Decision made.** Run it hourly while any collection job is active.

---

## A-008 · First live pilot attempt fails with HTTP 500

**1. What we did.** Ran the M1 pilot for the first time on the Mac. It failed
immediately with repeated HTTP 500 responses from the archive.org scrape API.

**2. Why we did it.** This is the pilot whose entire purpose is to surface
problems at 300 documents rather than at 170,725.

**3. Command/script used.**
`python3 marathi/scripts/collect_archive_gr.py --limit 300 --page-size 200`

**4. Input source.** `https://archive.org/services/search/v1/scrape`

**5. Output produced.**

```
[500] server error, retry in 5s
[500] server error, retry in 5s
```

**6. Important numbers.** Zero documents collected. The same query issued from a
different machine at the same time returned **HTTP 200, total=170,725, cursor
present**, at both `count=100` and `count=200`.

**7. Why the result matters.** The endpoint is healthy; the failure is specific
to our request. That distinction is the whole diagnosis — it rules out "the
source is down" and points at something in what we send or where we send it
from.

Two details from the output are informative on their own. Both messages printed
"retry in 5s", but the backoff doubles within a single call (5, 10, 20…). Two
identical delays therefore mean two *separate* calls each on their first
attempt, not one call retrying — which is a different situation with a different
cause. The log format was too ambiguous to tell these apart, which is a defect
in our instrumentation, not just in the source.

**8. Problems encountered.** Three, in order of importance:
   - The 500 itself, cause not yet established.
   - Retry messages that could not distinguish one-call-retrying from
     two-calls-failing.
   - No visibility into the response body, so a 500 carrying an error message
     looked identical to a 500 from load shedding.

**9. How we fixed it.**
   - Wrote `tools/diagnose_source.py`, which varies **one thing at a time**:
     five User-Agent strings, four page sizes, the advancedsearch fallback, and
     a real document download. It writes nothing and changes nothing.
   - Retry messages now carry the attempt number (`[500 3/8]`), removing the
     ambiguity permanently.
   - 5xx responses now print the first 200 characters of the body.
   - `MAX_RETRIES` raised 5 → 8.
   - Changed the User-Agent to a shorter contactable form, and added an
     `advancedsearch.php` fallback enumerator so a scrape-API outage degrades
     the pilot instead of killing it.

**10. Decision made.** Diagnose before changing the collector's behaviour
further. The leading hypothesis is the User-Agent string; the alternative is
transient load shedding. These have different fixes, and the diagnostic
distinguishes them.

**Likely viva question:** *"What happened when your collector hit an error?"* —
It backed off exponentially, saved its checkpoint, and exited cleanly with zero
data loss. We then found the endpoint was healthy from elsewhere, which told us
the fault was in our request rather than the source, and we built a diagnostic
that isolates one variable at a time instead of changing several things at once
and hoping.

**Design note worth stating.** The pilot did exactly its job. It cost 300
documents' worth of time to discover an instrumentation weakness that would have
been far more expensive to diagnose 100,000 documents into a full run.

---

## A-009 · Diagnostic clears the 500s but exposes a 404 in our own URL construction

**1. What we did.** Ran `tools/diagnose_source.py` on the Mac and read all four
probes.

**2. Why we did it.** To establish whether the HTTP 500s from A-008 were caused
by our User-Agent, our network, or transient load on archive.org.

**3. Command/script used.** `python3 tools/diagnose_source.py`

**4. Input source.** archive.org scrape API, advancedsearch API, and one sample
item `in.gov.maharashtra.gr.202607071620477816`.

**5. Output produced.**

| Probe | Result |
|---|---|
| 1 — five User-Agents | **all HTTP 200** |
| 2 — page sizes 100/200/500/1000 | all HTTP 200, `total=170,796`, cursor present |
| 3 — advancedsearch fallback | HTTP 200, `numFound=170,796` |
| 4 — document download | **HTTP 404** |

**6. Important numbers.** Item count has grown from 170,725 to **170,796** since
selection — the collection is still being added to. Probes 1–3 succeeded on
every variant; probe 4 failed on the only thing that actually matters.

**7. Why the result matters.** Two separate conclusions, and they point in
opposite directions:

- The 500s were **transient load shedding**. Not our User-Agent, not our
  network. No code change needed for that; the existing backoff handles it.
- But the document download was **404**, which is a real bug in our collector,
  and it would have failed on all 170,796 items.

Root cause: the collector constructed the text-file URL as
`<identifier>_djvu.txt`. The actual file is `202607071620477816_djvu.txt` — the
numeric part only, without the `in.gov.maharashtra.gr.` prefix. Confirmed
against the item metadata.

The convention is **not consistent across archive.org**. For the Konkani book
`konkanibhashaman0000jbmo` the text file *is* `konkanibhashaman0000jbmo_djvu.txt`,
i.e. the full identifier. That earlier success is exactly what made the wrong
assumption look safe. A filename that cannot be constructed has to be looked up.

**8. Problems encountered.** Two, and the second is the more embarrassing:

   - The 404 bug itself.
   - **The diagnostic printed "Every User-Agent succeeded → the earlier 500s
     were transient" as its verdict while probe 4 was failing.** The verdict
     logic only considered the User-Agent results. It reported success over a
     failure.

**9. How we fixed it.**
   - `fetch_document()` now calls `https://archive.org/metadata/<id>` first,
     finds the file whose `format` is `DjVuTXT` (falling back to a `_djvu.txt`
     suffix match), and downloads that exact name. One extra request per
     document, and it also returns `licenseurl` / `rights` /
     `possible-copyright-status` for the provenance record — fields that are
     absent on these government items, which is logged as `not_stated` rather
     than silently assumed to mean public domain.
   - Probe 4 now tests the naive guess **and** the metadata lookup side by side,
     so the difference between them is visible rather than inferred.
   - The verdict now reports the download result **first** and gates the overall
     conclusion on it. Enumeration succeeding is worthless if documents cannot
     be fetched.

**10. Decision made.** Never construct an archive.org file URL; always resolve it
from item metadata. Re-run the pilot with the corrected collector.

**Likely viva question:** *"How did you validate your data collection code?"* —
We built a diagnostic that varies one factor at a time and ran it before the
full crawl. It cleared three hypotheses about an intermittent failure and, more
usefully, caught a URL-construction bug that would have produced 170,796
consecutive 404s. It also caught a flaw in itself: the verdict was summarising
only part of its own evidence, which we fixed by making the download result gate
the conclusion.

**Lesson, stated plainly.** A green summary line is not evidence. Both the
dedup self-test in A-005 and this diagnostic reported success while the thing
they were meant to check was failing. Read the individual measurements, not the
verdict.

---

## A-010 · Align the repository with the layout specified in the project PDF

**1. What we did.** Re-read the "Repository layout" section of the specification
and added the missing per-language directories, plus a real `README.md`.

**2. Why we did it.** The spec gives an explicit example layout with
`tokenizer/ model/ train/ eval/ configs/` inside each language directory, and
requires a top-level `README.md` carrying reproduction steps and Google Drive
links. Our `README.md` contained only the GitHub Classroom badge (202 bytes),
and `model/`, `train/`, `eval/` did not exist.

**3. Command/script used.** Directory creation plus `.gitkeep` files (git does
not track empty directories).

**4. Input source.** `LMA_Individual_Project_v1.pdf`, page 3–4.

**5. Output produced.** `marathi/{model,train,eval}/`,
`konkani/{model,train,eval}/`, and a full `README.md`.

**6. Important numbers.** README grew from 202 bytes to a complete document with
layout, Drive-link table, reproduction steps and report index.

**7. Why the result matters.** The README is directly graded — the spec says
Drive links go in it and that permissions must let TAs open them without
requesting access. It also says "if it is not in the branch, it is not graded".
An empty README is lost marks for work we have actually done.

The spec permits subfolder names inside a language directory to differ, so our
existing `data/` and `scripts/` are fine; adding `model/`, `train/` and `eval/`
makes the layout visibly match the example and prepares Phase 2.

**8. Problems encountered.** The Drive links cannot be filled in yet — the
corpora do not exist. The table is present with `pending` markers so the gap is
visible rather than forgotten.

**9. How we fixed it.** N/A.

**10. Decision made.** Fill the Drive-link table as each artifact is produced,
not in one pass at the end.

---

## A-011 · Abandon archive.org as the primary Marathi source; pivot to news scraping

**1. What we did.** Measured M1 (Maharashtra GRs on the Internet Archive) to
destruction, concluded it could not meet the deadline, and built M2, a
sitemap-driven news collector, as the primary manual Marathi source.

**2. Why we did it.** M1 was correct but far too slow. The decision was made on
measurement, not impatience.

**3. Command/script used.**
`python3 marathi/scripts/collect_archive_gr.py --limit 300 --page-size 500`
then `python3 marathi/scripts/collect_news.py --probe` and `--limit 200`.

**4. Input source.** archive.org; then eight candidate Marathi news sites.

**5. Output produced.**

| Metric | M1 archive.org | M2 news |
|---|---|---|
| Throughput | **2.1 docs/min** | **758 docs/min** |
| Fetch failure rate | 61–68% | ~0% |
| Words per document | 598 | 384 |
| Projected time for the target | **393 hours** | **3.1 hours** |
| Rejections | 80 fetch_failed of 90 | 3 too_short of 203 |

M1's second run then returned **0 documents in 10.7 minutes** after the scrape
API became entirely unavailable.

**6. Important numbers.** 758 vs 2.1 documents per minute — a factor of ~360.
Acceptance rate 98.5% on M2 against roughly 10% on M1.

**7. Why the result matters.** The Phase 1 manual requirement is ~100M Marathi
tokens ≈ 55M words. At M1's measured rate that needs 393 hours; four days were
available. M1 was therefore not a slow option, it was an impossible one, and
continuing with it would have failed the 20% manual requirement outright.

The cause was never our code — M1's rejection breakdown was 80 `fetch_failed`
against 6 `langid_undecided` and 4 `not_enough_devanagari`. The pipeline worked
perfectly; the server did not. That distinction is what justified keeping the
collector and changing the source, rather than debugging further.

**8. Problems encountered.**
   - archive.org degraded to the point of unusability, and its health is
     entirely outside our control.
   - A **correctness bug found while investigating**: identifiers were marked
     "seen" *before* fetching. With 61% of fetches failing, that would have
     permanently discarded 61% of the collection on a transient outage.
   - The first `--probe` judged each site on a **single** sitemap URL, and
     reported 4 of 8 sites usable.
   - Byline and timestamp furniture leaked into article text:
     `By ऑनलाइन लोकमत | Updated: August 16, 2026 00:25 IST 2026-08-16T00:25:11+5:30 - विकास…`

**9. How we fixed it.**
   - Timeouts cut from 45s to (5s connect, 12s read); one attempt instead of
     eight; workers raised 6 → 12. Failed items are no longer marked seen, so a
     later pass retries them.
   - M1 is **kept and still running in the background**. Every document it
     returns is genuine manual data and it costs nothing to leave grinding.
   - The probe now samples **8 URLs per site**. Re-running it took the verdict
     from 4/8 to **7/8 usable** — three of the four apparent failures were
     unlucky samples (a panchang table, a horoscope, a video page), not broken
     extraction.
   - Added URL-pattern exclusion for horoscopes, videos, galleries, AMP
     web-stories, panchang and live blogs.
   - Added byline/timestamp stripping, verified against the exact lokmat string
     above.

**10. Decision made.** M2 becomes the primary manual Marathi source; M1 is
demoted to a secondary background source, retained with its measurements
recorded. See decisions D-012 and D-013.

**Likely viva question:** *"Why did you change your data source partway
through?"* — Because we measured it. The Internet Archive source projected 393
hours for the required volume against a four-day deadline, with 61–68% of
fetches failing on a server whose health we do not control. We kept the
collector, verified the fault was external rather than in our pipeline, and
built a second source that measured 758 documents/minute on the same
infrastructure. The old source still runs in the background and its data still
counts.

**Method note worth stating.** The single-sample probe nearly cost us three
usable sites. One observation cannot distinguish "this source does not work"
from "this particular item has no content". That is the third time in this
project a summary has been more confident than its evidence — the dedup
self-test, the diagnostic verdict, and now the probe.

---

## A-012 · Audit the actual text, not just the counts

**1. What we did.** Built `tools/audit_quality.py` and ran it over both corpora.
It prints real documents and measures OCR noise, junk characters, self-repetition,
repeated opening phrases, script purity and language distribution per source.

**2. Why we did it.** Every check up to this point measured a *count* -
documents per minute, words, duplicate rate, manual ratio, langid distribution.
All of those can look healthy while the corpus is OCR fragments and website
furniture. Discovering that after training would be expensive; after the
deadline, fatal.

**3. Command/script used.**
`python3 tools/audit_quality.py --language marathi --samples 3`

**4. Input source.** The live corpora: 14 Marathi shards, 8 Konkani shards.

**5. Output produced.**

| Source | Devanagari | OCR noise | Self-repetition | Top boilerplate | langid |
|---|---|---|---|---|---|
| Marathi GRs | 86.0% | 9.8% | 4.1% | 0.6% | mr 100% |
| Marathi news | 96.4% | 5.1% | 0.6% | 0.1% | mr 100% |
| Konkani books | 97.1% | 9.0% | 0.2% | 0.1% | kok 97.2% |
| Konkani Wikipedia | 94.2% | 5.6% | 1.1% | 0.1% | kok 95.1% |

Zero automated flags on either language.

**6. Important numbers.** Boilerplate is the one that matters most: the most
common opening phrase covers only **0.6%** of GR documents and **0.1%** of news
articles. With 170k formulaic government documents, that is the number that
proves they are genuinely distinct rather than one template repeated.

**7. Why the result matters.** The samples read as authentic language - real
finance-department circulars in Marathi, a real report on an OBC census protest,
a Konkani travel account describing Buddhist sculpture. That is the evidence no
metric can provide, and it is why the tool prints documents rather than only
statistics.

**8. Problems encountered.** Two defects surfaced that the count-based checks had
missed entirely, both recorded below as A-013 and A-014.

**9. How we fixed them.** See those entries.

**10. Decision made.** Run this audit before committing to any long collection
run from a new source.

**Likely viva question:** *"How do you know your corpus is good, not just
large?"* — We measured OCR noise, boilerplate repetition, self-repetition, script
purity and language per source, and we read samples from each. The boilerplate
number is the sharpest: 0.6% maximum on the government corpus means the
documents are distinct, not one form letter repeated 170,000 times.

---

## A-013 · A dry run was consuming the work it was previewing

**1. What we did.** Ran `collect_archive_books.py --dry-run`, then ran it for
real. The real run collected **zero** documents.

**2. Why we did it.** The dry run is meant to be a safe preview.

**3. Command/script used.**
`python3 konkani/scripts/collect_archive_books.py --dry-run` then without it.

**4. Input source.** archive.org Konkani book items.

**5. Output produced.** Dry run: 4 books, 50 segments, 58,529 words. Real run:
**0 books, 0 segments, 0 words**.

**6. Important numbers.** 14 identifiers examined, 14 skipped.

**7. Why the result matters.** The dry run called `checkpoint.mark_seen()` and
persisted the checkpoint, so all 14 books were recorded as already processed. The
real run skipped every one. **A preview that silently consumes the work it is
previewing is worse than having no preview at all** - it looks like it succeeded.

**8. Problems encountered.** Checking the other ingesters revealed the *same bug*
in `ingest_books_corpus.py` and `ingest_indiccorp.py`. The second one had not
been run yet - a dry run there would have quietly poisoned the IndicCorp ingest.

**9. How we fixed it.** Under `--dry-run` all three now write their checkpoint
into a temporary directory that is discarded.

**10. Decision made.** After re-running: 5 books, 52 segments, **60,925 words**.

**Method note.** The fix was found by grepping every script for the same pattern
rather than fixing only the one that failed. One observed bug of a given shape
usually means several.

---

## A-014 · Nested wikitext was deleting 10% of the Konkani manual corpus

**1. What we did.** Fixed residual wikitext markup that the quality audit found
surviving in the "filtered" Konkani Wikipedia corpus, then re-ingested.

**2. Why we did it.** Sample output contained:

```
{{double image|right|flag of India.svg|195|Emblem of India.svg|84| चो बावटो| ...}}
: the invitation, Bowyer Bible.]] : the man
```

**3. Command/script used.**
`python3 konkani/scripts/ingest_wikipedia_manual.py --fresh`

**4. Input source.** `konkani/data/processed/konkani_wikipedia_filtered.txt`.

**5. Output produced.**

| | before | after | change |
|---|---:|---:|---:|
| pages accepted | 2,223 | **2,459** | +236 |
| words | 1,129,889 | **1,395,235** | +265,346 |
| words per page | 508 | 567 | +59 |
| near-duplicates removed | **228** | **1** | −227 |

**6. Important numbers.** Word count went **up** by 265,346 after *removing*
text. That is the finding.

**7. Why the result matters.** The root cause is that the original cleaner used
a single-pass `\{\{[^{}]*\}\}`. That regex cannot match a **nested** template -
the inner braces are consumed first and the outer pair is stranded.

But the consequence is the interesting part. Those 228 pages were never
duplicates. They shared wikitext boilerplate - the same infobox templates and
file links - and that shared markup pushed their MinHash similarity past the
0.85 threshold, so the deduplicator deleted them as near-duplicates.
**Boilerplate was destroying ~10% of the Konkani manual corpus**, and because
Konkani is capped at 5x its manual total, each deleted page also cost five words
of corpus ceiling.

This is exactly the mechanism asserted in decision D-014 when byline furniture
was stripped from the news crawler. Here it is measured rather than argued.

**8. Problems encountered.** Re-running the script appended a *second* copy of
every document to the manifest and shards, silently doubling the manual word
total - the number that decides how much downloaded data the corpus may hold.

**9. How we fixed it.** The script now refuses to run over existing output unless
`--fresh` is passed, which deletes the previous copy first. The web collectors
are immune to this because their checkpoints hold a seen-set; scripts that read a
fixed input file have no such protection and needed an explicit guard.

**10. Decision made.** D-015 and D-016 below.

**Likely viva question:** *"Why does boilerplate removal matter?"* — Two reasons,
and the second is not obvious. It stops the model learning that articles begin
with English file names. And it stops the deduplicator confusing shared
furniture for shared content: before the fix, 228 distinct Konkani articles were
being deleted as near-duplicates purely because they carried the same infobox.

---

## Next action

Run the diagnostic, then the M1 pilot, on the Mac:

```bash
cd ~/Desktop/individual-project-Pakkabhabad18
python3 tools/diagnose_source.py
python3 marathi/scripts/collect_archive_gr.py --limit 300 --page-size 200
python3 tools/health_check.py --all
```

Nothing scales until this reports. What we are looking for: the real
distribution of rejection reasons, the real near-duplicate rate, and words per
accepted document — which together give the first honest estimate of how many
manual Marathi tokens M1 can actually supply.
