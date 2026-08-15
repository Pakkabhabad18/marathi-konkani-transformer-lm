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
