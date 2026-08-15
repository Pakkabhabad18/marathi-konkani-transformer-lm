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
