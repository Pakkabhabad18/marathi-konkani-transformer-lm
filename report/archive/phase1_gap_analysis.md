# Phase 1 — Gap Analysis

**Project:** Language Models and Agents, Monsoon 2026 — Individual Project
**Model H:** Marathi · **Model L:** Konkani (Devanagari)
**Branch:** `phase-1` · **Repo:** `CL3-410/individual-project-Pakkabhabad18`
**Audit date:** 14 August 2026 · **Phase 1 deadline:** 19 August 2026, 23:59 (5 days)

This document audits the repository as it exists against every Phase 1 requirement in
`LMA_Individual_Project_v1.pdf`. Nothing is marked complete without evidence that was
actually inspected or recomputed during the audit.

---

## 0. Audit method

What was inspected:

- The full official PDF (all 12 pages), not just the Phase 1 section.
- Every file in the working tree except `.venv/` and `.git/` internals (65 project entries).
- All 15 Python scripts under `konkani/scripts/`, read in full.
- `report/phase1_konkani_progress.md`, `README.md`, `.gitignore`, `requirements.txt`.
- Git state: `.git/config`, `HEAD`, `logs/HEAD`, local and remote refs, total object-store size.
- Both trained tokenizer models, **loaded and executed** to verify their actual behaviour.
- Public dataset cards for `omdeep22/Konkani_books_corpus-v2`, `ai4bharat/IndicCorpV2`,
  `ai4bharat/sangraha`.

What could **not** be verified directly: the 61.5 MB and 62 MB raw sample files under
`konkani/data/raw/` (transfer to the audit environment timed out). Claims that depend on them
are derived arithmetically from the reported statistics instead, and that is stated where it
applies.

---

## 1. Executive summary

The existing work is real and worth keeping. The Wikipedia collector, the filter, the corpus
audit script and the two tokenizers all run and produce genuine numbers. Nothing should be
deleted.

However, the audit found **four defects that change the conclusions already written into
`report/phase1_konkani_progress.md`**, and **three requirements that are not started at all**.
In priority order:

| # | Finding | Severity |
|---|---------|----------|
| A | Marathi (Model H) pipeline does not exist — empty directories only. Half of Phase 1 is missing. | **Critical** |
| B | The ≥20% manual-collection rule applies to **both** languages. No plan exists for ~100M manual Marathi tokens; Konkani manual stands at ~2.6M against a ~20M requirement. | **Critical** |
| C | Both tokenizers were trained **without `byte_fallback`**, so all reported UNK rates are artefacts of missing Latin coverage, not evidence about script mixing. The "must be mixed-script" conclusion does not follow from the data presented. | **High** |
| D | `analyze_books_corpus.py` divides Devanagari characters by **total** characters including whitespace. The corpus is not 83.54% Devanagari — it is **99.78%** Devanagari. | **High** |
| E | Local commits have not been pushed; `origin/phase-1` is stale. Grading reads the *remote* branch. | **High** |
| F | No deduplication, no Unicode normalization, no train/val/test splits anywhere. | **High** |
| G | No cross-corpus contamination control between Marathi and Konkani, which the spec forbids sharing documents between. | **High** |

---

## 2. Requirement-by-requirement analysis

Status key: **DONE** · **PARTIAL** · **MISSING** · **WRONG** (implemented, but incorrectly)

### 2.1 Language selection (§1.1)

| Item | Status | Evidence | Gap / fix | Priority |
|---|---|---|---|---|
| Model L from allowed list | **DONE** | Konkani is on the permitted list (Assamese, Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri, Mizo, Nepali, Sindhi). | Written justification for the choice is not yet in any report. Add 3–4 lines. | P2 |
| Model H non-English Indian, justified | **PARTIAL** | `marathi/` directory tree exists; no content, no justification. | Marathi qualifies easily. Needs a short written justification citing available public corpus sizes. | P2 |

**Risk specific to this pair.** Marathi and Konkani are closely related, both written in
Devanagari, and are routinely confused in automatically-collected corpora. During the audit the
*Konkani* tokenizer was run on Marathi text:

```
Marathi input:  मी घरी जातो आणि जेवण करतो. महाराष्ट्रातील शेतकरी संकटात आहेत.
Konkani BPE:    ['▁मी','▁घरी','▁जातो','▁आणि','▁जेवण','▁करतो','.','▁महाराष्ट्र','ाती','ल',...]
                16 tokens, 0 UNK, 3.81 chars/token
```

It tokenizes Marathi as cleanly as it tokenizes Konkani. This is not a tokenizer bug — it is
direct evidence of how orthographically close the two languages are, and therefore how easily
Marathi text can leak into a "Konkani" corpus and vice versa. See §2.5.

### 2.2 Dataset requirements (§1.2)

| Item | Status | Evidence | Gap / fix | Priority |
|---|---|---|---|---|
| Konkani corpus collected | **PARTIAL** | Books corpus audited (8,222,553 usable records, 61,805,534 words); Wikipedia self-collection (3,999 pages retained after filter). | ~92M preliminary tokens against a 500M target. Needs non-Wikipedia expansion. | P1 |
| Marathi corpus collected | **MISSING** | `marathi/data/{raw,processed,manual,splits}` all empty. No scripts. | Entire pipeline to build. | **P0** |
| ~500M tokens per language | **PARTIAL / at risk** | Konkani ~92M (preliminary, mixed tokenizers). Marathi 0. | Marathi is reachable. Konkani is not — see §3 for the honest ceiling and the justification to write. | P1 |
| ≥20% manual, **both** languages | **MISSING** | Konkani manual = Wikipedia only (~5.16M mixed-script tokens, of which only the Devanagari subset counts under our script decision ≈ 2.6M). Marathi manual = 0. | Largest single piece of remaining work. See §3. | **P0** |
| No shared documents across corpora | **MISSING** | No language-ID filter, no cross-corpus dedup anywhere. | Required by spec. See §2.5. | P1 |
| Invalid data removal | **PARTIAL** | `filter_wikipedia_corpus.py` drops pages under 50 words / 100 chars. `analyze_books_corpus.py` skips empty and `--- SOURCE:` marker records. | Source-appropriate filters needed per source, not one generic rule. No boilerplate or metadata stripping for the books corpus. | P1 |
| Duplicate removal | **MISSING** | No dedup code exists. | Exact-hash dedup plus near-duplicate (MinHash/SimHash) pass. Especially important: the books corpus is line-fragmented (7.52 words/record), which produces very high exact-duplicate rates on short lines. | P1 |
| Unicode normalization | **MISSING** | No NFC/NFKC anywhere. `clean_text()` only collapses whitespace. | Devanagari has multiple encodings for the same grapheme (nukta composition, ZWJ/ZWNJ). Without NFC the tokenizer learns duplicate pieces. Must run before tokenizer training. | **P0** |
| Corpus statistics | **PARTIAL** | `corpus_audit.py` is correct. `analyze_books_corpus.py` is **wrong** — see §2.3. | Fix the bug; add per-source stats tables. | P1 |
| Train/val/test splits | **MISSING** | `konkani/data/splits/` and `marathi/data/splits/` are empty. | Explicit deliverable #4. Must be document-level, not line-level, to avoid leakage. | P1 |
| Manual vs downloaded token split reported | **MISSING** | `report/phase1_konkani_progress.md` has no manual/downloaded column. | Required by spec ("report the manual vs. downloaded token split in your dataset statistics"). | P1 |

### 2.3 Defect detail: the Devanagari percentage is wrong

`konkani/scripts/analyze_books_corpus.py`:

```python
non_whitespace = total_characters          # <-- this is TOTAL characters
devanagari_percentage = (total_devanagari / non_whitespace) * 100
```

The variable is named `non_whitespace` but is assigned the total character count, whitespace
included. `corpus_audit.py` computes the same statistic correctly, so the two scripts disagree.

The dataset card for `omdeep22/Konkani_books_corpus-v2` warns of "large gaps between words and
excessive line breaks" from OCR — so whitespace is not a rounding error here.

Recomputing from the numbers already in the progress report:

```
total characters      379,610,529
Devanagari characters 317,109,283
non-Devanagari         62,501,246
words                  61,805,534
                       ------------
non-Devanagari per word     1.0113
```

There is almost exactly **one** non-Devanagari character per word — i.e. the inter-word spaces
and nothing else. Taking whitespace out of the denominator:

```
Devanagari as % of non-whitespace ≈ 317,109,283 / 317,804,995 = 99.78%
```

**Consequence.** The corpus is essentially pure Devanagari, not 83.54%. The reported figure
created a false impression of ~16% non-Devanagari contamination and fed into the argument for a
mixed-script tokenizer. That argument has to be rebuilt on the corrected numbers.

### 2.4 Tokenizer training (§1.3)

| Item | Status | Evidence | Gap / fix | Priority |
|---|---|---|---|---|
| Trained from scratch, no pretrained | **DONE** | SentencePiece BPE trained locally in `train_preliminary_tokenizer.py` / `train_mixed_tokenizer.py`. Compliant. | — | — |
| Separate tokenizer per language | **PARTIAL** | Two Konkani tokenizers exist. No Marathi tokenizer. | Build Marathi tokenizer. | **P0** |
| Vocabulary size reported | **DONE** | 32,000, verified by loading both models (`get_piece_size() == 32000`). | Justify the choice against fertility, as the spec asks. | P2 |
| Token-frequency statistics | **MISSING** | `tokenizer_stats.py` reports totals and rates but no frequency distribution. | Add frequency histogram / top-k pieces / coverage curve. | P1 |
| Average characters per token | **DONE** | Computed in `tokenizer_stats.py`. | Recompute on **held-out** text, not training data — see below. | P1 |
| Tokenization examples | **MISSING** | Not produced anywhere. | Explicit reporting requirement. Cheap to add. | P2 |
| Unknown-token statistics | **WRONG** | See below. | Retrain with `byte_fallback=True`, then report UNK ≈ 0 and explain *why*. | **P0** |
| Fertility / UNK comparison on held-out text | **WRONG** | `tokenizer_stats.py` evaluates on `tokenizer_sample.txt` — the tokenizer's own training file. | Held-out set required by spec. Current fertility figures are optimistic. | P1 |

#### Defect detail: `byte_fallback` is off

Both tokenizers were trained with `character_coverage=0.9995` and **no** `byte_fallback`
argument. Verified by loading each model and counting byte pieces in the vocabulary:

```
preliminary_konkani_bpe.model        vocab 32000   byte-fallback pieces: 0   -> OFF
preliminary_konkani_mixed_bpe.model  vocab 32000   byte-fallback pieces: 0   -> OFF
```

Running both models on controlled inputs reproduces the reported behaviour exactly and explains
it:

| Input | Devanagari-heavy tokenizer | Mixed tokenizer |
|---|---|---|
| Konkani (Devanagari) | 11 tokens, **0% UNK** | 11 tokens, **0% UNK** |
| Konkani (Roman) | 20 tokens, **45.0% UNK** | 24 tokens, 8.3% UNK |
| English | 19 tokens, **47.4% UNK** | 24 tokens, 4.2% UNK |
| Marathi | 16 tokens, **0% UNK** | 16 tokens, 0% UNK |

The 46.36% UNK rate in the progress report is therefore **not** a finding about Konkani
orthography. It is the mechanical consequence of asking a tokenizer with no Latin characters in
its vocabulary, and no byte fallback, to encode Latin text. Every Latin character becomes `<unk>`.

Two things follow:

1. **A correctly configured subword tokenizer has ~0% UNK by construction.** With
   `byte_fallback=True`, SentencePiece emits `<0xNN>` byte pieces for anything it cannot cover,
   so any UTF-8 string is representable. Reporting a 7.19% UNK rate invites the viva question
   *"why can your tokenizer not represent its own training data?"*, which has no good answer.
2. **The metric that actually decides script coverage is fertility**, i.e. tokens/word and
   chars/token, not UNK rate. With byte fallback on, poorly-covered script gets *expensive*
   (many tokens per word) rather than *unrepresentable*. That is the comparison the spec is
   asking for when it says "choose using fertility / unknown-token rate on held-out text".

**Note on the mixed-script conclusion.** It may still be right to include Roman Konkani — but
the evidence presented does not establish it, and under the decision recorded for this project
(Devanagari only, Romi documented as an excluded sub-corpus) the question is now moot for the
main corpus. The experiment stays in the repo as a documented negative result about tokenizer
configuration, which is a *better* viva story than the original claim.

### 2.5 Cross-corpus independence (§ Overview, §1.2)

The spec states: *"Do not share documents across corpora or concatenate languages"* and
*"The two resulting models must not share data, tokenizer, vocabulary, or weights."*

Currently there is nothing enforcing this. Given the Marathi/Konkani similarity demonstrated in
§2.1, this is a live risk, not a theoretical one:

- Public "Konkani" web crawls frequently contain Marathi.
- The books corpus provenance is described only as "digitized books, literature, and long-form
  cultural texts" — no per-book manifest, so its language purity is unverified.
- Marathi crawls from Maharashtra/Goa news sources will contain Konkani passages.

**Required:** a language-ID pass (trained on our own verified seed text, since off-the-shelf
langid models handle the gom/mar distinction poorly) plus a cross-corpus document-hash check
proving zero overlap. This is a deliverable-quality result: "0 shared documents, verified by
hash over N and M documents" is exactly the kind of evidence the spec asks for.

### 2.6 Repository, git, and deliverables

| Item | Status | Evidence | Gap / fix | Priority |
|---|---|---|---|---|
| Per-language directory layout | **DONE** | `konkani/` and `marathi/` both present with `configs/ data/ scripts/ tokenizer/`. Matches the spec's example layout. | — | — |
| Large artifacts kept out of git | **DONE** | `.gitignore` excludes `.venv/`, `__pycache__/`, `konkani/data/`. Verified: entire `.git` object store is **1.49 MB** — no corpus data was ever committed. | Add `marathi/data/` to `.gitignore` **before** any Marathi data lands, or it will be committed by accident. | **P0** |
| README with Drive links | **MISSING** | `README.md` contains only the GitHub Classroom badge (202 bytes). | Explicitly graded: "Put Google Drive links for datasets and checkpoints in the top-level README", with permissions open so TAs need not request access. | P1 |
| Meaningful commit history | **DONE** | 5 commits with descriptive messages: environment setup, audit script, corpus pipeline, tokenizers, gitignore. Satisfies "commits must reflect actual progress". | Keep the cadence. | — |
| **Work pushed to `origin/phase-1`** | **WRONG** | `refs/heads/phase-1` and `refs/remotes/origin/phase-1` both read `8a11864…`, but that ref was last written by a `git fetch`/`push` at clone time. Local commits after the clone are **not** confirmed on the remote. | **Verify with `git status -sb` and push today.** The branch is graded as it stands on GitHub at the deadline; local commits score zero. | **P0** |
| `__pycache__` committed | **PARTIAL** | `.gitignore` lists `__pycache__/`, but 14 `.pyc` files exist in the working tree under `konkani/scripts/__pycache__/`. | Confirm they are untracked (they were likely added before the ignore rule). `git rm -r --cached` if tracked. | P2 |
| Phase 1 deliverables 1–7 | **PARTIAL** | 1 Collection scripts: Konkani only. 2 Preprocessing: partial. 3 Stats reports: partial + one wrong figure. 4 Splits: none. 5 Tokenizer training code: Konkani only. 6 Vocab files: Konkani only. 7 Tokenizer models: Konkani only. | Marathi side of every one of the seven. | **P0** |

### 2.7 Collection pipeline robustness (§10 of the working brief)

| Item | Status | Notes |
|---|---|---|
| Checkpointing | **PARTIAL** | `resume_wikipedia_collection.py` resumes by re-traversing from the start and skipping known page IDs. It states this itself in its own output. Works, but O(n) wasted requests on every resume. |
| Rate-limit handling | **DONE** | Handles HTTP 429 with `Retry-After`, retries on timeout, sleeps 2s between batches, sets a real User-Agent. Genuinely good practice — keep this pattern. |
| Continuation-token persistence | **MISSING** | Not saved to disk. This is the root cause of the O(n) resume. Fix: write the continuation dict to a JSON checkpoint after each batch. |
| Duplicate avoidance | **DONE** | Page-ID set, loaded from existing metadata CSV. |
| Incremental flush | **DONE** | `text_file.flush()` / `metadata_file.flush()` per record — survives a kill. |
| Failure logging | **MISSING** | Errors print to stdout only; nothing persisted. |
| Two near-duplicate scripts | **NOTED** | `collect_wikipedia_sample.py` and `resume_wikipedia_collection.py` are ~85% identical. Should be one script with a `--resume` flag. Low priority — it is not wrong, just duplicated. |

---

## 3. The token budget — the arithmetic that drives everything else

The manual requirement is a **ratio**, not an absolute. If `M` is manual tokens and `T` is total
final training tokens, the spec requires `M / T ≥ 0.20`, i.e.

> **T ≤ 5 × M**

This has a consequence that is easy to miss: **adding more downloaded data can make you
non-compliant.** Past a point, extra public corpus does not help — it dilutes the manual
fraction. The size of each corpus is therefore capped by how much we collect by hand.

### Marathi (Model H)

| Bucket | Source | Tokens | Type |
|---|---|---|---|
| Downloaded | `ai4bharat/IndicCorpV2` (mar_Deva, ~27.8M rows, CC-0) | ample | downloaded |
| Downloaded | `ai4bharat/sangraha` verified (2,827M) + unverified (652M), CC-BY-4.0 | ample | downloaded |
| Manual | News / literary / government scraping and OCR by us | **~100M needed** | manual |
| | **Target total** | **~500M** | 20% manual |

Marathi has far more public text than we need, so the binding constraint is purely the manual
100M. At a Devanagari BPE fertility of roughly 2 tokens/word, 100M tokens ≈ 50M words ≈ **125k
articles at 400 words each**. That is achievable with a resumable sitemap-driven crawler running
continuously, but it must start immediately — it is the long pole for the deadline.

> Do **not** use Sangraha's *synthetic* split (10,817M tokens). It is machine-translated and
> romanized WikiMedia content, not naturally-occurring Marathi.

### Konkani (Model L)

| Bucket | Source | Tokens | Type |
|---|---|---|---|
| Downloaded | `omdeep22/Konkani_books_corpus-v2` (MIT) | ~87M preliminary | downloaded |
| Downloaded | `ai4bharat/sangraha` gom — **10.1M total, all splits** | ~10M | downloaded |
| Manual | Konkani Wikipedia, Devanagari subset only | ~2.6M | manual (weak) |
| Manual | **To be collected** | **~20M needed** | manual |

Sangraha — the largest systematic Indic corpus effort available — contains only **10.1M tokens**
of Konkani across all of its splits. That single figure is the strongest possible evidence for
the shortfall justification the spec explicitly permits.

Applying `T ≤ 5 × M`: if manual collection reaches ~20M tokens, the total Konkani corpus is
capped at ~100M tokens, which the books corpus alone nearly fills. **The realistic Konkani plan
is therefore ~100M total tokens with ~20M manual — not 500M** — with the shortfall reported
exactly and justified on evidence. Chasing 500M for Konkani is not achievable from legitimate
sources and attempting it would mean inflating counts, which the spec and the working brief both
forbid.

---

## 4. Prioritised work plan

**P0 — must happen in the next 24–48 hours**

1. Verify and push `phase-1` to GitHub. Confirm `git status -sb` shows nothing ahead of origin.
2. Add `marathi/data/` to `.gitignore` before any Marathi data is generated.
3. Start the Marathi manual collection crawler. It runs for days; everything else is faster.
4. Add Unicode NFC normalization to the preprocessing path, before any tokenizer is retrained.
5. Retrain tokenizers with `byte_fallback=True`; re-measure fertility on held-out text.

**P1 — this week**

6. Marathi downloaded-corpus ingestion (IndicCorpV2 + Sangraha verified/unverified).
7. Deduplication (exact then near-duplicate) per language.
8. Konkani manual collection from non-Wikipedia sources (see `phase1_source_inventory.md`).
9. Cross-corpus language-ID and contamination check.
10. Document-level train/val/test splits per language.
11. README with Google Drive links and open permissions.

**P2 — before submission**

12. Token-frequency statistics, tokenization examples, chars/token tables per language.
13. Written justifications: language choice, vocab size, Konkani shortfall.
14. Merge the two Wikipedia collectors into one script with `--resume`.
15. Final single-tokenizer recount producing one internally consistent official token count.

---

## 5. What must not be repeated in the final report

These figures are in `report/phase1_konkani_progress.md` and are now known to be misleading.
They should be corrected there, not silently dropped — a documented correction is stronger
evidence of understanding than a clean number.

| Claim currently written | Status | Replacement |
|---|---|---|
| "Devanagari percentage: 83.54%" | Wrong (whitespace in denominator) | ~99.78% of non-whitespace characters |
| "Wikipedia UNK rate = 46.36%" | Artefact of `byte_fallback=False` | Retrain with byte fallback; report UNK ≈ 0 and compare fertility instead |
| "UNK rate: 7.19%" | Same artefact | Same |
| "Conclusion: tokenizer should represent both Devanagari and Roman" | Not supported by the evidence given | Decision: Devanagari-only corpus; Romi documented as an excluded sub-corpus |
| "~91.91M tokens" | Mixes two different preliminary tokenizers | One final tokenizer, one recount, one official number |
| "Estimated tokens: 86.75M" | Fertility measured on the tokenizer's own training data | Re-measure on held-out text |
