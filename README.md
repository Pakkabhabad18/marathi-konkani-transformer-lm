[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Language Models and Agents — Individual Project

Two completely independent monolingual decoder-only Transformer language models,
built from scratch.

| | Language | Tier |
|---|---|---|
| **Model H** | Marathi (`mr`) | higher-resource |
| **Model L** | Konkani (`kok`, Devanagari) | lower-resource |

The two models share **no** data, tokenizer, vocabulary or weights. Only
language-agnostic utility code under `common/` is shared; see
`report/phase1_decisions.md` decision D-009 for why that is compatible with the
independence requirement.

---

## Repository layout

```
├── README.md                 reproduction steps + Google Drive links
├── common/                   shared, language-agnostic utility code only
│   ├── textnorm.py             Unicode NFC + whitespace normalization
│   ├── scriptid.py             script profiling + Marathi/Konkani language ID
│   ├── manifest.py             per-document provenance + manual/downloaded accounting
│   ├── checkpoint.py           crash-safe resumable checkpoints
│   └── dedup.py                exact (SHA-256) + near-duplicate (MinHash/LSH)
├── tools/
│   ├── health_check.py         monitors long-running collection jobs
│   └── diagnose_source.py      isolates why a source is failing
├── marathi/                  Model H — self-contained
│   ├── scripts/                collection + preprocessing
│   ├── data/                   NOT in git (see Google Drive links below)
│   ├── tokenizer/              tokenizer model + vocabulary
│   ├── configs/                model / training configuration
│   ├── model/                  Transformer implementation (Phase 2)
│   ├── train/                  training scripts (Phase 2)
│   └── eval/                   evaluation scripts (Phase 2)
├── konkani/                  Model L — self-contained, same structure
└── report/                   per-phase reports, tables, figures
```

Corpus data, tokenized corpora and model checkpoints are **not** committed to
git, per the project instructions. See the Drive links below.

---

## Google Drive links

> **Status: to be added before the Phase 1 deadline.** Permissions will be set
> so graders can open them without requesting access.

| Artifact | Phase | Link |
|---|---|---|
| Marathi corpus (raw + processed + splits) | 1 | _pending_ |
| Konkani corpus (raw + processed + splits) | 1 | _pending_ |
| Marathi tokenizer model + vocabulary | 1 | _pending_ |
| Konkani tokenizer model + vocabulary | 1 | _pending_ |
| Model H pretrained checkpoints | 2 | _pending_ |
| Model L pretrained checkpoints | 2 | _pending_ |
| Model H / L finetuned checkpoints | 3 | _pending_ |

---

## Reproduction steps

### Requirements

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12. Collection and preprocessing need only `requests` and the standard
library; tokenizer training additionally needs `sentencepiece`.

### Verify the shared library

Every module in `common/` self-tests. Run before trusting any pipeline output:

```bash
python3 common/textnorm.py
python3 common/scriptid.py
python3 common/manifest.py
python3 common/checkpoint.py
python3 common/dedup.py
```

Each prints `... self-test: all assertions passed`.

### Phase 1 — data collection

```bash
# Diagnose a source before collecting from it
python3 tools/diagnose_source.py

# Collect (resumable: Ctrl-C and re-run to continue, never restarts)
python3 marathi/scripts/collect_archive_gr.py --limit 300     # pilot
python3 marathi/scripts/collect_archive_gr.py                 # full run

# Monitor a running job from a second terminal
python3 tools/health_check.py --all
```

Konkani Wikipedia collection (documented secondary/experimental source):

```bash
python3 konkani/scripts/collect_wikipedia_sample.py
python3 konkani/scripts/resume_wikipedia_collection.py
python3 konkani/scripts/filter_wikipedia_corpus.py
```

---

## Reports

| Document | Contents |
|---|---|
| `report/phase1_runbook.md` | every command in order, with expected output |
| `report/phase1_gap_analysis.md` | requirement-by-requirement audit against the spec |
| `report/phase1_source_inventory.md` | every candidate and accepted source, with evidence |
| `report/phase1_execution_plan.md` | what we build, in what order, and why |
| `report/phase1_decisions.md` | design decisions and their justification |
| `report/phase1_viva_log.md` | full narrative log of every major step |
| `report/phase1_konkani_progress.md` | original Konkani exploration (preserved; some figures superseded — see decisions D-003, D-004) |

---

## AI tool usage

AI assistance was used for code drafting, auditing and documentation. Every
design decision is recorded with its justification in
`report/phase1_decisions.md`, and every step with its measured result in
`report/phase1_viva_log.md`, including the bugs found and how they were
diagnosed and corrected.
