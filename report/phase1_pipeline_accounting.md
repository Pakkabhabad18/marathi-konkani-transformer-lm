# Phase 1 — Pipeline Accounting

Raw collected → cleaned → deduplicated → final training tokens.
Stages 1–5 in **words** (tokenizer-independent); stage 6 in **tokens**.

| Stage | Marathi | Konkani |
|---|---:|---:|
| documents fetched | 2,237,944 | 372,836 |
| documents accepted | 2,887,867 | 323,112 |
| words accepted | 389,218,163 | 266,211,363 |
| manual words | 177,781,779 | 64,435,242 |
| downloaded words (real) | 211,436,384 | 116,071,660 |
| synthetic words (MT/LLM) | 0 | 85,704,461 |
| real words (manual+dl) | 389,218,163 | 180,506,902 |
| final training tokens | 872,024,099 | 506,259,368 |
| manual tokens | 475,466,104 | 159,563,967 |

`documents fetched` and `documents accepted` are counted at different
granularities and are not two stages of one funnel. Fetched is one row per work
item a collector requested — an archive.org book, a news URL, a corpus shard —
read from the collector checkpoints, and it includes items that were later
rejected. Accepted is one row per document in the manifests, after cleaning has
split multi-document items: a single OCR'd book becomes many documents, one
corpus shard becomes many rows.

The two therefore move independently. Marathi fetched 2,237,944 items, rejected
137,863, and the surviving 2,100,081 items yielded 2,887,867 documents — most of
the growth from `marathi_indiccorp`, where 1,848,028 shard items produced
2,635,630 documents. Konkani fetched 372,836 items and rejected 236,567 (63%),
and the surviving 136,269 items yielded 323,112 documents — `konkani_archive_books`
alone turned 2,576 OCR'd volumes into 53,843 documents.

The rows that do form a funnel are the word and token counts.
