# Phase 1 — Konkani Corpus Progress

## Current Objective

Build a large-scale Konkani corpus for Phase 1, targeting
500M training tokens.

## 1. Existing Konkani Books Corpus

Dataset:
`omdeep22/Konkani_books_corpus-v2`

Corpus audit:

- Total records: 8,228,786
- Usable records: 8,222,553
- Total characters: 379,610,529
- Total words: 61,805,534
- Devanagari characters: 317,109,283
- Devanagari percentage: 83.54%

Preliminary tokenizer estimate:

- Estimated tokens: 86.75M

Note:
This token count is a preliminary estimate and must be
recomputed using the final tokenizer before reporting the
official training-token count.

## 2. Self-Collected Konkani Wikipedia

Collection:

- Pages collected: 4,933
- Words: 2,275,434
- Characters: 15,979,682

Quality filtering:

- Pages retained: 3,999
- Pages removed: 934
- Retention: 81.07%
- Filtered words: 2,252,016
- Filtered characters: 15,797,232

Script distribution of raw collection:

- Devanagari: 2,613 pages (52.97%)
- Roman: 1,727 pages (35.01%)
- Mixed: 593 pages (12.02%)

Preliminary mixed-tokenizer result:

- SentencePiece tokens: 5,160,921
- Tokens/word: 2.2917
- UNK tokens: 370,911
- UNK rate: 7.19%

## 3. Tokenizer Experiments

### Devanagari-heavy preliminary tokenizer

Wikipedia sample:

- Tokens: 171,880
- UNK rate: 46.36%

### Mixed-script preliminary tokenizer

Wikipedia sample:

- Tokens: 250,334
- UNK rate: 6.60%

Conclusion:

Konkani tokenizer training should represent both
Devanagari and Roman-script Konkani.

## 4. Current Token Estimate

Books Corpus:
~86.75M preliminary tokens

Wikipedia:
5.16M preliminary mixed-tokenizer tokens

Current baseline:
~91.91M tokens

This is NOT the final official training-token count.

## 5. Remaining Work

- Expand Konkani corpus
- Identify additional eligible sources
- Collect self-collected data
- Improve language/script filtering
- Deduplicate corpus
- Build final mixed-script tokenizer
- Recount entire corpus using final tokenizer
- Construct train/validation/test splits
- Verify 500M-token training requirement
- Verify self-collected-data requirement
- Prepare Phase 1 report and reproducibility artifacts
