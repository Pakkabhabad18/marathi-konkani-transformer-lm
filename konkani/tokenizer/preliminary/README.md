# Superseded tokenizers

Early Konkani tokenizers, kept because the decisions log refers to them and
because the measurements taken with them are part of how the final tokenizer was
arrived at. They are not deliverables and must not be used.

| file | what it was | why it was replaced |
|---|---|---|
| `preliminary_konkani_bpe.*` | first Konkani BPE, trained on the Wikipedia-only corpus | corpus was ~2.2M words, unrepresentative of the final corpus; fertility measured on training data rather than held-out text (D-007) |
| `preliminary_konkani_mixed_bpe.*` | mixed Devanagari + Romi experiment | superseded by the Devanagari-only script decision |

The Phase 1 deliverables are `konkani/tokenizer/konkani_bpe.model` and
`konkani/tokenizer/konkani_bpe.vocab`, vocabulary 2,500.
