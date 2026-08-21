"""
Preliminary "mixed" Konkani tokenizer experiment (superseded).

Variant of `train_preliminary_tokenizer.py` trained over a sample mixing books
text with Wikipedia text, to see whether domain mixing changed subword
segmentation. Same unjustified vocab_size=32000.

Superseded by `tools/build_tokenizer.py`. Retained for provenance; its output
is not used by any Phase 1 artifact.
"""

import sentencepiece as spm


INPUT_FILE = (
    "konkani/data/raw/"
    "tokenizer_mixed_sample.txt"
)

MODEL_PREFIX = (
    "konkani/tokenizer/"
    "preliminary_konkani_mixed_bpe"
)

VOCAB_SIZE = 32000


def main():

    print("=" * 60)
    print("TRAINING MIXED-SCRIPT KONKANI TOKENIZER")
    print("=" * 60)

    print(f"Input:      {INPUT_FILE}")
    print(f"Vocabulary: {VOCAB_SIZE}")
    print("Algorithm:  BPE")
    print()

    spm.SentencePieceTrainer.train(
        input=INPUT_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        train_extremely_large_corpus=True,
        input_sentence_size=500000,
        shuffle_input_sentence=True,
        max_sentencepiece_length=16,
        num_threads=8,
    )

    print("\nTokenizer training complete.")

    print(
        f"Model: {MODEL_PREFIX}.model"
    )

    print(
        f"Vocabulary: {MODEL_PREFIX}.vocab"
    )


if __name__ == "__main__":
    main()
