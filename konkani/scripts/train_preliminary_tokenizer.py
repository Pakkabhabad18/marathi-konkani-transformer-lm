"""
Preliminary Konkani tokenizer experiment (superseded).

First tokenizer trained during early exploration, at vocab_size=32000 over a
500k-record sample of the books corpus. Used to get an initial feel for
Devanagari subword behaviour before any real corpus existed.

Superseded by `tools/build_tokenizer.py`, which sweeps candidate vocabulary
sizes, measures fertility, unknown-token rate, vocabulary utilisation and hapax
count on held-out text, and selects by a stated rule. The 32000 vocabulary here
was never justified against the parameter budget; the final tokenizers use
2,500 (see D-032).

Retained for provenance. Its output is not used by any Phase 1 artifact.
"""

import sentencepiece as spm
from pathlib import Path


INPUT_FILE = "konkani/data/raw/tokenizer_sample.txt"
MODEL_PREFIX = "konkani/tokenizer/preliminary_konkani_bpe"
VOCAB_SIZE = 32000


def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    print("=" * 60)
    print("TRAINING PRELIMINARY KONKANI TOKENIZER")
    print("=" * 60)

    print(f"Input:       {INPUT_FILE}")
    print(f"Vocabulary:  {VOCAB_SIZE}")
    print("Algorithm:   BPE")
    print("Model:       SentencePiece")
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
    print(f"Model: {MODEL_PREFIX}.model")
    print(f"Vocabulary: {MODEL_PREFIX}.vocab")


if __name__ == "__main__":
    main()
