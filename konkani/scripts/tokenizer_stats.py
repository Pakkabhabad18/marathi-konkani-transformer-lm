"""
Statistics for the preliminary tokenizer (superseded).

Reports token counts for `preliminary_konkani_bpe.model` over the early
sample file.

Superseded by `tools/build_tokenizer.py`, which reports the same statistics
plus vocabulary utilisation and hapax count, and — importantly — measures them
on HELD-OUT text rather than on the training file. Measuring fertility on the
data the tokenizer was fitted to understates it, which is the flaw this script
has and the reason it was replaced.

Retained for provenance.
"""

import sentencepiece as spm


MODEL_FILE = "konkani/tokenizer/preliminary_konkani_bpe.model"
INPUT_FILE = "konkani/data/raw/tokenizer_sample.txt"


def main():
    print("=" * 60)
    print("KONKANI PRELIMINARY TOKENIZER STATISTICS")
    print("=" * 60)

    tokenizer = spm.SentencePieceProcessor(
        model_file=MODEL_FILE
    )

    total_words = 0
    total_tokens = 0
    total_characters = 0
    total_unk = 0
    total_lines = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            text = line.strip()

            if not text:
                continue

            total_lines += 1

            words = text.split()
            tokens = tokenizer.encode(
                text,
                out_type=int,
            )

            total_words += len(words)
            total_tokens += len(tokens)
            total_characters += len(text)

            total_unk += sum(
                token_id == tokenizer.unk_id()
                for token_id in tokens
            )

    tokens_per_word = (
        total_tokens / total_words
        if total_words
        else 0
    )

    characters_per_token = (
        total_characters / total_tokens
        if total_tokens
        else 0
    )

    unk_rate = (
        total_unk / total_tokens * 100
        if total_tokens
        else 0
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Lines:                 {total_lines:,}")
    print(f"Whitespace words:      {total_words:,}")
    print(f"SentencePiece tokens:  {total_tokens:,}")
    print(f"Characters:            {total_characters:,}")

    print(
        f"\nTokens / word:         "
        f"{tokens_per_word:.4f}"
    )

    print(
        f"Characters / token:    "
        f"{characters_per_token:.4f}"
    )

    print(
        f"UNK tokens:            "
        f"{total_unk:,}"
    )

    print(
        f"UNK rate:              "
        f"{unk_rate:.6f}%"
    )

    print("\n" + "=" * 60)
    print("FULL CORPUS ESTIMATE")
    print("=" * 60)

    # Full corpus statistics from our previous audit.
    full_corpus_words = 61_805_534

    estimated_full_tokens = (
        full_corpus_words * tokens_per_word
    )

    print(
        f"Full corpus words:     "
        f"{full_corpus_words:,}"
    )

    print(
        f"Estimated tokens:      "
        f"{estimated_full_tokens:,.0f}"
    )

    print(
        f"Estimated millions:    "
        f"{estimated_full_tokens / 1_000_000:.2f}M"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
