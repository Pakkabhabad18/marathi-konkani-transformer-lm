"""
Preliminary tokenizer statistics over the Wikipedia sample (superseded).

Applied the mixed preliminary tokenizer to the raw Wikipedia sample. Same
train-on/measure-on flaw as `tokenizer_stats.py`.

Superseded by `tools/build_tokenizer.py`. Retained for provenance.
"""

import sentencepiece as spm


MODEL_FILE = (
    "konkani/tokenizer/"
    "preliminary_konkani_mixed_bpe.model"
)

INPUT_FILE = (
    "konkani/data/manual/"
    "konkani_wikipedia_sample.txt"
)


def main():

    print("=" * 60)
    print("KONKANI WIKIPEDIA TOKENIZER STATISTICS")
    print("=" * 60)

    print(f"Tokenizer: {MODEL_FILE}")

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

    print(
        f"Pages/lines:          "
        f"{total_lines:,}"
    )

    print(
        f"Whitespace words:     "
        f"{total_words:,}"
    )

    print(
        f"SentencePiece tokens: "
        f"{total_tokens:,}"
    )

    print(
        f"Characters:           "
        f"{total_characters:,}"
    )

    print(
        f"\nTokens / word:        "
        f"{tokens_per_word:.4f}"
    )

    print(
        f"Characters / token:   "
        f"{characters_per_token:.4f}"
    )

    print(
        f"UNK tokens:           "
        f"{total_unk:,}"
    )

    print(
        f"UNK rate:             "
        f"{unk_rate:.6f}%"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
