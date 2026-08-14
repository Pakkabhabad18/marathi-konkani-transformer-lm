import csv
import re
from pathlib import Path


INPUT_TEXT = Path(
    "konkani/data/manual/"
    "konkani_wikipedia_sample.txt"
)

INPUT_METADATA = Path(
    "konkani/data/manual/"
    "konkani_wikipedia_sample_metadata.csv"
)

OUTPUT_TEXT = Path(
    "konkani/data/processed/"
    "konkani_wikipedia_filtered.txt"
)

OUTPUT_METADATA = Path(
    "konkani/data/processed/"
    "konkani_wikipedia_filtered_metadata.csv"
)

MIN_WORDS = 50
MIN_CHARACTERS = 100


def main():

    print("=" * 60)
    print("KONKANI WIKIPEDIA QUALITY FILTER")
    print("=" * 60)

    OUTPUT_TEXT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        INPUT_TEXT,
        "r",
        encoding="utf-8",
    ) as text_file:

        texts = [
            line.strip()
            for line in text_file
        ]

    with open(
        INPUT_METADATA,
        "r",
        encoding="utf-8",
    ) as metadata_file:

        rows = list(
            csv.DictReader(
                metadata_file
            )
        )

    if len(texts) != len(rows):

        raise ValueError(
            f"Text/metadata mismatch: "
            f"{len(texts)} text lines vs "
            f"{len(rows)} metadata rows"
        )

    kept = []
    removed_short = []

    for text, row in zip(
        texts,
        rows,
    ):

        words = len(text.split())
        characters = len(text)

        if words < MIN_WORDS:

            removed_short.append(
                (text, row)
            )

            continue

        if characters < MIN_CHARACTERS:

            removed_short.append(
                (text, row)
            )

            continue

        kept.append(
            (text, row)
        )

    with open(
        OUTPUT_TEXT,
        "w",
        encoding="utf-8",
    ) as text_file:

        for text, _ in kept:

            text_file.write(
                text + "\n"
            )

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
        newline="",
    ) as metadata_file:

        if rows:

            fieldnames = rows[0].keys()

            writer = csv.DictWriter(
                metadata_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for _, row in kept:

                writer.writerow(row)

    total_words = sum(
        len(text.split())
        for text, _ in kept
    )

    total_characters = sum(
        len(text)
        for text, _ in kept
    )

    print("\n" + "-" * 60)
    print("FILTER RESULTS")
    print("-" * 60)

    print(
        f"Original pages:     "
        f"{len(rows):,}"
    )

    print(
        f"Pages kept:         "
        f"{len(kept):,}"
    )

    print(
        f"Pages removed:      "
        f"{len(removed_short):,}"
    )

    print(
        f"Retention rate:     "
        f"{len(kept) / len(rows) * 100:.2f}%"
    )

    print(
        f"\nFiltered words:     "
        f"{total_words:,}"
    )

    print(
        f"Filtered characters:"
        f" {total_characters:,}"
    )

    print("\nOutput:")
    print(OUTPUT_TEXT)
    print(OUTPUT_METADATA)

    print("=" * 60)


if __name__ == "__main__":
    main()
