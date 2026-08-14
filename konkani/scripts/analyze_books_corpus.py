from datasets import load_dataset
import re
import time


DATASET_NAME = "omdeep22/Konkani_books_corpus-v2"

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def devanagari_count(text):
    """Count Devanagari characters in text."""
    return sum(
        bool(DEVANAGARI_RE.fullmatch(char))
        for char in text
    )


def is_metadata(text):
    """Check whether a record is a source-marker record."""
    return text.strip().startswith("--- SOURCE:")


def main():
    """Analyze the complete streamed Konkani Books Corpus."""

    print("=" * 70)
    print("KONKANI BOOKS CORPUS - FULL STATISTICS")
    print("=" * 70)

    print("\nLoading dataset in streaming mode...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    total_records = 0
    empty_records = 0
    metadata_records = 0
    usable_records = 0

    total_characters = 0
    total_words = 0
    total_devanagari = 0

    start_time = time.time()

    for example in dataset:

        total_records += 1

        text = example.get("text", "")

        if not text.strip():
            empty_records += 1
            continue

        if is_metadata(text):
            metadata_records += 1
            continue

        usable_records += 1

        total_characters += len(text)
        total_words += len(text.split())
        total_devanagari += devanagari_count(text)

        if total_records % 100000 == 0:
            elapsed = time.time() - start_time

            print(
                f"Processed: {total_records:,} records | "
                f"Usable: {usable_records:,} | "
                f"Time: {elapsed:.1f}s"
            )

    elapsed = time.time() - start_time

    non_whitespace = total_characters

    if non_whitespace > 0:
        devanagari_percentage = (
            total_devanagari / non_whitespace
        ) * 100
    else:
        devanagari_percentage = 0.0

    average_chars = (
        total_characters / usable_records
        if usable_records
        else 0
    )

    average_words = (
        total_words / usable_records
        if usable_records
        else 0
    )

    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    print(f"Total records:              {total_records:,}")
    print(f"Empty records:              {empty_records:,}")
    print(f"Metadata records:           {metadata_records:,}")
    print(f"Usable records:             {usable_records:,}")

    print(f"\nTotal characters:           {total_characters:,}")
    print(f"Total words:                {total_words:,}")
    print(f"Devanagari characters:      {total_devanagari:,}")

    print(
        f"Devanagari percentage:      "
        f"{devanagari_percentage:.2f}%"
    )

    print(
        f"\nAverage characters/record:  "
        f"{average_chars:.2f}"
    )

    print(
        f"Average words/record:       "
        f"{average_words:.2f}"
    )

    print(f"\nProcessing time:            {elapsed:.2f} seconds")

    print("=" * 70)


if __name__ == "__main__":
    main()
