import re
from datasets import load_dataset


DATASET_NAME = "omdeep22/Konkani_books_corpus-v2"

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def devanagari_percentage(text):
    """Calculate the percentage of non-whitespace characters
    belonging to the Devanagari Unicode block.
    """
    non_whitespace = [
        char for char in text
        if not char.isspace()
    ]

    if not non_whitespace:
        return 0.0

    devanagari = sum(
        bool(DEVANAGARI_RE.fullmatch(char))
        for char in non_whitespace
    )

    return 100 * devanagari / len(non_whitespace)


def is_metadata(text):
    """Identify obvious source-marker metadata records."""
    return text.strip().startswith("--- SOURCE:")


def main():
    """Stream the dataset and inspect real text records."""

    print("=" * 60)
    print("KONKANI BOOKS CORPUS - REAL TEXT INSPECTION")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    found = 0
    scanned = 0

    for example in dataset:
        scanned += 1

        text = example.get("text", "")

        if not text.strip():
            continue

        if is_metadata(text):
            continue

        found += 1

        print("\n" + "-" * 60)
        print(f"Record #{found}")
        print(f"Characters: {len(text):,}")
        print(
            f"Devanagari: "
            f"{devanagari_percentage(text):.2f}%"
        )

        print("\nText preview:")
        print(text[:1000])

        if found >= 10:
            break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Records scanned: {scanned:,}")
    print(f"Real text records shown: {found:,}")


if __name__ == "__main__":
    main()
