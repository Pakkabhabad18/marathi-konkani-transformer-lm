"""
Sample extractor for the preliminary tokenizer experiments (superseded).

Pulled 500,000 records from `omdeep22/Konkani_books_corpus-v2` into a flat text
file, skipping the dataset's `--- SOURCE:` metadata lines, to give the early
tokenizer experiments something to train on before a real corpus existed.

Superseded by `konkani/scripts/ingest_books_corpus.py`, which ingests the same
dataset through the full quality pipeline and writes provenance manifests.
Retained for provenance.
"""

from datasets import load_dataset


DATASET_NAME = "omdeep22/Konkani_books_corpus-v2"

OUTPUT_FILE = "konkani/data/raw/tokenizer_sample.txt"

TARGET_RECORDS = 500_000


def is_metadata(text):
    return text.strip().startswith("--- SOURCE:")


def main():
    print("=" * 60)
    print("CREATING KONKANI TOKENIZER SAMPLE")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    collected = 0
    scanned = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as output:

        for example in dataset:

            scanned += 1

            text = example.get("text", "").strip()

            if not text:
                continue

            if is_metadata(text):
                continue

            output.write(text + "\n")

            collected += 1

            if collected % 50_000 == 0:
                print(
                    f"Collected: "
                    f"{collected:,}/{TARGET_RECORDS:,}"
                )

            if collected >= TARGET_RECORDS:
                break

    print("\n" + "=" * 60)
    print("SAMPLE CREATED")
    print("=" * 60)

    print(f"Records collected: {collected:,}")
    print(f"Records scanned:   {scanned:,}")
    print(f"Output file:       {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
