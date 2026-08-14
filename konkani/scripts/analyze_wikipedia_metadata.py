import csv
from collections import Counter


INPUT_FILE = (
    "konkani/data/manual/"
    "konkani_wikipedia_sample_metadata.csv"
)


def main():

    print("=" * 60)
    print("KONKANI WIKIPEDIA METADATA ANALYSIS")
    print("=" * 60)

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    print(
        f"\nTotal metadata records: "
        f"{len(rows):,}"
    )

    # --------------------------------------------------
    # Script distribution
    # --------------------------------------------------

    script_counts = Counter(
        row["script"]
        for row in rows
    )

    print("\n" + "-" * 60)
    print("SCRIPT DISTRIBUTION")
    print("-" * 60)

    for script, count in script_counts.most_common():

        percentage = (
            count / len(rows) * 100
        )

        print(
            f"{script:15s} "
            f"{count:6,d} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------
    # Word statistics
    # --------------------------------------------------

    total_words = sum(
        int(row["words"])
        for row in rows
    )

    total_chars = sum(
        int(row["clean_characters"])
        for row in rows
    )

    print("\n" + "-" * 60)
    print("CORPUS TOTALS")
    print("-" * 60)

    print(
        f"Clean characters: "
        f"{total_chars:,}"
    )

    print(
        f"Words:            "
        f"{total_words:,}"
    )

    print(
        f"Average words/page: "
        f"{total_words / len(rows):.2f}"
    )

    # --------------------------------------------------
    # Script percentages
    # --------------------------------------------------

    avg_dev = sum(
        float(row["devanagari_percentage"])
        for row in rows
    ) / len(rows)

    avg_latin = sum(
        float(row["latin_percentage"])
        for row in rows
    ) / len(rows)

    print("\n" + "-" * 60)
    print("AVERAGE PAGE SCRIPT PERCENTAGES")
    print("-" * 60)

    print(
        f"Average Devanagari: "
        f"{avg_dev:.2f}%"
    )

    print(
        f"Average Latin:      "
        f"{avg_latin:.2f}%"
    )

    # --------------------------------------------------
    # Largest pages
    # --------------------------------------------------

    largest = sorted(
        rows,
        key=lambda row: int(
            row["clean_characters"]
        ),
        reverse=True,
    )

    print("\n" + "-" * 60)
    print("10 LARGEST PAGES")
    print("-" * 60)

    for row in largest[:10]:

        print(
            f"{int(row['clean_characters']):8,d} chars | "
            f"{int(row['words']):7,d} words | "
            f"{row['script']:10s} | "
            f"{row['title']}"
        )

    # --------------------------------------------------
    # Smallest pages
    # --------------------------------------------------

    smallest = sorted(
        rows,
        key=lambda row: int(
            row["clean_characters"]
        ),
    )

    print("\n" + "-" * 60)
    print("10 SMALLEST PAGES")
    print("-" * 60)

    for row in smallest[:10]:

        print(
            f"{int(row['clean_characters']):8,d} chars | "
            f"{int(row['words']):7,d} words | "
            f"{row['script']:10s} | "
            f"{row['title']}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
