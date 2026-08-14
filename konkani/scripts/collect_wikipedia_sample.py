import csv
import re
from pathlib import Path

import requests


API_URL = "https://gom.wikipedia.org/w/api.php"

TEXT_FILE = Path(
    "konkani/data/manual/konkani_wikipedia_sample.txt"
)

METADATA_FILE = Path(
    "konkani/data/manual/konkani_wikipedia_sample_metadata.csv"
)

TARGET_PAGES = 5000
BATCH_SIZE = 100

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def script_percentages(text):
    chars = [
        c for c in text
        if not c.isspace()
    ]

    if not chars:
        return 0.0, 0.0

    devanagari = sum(
        bool(DEVANAGARI_RE.fullmatch(c))
        for c in chars
    )

    latin = sum(
        bool(LATIN_RE.fullmatch(c))
        for c in chars
    )

    return (
        devanagari * 100 / len(chars),
        latin * 100 / len(chars),
    )


def clean_text(text):

    text = re.sub(
        r"<!--.*?-->",
        " ",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"<ref[^>]*>.*?</ref>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\{\{[^{}]*\}\}",
        " ",
        text,
    )

    text = re.sub(
        r"\[\[(?:File|Image):.*?\]\]",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"\[\[[^|\]]+\|([^\]]+)\]\]",
        r"\1",
        text,
    )

    text = re.sub(
        r"\[\[([^\]]+)\]\]",
        r"\1",
        text,
    )

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = text.replace("'''", "")
    text = text.replace("''", "")

    text = re.sub(
        r"={2,6}\s*(.*?)\s*={2,6}",
        r"\1",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def get_pages(continuation=None):

    params = {
        "action": "query",
        "generator": "allpages",
        "gaplimit": BATCH_SIZE,
        "gapnamespace": 0,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "formatversion": 2,
    }

    # Wikimedia may return several continuation
    # parameters, e.g. rvcontinue.
    if continuation:
        params.update(continuation)

    response = requests.get(
        API_URL,
        params=params,
        timeout=60,
        headers={
            "User-Agent":
                "LMA-Phase1-Student-Project/1.0"
        },
    )

    response.raise_for_status()

    return response.json()


def main():

    print("=" * 60)
    print("KONKANI WIKIPEDIA SELF-COLLECTION")
    print("=" * 60)

    TEXT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    collected = 0
    examined = 0
    redirects = 0
    empty = 0

    devanagari_pages = 0
    roman_pages = 0
    mixed_pages = 0

    continuation = None

    with open(
        TEXT_FILE,
        "w",
        encoding="utf-8",
    ) as text_file, open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as metadata_file:

        writer = csv.writer(metadata_file)

        writer.writerow([
            "page_id",
            "title",
            "url",
            "raw_characters",
            "clean_characters",
            "words",
            "devanagari_percentage",
            "latin_percentage",
            "script",
            "collection_type",
        ])

        while collected < TARGET_PAGES:

            data = get_pages(
                continuation
            )

            pages = (
                data
                .get("query", {})
                .get("pages", [])
            )

            print(
                f"\nReceived {len(pages)} pages"
            )

            for page in pages:

                examined += 1

                revisions = page.get(
                    "revisions",
                    [],
                )

                if not revisions:
                    empty += 1
                    continue

                content = (
                    revisions[0]
                    .get("slots", {})
                    .get("main", {})
                    .get("content", "")
                )

                if not content.strip():
                    empty += 1
                    continue

                if re.match(
                    r"^\s*#REDIRECT\b",
                    content,
                    flags=re.IGNORECASE,
                ):
                    redirects += 1
                    continue

                cleaned = clean_text(
                    content
                )

                if not cleaned:
                    empty += 1
                    continue

                dev_pct, latin_pct = (
                    script_percentages(cleaned)
                )

                if (
                    dev_pct >= 70
                    and latin_pct < 20
                ):
                    script = "Devanagari"
                    devanagari_pages += 1

                elif (
                    latin_pct >= 70
                    and dev_pct < 20
                ):
                    script = "Roman"
                    roman_pages += 1

                else:
                    script = "Mixed"
                    mixed_pages += 1

                title = page.get(
                    "title",
                    "",
                )

                url = (
                    "https://gom.wikipedia.org/wiki/"
                    + title.replace(" ", "_")
                )

                text_file.write(
                    cleaned + "\n"
                )

                writer.writerow([
                    page.get(
                        "pageid",
                        "",
                    ),
                    title,
                    url,
                    len(content),
                    len(cleaned),
                    len(cleaned.split()),
                    f"{dev_pct:.2f}",
                    f"{latin_pct:.2f}",
                    script,
                    "self-collected",
                ])

                collected += 1

                print(
                    f"Collected "
                    f"{collected}/{TARGET_PAGES}: "
                    f"{title} "
                    f"[{script}]"
                )

                if collected >= TARGET_PAGES:
                    break

            if collected >= TARGET_PAGES:
                break

            # IMPORTANT:
            # Keep all continuation parameters
            # except the generic "continue" key.
            raw_continuation = data.get(
                "continue",
                {},
            )

            continuation = {
                key: value
                for key, value
                in raw_continuation.items()
                if key != "continue"
            }

            if not continuation:
                print(
                    "No continuation token. "
                    "Collection finished."
                )
                break

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    print(
        f"Pages examined:       {examined:,}"
    )

    print(
        f"Pages collected:      {collected:,}"
    )

    print(
        f"Redirects skipped:     {redirects:,}"
    )

    print(
        f"Empty pages skipped:   {empty:,}"
    )

    print(
        f"Devanagari pages:      "
        f"{devanagari_pages:,}"
    )

    print(
        f"Roman pages:           "
        f"{roman_pages:,}"
    )

    print(
        f"Mixed-script pages:    "
        f"{mixed_pages:,}"
    )

    print(
        f"\nText file: {TEXT_FILE}"
    )

    print(
        f"Metadata:  {METADATA_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
