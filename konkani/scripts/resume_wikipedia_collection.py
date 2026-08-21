"""
Resumable variant of the Wikipedia API collector (superseded).

Written because `collect_wikipedia_sample.py` could not resume after an
interruption. It re-reads the existing output file to work out where to
continue — a workaround for the absence of a real checkpoint.

The lesson generalised into `common/checkpoint.py`, which stores job position
atomically, and every current collector uses it. Superseded by
`konkani/scripts/ingest_wikipedia_manual.py`. Retained for provenance.
"""

import csv
import re
import time
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
BATCH_SIZE = 50

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

    if continuation:
        params.update(continuation)

    while True:

        try:

            response = requests.get(
                API_URL,
                params=params,
                timeout=90,
                headers={
                    "User-Agent":
                        "LMA-Phase1-Student-Project/1.0"
                },
            )

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:
                    wait_time = int(
                        retry_after
                    )
                else:
                    wait_time = 30

                print(
                    "\nRate limited (HTTP 429)."
                )

                print(
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            response.raise_for_status()

            return response.json()

        except requests.exceptions.Timeout:

            print(
                "\nRequest timed out."
            )

            print(
                "Waiting 10 seconds and retrying..."
            )

            time.sleep(10)


def main():

    print("=" * 60)
    print("RESUMING KONKANI WIKIPEDIA COLLECTION")
    print("=" * 60)

    existing_pages = 0

    if TEXT_FILE.exists():

        with open(
            TEXT_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            existing_pages = sum(
                1 for line in file
                if line.strip()
            )

    print(
        f"Existing collected pages: "
        f"{existing_pages:,}"
    )

    print(
        f"Target pages: {TARGET_PAGES:,}"
    )

    if existing_pages >= TARGET_PAGES:

        print(
            "\nTarget already reached."
        )

        return

    print(
        "\nIMPORTANT:"
    )

    print(
        "The current collector does not "
        "persist the API continuation token."
    )

    print(
        "Therefore we cannot safely resume "
        "from exactly page 759."
    )

    print(
        "\nWe will restart the API traversal "
        "but preserve the existing data."
    )

    print(
        "Already-collected page IDs will be skipped."
    )

    existing_ids = set()

    if METADATA_FILE.exists():

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                page_id = row.get(
                    "page_id"
                )

                if page_id:
                    existing_ids.add(
                        page_id
                    )

    print(
        f"Existing page IDs loaded: "
        f"{len(existing_ids):,}"
    )

    collected = existing_pages
    examined = 0

    continuation = None

    with open(
        TEXT_FILE,
        "a",
        encoding="utf-8",
    ) as text_file, open(
        METADATA_FILE,
        "a",
        encoding="utf-8",
        newline="",
    ) as metadata_file:

        writer = csv.writer(
            metadata_file
        )

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

                page_id = str(
                    page.get(
                        "pageid",
                        ""
                    )
                )

                if page_id in existing_ids:
                    continue

                revisions = page.get(
                    "revisions",
                    [],
                )

                if not revisions:
                    continue

                content = (
                    revisions[0]
                    .get("slots", {})
                    .get("main", {})
                    .get("content", "")
                )

                if not content.strip():
                    continue

                if re.match(
                    r"^\s*#REDIRECT\b",
                    content,
                    flags=re.IGNORECASE,
                ):
                    continue

                cleaned = clean_text(
                    content
                )

                if not cleaned:
                    continue

                dev_pct, latin_pct = (
                    script_percentages(
                        cleaned
                    )
                )

                if (
                    dev_pct >= 70
                    and latin_pct < 20
                ):

                    script = "Devanagari"

                elif (
                    latin_pct >= 70
                    and dev_pct < 20
                ):

                    script = "Roman"

                else:

                    script = "Mixed"

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
                    page_id,
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

                text_file.flush()
                metadata_file.flush()

                existing_ids.add(
                    page_id
                )

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
                    "\nNo continuation token."
                )

                break

            # Be polite to Wikimedia.
            time.sleep(2)

    print("\n" + "=" * 60)
    print("COLLECTION STOPPED / COMPLETE")
    print("=" * 60)

    print(
        f"Previously collected: "
        f"{existing_pages:,}"
    )

    print(
        f"Currently collected:  "
        f"{collected:,}"
    )

    print(
        f"New pages added:       "
        f"{collected - existing_pages:,}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()

