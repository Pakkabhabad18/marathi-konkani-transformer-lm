"""
One-off probe of Wikipedia API batching (superseded).

As `debug_wikipedia_api.py`, but checking how the API paginates a multi-article
request and where continuation tokens appear. Diagnostic only.

Retained for provenance.
"""

import json
import requests


API_URL = "https://gom.wikipedia.org/w/api.php"


def main():

    params = {
        "action": "query",
        "generator": "allpages",
        "gaplimit": 100,
        "gapnamespace": 0,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "formatversion": 2,
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=60,
        headers={
            "User-Agent": "LMA-Phase1-Student-Project/1.0"
        },
    )

    response.raise_for_status()

    data = response.json()

    pages = (
        data
        .get("query", {})
        .get("pages", [])
    )

    print("=" * 60)
    print("WIKIPEDIA BATCH DEBUG")
    print("=" * 60)

    print(f"Pages returned: {len(pages)}")

    print("\nTop-level keys:")
    print(list(data.keys()))

    print("\nContinue information:")
    print(
        json.dumps(
            data.get("continue"),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nFirst 5 pages:")
    print("-" * 60)

    for page in pages[:5]:

        print("\nPAGE:")
        print("ID:", page.get("pageid"))
        print("Title:", page.get("title"))

        revisions = page.get(
            "revisions",
            [],
        )

        print(
            "Number of revisions:",
            len(revisions),
        )

        if revisions:

            print(
                "Revision keys:",
                list(revisions[0].keys()),
            )

            print(
                "Revision:",
                json.dumps(
                    revisions[0],
                    indent=2,
                    ensure_ascii=False,
                )[:3000],
            )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

