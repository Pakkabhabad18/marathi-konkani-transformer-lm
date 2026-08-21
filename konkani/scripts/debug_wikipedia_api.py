"""
One-off probe of the Konkani Wikipedia API (superseded).

Issued a single `allpages` query against `gom.wikipedia.org` and printed the raw
JSON, to work out the response shape before writing the collector. Diagnostic
only — it collects nothing.

Retained for provenance.
"""

import json
import requests


API_URL = "https://gom.wikipedia.org/w/api.php"


def main():
    params = {
        "action": "query",
        "generator": "allpages",
        "gaplimit": 1,
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

    print("HTTP status:", response.status_code)
    print("Content-Type:", response.headers.get("content-type"))

    response.raise_for_status()

    data = response.json()

    print("\n" + "=" * 60)
    print("RAW API STRUCTURE")
    print("=" * 60)

    print(json.dumps(data, indent=2, ensure_ascii=False)[:12000])


if __name__ == "__main__":
    main()
