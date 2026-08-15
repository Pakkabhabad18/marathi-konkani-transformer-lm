#!/usr/bin/env python3
"""
Diagnose why a source is failing, before changing the collector.

WHY THIS EXISTS
---------------
The M1 collector hit repeated HTTP 500s from the archive.org scrape API, while
the identical query succeeded when issued from a different machine. That means
the failure is specific to *our* request, not to the endpoint. Guessing which
part of the request is at fault wastes time; this script tests the variables one
at a time and prints the raw evidence.

It changes nothing and writes nothing. Run it, read it, then decide.

USAGE
-----
    python3 tools/diagnose_source.py
"""

from __future__ import annotations

import json
import sys
import time

import requests

SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
ADVANCED_URL = "https://archive.org/advancedsearch.php"
QUERY = "identifier:in.gov.maharashtra.gr.*"
SAMPLE_ITEM = "in.gov.maharashtra.gr.202607071620477816"

# The User-Agent currently used by the collector, plus alternatives to isolate
# whether the UA string is what triggers the 500.
USER_AGENTS = {
    "collector_current": "LMA-Phase1-Student-Project/1.0 (academic corpus construction)",
    "simple_token": "LMA-Phase1-Student-Project/1.0",
    "contactable": "lma-phase1-research/1.0 (student project; contact: pakkabhabad@gmail.com)",
    "browser_like": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "python_default": None,   # let requests send its own
}

SEP = "-" * 68


def show(label: str, response: requests.Response | None, error: str = "") -> bool:
    """Print one probe result. Returns True if it succeeded."""
    if error:
        print(f"  {label:22s} EXCEPTION  {error}")
        return False

    ok = response.status_code == 200
    print(f"  {label:22s} HTTP {response.status_code}  "
          f"{len(response.content):>8,} bytes  "
          f"server={response.headers.get('Server', '?')}")

    if not ok:
        body = response.text[:400].replace("\n", " ").strip()
        print(f"  {'':22s} body: {body or '(empty)'}")
        for header in ("Retry-After", "X-Rate-Limit", "Connection", "Content-Type"):
            if header in response.headers:
                print(f"  {'':22s} {header}: {response.headers[header]}")
    return ok


def probe_user_agents() -> dict[str, bool]:
    """Test the scrape API with each candidate User-Agent."""
    print(SEP)
    print("PROBE 1 - scrape API, varying only the User-Agent")
    print(SEP)

    results = {}
    for name, ua in USER_AGENTS.items():
        headers = {"User-Agent": ua} if ua else {}
        try:
            r = requests.get(
                SCRAPE_URL,
                params={"q": QUERY, "fields": "identifier", "count": 100},
                headers=headers,
                timeout=90,
            )
            results[name] = show(name, r)
        except requests.RequestException as exc:
            results[name] = show(name, None, f"{type(exc).__name__}: {exc}")
        time.sleep(2)
    return results


def probe_page_sizes(ua: str | None) -> None:
    """Test whether the requested page size matters."""
    print()
    print(SEP)
    print("PROBE 2 - scrape API, varying page size")
    print(SEP)

    headers = {"User-Agent": ua} if ua else {}
    for count in (100, 200, 500, 1000):
        try:
            r = requests.get(
                SCRAPE_URL,
                params={"q": QUERY, "fields": "identifier", "count": count},
                headers=headers,
                timeout=90,
            )
            if show(f"count={count}", r) and r.status_code == 200:
                data = r.json()
                print(f"  {'':22s} total={data.get('total'):,} "
                      f"items={len(data.get('items', []))} "
                      f"cursor={'yes' if data.get('cursor') else 'no'}")
        except requests.RequestException as exc:
            show(f"count={count}", None, f"{type(exc).__name__}: {exc}")
        time.sleep(2)


def probe_fallback(ua: str | None) -> None:
    """Test the advancedsearch endpoint as an alternative enumerator."""
    print()
    print(SEP)
    print("PROBE 3 - advancedsearch.php fallback")
    print(SEP)

    headers = {"User-Agent": ua} if ua else {}
    try:
        r = requests.get(
            ADVANCED_URL,
            params={"q": QUERY, "fl[]": "identifier", "rows": 100,
                    "page": 1, "output": "json"},
            headers=headers,
            timeout=90,
        )
        if show("advancedsearch", r) and r.status_code == 200:
            data = r.json()
            resp = data.get("response", {})
            print(f"  {'':22s} numFound={resp.get('numFound'):,} "
                  f"docs={len(resp.get('docs', []))}")
    except requests.RequestException as exc:
        show("advancedsearch", None, f"{type(exc).__name__}: {exc}")


def probe_download(ua: str | None) -> bool:
    """Resolve and fetch one document's OCR text. Returns True on success.

    Deliberately does BOTH the naive guess and the metadata lookup, because the
    difference between them is the bug this probe exists to catch: an item's
    text file is not reliably named after the item identifier.
    """
    print()
    print(SEP)
    print("PROBE 4 - document download (this is what the bulk of the job does)")
    print(SEP)

    headers = {"User-Agent": ua} if ua else {}

    # 4a. The naive construction: <identifier>_djvu.txt
    naive_url = f"https://archive.org/download/{SAMPLE_ITEM}/{SAMPLE_ITEM}_djvu.txt"
    try:
        r = requests.get(naive_url, headers=headers, timeout=90)
        naive_ok = r.status_code == 200
        print(f"  {'guessed name':22s} HTTP {r.status_code}"
              f"{'  <- constructing the filename does NOT work' if not naive_ok else ''}")
    except requests.RequestException as exc:
        naive_ok = False
        print(f"  {'guessed name':22s} EXCEPTION {exc}")

    # 4b. The correct route: ask the metadata endpoint what the file is called.
    try:
        m = requests.get(f"https://archive.org/metadata/{SAMPLE_ITEM}",
                         headers=headers, timeout=90)
        if m.status_code != 200:
            show("metadata", m)
            return False

        files = m.json().get("files", []) or []
        name = next((f.get("name") for f in files if f.get("format") == "DjVuTXT"), None)
        if not name:
            name = next((f.get("name") for f in files
                         if str(f.get("name", "")).endswith("_djvu.txt")), None)

        if not name:
            print(f"  {'metadata lookup':22s} no DjVuTXT file in this item")
            return False

        print(f"  {'metadata lookup':22s} text file is '{name}'")

        r = requests.get(f"https://archive.org/download/{SAMPLE_ITEM}/{name}",
                         headers=headers, timeout=90)
        if not show("resolved name", r) or r.status_code != 200:
            return False

        r.encoding = "utf-8"
        text = r.text
        non_ws = sum(1 for c in text if not c.isspace())
        deva = sum(1 for c in text if "ऀ" <= c <= "ॿ")
        print(f"  {'':22s} final URL: {r.url}")
        print(f"  {'':22s} chars={len(text):,} words={len(text.split()):,} "
              f"devanagari={deva / max(non_ws, 1):.1%} of non-whitespace")
        print(f"  {'':22s} preview: {' '.join(text.split())[:200]}")
        return True

    except requests.RequestException as exc:
        show("metadata", None, f"{type(exc).__name__}: {exc}")
        return False


def main() -> int:
    print("=" * 68)
    print("SOURCE DIAGNOSTIC - archive.org")
    print(f"requests {requests.__version__} | python {sys.version.split()[0]}")
    print("=" * 68)
    print()

    ua_results = probe_user_agents()
    working = [name for name, ok in ua_results.items() if ok]

    best_ua = USER_AGENTS[working[0]] if working else USER_AGENTS["contactable"]
    probe_page_sizes(best_ua)
    probe_fallback(best_ua)
    download_ok = probe_download(best_ua)

    print()
    print("=" * 68)
    print("VERDICT")
    print("=" * 68)

    # Enumeration and download are separate failure modes. An earlier version of
    # this script reported "everything succeeded" while PROBE 4 was returning
    # 404, because the verdict only looked at the User-Agent results. Listing
    # items is useless if the documents themselves cannot be fetched, so the
    # download result is reported first and gates the overall conclusion.
    if not download_ok:
        print("  DOCUMENT DOWNLOAD FAILED - this blocks collection regardless of")
        print("  whether the listing API works.")
        print("  -> If 'guessed name' 404s but 'resolved name' succeeds, the")
        print("     collector must look the filename up via /metadata/<id>.")
        print("  -> If both fail, the item may have no OCR text layer; try")
        print("     another identifier before concluding the source is unusable.")
        print()

    if not working:
        print("  Every User-Agent failed.")
        print("  -> The problem is NOT the User-Agent. Likely causes, in order:")
        print("     1. Your network or ISP is blocking/proxying archive.org.")
        print("     2. archive.org is having an outage (check status).")
        print("     3. A local proxy or VPN is intercepting HTTPS.")
        print("  -> Test from the shell:  curl -sS -o /dev/null -w '%{http_code}\\n' \\")
        print("       'https://archive.org/services/search/v1/scrape?q=identifier%3Atest&count=100'")
    elif len(working) == len(ua_results):
        print("  Every User-Agent succeeded.")
        print("  -> The earlier 500s were transient. archive.org sheds load this way.")
        print("  -> Fix is patience, not code: keep the backoff, raise MAX_RETRIES,")
        print("     and re-run. The checkpoint means nothing is lost.")
    else:
        print(f"  Some User-Agents work, some do not. Working: {', '.join(working)}")
        print(f"  -> Set USER_AGENT in the collector to the '{working[0]}' value.")

    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
