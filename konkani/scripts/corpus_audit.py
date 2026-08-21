"""
Early plain-text corpus audit (superseded).

Reported basic statistics — document counts, word counts, Devanagari and Latin
character shares — for a flat text file. Its own docstring noted that
"tokenizer-level statistics will be added later"; they were, elsewhere.

Superseded by `tools/corpus_stats.py`, which reads the manifests rather than a
flat file, so it can separate manual, downloaded and synthetic text, and by
`tools/pipeline_accounting.py`, which reconciles every stage. Retained for
provenance.
"""

import argparse
import re
from pathlib import Path


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def read_text(file_path: Path) -> str:
    """Read a UTF-8 text file and return its contents."""
    return file_path.read_text(encoding="utf-8")


def audit_corpus(text: str) -> dict:
    """Calculate basic corpus statistics."""
    lines = text.splitlines()

    words = text.split()

    devanagari_chars = len(DEVANAGARI_RE.findall(text))
    latin_chars = len(LATIN_RE.findall(text))

    total_non_whitespace = sum(
        1 for char in text if not char.isspace()
    )

    if total_non_whitespace > 0:
        devanagari_percentage = (
            devanagari_chars / total_non_whitespace
        ) * 100
    else:
        devanagari_percentage = 0.0

    return {
        "documents_or_lines": len(lines),
        "characters": len(text),
        "non_whitespace_characters": total_non_whitespace,
        "whitespace_separated_words": len(words),
        "devanagari_characters": devanagari_chars,
        "latin_characters": latin_chars,
        "devanagari_percentage": devanagari_percentage,
    }


def print_statistics(file_path: Path, stats: dict) -> None:
    """Print corpus statistics in a readable format."""
    print("\n" + "=" * 50)
    print("CORPUS AUDIT")
    print("=" * 50)

    print(f"File: {file_path}")
    print(f"Lines/Documents: {stats['documents_or_lines']:,}")
    print(f"Characters: {stats['characters']:,}")
    print(
        f"Non-whitespace characters: "
        f"{stats['non_whitespace_characters']:,}"
    )
    print(
        f"Whitespace-separated words: "
        f"{stats['whitespace_separated_words']:,}"
    )
    print(
        f"Devanagari characters: "
        f"{stats['devanagari_characters']:,}"
    )
    print(
        f"Latin characters: "
        f"{stats['latin_characters']:,}"
    )
    print(
        f"Devanagari percentage: "
        f"{stats['devanagari_percentage']:.2f}%"
    )

    print("=" * 50)


def main():
    """Run the corpus audit from the command line."""
    parser = argparse.ArgumentParser(
        description="Audit a UTF-8 text corpus."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to a UTF-8 text file.",
    )

    args = parser.parse_args()

    if not args.file.exists():
        raise FileNotFoundError(
            f"File not found: {args.file}"
        )

    if not args.file.is_file():
        raise ValueError(
            f"Expected a file, but received: {args.file}"
        )

    text = read_text(args.file)

    stats = audit_corpus(text)

    print_statistics(args.file, stats)


if __name__ == "__main__":
    main()
