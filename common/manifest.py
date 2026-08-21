"""
Per-document provenance manifest and manual/downloaded token accounting.

WHY THIS EXISTS
---------------
The binding constraint of Phase 1 is not the token target, it is the ratio:

    manual_tokens / total_tokens >= 0.20     =>     total <= 5 * manual

That means every token in the corpus has to be attributable to a source, and
every source has to be classified as manual or downloaded, at the moment it is
collected. Reconstructing this afterwards from a pile of text files is not
possible, so the manifest is written as the data arrives.

The specification also requires reporting the manual vs downloaded split in the
dataset statistics, and requires evidence for the manual claim. A per-document
manifest row *is* that evidence.

WHAT COUNTS AS MANUAL
---------------------
Per the project brief: OCR from books/PDFs, scraping and cleaning pages we
gather ourselves, typed or transcribed text. A ready-made corpus does not become
manual because we downloaded and cleaned it. CollectionType makes this a typed
choice at the call site rather than a string that can drift.

THE `tokens` FIELD IS DELIBERATELY NULL AT COLLECTION TIME
----------------------------------------------------------
Token counts depend on the tokenizer. Phase 1 requires one final tokenizer per
language and one internally consistent count. Writing a token count here, using
whatever preliminary tokenizer existed on the day, is exactly how the earlier
"~91.91M tokens" figure ended up mixing two different tokenizers. So collection
records `words` and `chars` (tokenizer-independent), leaves `tokens` as null,
and a single later pass fills it in for the whole corpus at once.
"""

from __future__ import annotations

import json
import time as _time
import hashlib
import os
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional


class CollectionType(str, Enum):
    """Manual vs downloaded, as a typed choice rather than a free-text string."""

    MANUAL_OCR = "manual_ocr"
    MANUAL_SCRAPE = "manual_scrape"
    MANUAL_TRANSCRIBED = "manual_transcribed"
    DOWNLOADED_DATASET = "downloaded_dataset"
    # Machine-translated / synthetic. Permitted by the TAs on 18 Aug 2026 ONLY
    # as a last resort after real sources are exhausted, and only with written
    # justification. It is a separate member rather than a reuse of
    # DOWNLOADED_DATASET so that MT text can never be silently folded into the
    # downloaded figure: `is_manual` is False for it (it does not start with
    # "manual_"), so it cannot count toward the 20% floor, while the distinct
    # value keeps it separable in every per-source report.
    MACHINE_TRANSLATED = "machine_translated"

    @property
    def is_manual(self) -> bool:
        return self.value.startswith("manual_")

    @property
    def is_synthetic(self) -> bool:
        return self.value == "machine_translated"


def content_hash(text: str) -> str:
    """SHA-256 of the normalized text. Used for exact dedup and cross-corpus checks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class DocumentRecord:
    """One row of provenance. Field names match the schema fixed in the brief."""

    source_name: str
    source_url: str
    collection_method: str
    access_date: str
    raw_chars: int
    clean_chars: int
    words: int
    tokens: Optional[int]
    preprocessing_applied: list
    script: str
    langid_score: Optional[float]
    content_hash: str

    # Extra fields carried for auditing; not part of the required schema.
    doc_id: str = ""
    language: str = ""
    collection_type: str = ""
    is_manual: bool = False
    langid_label: str = ""
    devanagari_ratio: Optional[float] = None
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def make_record(
    *,
    text: str,
    raw_text: str,
    source_name: str,
    source_url: str,
    collection_type: CollectionType,
    language: str,
    preprocessing_applied: list,
    script: str,
    langid_score: Optional[float] = None,
    langid_label: str = "",
    devanagari_ratio: Optional[float] = None,
    doc_id: str = "",
    notes: str = "",
    access_date: Optional[str] = None,
) -> DocumentRecord:
    """Build a manifest row from a cleaned document and its raw original."""
    return DocumentRecord(
        source_name=source_name,
        source_url=source_url,
        collection_method=collection_type.value,
        access_date=access_date or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        raw_chars=len(raw_text),
        clean_chars=len(text),
        words=len(text.split()),
        tokens=None,
        preprocessing_applied=list(preprocessing_applied),
        script=script,
        langid_score=langid_score,
        content_hash=content_hash(text),
        doc_id=doc_id,
        language=language,
        collection_type=collection_type.value,
        is_manual=collection_type.is_manual,
        langid_label=langid_label,
        devanagari_ratio=devanagari_ratio,
        notes=notes,
    )


class ManifestWriter:
    """Append-only JSONL manifest, flushed per record so a kill loses nothing.

    JSONL rather than CSV because `preprocessing_applied` is a list and CSV would
    force it into a delimited string that has to be parsed back.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.written = 0

    def write(self, record: DocumentRecord) -> None:
        self._fh.write(record.to_json() + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.written += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_manifest(path: str | Path, attempts: int = 5) -> Iterator[dict]:
    """Stream manifest rows. Tolerates a truncated final line from a hard kill.

    RETRIES ON ETIMEDOUT (errno 60)
    -------------------------------
    This repository lives under an iCloud-synced Desktop. iCloud evicts file
    contents to free space and leaves a "dataless" placeholder behind; the
    first read of such a file blocks while macOS re-downloads it, and if that
    takes too long the read fails with OSError errno 60 (ETIMEDOUT).

    That is a TRANSIENT condition - the download continues in the background
    and the next attempt usually succeeds - but streaming line-by-line turns it
    into a crash halfway through a 149 MB manifest, which killed a full stats
    run. So the file is read whole, with retries and backoff, and only then
    parsed. Reading whole costs memory proportional to one manifest, which is
    bounded and small next to the corpus itself.

    A persistent failure still raises: silently skipping a manifest would make
    every downstream total quietly wrong, which is far worse than stopping.
    """
    p = Path(path)
    if not p.exists():
        return

    blob = None
    for attempt in range(1, attempts + 1):
        try:
            blob = p.read_text(encoding="utf-8", errors="replace")
            break
        except OSError as exc:
            if attempt == attempts:
                raise OSError(
                    f"{p}: unreadable after {attempts} attempts ({exc}). "
                    f"If this is an iCloud placeholder, force the download "
                    f"with:  brctl download '{p}'") from exc
            wait = 2 ** (attempt - 1)
            print(f"  [retry {attempt}/{attempts}] {p.name}: {exc.__class__.__name__} "
                  f"- waiting {wait}s for iCloud to materialise the file",
                  flush=True)
            _time.sleep(wait)

    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


@dataclass
class Accounting:
    """Aggregated manual vs downloaded totals for one language."""

    documents: int = 0
    manual_documents: int = 0
    downloaded_documents: int = 0
    words: int = 0
    manual_words: int = 0
    downloaded_words: int = 0
    chars: int = 0
    tokens: int = 0
    manual_tokens: int = 0
    downloaded_tokens: int = 0
    by_source: dict = field(default_factory=dict)

    @property
    def manual_word_fraction(self) -> float:
        return self.manual_words / self.words if self.words else 0.0

    @property
    def manual_token_fraction(self) -> float:
        return self.manual_tokens / self.tokens if self.tokens else 0.0

    def headroom(self) -> int:
        """How many more total words the corpus may hold at the current manual level.

        Because total <= 5 * manual, this returns 5*manual_words - words. A
        negative value means the corpus is already over-diluted and either more
        manual data is needed or downloaded data must be dropped.
        """
        return 5 * self.manual_words - self.words


def summarize(manifest_path: str | Path) -> Accounting:
    """Aggregate a manifest into manual/downloaded totals, overall and per source."""
    acc = Accounting()
    for row in read_manifest(manifest_path):
        is_manual = bool(row.get("is_manual"))
        words = int(row.get("words") or 0)
        chars = int(row.get("clean_chars") or 0)
        tokens = int(row.get("tokens") or 0)
        src = row.get("source_name", "unknown")

        acc.documents += 1
        acc.words += words
        acc.chars += chars
        acc.tokens += tokens

        if is_manual:
            acc.manual_documents += 1
            acc.manual_words += words
            acc.manual_tokens += tokens
        else:
            acc.downloaded_documents += 1
            acc.downloaded_words += words
            acc.downloaded_tokens += tokens

        s = acc.by_source.setdefault(
            src, {"documents": 0, "words": 0, "tokens": 0, "is_manual": is_manual}
        )
        s["documents"] += 1
        s["words"] += words
        s["tokens"] += tokens

    return acc


def atomic_write_json(path: str | Path, data) -> None:
    """Write JSON so that an interrupted write cannot corrupt the existing file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


if __name__ == "__main__":
    import tempfile as _tf

    assert CollectionType.MANUAL_OCR.is_manual
    assert not CollectionType.DOWNLOADED_DATASET.is_manual

    with _tf.TemporaryDirectory() as d:
        mpath = Path(d) / "m.jsonl"
        with ManifestWriter(mpath) as w:
            for i in range(3):
                w.write(make_record(
                    text="गोंयची भास कोंकणी आसा",
                    raw_text="गोंयची   भास   कोंकणी   आसा",
                    source_name="pilot_manual",
                    source_url=f"https://example.org/{i}",
                    collection_type=CollectionType.MANUAL_OCR,
                    language="kok",
                    preprocessing_applied=["unicode_nfc", "normalize_whitespace"],
                    script="Devanagari",
                ))
            w.write(make_record(
                text="डाउनलोड केलेला मजकूर आहे",
                raw_text="डाउनलोड केलेला मजकूर आहे",
                source_name="public_corpus",
                source_url="hf://example",
                collection_type=CollectionType.DOWNLOADED_DATASET,
                language="kok",
                preprocessing_applied=["unicode_nfc"],
                script="Devanagari",
            ))

        acc = summarize(mpath)
        assert acc.documents == 4, acc.documents
        assert acc.manual_documents == 3
        assert acc.downloaded_documents == 1
        print(f"docs={acc.documents} manual_words={acc.manual_words} "
              f"total_words={acc.words} manual_frac={acc.manual_word_fraction:.1%} "
              f"headroom={acc.headroom()} words")
        assert len(acc.by_source) == 2

    print("manifest self-test: all assertions passed")
