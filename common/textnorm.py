"""
Unicode and whitespace normalization shared by both language pipelines.

WHY THIS EXISTS
---------------
Devanagari can encode the same visible grapheme in more than one way. The most
common case is the nukta: क़ can be a single precomposed code point (U+0958) or
a base letter plus a combining nukta (U+0915 U+093C). To a human they are
identical; to a tokenizer they are two different strings, so BPE learns two
separate sets of merges for the same word and the vocabulary is silently wasted.
NFC composition collapses these to one canonical form.

NOTE ON WHAT WE DO *NOT* STRIP
------------------------------
ZWJ (U+200D) and ZWNJ (U+200C) are NOT removed. In Devanagari they are
meaningful: ZWNJ forces a half-form/explicit-virama rendering and ZWJ requests
a conjunct. Stripping them changes how words are written. Many naive cleaning
scripts delete all "invisible" characters and corrupt Indic text this way.
We remove only characters that carry no linguistic content: BOM, zero-width
space, soft hyphen, and C0/C1 control characters other than newline and tab.

This module is language-agnostic utility code. It is shared between the Marathi
and Konkani pipelines deliberately: the project specification forbids sharing
*data, tokenizers, vocabularies and weights* between the two models, and says
nothing against sharing ordinary library code. Sharing the normalizer is in
fact desirable, because it guarantees both corpora are normalized identically
and therefore that the cross-corpus contamination check compares like with like.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that are invisible and carry no linguistic content in Devanagari.
# ZWJ (200D) and ZWNJ (200C) are deliberately absent from this set - see docstring.
_ZERO_WIDTH = {
    "﻿",  # BOM / zero-width no-break space
    "​",  # zero-width space
    "­",  # soft hyphen
    "⁠",  # word joiner
}

# C0 and C1 controls except \t and \n, which we handle as whitespace.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Unicode whitespace variants that should all become a plain space.
_SPACE_VARIANTS_RE = re.compile(r"[   -   　]")

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SPACE_BEFORE_NEWLINE_RE = re.compile(r"[ \t]+\n")
_NEWLINE_AFTER_SPACE_RE = re.compile(r"\n[ \t]+")

# Devanagari danda and double danda are sentence terminators. OCR often leaves
# them stuck to the following word; we make spacing consistent around them.
_DANDA_RE = re.compile(r"\s*([।॥])\s*")


def normalize_unicode(text: str) -> str:
    """Apply NFC composition and drop non-linguistic invisible characters.

    NFC is chosen over NFKC on purpose. NFKC applies *compatibility* mappings,
    which would rewrite Devanagari digits into ASCII digits and destroy some
    ligature distinctions. We want canonical equivalence only.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    for char in _ZERO_WIDTH:
        if char in text:
            text = text.replace(char, "")

    text = _CONTROL_RE.sub("", text)
    return text


def normalize_whitespace(text: str, *, keep_paragraphs: bool = True) -> str:
    """Collapse runs of whitespace without destroying paragraph structure.

    The Konkani books corpus card explicitly warns of "large gaps between words
    and excessive line breaks" from the digitisation process, so this is not a
    cosmetic step: unnormalized runs of spaces become their own BPE tokens and
    inflate the token count with content-free pieces.

    Args:
        text: input string, ideally already Unicode-normalized.
        keep_paragraphs: if True, a blank line is preserved as a paragraph
            boundary (collapsed to exactly one blank line). If False, all
            newlines become single spaces, producing one long line.
    """
    if not text:
        return ""

    text = _SPACE_VARIANTS_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if keep_paragraphs:
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = _SPACE_BEFORE_NEWLINE_RE.sub("\n", text)
        text = _NEWLINE_AFTER_SPACE_RE.sub("\n", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_danda(text: str) -> str:
    """Put exactly one space after a danda and none before it."""
    return _DANDA_RE.sub(r"\1 ", text).strip()


def normalize(text: str, *, keep_paragraphs: bool = True, danda: bool = True) -> str:
    """Full normalization pipeline: Unicode, then whitespace, then punctuation.

    This is the single entry point every collector and preprocessor should call,
    so that every document in both corpora passes through identical steps.
    """
    text = normalize_unicode(text)
    if danda:
        text = normalize_danda(text)
    text = normalize_whitespace(text, keep_paragraphs=keep_paragraphs)
    return text


# The ordered list of steps applied by normalize(), recorded in each document's
# manifest row as `preprocessing_applied` so provenance is auditable later.
NORMALIZATION_STEPS = [
    "unicode_nfc",
    "strip_zero_width",
    "strip_control_chars",
    "normalize_danda_spacing",
    "normalize_whitespace",
]


if __name__ == "__main__":
    # Self-test: decomposed vs precomposed nukta must converge under NFC.
    decomposed = "क़"       # KA + NUKTA
    precomposed = "क़"            # QA
    assert normalize_unicode(decomposed) == normalize_unicode(precomposed), "NFC failed"

    # ZWNJ must survive normalization.
    zwnj_text = "क्‌ष"
    assert "‌" in normalize(zwnj_text), "ZWNJ was destroyed"

    # Zero-width space must not survive.
    assert "​" not in normalize("क​ख")

    # OCR-style gaps collapse.
    assert normalize("गोंय     चि     भास") == "गोंय चि भास"

    # Paragraph structure survives, excess blank lines do not.
    assert normalize("एक\n\n\n\nदोन") == "एक\n\nदोन"

    print("textnorm self-test: all assertions passed")
