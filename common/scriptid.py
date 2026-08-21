"""
Script detection and Marathi/Konkani discrimination.

TWO SEPARATE PROBLEMS
---------------------
1. *Script* identification: which writing system are these characters in?
   This is easy and exact - Unicode blocks answer it definitively.

2. *Language* identification within the same script: is this Devanagari text
   Marathi or Konkani? This is hard. The two languages are closely related,
   share Devanagari, share most of their character inventory, and share a large
   amount of vocabulary. During the Phase 1 audit we loaded the Konkani
   tokenizer and ran it on Marathi text: it produced 16 tokens, 0 unknowns and
   3.81 chars/token - statistically indistinguishable from its behaviour on
   Konkani. Character statistics alone therefore cannot separate them.

   Off-the-shelf language identifiers are also unreliable here, because most are
   trained with little or no Konkani and will happily label Konkani as Marathi
   or Hindi.

   The approach used here is a *closed-class function-word* discriminator. High
   frequency grammatical words - the copula, the conjunction, pronouns - are the
   most stable difference between the two languages and are almost impossible to
   avoid in running text. Content words are borrowed freely between the two;
   function words are not.

   Worked example (the clearest single contrast):
       "and"  ->  Marathi आणि   vs  Konkani आनी
       "is"   ->  Marathi आहे   vs  Konkani आसा
       "I"    ->  Marathi मी    vs  Konkani हांव

   This is version 0. It is deliberately simple, fully explainable, and its
   limitations are documented below. It is used as a *filter with a margin*,
   not as a classifier that must be right on every document.

LIMITATIONS (state these in the viva)
-------------------------------------
- Very short documents contain few function words and get low-confidence scores.
  We abstain rather than guess: score near zero means "undecidable", and callers
  should route those documents to a review bucket, not to a corpus.
- Quoted speech, or a Marathi article about Konkani, can carry markers of both.
  The margin threshold exists for exactly this case.
- It cannot detect Hindi, which shares many of these forms with Marathi. A
  separate Hindi guard is needed if Hindi contamination is suspected.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ꣠-ꣿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
KANNADA_RE = re.compile(r"[ಀ-೿]")
ASCII_DIGIT_RE = re.compile(r"[0-9]")
DEVA_DIGIT_RE = re.compile(r"[०-९]")

_WORD_RE = re.compile(r"[ऀ-ॿ]+")

# Closed-class markers. Each entry is a whole word, matched with boundaries.
# Chosen because they are (a) very frequent, (b) grammatical rather than lexical,
# and (c) differ between the two languages in form, not just in spelling variant.
MARATHI_MARKERS = {
    "आहे", "आहेत", "होता", "होती", "होते", "नाही", "नव्हते",
    "आणि", "किंवा", "परंतु", "म्हणून", "म्हणजे",
    "मी", "तू", "तो", "ती", "आम्ही", "तुम्ही", "त्यांनी", "यांनी",
    "मध्ये", "मधून", "साठी", "कडून", "पर्यंत", "नंतर",
    "करण्यात", "असून", "केली", "केले", "झाला", "झाली", "आला",
    "त्याने", "तिने", "काही", "सर्व", "पण",
}

KONKANI_MARKERS = {
    "आसा", "आसात", "आशिल्लो", "आशिल्ली", "आशिल्लें", "ना", "नासतना",
    "आनी", "वा", "पूण", "देखून", "म्हळ्यार", "जाल्यार",
    "हांव", "तूं", "तो", "ती", "आमी", "तुमी", "तांणी", "हांणी",
    "मदीं", "खातीर", "थावन", "मेरेन", "उपरांत",
    "केल्या", "केलें", "जालो", "जाली", "जालें", "आयलो",
    "ताणें", "तिणें", "कांय", "सगळें", "हें", "तें", "म्हण",
}

# Words that appear in both lists are removed - they discriminate nothing.
_AMBIGUOUS = MARATHI_MARKERS & KONKANI_MARKERS
MARATHI_MARKERS = MARATHI_MARKERS - _AMBIGUOUS
KONKANI_MARKERS = KONKANI_MARKERS - _AMBIGUOUS


@dataclass
class ScriptProfile:
    """Character-level composition of a document, all ratios over non-whitespace."""

    total_chars: int
    non_whitespace_chars: int
    devanagari: int
    latin: int
    kannada: int
    ascii_digits: int
    devanagari_digits: int
    devanagari_ratio: float
    latin_ratio: float
    kannada_ratio: float
    script: str

    def to_dict(self) -> dict:
        return asdict(self)


def profile_script(text: str) -> ScriptProfile:
    """Compute script composition.

    All ratios use NON-WHITESPACE characters as the denominator. This is the bug
    that was found in the Phase 1 audit: the original analyze_books_corpus.py
    divided by total characters including spaces, which reported the Konkani
    books corpus as 83.54% Devanagari when the true figure over non-whitespace
    characters is 99.78%.
    """
    total = len(text)
    non_ws = sum(1 for c in text if not c.isspace())

    if non_ws == 0:
        return ScriptProfile(total, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, "empty")

    deva = len(DEVANAGARI_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    kannada = len(KANNADA_RE.findall(text))
    ascii_d = len(ASCII_DIGIT_RE.findall(text))
    deva_d = len(DEVA_DIGIT_RE.findall(text))

    deva_ratio = deva / non_ws
    latin_ratio = latin / non_ws
    kannada_ratio = kannada / non_ws

    if deva_ratio >= 0.70 and latin_ratio < 0.20:
        script = "Devanagari"
    elif latin_ratio >= 0.70 and deva_ratio < 0.20:
        script = "Latin"
    elif kannada_ratio >= 0.50:
        script = "Kannada"
    else:
        script = "Mixed"

    return ScriptProfile(
        total_chars=total,
        non_whitespace_chars=non_ws,
        devanagari=deva,
        latin=latin,
        kannada=kannada,
        ascii_digits=ascii_d,
        devanagari_digits=deva_d,
        devanagari_ratio=round(deva_ratio, 6),
        latin_ratio=round(latin_ratio, 6),
        kannada_ratio=round(kannada_ratio, 6),
        script=script,
    )


@dataclass
class LangIdResult:
    """Outcome of the Marathi/Konkani discriminator."""

    label: str          # "mr" | "kok" | "undecided"
    score: float        # signed: +1 fully Marathi, -1 fully Konkani, 0 undecidable
    marathi_hits: int
    konkani_hits: int
    total_words: int
    confident: bool

    def to_dict(self) -> dict:
        return asdict(self)


def identify_marathi_konkani(
    text: str,
    *,
    min_markers: int = 5,
    margin: float = 0.30,
) -> LangIdResult:
    """Discriminate Marathi from Konkani using closed-class function words.

    Args:
        text: Devanagari text, already Unicode-normalized.
        min_markers: minimum total marker hits before any verdict is given.
            Below this the document is "undecided" - we abstain instead of
            guessing on insufficient evidence.
        margin: required separation. score = (m - k) / (m + k); a verdict needs
            |score| >= margin, so a document with a near-even split of markers
            is reported as undecided rather than assigned to a corpus.

    Returns:
        LangIdResult. The `score` field is what gets written into the manifest's
        `langid_score` column: positive means Marathi-leaning, negative means
        Konkani-leaning, magnitude is confidence.
    """
    # Split into WHOLE Devanagari words. _WORD_RE is [ऀ-ॿ]+, so punctuation,
    # Latin text and digits are not words here. Whole-word matching is
    # essential: "आणि" (Marathi "and") is a substring of longer words, and
    # substring matching would count those as evidence.
    words = _WORD_RE.findall(text)
    if not words:
        # No Devanagari at all - there is nothing to discriminate on. Abstain
        # rather than return a default label, which a caller might trust.
        return LangIdResult("undecided", 0.0, 0, 0, 0, False)

    # Count each distinct word once, then multiply by its frequency below.
    # Counting frequencies (not just presence) is deliberate: a document that
    # says आहे twenty times is stronger evidence of Marathi than one that says
    # it once, and the score should reflect that.
    word_set_counts = {}
    for w in words:
        word_set_counts[w] = word_set_counts.get(w, 0) + 1

    # Total marker OCCURRENCES on each side. MARATHI_MARKERS and
    # KONKANI_MARKERS have already had their intersection removed at import
    # time, so a word can contribute to at most one side - no double counting.
    m_hits = sum(c for w, c in word_set_counts.items() if w in MARATHI_MARKERS)
    k_hits = sum(c for w, c in word_set_counts.items() if w in KONKANI_MARKERS)
    total_hits = m_hits + k_hits

    # EVIDENCE GATE. Below min_markers there is not enough signal to decide.
    # This is why the discriminator abstained on 22-word IndicCorp fragments
    # (D-034) and only gave verdicts once they were packed into 300-word
    # documents - the abstention was correct behaviour, not a failure.
    if total_hits < min_markers:
        return LangIdResult("undecided", 0.0, m_hits, k_hits, len(words), False)

    # Signed, normalised score in [-1, +1].
    #   +1 : every marker found was Marathi
    #   -1 : every marker found was Konkani
    #    0 : an even split
    # Normalising by total_hits makes the score comparable across documents of
    # very different lengths, which is what let us calibrate gom.txt against
    # two reference corpora in D-035.
    score = (m_hits - k_hits) / total_hits

    # MARGIN GATE. A verdict needs |score| >= margin. A document with a
    # near-even split - quoted speech, a Marathi article about Konkani, or a
    # crawled page mixing both - lands in the dead band and is reported
    # undecided rather than being forced into one corpus.
    if score >= margin:
        return LangIdResult("mr", round(score, 4), m_hits, k_hits, len(words), True)
    if score <= -margin:
        return LangIdResult("kok", round(score, 4), m_hits, k_hits, len(words), True)
    return LangIdResult("undecided", round(score, 4), m_hits, k_hits, len(words), False)


if __name__ == "__main__":
    marathi = (
        "महाराष्ट्रातील शेतकरी संकटात आहेत आणि सरकारने मदत करण्यात यावी अशी मागणी "
        "त्यांनी केली आहे. या भागात पाऊस झाला नाही म्हणून पिके वाळली आहेत."
    )
    konkani = (
        "गोंयची भास कोंकणी आसा आनी तिचो इतिहास पोरनो आसा. हांव घरा वतां आनी जेवण "
        "करतां. तांणी सांगलें की हें काम बरें जालें म्हण."
    )

    for name, txt in (("Marathi", marathi), ("Konkani", konkani)):
        prof = profile_script(txt)
        lid = identify_marathi_konkani(txt)
        print(f"{name:8s} script={prof.script:11s} deva={prof.devanagari_ratio:.3f} "
              f"-> label={lid.label:9s} score={lid.score:+.3f} "
              f"(m={lid.marathi_hits}, k={lid.konkani_hits}) confident={lid.confident}")

    assert identify_marathi_konkani(marathi).label == "mr"
    assert identify_marathi_konkani(konkani).label == "kok"
    assert identify_marathi_konkani("गोंय").label == "undecided", "must abstain when short"

    # The audit's whitespace bug, reproduced and fixed.
    spaced = "गोंय     चि     भास"
    assert profile_script(spaced).devanagari_ratio == 1.0, "ratio must ignore whitespace"

    print("scriptid self-test: all assertions passed")
