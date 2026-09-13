"""
Comparative and transitive reasoning data: lexicon and item construction.

Phase 3 asks for a synthetic finetuning corpus per language, generated
programmatically so the ground-truth label is known by construction rather than
annotated afterwards. This module holds the language resources and the item
builders; `tools/make_reasoning_data.py` is the command-line front end.

WHY EVERY SURFACE FORM HERE WAS MEASURED, NOT WRITTEN FROM MEMORY
-----------------------------------------------------------------
The finetuning text has to be grammatical Marathi and grammatical Konkani. Two
things make that harder than it sounds, and both were found by measuring the
Phase 1 corpora rather than by reasoning about grammar:

1. Proper names take an oblique stem before a postposition, and the stem is
   irregular. राम -> रामा, सीता -> सीते, कविता -> कविते, but राहुल -> राहुल in
   Marathi and राहुल -> राहुला in Konkani. The same name therefore inflects
   differently in the two languages. Every stem in NAMES below is the one that
   accounts for the large majority of that name's genitive occurrences in that
   language's own training split.

2. The genitive suffix and the interrogative agree with the *possessed* noun,
   not with the possessor. उंचाय is feminine (ची उंचाय: 455 hits, चें उंचाय: 1),
   मोल and वजन are neuter (चें मोल: 1257, चें वजन: 365). So ATTRIBUTES carries a
   gender and that single field drives both the genitive suffix and the question
   word, which is why the two can never disagree.

Corpus evidence for every item is in report/phase3_lexicon_evidence.json.

DIGITS
------
Marathi writes 80.2% of its digits in Devanagari; Konkani writes 60.5% in Latin.
Each language's data follows its own corpus convention, so the numerals in a
finetuning example are the numerals that model actually saw during pretraining.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Entity names, with the oblique stem taken from each language's own corpus.
# `stem` + a genitive suffix gives the possessive ("रामा" + "ची" -> "रामाची");
# `stem` + the comparative particle gives the standard-of-comparison form
# ("रामा" + "पेक्षा" -> "रामापेक्षा"). One stem, both jobs.
# --------------------------------------------------------------------------
NAMES: dict[str, dict[str, str]] = {
    "marathi": {
        "विजय": "विजया", "गीता": "गीता", "राम": "रामा", "गणेश": "गणेशा",
        "शंकर": "शंकरा", "लता": "लते", "कविता": "कविते", "कृष्ण": "कृष्णा",
        "सचिन": "सचिन", "प्रकाश": "प्रकाशा", "विनोद": "विनोदा", "राहुल": "राहुल",
        "रेखा": "रेखा", "गोविंद": "गोविंदा", "सीता": "सीते", "अजय": "अजय",
        "नारायण": "नारायणा", "दत्ता": "दत्ता", "प्रिया": "प्रिया",
        "संजय": "संजय", "माधुरी": "माधुरी", "दीपक": "दीपक",
    },
    "konkani": {
        "कविता": "कविते", "गीता": "गीता", "राम": "रामा", "कृष्ण": "कृष्णा",
        "शंकर": "शंकरा", "लता": "लते", "विनोद": "विनोदा", "प्रकाश": "प्रकाशा",
        "नारायण": "नारायणा", "मीरा": "मीरा", "सीता": "सीते", "अशोक": "अशोका",
        "रमेश": "रमेशा", "विजय": "विजया", "नीता": "नीता", "दत्ता": "दत्ता",
        "गणेश": "गणेशा", "मोहन": "मोहना", "महेश": "महेशा", "रेखा": "रेखे",
        "बाळू": "बाळू", "प्रिया": "प्रिया", "अनिता": "अनिता", "किशोर": "किशोरा",
        "मंगल": "मंगला", "सुधा": "सुधा", "गोविंद": "गोविंदा", "राहुल": "राहुला",
        "अनिल": "अनिला", "स्वाती": "स्वाती", "सुरेश": "सुरेशा", "सुनीता": "सुनीता",
    },
}


@dataclass(frozen=True)
class Attribute:
    """One measurable property, with everything agreement depends on.

    `gender` is the gender of `noun`, and it selects both the genitive suffix
    and the interrogative. Keeping them in one place is what stops a template
    from pairing a feminine noun with a neuter question word.
    """
    key: str
    noun: str
    gender: str          # "f" or "n"
    unit: str            # written after the number; "" for none
    low: int
    high: int


ATTRIBUTES: dict[str, list[Attribute]] = {
    "marathi": [
        Attribute("height", "उंची", "f", "फूट", 4, 7),
        Attribute("age",    "वय",   "n", "वर्षे", 8, 60),
        Attribute("weight", "वजन",  "n", "किलो", 30, 90),
    ],
    "konkani": [
        Attribute("height", "उंचाय", "f", "फूट", 4, 7),
        Attribute("age",    "पिराय", "f", "वर्सां", 8, 60),
        Attribute("weight", "वजन",   "n", "किलो", 30, 90),
    ],
}

# Genitive suffix and interrogative, by the gender of the possessed noun.
GENITIVE = {
    "marathi": {"f": "ची", "n": "चे"},
    "konkani": {"f": "ची", "n": "चें"},
}
INTERROGATIVE = {
    "marathi": {"f": "कोणाची", "n": "कोणाचे"},
    "konkani": {"f": "कोणाची", "n": "कोणाचें"},
}

# The equality word also agrees in Konkani: सारकी (f, 6,289 hits) against
# सारकें (n, 5,046), and the agreeing pairs are the ones that occur -
# "पिराय सारकी", "वजन सारकें". Marathi's समान is invariant, so both genders
# map to it. This means Konkani has two possible equality answers and Marathi
# one, which the dataset statistics report.
EQUAL = {
    "marathi": {"f": "समान", "n": "समान"},
    "konkani": {"f": "सारकी", "n": "सारकें"},
}

LEXICON = {
    "marathi": {
        "question_marker": "प्रश्न:",
        "answer_marker": "उत्तर:",
        "copula": "आहे",
        "therefore": "म्हणून",
        "more": "जास्त",
        "less": "कमी",
        "most": "सर्वात जास्त",
        "least": "सर्वात कमी",
        "particle": "पेक्षा",      # attaches directly to the oblique stem
        "particle_spaced": False,
        "digits": "devanagari",
    },
    "konkani": {
        "question_marker": "प्रस्न:",
        "answer_marker": "जाप:",
        "copula": "आसा",
        "therefore": "देखून",
        "more": "चड",
        "less": "उणें",
        "most": "सगळ्यांत चड",
        "least": "सगळ्यांत उणें",
        "particle": "परस",         # written as a separate word
        "particle_spaced": True,
        "digits": "latin",
    },
}

_DEV_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def number(value: int, language: str) -> str:
    """Render an integer in the numeral system that language's corpus prefers."""
    text = str(value)
    return text.translate(_DEV_DIGITS) if LEXICON[language]["digits"] == "devanagari" else text


def possessive(name: str, attr: Attribute, language: str) -> str:
    """'राम' + उंची -> 'रामाची उंची'. Suffix chosen by the noun's gender."""
    stem = NAMES[language][name]
    return f"{stem}{GENITIVE[language][attr.gender]} {attr.noun}"


def standard(name: str, language: str) -> str:
    """The comparison form: 'रामापेक्षा' in Marathi, 'रामा परस' in Konkani."""
    stem = NAMES[language][name]
    lex = LEXICON[language]
    sep = " " if lex["particle_spaced"] else ""
    return f"{stem}{sep}{lex['particle']}"


# --------------------------------------------------------------------------
# Templates
#
# A "family" is a reasoning structure (compare two, pick the extreme of three,
# chain inequalities). A "pattern" is a way of putting that structure into
# words. Holding out a pattern for the test split is what the specification
# means by held-out relation patterns: the test set then asks the same
# reasoning in wording the model never saw, so a model that memorised surface
# strings scores zero while a model that learned the relation does not.
#
# p_value   facts are numbers            "रामाची उंची ५ फूट आहे."
# p_more    relation stated as 'greater' "रामाची उंची विजयापेक्षा जास्त आहे."
# p_less    the SAME relation inverted   "विजयाची उंची रामापेक्षा कमी आहे."
#
# p_less is held out. It states exactly the relations p_more states, in the
# opposite direction, so it is solvable only by a model that represents the
# ordering rather than the word order.
# --------------------------------------------------------------------------
FAMILIES = ["compare_two", "superlative_three", "transitive_2hop",
            "transitive_3hop", "equality"]
PATTERNS = ["p_value", "p_more", "p_less"]
HELD_OUT_PATTERNS = ["p_less"]


@dataclass
class Item:
    """One finetuning example, with the label known by construction."""
    item_id: str
    language: str
    family: str
    pattern: str
    attribute: str
    entities: list[str]
    values: list[int] | None
    prompt: str                      # question only; loss is masked over this
    answer: str                      # the exact-match target
    answer_marker: str               # "उत्तर:" / "जाप:"
    rationale: str = ""
    metadata: dict = field(default_factory=dict)

    def completion(self, with_rationale: bool) -> str:
        """What the model must produce. Loss is computed over exactly this span.

        With rationale, the chain comes *before* the answer marker: the marker
        has to be the last thing before the answer or it stops being a reliable
        delimiter, and scoring takes whatever follows its final occurrence.
        """
        if with_rationale and self.rationale:
            return f"{self.rationale} {self.answer_marker} {self.answer}"
        return f"{self.answer_marker} {self.answer}"

    def text(self, with_rationale: bool) -> str:
        return f"{self.prompt} {self.completion(with_rationale)}"


def _fact_value(name: str, attr: Attribute, value: int, language: str) -> str:
    lex = LEXICON[language]
    unit = f" {attr.unit}" if attr.unit else ""
    return (f"{possessive(name, attr, language)} "
            f"{number(value, language)}{unit} {lex['copula']}.")


def _fact_relation(hi: str, lo: str, attr: Attribute, language: str,
                   inverted: bool) -> str:
    """State hi > lo. `inverted` says it from the smaller side instead."""
    lex = LEXICON[language]
    if inverted:
        return (f"{possessive(lo, attr, language)} {standard(hi, language)} "
                f"{lex['less']} {lex['copula']}.")
    return (f"{possessive(hi, attr, language)} {standard(lo, language)} "
            f"{lex['more']} {lex['copula']}.")


def _question(attr: Attribute, language: str, extreme: str, n_entities: int) -> str:
    """extreme is 'max' or 'min'. Two entities take the plain comparative,
    three or more take the superlative, which is how the languages work."""
    lex = LEXICON[language]
    if n_entities <= 2:
        word = lex["more"] if extreme == "max" else lex["less"]
    else:
        word = lex["most"] if extreme == "max" else lex["least"]
    who = INTERROGATIVE[language][attr.gender]
    return f"{word} {attr.noun} {who} {lex['copula']}?"


def _rationale(order: list[str], attr: Attribute, language: str,
               winner: str, extreme: str) -> str:
    """A one-line ordering followed by the conclusion, in the target language.

    The chain uses bare nominative names: a genitive form standing alone either
    side of '>' is not a well-formed noun phrase. The conclusion is a complete
    sentence, possessive and copula included, so the rationale reads as language
    rather than as notation with a word glued on the end.
    """
    lex = LEXICON[language]
    # The chain is a ranked list, highest first, headed by the attribute name.
    # An earlier version wrote it as "A > B > C"; ">" is not in either 2,500
    # piece vocabulary, so every rationale spent byte-fallback tokens on ASCII
    # punctuation - 2.56% of all tokens, against 0.0000% for the plain variant.
    # A comma-separated list carries the same ordering in characters both
    # tokenizers already know, and the sentence that follows states the
    # direction so the list cannot be read backwards.
    chain = f"{attr.noun}: " + ", ".join(order)
    if len(order) > 2:
        word = lex["most"] if extreme == "max" else lex["least"]
    else:
        word = lex["more"] if extreme == "max" else lex["less"]
    return (f"{chain}. {lex['therefore']} {word} {attr.noun} "
            f"{NAMES[language][winner]}{GENITIVE[language][attr.gender]} "
            f"{lex['copula']}.")


# Which wordings each reasoning structure admits. compare_two and equality are
# stated with numbers because stating them as a relation would put the answer
# in the question. The transitive families are relational by definition: giving
# numbers would collapse a chain of inequalities into a lookup.
FAMILY_PATTERNS = {
    "compare_two":       ["p_value"],
    "superlative_three": ["p_value"],
    "equality":          ["p_value"],
    "transitive_2hop":   ["p_more", "p_less"],
    "transitive_3hop":   ["p_more", "p_less"],
}
FAMILY_SIZE = {"compare_two": 2, "superlative_three": 3, "equality": 2,
               "transitive_2hop": 3, "transitive_3hop": 4}


def _assemble(lex: dict, facts: list[str], question: str) -> str:
    """The masked part: marker, facts, question. No answer marker here - it
    belongs to the completion so that loss covers it."""
    return f"{lex['question_marker']} {' '.join(facts)} {question}"


def build_item(rng: random.Random, language: str, family: str, pattern: str,
               names: list[str], item_id: str) -> Item | None:
    """Construct one item. Returns None if the draw was degenerate.

    Values are redrawn until they are distinct (or deliberately equal, for the
    equality family), because a tie in a family whose answer is a single name
    has no well-defined label — and a dataset whose labels are sometimes
    ambiguous cannot be scored by exact match.
    """
    lex = LEXICON[language]
    attr = rng.choice(ATTRIBUTES[language])
    k = FAMILY_SIZE[family]
    if len(names) < k:
        return None
    chosen = rng.sample(names, k)
    extreme = rng.choice(["max", "min"])

    if family == "equality":
        value = rng.randint(attr.low, attr.high)
        values = [value, value]
        facts = [_fact_value(n, attr, value, language) for n in chosen]
        answer = EQUAL[language][attr.gender]
        # Equality gets a rationale too, so that in the chain-of-thought variant
        # every item has the same shape. An item with no chain among items that
        # all have one is a second, accidental signal about the answer.
        order = chosen
        rationale = (f"{attr.noun}: {chosen[0]}, {chosen[1]}. "
                     f"{lex['therefore']} {attr.noun} {answer} {lex['copula']}.")
    elif pattern == "p_value":
        for _ in range(20):
            values = [rng.randint(attr.low, attr.high) for _ in range(k)]
            if len(set(values)) == k:
                break
        else:
            return None
        facts = [_fact_value(n, attr, v, language) for n, v in zip(chosen, values)]
        order = [n for n, _ in sorted(zip(chosen, values), key=lambda p: -p[1])]
        answer = order[0] if extreme == "max" else order[-1]
        rationale = _rationale(order, attr, language, answer, extreme)
    else:
        # Relational: `chosen` is already the descending order by construction.
        values = None
        order = chosen
        inverted = (pattern == "p_less")
        facts = [_fact_relation(order[i], order[i + 1], attr, language, inverted)
                 for i in range(k - 1)]
        # The stated order of the facts is shuffled so the answer cannot be read
        # off the first or last sentence without composing the chain.
        rng.shuffle(facts)
        answer = order[0] if extreme == "max" else order[-1]
        rationale = _rationale(order, attr, language, answer, extreme)

    question = _question(attr, language, extreme, k)
    return Item(
        item_id=item_id, language=language, family=family, pattern=pattern,
        attribute=attr.key, entities=list(chosen), values=values,
        prompt=_assemble(lex, facts, question), answer=answer,
        answer_marker=lex["answer_marker"], rationale=rationale,
        metadata={"extreme": extreme, "order": order, "n_entities": k},
    )


def split_names(language: str, test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Disjoint entity pools. Test items use names the model never saw in
    training, so a model that memorised 'रामा -> answer' scores nothing."""
    rng = random.Random(seed)
    names = sorted(NAMES[language])
    rng.shuffle(names)
    n_test = max(4, round(len(names) * test_fraction))
    return names[n_test:], names[:n_test]


def generate(language: str, n_items: int, names: list[str], patterns: list[str],
             seed: int, families: list[str] | None = None) -> list[Item]:
    """Draw items round-robin over families so no structure dominates.

    Exact prompt duplicates are dropped: with a small entity pool the same draw
    recurs, and duplicated prompts would inflate the apparent dataset size while
    teaching nothing new. The attempt cap stops the loop when the template space
    is exhausted rather than spinning forever.
    """
    rng = random.Random(seed)
    families = families or FAMILIES
    usable = [(f, p) for f in families for p in FAMILY_PATTERNS[f]
              if p in patterns and len(names) >= FAMILY_SIZE[f]]
    if not usable:
        raise ValueError(f"no family/pattern combination available for {language}")

    items, seen, attempts = [], set(), 0
    limit = n_items * 60
    while len(items) < n_items and attempts < limit:
        attempts += 1
        family, pattern = usable[len(items) % len(usable)]
        item = build_item(rng, language, family, pattern, names,
                          f"{language[:2]}-{len(items):06d}")
        if item is None or item.prompt in seen:
            continue
        seen.add(item.prompt)
        items.append(item)
    return items
