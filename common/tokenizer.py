"""
Tokenizer training and evaluation, shared implementation.

Only the *code* is shared. Each language trains its own tokenizer on its own
corpus and produces its own vocabulary; nothing is shared between Model H and
Model L at the data or vocabulary level. See decision D-009.

THE BYTE-FALLBACK DECISION
--------------------------
`byte_fallback=True` is not optional here, and it is the single most important
setting in this file.

The Phase 1 audit found both existing preliminary tokenizers had been trained
without it. Verified by counting byte pieces in each vocabulary: **0**. The
consequence is that any character outside the learned inventory becomes `<unk>`,
which is where the reported 46.36% and 7.19% unknown-token rates came from -
they were artefacts of missing Latin coverage, not facts about Konkani.

With byte fallback on, SentencePiece adds 256 `<0xNN>` pieces, so any UTF-8
string is representable and the unknown-token rate is ~0 **by construction**.

WHY THIS CHANGES THE EVALUATION METRIC
--------------------------------------
Once UNK is ~0 everywhere, UNK rate can no longer discriminate between
tokenizers. The metric that actually decides vocabulary size and script coverage
is **fertility**: tokens per word, and characters per token, measured on
held-out text.

The intuition: with byte fallback, poorly-covered text does not become
unrepresentable, it becomes *expensive*. A word the tokenizer knows well costs
one or two tokens; a word it has never seen costs one token per byte. Fertility
measures that cost directly. The specification asks us to choose vocabulary size
"using fertility / unknown-token rate on held-out text" - this is why fertility
is the operative half of that pair.

WHY HELD-OUT TEXT
-----------------
The old `tokenizer_stats.py` measured fertility on `tokenizer_sample.txt` - the
tokenizer's own training file. A tokenizer is optimised for exactly that text,
so measuring on it reports the best number it will ever produce. Every figure
here comes from text the tokenizer has never seen.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import sentencepiece as spm


@dataclass
class FertilityReport:
    """Held-out tokenizer measurements. All fields are reported in Phase 1."""

    vocab_size: int
    documents: int
    words: int
    characters: int
    tokens: int
    unk_tokens: int
    tokens_per_word: float
    characters_per_token: float
    unk_rate: float
    byte_pieces: int
    single_token_word_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def split_train_heldout(
    lines: list[str], heldout_size: int = 5000, seed: int = 20260819
) -> tuple[list[str], list[str]]:
    """Reserve a random held-out slice for evaluation. Deterministic given seed."""
    rng = random.Random(seed)
    indices = list(range(len(lines)))
    rng.shuffle(indices)
    heldout_idx = set(indices[:min(heldout_size, len(lines) // 10)])
    train = [l for i, l in enumerate(lines) if i not in heldout_idx]
    heldout = [lines[i] for i in sorted(heldout_idx)]
    return train, heldout


def train_tokenizer(
    *,
    input_path: str | Path,
    model_prefix: str | Path,
    vocab_size: int,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    input_sentence_size: int = 3_000_000,
    num_threads: int = 8,
) -> Path:
    """Train one SentencePiece tokenizer from scratch.

    No pretrained tokenizer is used or adapted - this trains on our corpus only,
    as the specification requires.
    """
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        # ---- the setting the audit found missing ----
        byte_fallback=True,
        # ---------------------------------------------
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        max_sentencepiece_length=16,
        num_threads=num_threads,
        normalization_rule_name="identity",   # we already applied NFC ourselves
    )
    return model_prefix.with_suffix(".model")


def count_byte_pieces(sp: spm.SentencePieceProcessor) -> int:
    """Number of <0xNN> byte pieces. Must be 256 when byte_fallback is on."""
    return sum(1 for i in range(sp.get_piece_size())
               if sp.id_to_piece(i).startswith("<0x"))


def evaluate_tokenizer(model_path: str | Path, heldout: list[str]) -> FertilityReport:
    """Measure fertility and unknown-token statistics on unseen text."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    words = tokens = chars = unk = docs = 0
    single_token_words = 0
    total_words_for_rate = 0

    for line in heldout:
        text = line.strip()
        if not text:
            continue
        docs += 1
        line_words = text.split()
        ids = sp.encode(text)

        words += len(line_words)
        tokens += len(ids)
        chars += len(text)
        unk += sum(1 for i in ids if i == sp.unk_id())

        # How often is a whole word a single token? A direct readout of how well
        # the vocabulary fits this language's morphology.
        for w in line_words[:40]:
            total_words_for_rate += 1
            if len(sp.encode(w)) == 1:
                single_token_words += 1

    return FertilityReport(
        vocab_size=sp.get_piece_size(),
        documents=docs,
        words=words,
        characters=chars,
        tokens=tokens,
        unk_tokens=unk,
        tokens_per_word=tokens / words if words else 0.0,
        characters_per_token=chars / tokens if tokens else 0.0,
        unk_rate=unk / tokens if tokens else 0.0,
        byte_pieces=count_byte_pieces(sp),
        single_token_word_rate=(single_token_words / total_words_for_rate
                                if total_words_for_rate else 0.0),
    )


def token_frequency_stats(model_path: str | Path, heldout: list[str],
                          top_k: int = 25) -> dict:
    """Token-frequency statistics, a required Phase 1 report item."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    counter: Counter = Counter()

    for line in heldout:
        counter.update(sp.encode(line.strip()))

    total = sum(counter.values())
    vocab = sp.get_piece_size()
    used = len(counter)

    # Coverage curve: how much of the token stream the most frequent pieces cover.
    ordered = [c for _, c in counter.most_common()]
    coverage = {}
    running = 0
    marks = {100, 1000, 5000, 10000}
    for i, c in enumerate(ordered, 1):
        running += c
        if i in marks:
            coverage[f"top_{i}"] = running / total if total else 0.0

    return {
        "vocabulary_size": vocab,
        "distinct_tokens_seen": used,
        "vocabulary_utilisation": used / vocab if vocab else 0.0,
        "total_tokens": total,
        "coverage": coverage,
        "most_frequent": [
            {"id": tid, "piece": sp.id_to_piece(tid), "count": c,
             "share": c / total if total else 0.0}
            for tid, c in counter.most_common(top_k)
        ],
        "hapax_tokens": sum(1 for c in counter.values() if c == 1),
    }


def tokenization_examples(model_path: str | Path, samples: list[str]) -> list[dict]:
    """Worked tokenization examples, a required Phase 1 report item."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    out = []
    for text in samples:
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text)
        out.append({
            "text": text,
            "pieces": pieces,
            "n_tokens": len(ids),
            "n_words": len(text.split()),
            "tokens_per_word": len(ids) / max(len(text.split()), 1),
            "unk": sum(1 for i in ids if i == sp.unk_id()),
            "roundtrip_ok": sp.decode(ids) == text,
        })
    return out


def sweep_vocab_sizes(
    *,
    train_path: str | Path,
    heldout: list[str],
    output_dir: str | Path,
    sizes: list[int],
    prefix: str,
) -> list[FertilityReport]:
    """Train one tokenizer per candidate vocabulary size and compare on held-out text.

    This is how vocabulary size gets *chosen* rather than assumed. The earlier
    work used 32,000 with no comparison behind it.

    The trade-off being measured: a larger vocabulary lowers fertility (fewer
    tokens per word, so more text fits in a fixed context window) but spends more
    of a ~25M-parameter budget on the embedding and output layers. At vocab V and
    model dimension d, tied embeddings cost V x d parameters - at V=32k, d=512
    that is 16.4M parameters, well over half the budget. So the smallest
    vocabulary whose fertility is close to the best is usually the right choice,
    not the largest.
    """
    output_dir = Path(output_dir)
    reports = []
    for size in sizes:
        model_prefix = output_dir / f"{prefix}_v{size}"
        print(f"  training vocab_size={size:,} ...", flush=True)
        try:
            model = train_tokenizer(input_path=train_path,
                                    model_prefix=model_prefix,
                                    vocab_size=size)
        except RuntimeError as exc:
            # SentencePiece refuses a vocabulary larger than the number of
            # distinct pieces the corpus can support, and it is a hard error.
            # This matters most for Konkani, whose corpus is genuinely small:
            # a crash mid-sweep would lose every candidate already measured.
            # Skip the candidate, report the ceiling, keep going.
            ceiling = _parse_vocab_ceiling(str(exc))
            if ceiling is None:
                raise
            print(f"    SKIPPED: corpus supports at most {ceiling:,} pieces. "
                  f"This is a data-size limit, not a configuration error.",
                  flush=True)
            continue

        report = evaluate_tokenizer(model, heldout)
        reports.append(report)
        print(f"    tokens/word={report.tokens_per_word:.4f}  "
              f"chars/token={report.characters_per_token:.4f}  "
              f"unk={report.unk_rate:.6%}  "
              f"byte_pieces={report.byte_pieces}  "
              f"whole-word={report.single_token_word_rate:.1%}", flush=True)

    if not reports:
        raise SystemExit(
            "No candidate vocabulary size could be trained. The corpus is too "
            "small or too repetitive. Collect more data, or pass smaller sizes "
            "with --vocab-sizes."
        )
    return reports


_VOCAB_CEILING_RE = re.compile(r"set it to a value <=\s*(\d+)")


def _parse_vocab_ceiling(message: str) -> int | None:
    """Extract the maximum supported vocabulary size from SentencePiece's error."""
    match = _VOCAB_CEILING_RE.search(message)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    import tempfile

    # End-to-end check on synthetic Devanagari text: does byte_fallback actually
    # produce 256 byte pieces, and is UNK genuinely ~0 on unseen script?
    marathi = ("महाराष्ट्रातील शेतकरी संकटात आहेत आणि सरकारने तातडीने मदत करण्यात यावी "
               "अशी मागणी त्यांनी केली आहे या भागात पाऊस झाला नाही म्हणून पिके वाळली आहेत ")
    lines = [marathi * 2 for _ in range(400)]

    with tempfile.TemporaryDirectory() as d:
        train_file = Path(d) / "train.txt"
        train_file.write_text("\n".join(lines), encoding="utf-8")

        model = train_tokenizer(input_path=train_file,
                                model_prefix=Path(d) / "test",
                                vocab_size=300,
                                input_sentence_size=1000)

        sp = spm.SentencePieceProcessor(model_file=str(model))
        byte_pieces = count_byte_pieces(sp)
        print(f"byte pieces in vocabulary: {byte_pieces}")
        assert byte_pieces == 256, f"byte_fallback is OFF - got {byte_pieces} byte pieces"

        # Text in a script the tokenizer never saw. Without byte fallback this
        # would be all <unk>; with it, UNK must be zero.
        unseen = ["English text the tokenizer never saw during training.",
                  "ಕನ್ನಡ ಪಠ್ಯ",
                  "日本語のテキスト"]
        report = evaluate_tokenizer(model, unseen)
        print(f"unseen-script UNK rate: {report.unk_rate:.6%} "
              f"({report.unk_tokens} unk tokens)")
        assert report.unk_rate == 0.0, "byte fallback failed to cover unseen script"

        ex = tokenization_examples(model, [marathi.strip()])[0]
        print(f"roundtrip decode exact: {ex['roundtrip_ok']}")
        assert ex["roundtrip_ok"], "encode/decode is not lossless"

    print("tokenizer self-test: all assertions passed")
