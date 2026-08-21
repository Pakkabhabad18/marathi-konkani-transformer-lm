#!/usr/bin/env python3
"""
Build one language's tokenizer: sweep vocabulary sizes, choose, train, report.

Produces every Phase 1 tokenizer deliverable for a language:
  - tokenizer model file            -> <lang>/tokenizer/<lang>_bpe.model
  - vocabulary file                 -> <lang>/tokenizer/<lang>_bpe.vocab
  - vocabulary size, chosen not assumed
  - token-frequency statistics
  - average characters per token
  - tokenization examples
  - unknown-token statistics
  - held-out fertility comparison across candidate vocabulary sizes

The two languages are trained completely separately. This script is run once per
language and never mixes their corpora; the shared code is implementation only
(decision D-009).

USAGE
-----
    python3 tools/build_tokenizer.py --language marathi
    python3 tools/build_tokenizer.py --language konkani --vocab-sizes 8000,16000,32000
    python3 tools/build_tokenizer.py --language marathi --quick    # small sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.tokenizer import (                                    # noqa: E402
    evaluate_tokenizer,
    split_train_heldout,
    sweep_vocab_sizes,
    token_frequency_stats,
    tokenization_examples,
    train_tokenizer,
)

DEFAULT_SIZES = [8000, 16000, 24000, 32000, 48000]
QUICK_SIZES = [8000, 16000, 32000]

# Sentences used for the worked tokenization examples in the report. Written in
# each language's own script and natural phrasing, per the specification.
EXAMPLE_SENTENCES = {
    "marathi": [
        "महाराष्ट्रातील शेतकऱ्यांनी सरकारकडे तातडीने मदतीची मागणी केली आहे.",
        "पुणे शहरात आज सकाळी जोरदार पाऊस झाला आणि वाहतूक विस्कळीत झाली.",
        "या पुस्तकात लेखकाने आपले बालपणीचे अनुभव अतिशय सुंदर शब्दांत मांडले आहेत.",
        "संगणक विज्ञान आणि कृत्रिम बुद्धिमत्ता या विषयांचा अभ्यास वाढत आहे.",
    ],
    "konkani": [
        "गोंयची राजभास कोंकणी आसा आनी तिचो इतिहास खूब पोरनो आसा.",
        "हांव सकाळीं उठून घरा भायर वचून काम करतां आनी सांजवेळार परत येतां.",
        "ह्या पुस्तकांत बरोवप्यान आपले ल्हानपणाचे अणभव बरे तरेन बरयल्यात.",
        "गोंयांत कोंकणी शिकोवपाचें काम शाळांनी बरें तरेन चलता.",
    ],
}


def gather_corpus(language: str, max_lines: int = 0) -> list[str]:
    """Collect every text shard for one language.

    Reads only from that language's own data directory. There is no code path
    here that can mix corpora - the language name is the only input.
    """
    data_dir = REPO_ROOT / language / "data"
    shards = sorted(data_dir.rglob("shard_*.txt"))

    if not shards:
        raise SystemExit(
            f"No shards found under {data_dir}.\n"
            f"Run the collection scripts for {language} first."
        )

    print(f"Found {len(shards)} shard file(s) for {language}:")
    for shard in shards[:6]:
        print(f"  {shard.relative_to(REPO_ROOT)}")
    if len(shards) > 6:
        print(f"  ... and {len(shards) - 6} more")

    lines: list[str] = []
    for shard in shards:
        with open(shard, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    lines.append(line)
                    if max_lines and len(lines) >= max_lines:
                        return lines
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a language's tokenizer.")
    parser.add_argument("--language", required=True, choices=["marathi", "konkani"])
    parser.add_argument("--vocab-sizes", default="",
                        help="comma-separated candidates (default: 8k..48k)")
    parser.add_argument("--quick", action="store_true",
                        help="smaller sweep, for a fast first pass")
    parser.add_argument("--heldout", type=int, default=5000,
                        help="documents reserved for evaluation")
    parser.add_argument("--max-lines", type=int, default=0,
                        help="cap corpus lines read (0 = all)")
    parser.add_argument("--fertility-tolerance", type=float, default=0.02,
                        help="accept the smallest vocab within this fraction of best")
    # A sweep exists to produce evidence, not to pick the deliverable. Without
    # this flag every sweep retrains and REPLACES <lang>/tokenizer/<lang>_bpe.*,
    # which silently invalidates every token count measured with the previous
    # model. --sweep-only writes the comparison table and stops.
    parser.add_argument("--sweep-only", action="store_true",
                        help="measure candidates and write the sweep report; "
                             "do not train or install a final tokenizer")
    args = parser.parse_args()

    lang = args.language
    if args.vocab_sizes:
        sizes = [int(s) for s in args.vocab_sizes.split(",") if s.strip()]
    else:
        sizes = QUICK_SIZES if args.quick else DEFAULT_SIZES

    tok_dir = REPO_ROOT / lang / "tokenizer"
    # Sweep working files live under <lang>/data/, which is gitignored.
    #
    # They were previously written to <lang>/tokenizer/sweep/, which is NOT
    # ignored - so a 435 MB concatenated training file was committed and GitHub
    # rejected the push (100 MB per-file limit). The specification also requires
    # large artifacts to go to Google Drive rather than git.
    #
    # Only the final .model and .vocab stay under tokenizer/: those are Phase 1
    # deliverables and are ~1 MB each.
    work_dir = REPO_ROOT / lang / "data" / "tokenizer_sweep"
    tok_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"TOKENIZER BUILD - {lang}")
    print(f"Candidate vocabulary sizes: {sizes}")
    print("=" * 70)

    lines = gather_corpus(lang, args.max_lines)
    print(f"\nCorpus lines: {len(lines):,}")
    print(f"Corpus words: {sum(len(l.split()) for l in lines):,}")

    # HELD-OUT SPLIT.
    # Evaluation text must be text the tokenizer has never seen. Measuring
    # fertility on the training file - which the earlier tokenizer_stats.py did -
    # reports the most favourable number a tokenizer will ever produce.
    train_lines, heldout = split_train_heldout(lines, args.heldout)
    print(f"Training lines: {len(train_lines):,} | Held-out lines: {len(heldout):,}")

    if len(train_lines) < 1000:
        raise SystemExit("Corpus too small to train a tokenizer. Collect more data.")

    train_path = work_dir / "train_input.txt"
    train_path.write_text("\n".join(train_lines), encoding="utf-8")

    print(f"\n--- Vocabulary sweep on held-out text ---")
    reports = sweep_vocab_sizes(
        train_path=train_path, heldout=heldout,
        output_dir=work_dir, sizes=sizes, prefix=lang,
    )

    print("\n" + "-" * 70)
    print(f"{'vocab':>8} {'tok/word':>10} {'chars/tok':>11} {'UNK':>10} "
          f"{'whole-word':>11} {'bytes':>7}")
    print("-" * 70)
    for r in reports:
        print(f"{r.vocab_size:>8,} {r.tokens_per_word:>10.4f} "
              f"{r.characters_per_token:>11.4f} {r.unk_rate:>9.4%} "
              f"{r.single_token_word_rate:>10.1%} {r.byte_pieces:>7}")

    # CHOOSE. Lower fertility is better, but a larger vocabulary spends more of a
    # ~25M-parameter budget on embeddings. So take the SMALLEST vocabulary whose
    # fertility is within tolerance of the best - not simply the best.
    best = min(reports, key=lambda r: r.tokens_per_word)
    threshold = best.tokens_per_word * (1 + args.fertility_tolerance)
    chosen = min((r for r in reports if r.tokens_per_word <= threshold),
                 key=lambda r: r.vocab_size)

    print("-" * 70)
    print(f"Lowest fertility : {best.vocab_size:,} "
          f"({best.tokens_per_word:.4f} tokens/word)")
    print(f"CHOSEN           : {chosen.vocab_size:,} "
          f"({chosen.tokens_per_word:.4f} tokens/word)")
    print(f"  smallest vocabulary within {args.fertility_tolerance:.0%} of the best,")
    print(f"  because embedding parameters scale with vocabulary size.")

    for r in reports:
        if r.byte_pieces != 256:
            print(f"\n!! vocab {r.vocab_size}: {r.byte_pieces} byte pieces, expected 256")
            print("   byte_fallback is not active - do not use this tokenizer.")
            return 1

    if args.sweep_only:
        # Evidence only. The final vocabulary for this project was chosen at
        # 2,500 on the parameter-budget argument in D-043, which this table
        # supports but does not by itself produce: `chosen` above is the
        # fertility-tolerance rule, and it selects a larger vocabulary. Writing
        # the two to separate files keeps that distinction visible.
        sweep_payload = {
            "language": lang,
            "purpose": ("vocabulary sweep evidence for D-043; the deliverable "
                        "tokenizer is built separately at --vocab-sizes 2500"),
            "corpus_lines": len(lines),
            "training_lines": len(train_lines),
            "heldout_lines": len(heldout),
            "heldout_documents": args.heldout,
            "candidates": [r.to_dict() for r in reports],
            "lowest_fertility_vocab": best.vocab_size,
            "fertility_tolerance": args.fertility_tolerance,
            "vocab_chosen_by_tolerance_rule": chosen.vocab_size,
            "vocab_used_in_project": 2500,
            "note": ("The tolerance rule and the project's choice differ. The "
                     "project weights the ~25M parameter budget, which this "
                     "script does not model. See D-043."),
        }
        sweep_json = REPO_ROOT / "report" / f"phase1_tokenizer_sweep_{lang}.json"
        sweep_json.parent.mkdir(parents=True, exist_ok=True)
        sweep_json.write_text(json.dumps(sweep_payload, ensure_ascii=False,
                                         indent=2), encoding="utf-8")
        print(f"\n  sweep report written: {sweep_json.relative_to(REPO_ROOT)}")
        print("  --sweep-only: final tokenizer NOT retrained or replaced.")
        return 0

    # FINAL MODEL at the chosen size, written to the language's tokenizer dir.
    print(f"\n--- Training final tokenizer at vocab_size={chosen.vocab_size:,} ---")
    final_prefix = tok_dir / f"{lang}_bpe"
    final_model = train_tokenizer(input_path=train_path,
                                  model_prefix=final_prefix,
                                  vocab_size=chosen.vocab_size)

    final_report = evaluate_tokenizer(final_model, heldout)
    freq = token_frequency_stats(final_model, heldout)
    examples = tokenization_examples(final_model, EXAMPLE_SENTENCES[lang])

    heldout_path = work_dir / f"{lang}_heldout.txt"
    heldout_path.write_text("\n".join(heldout), encoding="utf-8")

    payload = {
        "language": lang,
        "model_file": str(final_model.relative_to(REPO_ROOT)),
        "vocab_file": str(final_prefix.with_suffix(".vocab").relative_to(REPO_ROOT)),
        "heldout_file": str(heldout_path.relative_to(REPO_ROOT)),
        "corpus_lines": len(lines),
        "training_lines": len(train_lines),
        "heldout_lines": len(heldout),
        "chosen_vocab_size": chosen.vocab_size,
        "selection_rule": (
            f"smallest vocabulary within {args.fertility_tolerance:.0%} "
            f"fertility of the best candidate"
        ),
        "sweep": [r.to_dict() for r in reports],
        "final_heldout_report": final_report.to_dict(),
        "token_frequency": freq,
        "examples": examples,
    }

    out_json = REPO_ROOT / "report" / f"phase1_tokenizer_{lang}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"FINAL TOKENIZER - {lang}")
    print("=" * 70)
    print(f"  vocabulary size        {final_report.vocab_size:,}")
    print(f"  byte-fallback pieces   {final_report.byte_pieces}  (must be 256)")
    print(f"  tokens per word        {final_report.tokens_per_word:.4f}   [held-out]")
    print(f"  characters per token   {final_report.characters_per_token:.4f}   [held-out]")
    print(f"  unknown-token rate     {final_report.unk_rate:.6%}   [held-out]")
    print(f"  whole-word tokens      {final_report.single_token_word_rate:.1%}")
    print(f"  vocabulary utilisation {freq['vocabulary_utilisation']:.1%}")
    print(f"  hapax tokens           {freq['hapax_tokens']:,}")

    print("\n  Most frequent pieces:")
    for item in freq["most_frequent"][:10]:
        print(f"    {item['piece']!r:22s} {item['count']:>9,}  {item['share']:.3%}")

    print("\n  Tokenization examples:")
    for ex in examples[:2]:
        print(f"    {ex['text'][:62]}")
        print(f"      -> {ex['n_tokens']} tokens "
              f"({ex['tokens_per_word']:.2f}/word), unk={ex['unk']}, "
              f"roundtrip={'ok' if ex['roundtrip_ok'] else 'FAILED'}")
        print(f"      {ex['pieces'][:12]}")

    print(f"\n  Model      {final_model.relative_to(REPO_ROOT)}")
    print(f"  Vocabulary {final_prefix.with_suffix('.vocab').relative_to(REPO_ROOT)}")
    print(f"  Report     {out_json.relative_to(REPO_ROOT)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
