#!/usr/bin/env python3
"""
Encode a cleaned text split into a flat array of token ids on disk.

WHY THIS STEP EXISTS
--------------------
Training reads token ids, not text. Encoding on the fly would re-run
SentencePiece over the same 3-5 GB of text on every epoch and every restart,
and would make throughput depend on tokenizer speed rather than on the GPU. So
the corpus is tokenized once, written as a flat binary array, and memory-mapped
at training time. The training loop then samples a window by slicing an array -
no parsing, no decoding, no Python object per document.

WHY uint16
----------
Both vocabularies are 2,500, comfortably below 65,536, so every token id fits in
two bytes. That halves the file against uint32 and quarters it against int64:
500M tokens is 1.0 GB rather than 4.0 GB. The size matters because this file has
to be uploaded to Kaggle as a dataset and then read inside a session with
limited disk. The script asserts the vocabulary really is under 65,536 rather
than assuming it, because a silent wraparound here would corrupt every token id
above the limit and would not show up until the loss refused to fall.

DOCUMENT BOUNDARIES
-------------------
Documents are one per line in the split files. They are concatenated into a
single stream with an end-of-sequence token between them, so the model can learn
that documents end. Without a separator the model is trained to continue from
the last sentence of one document into the first sentence of the next, which is
a pattern that does not exist in real text. If the tokenizer has no EOS id the
script stops rather than concatenating silently.

RESUMABLE
---------
Progress is checkpointed after every flush: the number of lines consumed and the
number of tokens written. Re-running appends from where it stopped. This is what
allows `--max-tokens 500000000` now and a larger budget later without re-encoding
the first 500M tokens - relevant because Model H trains on a 500M budget first
and may get a longer run afterwards.

USAGE
-----
    python3 tools/pack_tokens.py --language konkani --split train \\
        --max-tokens 500000000
    python3 tools/pack_tokens.py --language konkani --split val
    python3 tools/pack_tokens.py --language marathi --split train \\
        --max-tokens 500000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# uint16 holds 0..65535. Any vocabulary at or above this silently wraps.
UINT16_LIMIT = 65_536

# Lines are encoded in batches: SentencePiece's batch API amortises the call
# overhead, and 20k lines is large enough to matter without holding much memory.
BATCH_LINES = 20_000

# Tokens are buffered in memory and flushed in chunks so the file is written in
# large sequential writes rather than one write per document.
FLUSH_TOKENS = 8_000_000


def open_with_retry(path: Path, attempts: int = 5):
    """Open a file, retrying past iCloud's dataless-placeholder errors.

    Files under a synced folder can be evicted to metadata-only. The first read
    triggers a download and fails in the meantime with EDEADLK ("Resource
    deadlock avoided") or ETIMEDOUT. Both are transient, so back off and retry
    rather than treating them as a missing file - Phase 1 lost two runs to this
    being handled as a permanent error.
    """
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return path.open("r", encoding="utf-8")
        except OSError as exc:
            if attempt == attempts:
                raise
            print(f"  read failed ({exc.strerror}), retry {attempt}/{attempts} "
                  f"in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def load_tokenizer(language: str):
    """Load the language's SentencePiece model and report its special ids.

    Returns (processor, eos_id). Exits if the model has no EOS, because the
    caller needs one to separate documents and a missing separator is a silent
    corpus-level error rather than a crash.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        raise SystemExit(
            "sentencepiece is not installed. Run:\n"
            "    pip install sentencepiece")

    model_path = REPO_ROOT / language / "tokenizer" / f"{language}_bpe.model"
    if not model_path.exists():
        raise SystemExit(f"Tokenizer not found: {model_path}")

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()

    print(f"  tokenizer      {model_path.relative_to(REPO_ROOT)}")
    print(f"  vocabulary     {vocab_size:,}")
    print(f"  unk / bos / eos / pad ids   "
          f"{sp.unk_id()} / {sp.bos_id()} / {sp.eos_id()} / {sp.pad_id()}")

    if vocab_size >= UINT16_LIMIT:
        raise SystemExit(
            f"Vocabulary {vocab_size:,} does not fit in uint16. Token ids would "
            f"wrap around silently. Change the dtype in this script before "
            f"continuing.")

    eos_id = sp.eos_id()
    if eos_id is None or eos_id < 0:
        raise SystemExit(
            "This tokenizer has no EOS id, so documents cannot be separated in "
            "the packed stream. Retrain it with an eos_id, or choose an unused "
            "id here deliberately and record the choice - do not concatenate "
            "documents without a boundary.")

    return sp, eos_id


def pack(language: str, split: str, max_tokens: int, resume: bool) -> int:
    text_path = REPO_ROOT / language / "data" / "splits" / f"{split}.txt"
    if not text_path.exists():
        raise SystemExit(f"Split not found: {text_path}")

    out_dir = REPO_ROOT / language / "data" / "packed"
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{split}.bin"
    meta_path = out_dir / f"{split}.json"

    print("=" * 70)
    print(f"PACK TOKENS - {language} / {split}")
    print("=" * 70)
    print(f"  source         {text_path.relative_to(REPO_ROOT)} "
          f"({text_path.stat().st_size / 1e9:.2f} GB)")

    sp, eos_id = load_tokenizer(language)

    # Resume state. `lines_done` is how many input lines are already represented
    # in the .bin file; the file is opened in append mode and those lines are
    # skipped. Without both halves the output would either duplicate or lose a
    # section of the corpus.
    lines_done, tokens_done = 0, 0
    if resume and bin_path.exists() and meta_path.exists():
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        lines_done = int(prior.get("lines_consumed", 0))
        tokens_done = int(prior.get("tokens", 0))
        actual = bin_path.stat().st_size // 2
        if actual != tokens_done:
            raise SystemExit(
                f"Checkpoint disagrees with the file: metadata says "
                f"{tokens_done:,} tokens, file holds {actual:,}. Delete "
                f"{bin_path.name} and {meta_path.name} and re-run.")
        if tokens_done >= max_tokens:
            print(f"\n  already at {tokens_done:,} tokens "
                  f"(budget {max_tokens:,}); nothing to do")
            return 0
        print(f"  resuming       {tokens_done:,} tokens, "
              f"{lines_done:,} lines already packed")

    mode = "ab" if lines_done else "wb"
    buffer: list[int] = []
    batch: list[str] = []
    docs = lines_done
    tokens = tokens_done
    empty_lines = 0
    started = time.time()

    def flush(handle) -> None:
        """Write the buffered ids to disk as uint16 and clear the buffer."""
        nonlocal buffer
        if not buffer:
            return
        np.asarray(buffer, dtype=np.uint16).tofile(handle)
        handle.flush()
        buffer = []

    def write_meta() -> None:
        """Record progress so an interrupted run can resume exactly."""
        meta_path.write_text(json.dumps({
            "language": language,
            "split": split,
            "tokens": tokens,
            "documents": docs,
            "lines_consumed": docs,
            "dtype": "uint16",
            "eos_id": eos_id,
            "vocab_size": sp.get_piece_size(),
            "source": str(text_path.relative_to(REPO_ROOT)),
            "complete": tokens >= max_tokens,
        }, indent=2), encoding="utf-8")

    def encode_batch(handle) -> bool:
        """Encode one batch of lines. Returns False when the budget is reached."""
        nonlocal batch, tokens, docs
        if not batch:
            return True
        for ids in sp.encode(batch, out_type=int):
            buffer.extend(ids)
            buffer.append(eos_id)          # document boundary
            tokens += len(ids) + 1
            docs += 1
            if tokens >= max_tokens:
                flush(handle)
                batch = []
                return False
        batch = []
        if len(buffer) >= FLUSH_TOKENS:
            flush(handle)
            write_meta()
        return True

    with bin_path.open(mode) as out, open_with_retry(text_path) as fh:
        for line_no, line in enumerate(fh):
            # Skip lines already packed by a previous run.
            if line_no < lines_done:
                continue
            text = line.strip()
            if not text:
                empty_lines += 1
                docs += 1          # still consumed, so resume stays aligned
                continue
            batch.append(text)
            if len(batch) >= BATCH_LINES:
                if not encode_batch(out):
                    break
                elapsed = time.time() - started
                rate = (tokens - tokens_done) / elapsed if elapsed else 0
                pct = 100.0 * tokens / max_tokens
                print(f"  {tokens:>14,} tokens  {pct:5.1f}%  "
                      f"{docs:>10,} docs  {rate/1e6:5.2f}M tok/s", flush=True)
        else:
            encode_batch(out)
        flush(out)

    write_meta()

    elapsed = time.time() - started
    print("-" * 70)
    print(f"  tokens written {tokens:,}")
    print(f"  documents      {docs:,}")
    print(f"  empty lines    {empty_lines:,}")
    print(f"  file           {bin_path.relative_to(REPO_ROOT)} "
          f"({bin_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  elapsed        {elapsed/60:.1f} min")
    print(f"  budget reached {'yes' if tokens >= max_tokens else 'no (corpus exhausted)'}")
    print("=" * 70)
    return 0


def verify(language: str, split: str) -> int:
    """Re-read the packed file and check it against its own metadata.

    Cheap, and it catches the two failures that matter: a truncated write, and
    an id outside the vocabulary (which would mean the uint16 assumption broke).
    """
    out_dir = REPO_ROOT / language / "data" / "packed"
    bin_path, meta_path = out_dir / f"{split}.bin", out_dir / f"{split}.json"
    if not bin_path.exists():
        raise SystemExit(f"Not packed yet: {bin_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")

    print(f"VERIFY {language}/{split}")
    print(f"  tokens on disk     {len(arr):,}")
    print(f"  tokens in metadata {meta['tokens']:,}")
    ok_len = len(arr) == meta["tokens"]
    print(f"  lengths agree      {'yes' if ok_len else 'NO'}")

    hi = int(arr.max())
    ok_range = hi < meta["vocab_size"]
    print(f"  highest id         {hi} (vocabulary {meta['vocab_size']})")
    print(f"  ids within vocab   {'yes' if ok_range else 'NO'}")

    eos_count = int((arr == meta["eos_id"]).sum())
    print(f"  EOS tokens         {eos_count:,} "
          f"(documents {meta['documents']:,})")

    return 0 if (ok_len and ok_range) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, choices=["marathi", "konkani"])
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--max-tokens", type=int, default=0,
                   help="stop after this many tokens (0 = whole split). The "
                        "training budget is 500,000,000 per model.")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore any existing output and start over")
    p.add_argument("--verify", action="store_true",
                   help="check an existing packed file instead of packing")
    args = p.parse_args()

    if args.verify:
        return verify(args.language, args.split)

    budget = args.max_tokens if args.max_tokens > 0 else (1 << 62)
    return pack(args.language, args.split, budget, resume=not args.no_resume)


if __name__ == "__main__":
    raise SystemExit(main())
