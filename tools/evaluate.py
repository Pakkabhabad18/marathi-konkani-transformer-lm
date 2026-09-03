#!/usr/bin/env python3
"""
Evaluate one pretrained model: intrinsic language-modelling metrics and
generation quality.

Run once per language. Every number the specification asks for in section 2.3,
except the attention analysis, which is in tools/attention_analysis.py.

WHAT IS MEASURED, AND WHY EACH ONE IS HERE
------------------------------------------
Cross-entropy and perplexity on held-out text. The direct training objective.
Comparable between checkpoints of the *same* model; not comparable between Model
H and Model L, because their tokenizers segment differently and perplexity is
per token.

Bits per byte on the same text. The cross-language measure. Dividing the same
likelihood by UTF-8 bytes instead of tokens removes the tokenizer from the
denominator, so Marathi and Konkani can be placed side by side honestly.

Generation under greedy decoding and temperatures 0.5, 1.0 and 1.5, scored with
BLEU-4, chrF and ROUGE-L against the true continuation. Absolute values will be
low - there are many acceptable continuations of a prefix and we compare against
one - so the number that carries information is the difference between models and
between temperatures.

Diversity and repetition. A small language model's characteristic failure is
degeneration into a loop, and perplexity cannot see it: a repeated
high-probability phrase scores *well*. Distinct-1, Distinct-2 and the 4-gram
repetition rate are what catch it.

USAGE
-----
    python3 tools/evaluate.py --language konkani \\
        --checkpoint /path/to/pretrain_best.pt \\
        --data-dir /path/to/packed \\
        --tokenizer /path/to/konkani_bpe.model \\
        --out-dir report/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.data import PackedTokens, resolve_device                 # noqa: E402
from common.metrics import (bits_per_byte, corpus_bleu, corpus_chrf,  # noqa: E402
                            corpus_rouge_l, diversity_stats)
from common.model.config import ModelConfig                           # noqa: E402
from common.model.lm import DecoderLM                                 # noqa: E402

# Decoding settings the specification names explicitly. Greedy is temperature 0
# in our generate(): argmax rather than sampling, so it is deterministic and is
# the setting where degeneration is most visible.
DECODE_SETTINGS = [
    ("greedy", 0.0),
    ("temp_0.5", 0.5),
    ("temp_1.0", 1.0),
    ("temp_1.5", 1.5),
]

PROMPT_TOKENS = 64        # conditioning prefix taken from held-out text
CONTINUE_TOKENS = 128     # tokens the model must produce


def load_model(checkpoint: Path, device: torch.device):
    """Rebuild the model from the config stored inside its own checkpoint.

    The config travels with the weights rather than being passed in, so it is
    impossible to load a checkpoint into a differently shaped model and get
    silently wrong results. This is why train.py writes `model_config` into
    every checkpoint.
    """
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = DecoderLM(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    print(f"  checkpoint     {checkpoint}")
    print(f"  trained steps  {blob['step']:,}")
    print(f"  best val loss  {blob.get('best_val', float('nan')):.4f}")
    print(f"  parameters     {model.count_parameters():,}")
    return model, cfg, blob


@torch.no_grad()
def intrinsic(model, cfg, data: PackedTokens, sp, device, n_windows: int) -> dict:
    """Cross-entropy, perplexity and bits-per-byte over held-out windows.

    Windows are evenly spaced across the split rather than taken from the front,
    because the splits are source-stratified: the first N tokens would be one
    source, not the corpus.

    The loss is accumulated as a *sum* of per-token negative log-likelihood, not
    a mean of per-batch means. Batches differ in nothing here (all are full), but
    summing keeps the arithmetic correct for bits-per-byte, which needs total
    nats against total bytes.
    """
    T = cfg.context_length
    high = len(data) - T - 1
    starts = [int(i) for i in torch.linspace(0, high, n_windows).tolist()]

    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    batch_size = 16

    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        arr = torch.stack([
            torch.from_numpy(data.tokens[s:s + T + 1].astype("int64"))
            for s in chunk]).to(device)
        x, y = arr[:, :-1], arr[:, 1:]

        logits, _, _ = model(x)
        # reduction="sum" so the total is in nats over all predicted positions.
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += y.numel()

        # Bytes of the *target* text, decoded back through the tokenizer. This
        # has to be the same text the likelihood was computed over, or BPB is
        # measuring two different things.
        for row in y.tolist():
            total_bytes += len(sp.decode(row).encode("utf-8"))

    mean_nll = total_nll / total_tokens
    return {
        "windows": len(starts),
        "tokens_scored": total_tokens,
        "bytes_scored": total_bytes,
        "cross_entropy_nats": mean_nll,
        "perplexity": math.exp(mean_nll),
        "bits_per_byte": bits_per_byte(total_nll, total_bytes),
        "bytes_per_token": total_bytes / total_tokens,
    }


@torch.no_grad()
def generation(model, cfg, data: PackedTokens, sp, device,
               n_prompts: int) -> dict:
    """Generate continuations at each decoding setting and score them.

    Prompts are fixed held-out windows, identical across settings and across the
    two models, so any difference in score comes from the model or the
    temperature and not from which text was sampled.
    """
    high = len(data) - (PROMPT_TOKENS + CONTINUE_TOKENS) - 1
    starts = [int(i) for i in torch.linspace(0, high, n_prompts).tolist()]

    prompts, references = [], []
    for s in starts:
        window = data.tokens[s:s + PROMPT_TOKENS + CONTINUE_TOKENS].astype("int64")
        prompts.append(window[:PROMPT_TOKENS].tolist())
        references.append(window[PROMPT_TOKENS:].tolist())

    ref_text = [sp.decode(r) for r in references]
    ref_words = [t.split() for t in ref_text]

    results = {}
    for name, temperature in DECODE_SETTINGS:
        started = time.time()
        hyp_text, hyp_tokens = [], []

        for i in range(0, len(prompts), 8):
            batch = torch.tensor(prompts[i:i + 8], dtype=torch.long, device=device)
            out = model.generate(batch, max_new_tokens=CONTINUE_TOKENS,
                                 temperature=temperature, eos_id=None)
            # Keep only what the model produced, not the prompt it was given.
            for row in out[:, PROMPT_TOKENS:].tolist():
                hyp_tokens.append(row)
                hyp_text.append(sp.decode(row))

        hyp_words = [t.split() for t in hyp_text]

        results[name] = {
            "temperature": temperature,
            "bleu": corpus_bleu(hyp_words, ref_words),
            "chrf": corpus_chrf(hyp_text, ref_text),
            "rouge_l": corpus_rouge_l(hyp_words, ref_words),
            # Diversity is measured on token pieces rather than words: a model
            # looping on sub-word fragments would not show up in whitespace
            # tokens.
            "diversity": diversity_stats([[str(t) for t in row]
                                          for row in hyp_tokens]),
            "seconds": time.time() - started,
            "samples": [
                {"prompt": sp.decode(prompts[j]),
                 "generated": hyp_text[j],
                 "reference": ref_text[j]}
                for j in range(min(3, len(hyp_text)))
            ],
        }
        d = results[name]
        print(f"  {name:<10} BLEU {d['bleu']['bleu']:6.2f}  "
              f"chrF {d['chrf']['chrf']:6.2f}  "
              f"ROUGE-L {d['rouge_l']['rouge_l']:6.2f}  "
              f"dist-1 {d['diversity']['distinct_1']:.3f}  "
              f"dist-2 {d['diversity']['distinct_2']:.3f}  "
              f"rep-4 {d['diversity']['repetition_rate_4gram']:.3f}",
              flush=True)

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=["marathi", "konkani"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="path to the language's SentencePiece .model")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="test by default: val was used for model selection "
                         "during training, so reporting on it would be "
                         "optimistic")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--windows", type=int, default=512,
                    help="held-out windows for perplexity and BPB")
    ap.add_argument("--prompts", type=int, default=64,
                    help="prompts per decoding setting")
    args = ap.parse_args()

    import sentencepiece as spm

    device = resolve_device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"EVALUATION - {args.language}  (split: {args.split})")
    print("=" * 74)
    print(f"  device         {device}")

    model, cfg, blob = load_model(Path(args.checkpoint), device)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    data = PackedTokens(args.language, args.split, REPO_ROOT, args.data_dir)
    print(f"  data           {data.describe()}")

    if sp.get_piece_size() != cfg.vocab_size:
        raise SystemExit(
            f"Tokenizer vocabulary {sp.get_piece_size()} does not match the "
            f"model's {cfg.vocab_size}. Decoded text would be nonsense.")

    print("\n--- intrinsic ---")
    intr = intrinsic(model, cfg, data, sp, device, args.windows)
    print(f"  cross-entropy  {intr['cross_entropy_nats']:.4f} nats/token")
    print(f"  perplexity     {intr['perplexity']:.2f}")
    print(f"  bits per byte  {intr['bits_per_byte']:.4f}")
    print(f"  bytes/token    {intr['bytes_per_token']:.3f}")
    print(f"  scored         {intr['tokens_scored']:,} tokens, "
          f"{intr['bytes_scored']:,} bytes")

    print("\n--- generation ---")
    gen = generation(model, cfg, data, sp, device, args.prompts)

    payload = {
        "language": args.language,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "trained_steps": blob["step"],
        "best_val_loss": blob.get("best_val"),
        "model_config": blob["model_config"],
        "parameters": model.count_parameters(),
        "intrinsic": intr,
        "generation": gen,
    }
    out_json = out_dir / f"phase2_eval_{args.language}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  {out_json}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
