#!/usr/bin/env python3
"""
Reasoning evaluation: exact-match accuracy, pretrained against finetuned.

Phase 3 asks for "pretrained vs. finetuned accuracy (or exact match) on the
synthetic test set, qualitative successes/failures, and a comparison between
Model H and Model L on these reasoning tasks". This produces all of that, plus
the thing that makes the accuracy number trustworthy: a re-measurement of
held-out *pretraining* perplexity, so that any gain on the reasoning task can be
weighed against what it cost the language model.

    python3 tools/evaluate_reasoning.py --language konkani \\
        --checkpoint <ckpt>/konkani/finetune_plain_best.pt \\
        --compare-to  <ckpt>/konkani/pretrain_best.pt \\
        --lm-data-dir <packed>

WHY EXACT MATCH AND NOT A SOFT SCORE
------------------------------------
The dataset was generated programmatically, so every item has one correct answer
known by construction - an entity name, or the equality word. There is no
paraphrase problem and nothing to give partial credit for. A soft metric here
would only blur the result.

The answer is read as the text following the LAST occurrence of that language's
answer marker. Last rather than first, because in the chain-of-thought variant a
model can emit the marker inside its rationale before emitting the real one.

CHANCE LEVEL, WHICH THE ACCURACY NUMBER IS MEANINGLESS WITHOUT
--------------------------------------------------------------
These are multiple-choice questions in disguise: the answer is almost always one
of the entities named in the prompt. A model that picked uniformly at random from
those entities would score around 1/k for a k-entity item - roughly 40% averaged
over this test set, because two-entity items are common. Reporting 45% accuracy
without that baseline would look like learning when it is barely above guessing,
so `chance_accuracy` is computed per item and reported alongside.

FORMAT COMPLIANCE IS REPORTED SEPARATELY FROM CORRECTNESS
---------------------------------------------------------
A pretrained model has never seen this template and will mostly continue the
prompt as prose, never emitting the answer marker at all. That is a different
failure from emitting the marker and naming the wrong entity, and collapsing the
two into one accuracy number hides what finetuning actually changed. Both are
reported.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.data import PackedTokens, resolve_device                  # noqa: E402
from common.metrics import bits_per_byte                              # noqa: E402
from common.model.config import ModelConfig                           # noqa: E402
from common.model.lm import DecoderLM                                 # noqa: E402

EOS_ID = 2


def load_model(path: str, device: torch.device):
    """Rebuild from the config stored inside the checkpoint.

    Works unchanged for a pretrained or a finetuned checkpoint, because
    finetune.py writes the same payload shape that train.py does.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = DecoderLM(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg, blob


def load_items(language: str, data_dir: str | None) -> list[dict]:
    base = Path(data_dir) if data_dir else REPO_ROOT / language / "data" / "reasoning"
    with (base / "test.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def chance_accuracy(item: dict) -> float:
    """Probability of being right by guessing uniformly among the plausible answers.

    The plausible set is the entities named in the prompt, plus one extra option
    for the equality answer, which is available on every item because the model
    cannot know in advance that the values differ.
    """
    return 1.0 / (len(item["entities"]) + 1)


@torch.no_grad()
def run_generation(model, cfg, items, sp, device, max_new: int,
                   batch_size: int) -> list[dict]:
    """Greedy-decode a continuation for every test prompt.

    Prompts are bucketed by EXACT token length and batched within a bucket. The
    alternative - padding a batch to its longest prompt - would be wrong here:
    positions come from a learned absolute embedding table, so left-padding
    shifts every position index and right-padding puts pad tokens between the
    prompt and the text being generated.
    """
    encoded = [(i, sp.encode(it["prompt"])) for i, it in enumerate(items)]
    buckets: dict[int, list] = collections.defaultdict(list)
    for idx, ids in encoded:
        buckets[len(ids)].append((idx, ids))

    results: list[dict | None] = [None] * len(items)
    done, last_report = 0, 0
    for length in sorted(buckets):
        rows = buckets[length]
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            prompt = torch.tensor([ids for _, ids in chunk],
                                  dtype=torch.long, device=device)
            out = model.generate(prompt, max_new_tokens=max_new,
                                 temperature=0.0, eos_id=EOS_ID)
            for (idx, ids), row in zip(chunk, out[:, len(ids):].tolist()):
                # generate() only stops when EVERY row in the batch has emitted
                # EOS, so a row that finished early carries junk after its own
                # EOS. Cut each row at its own.
                if EOS_ID in row:
                    row = row[:row.index(EOS_ID)]
                results[idx] = {"generated_ids": row, "generated": sp.decode(row)}
            done += len(chunk)
            # Progress at coarse intervals rather than a carriage-return spinner:
            # this runs inside a Kaggle notebook where stdout is captured, and a
            # spinner there produces one long unreadable line.
            if done - last_report >= max(1, len(items) // 5) or done == len(items):
                print(f"    {done:>5}/{len(items)} generated", flush=True)
                last_report = done
    return results


def score(items: list[dict], generations: list[dict]) -> dict:
    """Exact match, with format compliance kept separate from correctness."""
    rows = []
    for it, gen in zip(items, generations):
        marker = it["answer_marker"]
        text = gen["generated"]
        has_marker = marker in text
        if has_marker:
            predicted = text.rsplit(marker, 1)[1].strip().split()
            predicted = predicted[0] if predicted else ""
        else:
            predicted = ""
        rows.append({
            "id": it["id"], "family": it["family"], "pattern": it["pattern"],
            "attribute": it["attribute"], "extreme": it["extreme"],
            "n_entities": it["n_entities"], "gold": it["answer"],
            "predicted": predicted, "correct": predicted == it["answer"],
            "emitted_marker": has_marker, "chance": chance_accuracy(it),
            "generated": text,
        })

    def agg(subset: list[dict]) -> dict:
        if not subset:
            return {}
        n = len(subset)
        return {
            "n": n,
            "accuracy": round(100.0 * sum(r["correct"] for r in subset) / n, 2),
            "format_compliance": round(
                100.0 * sum(r["emitted_marker"] for r in subset) / n, 2),
            "chance_accuracy": round(
                100.0 * sum(r["chance"] for r in subset) / n, 2),
        }

    def by(field: str) -> dict:
        out = {}
        for key in sorted({r[field] for r in rows}):
            out[str(key)] = agg([r for r in rows if r[field] == key])
        return out

    return {
        "overall": agg(rows),
        "by_family": by("family"),
        "by_pattern": by("pattern"),
        "by_attribute": by("attribute"),
        "by_n_entities": by("n_entities"),
        "rows": rows,
    }


@torch.no_grad()
def language_model_check(model, cfg, language: str, lm_data_dir: str, sp,
                         device, windows: int) -> dict:
    """Perplexity and bits-per-byte on the PRETRAINING test split.

    This is the catastrophic-forgetting measurement. Finetuning on a narrow
    template can make a model excellent at the template and much worse at the
    language, and nothing in the reasoning accuracy would show it.
    """
    data = PackedTokens(language, "test", REPO_ROOT, lm_data_dir)
    T = cfg.context_length
    high = len(data) - T - 1
    starts = [int(i) for i in torch.linspace(0, high, windows).tolist()]

    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    for i in range(0, len(starts), 16):
        arr = torch.stack([
            torch.from_numpy(data.tokens[s:s + T + 1].astype("int64"))
            for s in starts[i:i + 16]]).to(device)
        x, y = arr[:, :-1], arr[:, 1:]
        logits, _, _ = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += y.numel()
        for row in y.tolist():
            total_bytes += len(sp.decode(row).encode("utf-8"))

    mean = total_nll / total_tokens
    return {"windows": len(starts), "cross_entropy_nats": mean,
            "perplexity": math.exp(mean),
            "bits_per_byte": bits_per_byte(total_nll, total_bytes)}


def evaluate_one(label: str, path: str, items, sp, device, args) -> dict:
    print(f"\n--- {label}: {path} ---")
    model, cfg, blob = load_model(path, device)
    print(f"  step {blob['step']:,}  phase {blob.get('phase', 'pretrain')}  "
          f"parameters {model.count_parameters():,}")

    gens = run_generation(model, cfg, items, sp, device,
                          args.max_new_tokens, args.batch_size)
    result = score(items, gens)
    o = result["overall"]
    print(f"  accuracy          {o['accuracy']:6.2f}%   "
          f"(chance {o['chance_accuracy']:.2f}%)")
    print(f"  format compliance {o['format_compliance']:6.2f}%")
    for pattern, d in result["by_pattern"].items():
        held = "  <- held-out wording" if pattern == "p_less" else ""
        print(f"    {pattern:<9} n={d['n']:<5} acc {d['accuracy']:6.2f}%  "
              f"format {d['format_compliance']:6.2f}%{held}")

    result["checkpoint"] = path
    result["step"] = blob["step"]
    result["phase"] = blob.get("phase", "pretrain")

    if args.lm_data_dir:
        print("  language-model check on the pretraining test split ...")
        lm = language_model_check(model, cfg, args.language, args.lm_data_dir,
                                  sp, device, args.lm_windows)
        print(f"    perplexity {lm['perplexity']:.2f}   "
              f"bits/byte {lm['bits_per_byte']:.4f}")
        result["language_model"] = lm
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=["marathi", "konkani"])
    ap.add_argument("--checkpoint", required=True, help="the finetuned model")
    ap.add_argument("--compare-to", default=None,
                    help="the pretrained checkpoint, scored on the same items")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data-dir", default=None, help="reasoning test.jsonl lives here")
    ap.add_argument("--lm-data-dir", default=None,
                    help="packed pretraining .bin files; enables the "
                         "catastrophic-forgetting check")
    ap.add_argument("--lm-windows", type=int, default=256)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="plain")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import sentencepiece as spm
    device = resolve_device(args.device)
    tok = args.tokenizer or (REPO_ROOT / args.language / "tokenizer" /
                             f"{args.language}_bpe.model")
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    items = load_items(args.language, args.data_dir)

    print("=" * 74)
    print(f"REASONING EVALUATION - {args.language}")
    print("=" * 74)
    print(f"  device      {device}")
    print(f"  test items  {len(items):,}")

    payload = {"language": args.language, "test_items": len(items),
               "tag": args.tag}
    payload["finetuned"] = evaluate_one("finetuned", args.checkpoint, items,
                                        sp, device, args)
    if args.compare_to:
        payload["pretrained"] = evaluate_one("pretrained (baseline)",
                                             args.compare_to, items, sp, device,
                                             args)
        f, p = payload["finetuned"]["overall"], payload["pretrained"]["overall"]
        payload["delta"] = {
            "accuracy": round(f["accuracy"] - p["accuracy"], 2),
            "format_compliance": round(
                f["format_compliance"] - p["format_compliance"], 2),
        }
        print("\n--- pretrained -> finetuned ---")
        print(f"  accuracy          {p['accuracy']:6.2f}%  ->  {f['accuracy']:6.2f}%"
              f"   ({payload['delta']['accuracy']:+.2f})")
        print(f"  format compliance {p['format_compliance']:6.2f}%  ->  "
              f"{f['format_compliance']:6.2f}%"
              f"   ({payload['delta']['format_compliance']:+.2f})")
        if "language_model" in payload["finetuned"]:
            lp = payload["pretrained"]["language_model"]
            lf = payload["finetuned"]["language_model"]
            payload["delta"]["lm_perplexity"] = round(
                lf["perplexity"] - lp["perplexity"], 3)
            payload["delta"]["lm_bits_per_byte"] = round(
                lf["bits_per_byte"] - lp["bits_per_byte"], 5)
            print(f"  LM perplexity     {lp['perplexity']:6.2f}   ->  "
                  f"{lf['perplexity']:6.2f}"
                  f"   ({payload['delta']['lm_perplexity']:+.3f})  "
                  f"<- forgetting")

    # Qualitative material for the report: correct and incorrect examples,
    # taken in order rather than hand-picked.
    rows = payload["finetuned"]["rows"]
    payload["examples"] = {
        "correct": [r for r in rows if r["correct"]][:6],
        "incorrect": [r for r in rows if not r["correct"]][:6],
        "no_marker": [r for r in rows if not r["emitted_marker"]][:3],
    }

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"phase3_reasoning_eval_{args.language}_{args.tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  {out}")

    print("\n--- sample generations ---")
    for r in payload["examples"]["correct"][:2] + payload["examples"]["incorrect"][:2]:
        mark = "OK " if r["correct"] else "XX "
        print(f"  {mark}[{r['family']}/{r['pattern']}] gold={r['gold']}  "
              f"predicted={r['predicted']!r}")
        print(f"      {r['generated'][:110]}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
