#!/usr/bin/env python3
"""
Attention on comparative-reasoning prompts, pretrained against finetuned.

WHY THIS EXISTS SEPARATELY FROM tools/attention_analysis.py
-----------------------------------------------------------
`tools/attention_analysis.py` measures attention over the packed pretraining
corpus - news, government circulars, OCR'd books. That is the right input for
the Phase 2 question, which was what the language model learned in general.

It is the wrong input for the Phase 3 question. Specification 3.2 asks whether
finetuning changed local versus long-range attention or head specialisation
"especially on comparative-reasoning prompts", and a model finetuned to answer
"whose height is greater" may reorganise itself only on inputs that look like
that question. Averaging over ordinary prose would wash that out: the reasoning
prompts are a tiny, highly structured slice of input space, and the corpus
contains nothing shaped like them.

So this script runs the same measurements on the actual test prompts the model
was finetuned to answer, and adds one that only makes sense here: at the last
prompt position - the position from which the answer token is predicted - which
tokens of the question is the model looking at? That is the closest thing to
"where does it read the answer from" that attention weights can tell us.

WHAT IT REPORTS
    1. Per-layer entropy and mean attention distance, averaged over N reasoning
       prompts, pretrained against finetuned. Directly comparable to the Phase 2
       corpus-text figures, which lets the two be contrasted.
    2. Where the final position attends, bucketed into the parts of the prompt
       that mean different things: the entity names, the numeric values, the
       attribute word, and the question itself.
    3. Heatmaps for an early and a late layer, both checkpoints, with the real
       Devanagari pieces on the axes.

USAGE
    python3 tools/attention_reasoning.py --language marathi \\
      --pretrained marathi/model/marathi_pretrain_best.pt \\
      --finetuned  marathi/model/finetune_final_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.model.config import ModelConfig            # noqa: E402
from common.model.lm import DecoderLM                  # noqa: E402


def load_model(path: str, device: torch.device) -> tuple[DecoderLM, ModelConfig]:
    """Rebuild a model from the config stored inside its own checkpoint.

    Never from a separate config file: a checkpoint whose weights and config
    disagree either fails to load or - worse - loads into shapes that happen to
    fit and mean something else.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = DecoderLM(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg


def load_items(language: str, data_dir: str | None, family: str, limit: int) -> list[dict]:
    """The first `limit` test items of one family, in file order.

    In file order rather than sampled, so the figures and the statistics are
    reproducible without carrying a seed around.
    """
    path = Path(data_dir) if data_dir else REPO_ROOT / language / "data" / "reasoning"
    items = []
    with open(path / "test.jsonl", encoding="utf-8") as fh:
        for line in fh:
            it = json.loads(line)
            if family == "any" or it["family"] == family:
                items.append(it)
            if len(items) >= limit:
                break
    if not items:
        raise SystemExit(f"no test items of family {family!r} in {path}")
    return items


@torch.no_grad()
def statistics(model: DecoderLM, cfg: ModelConfig, items: list[dict], sp,
               device: torch.device) -> dict:
    """Per-head entropy and mean attention distance over reasoning prompts.

    Both quantities are computed exactly as tools/attention_analysis.py computes
    them over corpus text, so the two sets of numbers can be put side by side:

      entropy        of each query's distribution over the keys it may attend
                     to, in bits. High means diffuse, low means peaked.
      mean distance  how many positions back a query attends on average,
                     weighted by attention. Small means local.

    Position 0 is skipped for both. It can only attend to itself, so its entropy
    is 0 and its distance is 0 by construction, and including it would drag both
    averages toward zero by an amount that depends only on sequence length.
    """
    n_layers, n_heads = cfg.n_layers, cfg.n_heads
    ent = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    dist = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    seen = 0

    for it in items:
        ids = sp.encode(it["prompt"])[: cfg.context_length]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        _, _, attns = model(x, return_attention=True)
        T = x.shape[1]
        # Distance from query i to key j, as a non-negative offset. Only the
        # lower triangle carries weight, so j <= i everywhere that matters.
        idx = torch.arange(T, device=device)
        offset = (idx.view(-1, 1) - idx.view(1, -1)).clamp_min(0).float()

        for layer, attn in enumerate(attns):
            a = attn[0].float()                                   # (H, T, T)
            p = a.clamp_min(1e-12)
            # Masked positions are exactly 0 after a softmax over -inf, and
            # 0*log0 is 0 in the limit, so clamping keeps those terms at zero
            # instead of producing nan.
            e = -(a * p.log2()).sum(-1)                            # (H, T)
            d = (a * offset.unsqueeze(0)).sum(-1)                  # (H, T)
            ent[layer] += e[:, 1:].mean(-1).double().cpu()
            dist[layer] += d[:, 1:].mean(-1).double().cpu()
        seen += 1

    return {"prompts": seen,
            "entropy_bits": (ent / seen).tolist(),
            "mean_distance": (dist / seen).tolist()}


@torch.no_grad()
def answer_position_focus(model: DecoderLM, cfg: ModelConfig, items: list[dict],
                          sp, device: torch.device) -> dict:
    """Where does the model look from the position that must produce the answer?

    The last prompt token is the position whose output becomes the first answer
    token, so its attention row is the one that matters for the task. Each
    prompt's tokens are bucketed by what they mean, and the row's mass is summed
    per bucket, averaged over prompts and over all heads of the final layer.

    Buckets are found by matching decoded pieces against the item's own metadata
    rather than by position, because tokenisation splits names and numbers
    unpredictably.
    """
    last_layer = cfg.n_layers - 1
    totals = {"entity_names": 0.0, "numeric_values": 0.0,
              "attribute_word": 0.0, "question_tail": 0.0, "first_token": 0.0,
              "other": 0.0}
    seen = 0

    for it in items:
        ids = sp.encode(it["prompt"])[: cfg.context_length]
        pieces = [sp.id_to_piece(i).replace("▁", "") for i in ids]
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        _, _, attns = model(x, return_attention=True)
        row = attns[last_layer][0, :, -1, :].float().mean(0)       # (T,)

        # Which surface strings count as what, for this item.
        names = list(it.get("entities") or [])
        # The transitive families state relations rather than numbers, so
        # "values" is null for them. `or []` rather than a default, because the
        # key exists and holds None - a default would never fire.
        values = [str(v) for v in (it.get("values") or [])]
        # Devanagari digits, since Marathi writes most numbers that way.
        dev = str.maketrans("0123456789", "०१२३४५६७८९")
        values += [v.translate(dev) for v in values]
        tail_start = int(len(ids) * 0.8)                # the question clause

        for pos, piece in enumerate(pieces):
            w = float(row[pos])
            if pos == 0:
                totals["first_token"] += w
            elif piece and any(piece in n or n.startswith(piece) for n in names):
                totals["entity_names"] += w
            elif piece and any(piece in v for v in values):
                totals["numeric_values"] += w
            elif piece and piece in (it.get("attribute") or ""):
                totals["attribute_word"] += w
            elif pos >= tail_start:
                totals["question_tail"] += w
            else:
                totals["other"] += w
        seen += 1

    return {"prompts": seen,
            "layer": last_layer,
            "share": {k: round(100.0 * v / seen, 2) for k, v in totals.items()}}


def heatmaps(models: dict, cfg: ModelConfig, item: dict, sp,
             device: torch.device, out_dir: Path, language: str) -> list[str]:
    """One figure per (checkpoint, layer), four heads each, tokens on the axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Lohit Devanagari renders the script; DejaVu is the fallback for the ASCII
    # in axis titles. Without this every piece would draw as a hollow box.
    plt.rcParams["font.family"] = ["Lohit Devanagari", "DejaVu Sans"]

    ids = sp.encode(item["prompt"])[: cfg.context_length]
    pieces = [sp.id_to_piece(i).replace("▁", "") or "_" for i in ids]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    layers = sorted({0, cfg.n_layers - 1})
    saved = []

    for label, model in models.items():
        with torch.no_grad():
            _, _, attns = model(x, return_attention=True)
        for layer in layers:
            a = attns[layer][0].float().cpu().numpy()
            fig, axes = plt.subplots(1, 4, figsize=(22, 6))
            for head, ax in enumerate(axes):
                im = ax.imshow(a[head], cmap="viridis", aspect="auto",
                               vmin=0.0, vmax=float(a[head].max()))
                ax.set_title(f"head {head}")
                ax.set_xticks(range(len(pieces)))
                ax.set_yticks(range(len(pieces)))
                ax.set_xticklabels(pieces, rotation=90, fontsize=5)
                ax.set_yticklabels(pieces, fontsize=5)
                ax.set_xlabel("key position (attended to)")
                if head == 0:
                    ax.set_ylabel("query position (attending)")
                fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
            where = ("early" if layer == 0 else "late")
            fig.suptitle(f"{language.capitalize()} - {label}, {where} layer {layer}, "
                         f"on a {item['family']} reasoning prompt\n"
                         f"lower triangle only: causal masking")
            fig.tight_layout()
            path = out_dir / f"phase3_attn_reasoning_{language}_{label}_layer{layer}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path.name)
            print(f"  {path.name}")
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["marathi", "konkani"])
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--family", default="transitive_3hop",
                    help="which family to measure; the deepest chain by default")
    ap.add_argument("--prompts", type=int, default=64,
                    help="how many test prompts the statistics average over")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import sentencepiece as spm
    device = torch.device(args.device)
    tok = args.tokenizer or (REPO_ROOT / args.language / "tokenizer" /
                             f"{args.language}_bpe.model")
    sp = spm.SentencePieceProcessor(model_file=str(tok))

    print("=" * 74)
    print(f"ATTENTION ON REASONING PROMPTS - {args.language}")
    print("=" * 74)

    pre, cfg = load_model(args.pretrained, device)
    fin, cfg_f = load_model(args.finetuned, device)
    assert cfg.n_layers == cfg_f.n_layers and cfg.n_heads == cfg_f.n_heads
    items = load_items(args.language, args.data_dir, args.family, args.prompts)
    print(f"  family        {args.family}")
    print(f"  prompts       {len(items)}")
    print(f"  layers x head {cfg.n_layers} x {cfg.n_heads}")
    print(f"  example       {items[0]['prompt'][:70]}...")
    print(f"  gold answer   {items[0]['answer']}")

    stats = {"pretrained": statistics(pre, cfg, items, sp, device),
             "finetuned": statistics(fin, cfg, items, sp, device)}

    print(f"\n{'layer':>5}{'entropy pre':>13}{'post':>8}{'delta':>8} |"
          f"{'dist pre':>10}{'post':>8}{'delta':>8}")
    for L in range(cfg.n_layers):
        ep = sum(stats["pretrained"]["entropy_bits"][L]) / cfg.n_heads
        ef = sum(stats["finetuned"]["entropy_bits"][L]) / cfg.n_heads
        dp = sum(stats["pretrained"]["mean_distance"][L]) / cfg.n_heads
        df = sum(stats["finetuned"]["mean_distance"][L]) / cfg.n_heads
        print(f"{L:>5}{ep:>13.3f}{ef:>8.3f}{ef - ep:>+8.3f} |"
              f"{dp:>10.2f}{df:>8.2f}{df - dp:>+8.2f}")

    flat = lambda d, k: [v for row in d[k] for v in row]
    mp = sum(flat(stats["pretrained"], "entropy_bits")) / (cfg.n_layers * cfg.n_heads)
    mf = sum(flat(stats["finetuned"], "entropy_bits")) / (cfg.n_layers * cfg.n_heads)
    print(f"\n  mean entropy over all {cfg.n_layers * cfg.n_heads} heads: "
          f"{mp:.3f} -> {mf:.3f} ({mf - mp:+.3f} bits)")

    focus = {"pretrained": answer_position_focus(pre, cfg, items, sp, device),
             "finetuned": answer_position_focus(fin, cfg, items, sp, device)}
    print(f"\n--- where the answer position looks (final layer, all heads) ---")
    keys = list(focus["pretrained"]["share"])
    print(f"  {'bucket':<16}{'pretrained':>12}{'finetuned':>12}{'delta':>9}")
    for k in keys:
        a, b = focus["pretrained"]["share"][k], focus["finetuned"]["share"][k]
        print(f"  {k:<16}{a:>11.2f}%{b:>11.2f}%{b - a:>+8.2f}")

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- heatmaps ---")
    figs = heatmaps({"pretrained": pre, "finetuned": fin}, cfg, items[0], sp,
                    device, out_dir, args.language)

    payload = {"language": args.language, "family": args.family,
               "prompts_averaged": len(items),
               "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
               "example_prompt": items[0]["prompt"],
               "example_answer": items[0]["answer"],
               "statistics": stats, "answer_position_focus": focus,
               "figures": figs}
    report_dir = (Path(args.out_dir).parent if args.out_dir
                  else REPO_ROOT / "report")
    out = report_dir / f"phase3_attn_reasoning_{args.language}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  {out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
