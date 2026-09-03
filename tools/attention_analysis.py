#!/usr/bin/env python3
"""
Attention analysis: what the heads actually learned.

The specification asks for this because we implemented multi-head attention
ourselves, so we should be able to say what it does rather than only that it
runs. Three measurements, plus heatmaps.

ATTENTION ENTROPY, per head
---------------------------
For one query position, the attention weights are a probability distribution over
the positions it may attend to. Its Shannon entropy says how spread out that
distribution is:

    H = -sum_j p_j log2 p_j     bits

Zero bits means the head attends to exactly one position. log2(k) bits means it
spreads evenly over k positions. So entropy separates *selective* heads from
*diffuse* ones, which is a distinction that is invisible in a loss curve.

Entropy has to be normalised before heads at different positions are compared.
Under causal masking, query position t can attend to t+1 positions, so its
maximum possible entropy is log2(t+1) - a head at position 5 simply cannot reach
the entropy of a head at position 400. We report raw entropy in bits and also
entropy divided by log2(t+1), which is comparable across positions and lands in
[0, 1].

MEAN ATTENTION DISTANCE
-----------------------
    d = sum_j p_j * (t - j)

the expected number of positions a query looks back. Small d means a local head
attending to recent tokens; large d means a head pulling information from far
away. Contrasting early and late layers on this is the clearest structural
statement the analysis can make.

WHAT TO EXPECT
--------------
Early layers usually hold low-entropy, short-distance heads doing positional work
- attend to the previous token, attend to the start. Later layers usually hold
higher-entropy, longer-distance heads doing content-based work. Whether Model L,
trained on a smaller and partly synthetic corpus, develops the same structure is
one of the more interesting things this project can report, and it is a genuine
question rather than one with a known answer.

USAGE
-----
    python3 tools/attention_analysis.py --language konkani \\
        --checkpoint ~/Desktop/phase2_checkpoints/konkani/pretrain_best.pt \\
        --tokenizer konkani/tokenizer/konkani_bpe.model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.data import PackedTokens, resolve_device        # noqa: E402
from common.model.config import ModelConfig                 # noqa: E402
from common.model.lm import DecoderLM                       # noqa: E402

# Heatmaps over a short window. A 512x512 heatmap is unreadable at figure size;
# 48 positions is enough to show the diagonal structure that matters.
HEATMAP_POSITIONS = 48


@torch.no_grad()
def collect(model, cfg, data: PackedTokens, device, n_sequences: int,
            seq_len: int):
    """Run held-out sequences and gather per-head entropy and distance.

    Statistics are accumulated over sequences rather than computed on one, so a
    single unusual passage cannot dominate. Position 0 is excluded throughout: it
    can only attend to itself, so its entropy is 0 and its distance is 0 by
    definition, and including it drags every head's average toward zero for a
    reason that has nothing to do with what the head learned.
    """
    high = len(data) - seq_len - 1
    starts = [int(i) for i in torch.linspace(0, high, n_sequences).tolist()]

    L, H = cfg.n_layers, cfg.n_heads
    ent_sum = np.zeros((L, H))
    ent_norm_sum = np.zeros((L, H))
    dist_sum = np.zeros((L, H))
    count = 0

    # Positions 1..T-1 and their maximum entropies, precomputed once.
    t_idx = torch.arange(seq_len, device=device)
    max_ent = torch.log2((t_idx + 1).float().clamp(min=1))

    for s in starts:
        x = torch.from_numpy(
            data.tokens[s:s + seq_len].astype("int64")).unsqueeze(0).to(device)
        _, _, attns = model(x, return_attention=True)

        for layer, attn in enumerate(attns):
            a = attn[0].float()                       # (H, T, T)

            # Entropy per query position. Masked positions hold exactly 0 after
            # softmax over -inf, and 0 * log 0 is 0 in the limit, so clamping
            # before the log keeps those terms at zero rather than producing nan.
            p = a.clamp_min(1e-12)
            ent = -(a * torch.log2(p)).sum(dim=-1)    # (H, T)

            # Mean attention distance: expected look-back in positions.
            offsets = (t_idx.view(1, -1, 1) - t_idx.view(1, 1, -1)).float()
            dist = (a * offsets).sum(dim=-1)          # (H, T)

            # Skip position 0 - see the docstring.
            ent_sum[layer] += ent[:, 1:].mean(dim=-1).cpu().numpy()
            ent_norm_sum[layer] += (ent[:, 1:]
                                    / max_ent[1:].unsqueeze(0)).mean(dim=-1).cpu().numpy()
            dist_sum[layer] += dist[:, 1:].mean(dim=-1).cpu().numpy()
        count += 1

    return {
        "entropy_bits": ent_sum / count,
        "entropy_normalised": ent_norm_sum / count,
        "mean_distance": dist_sum / count,
        "sequences": count,
        "seq_len": seq_len,
    }


@torch.no_grad()
def heatmaps(model, cfg, data: PackedTokens, sp, device, language: str,
             out_dir: Path) -> list[str]:
    """Save attention heatmaps for an early and a late layer.

    Every figure carries a title, both axis labels and a colourbar - the
    specification says plots missing labels or legends may score zero, and a
    heatmap without a colour scale is genuinely unreadable anyway.

    Axis ticks are token positions rather than Devanagari text: matplotlib needs
    a Devanagari-capable font to render the script, and a missing glyph silently
    becomes a box. The tokenised sentence is written into the JSON instead, so
    the report can show text and figure together.
    """
    import matplotlib
    matplotlib.use("Agg")           # no display on a headless machine
    import matplotlib.pyplot as plt

    T = HEATMAP_POSITIONS
    # A window from the middle of the split rather than the start: the splits are
    # source-stratified, so the first tokens are all one source.
    start = len(data) // 2
    ids = data.tokens[start:start + T].astype("int64")
    x = torch.from_numpy(ids).unsqueeze(0).to(device)
    _, _, attns = model(x, return_attention=True)

    pieces = [sp.id_to_piece(int(i)) for i in ids]
    layers = sorted({0, cfg.n_layers // 2, cfg.n_layers - 1})
    heads = list(range(min(4, cfg.n_heads)))
    saved = []

    for layer in layers:
        a = attns[layer][0].float().cpu().numpy()      # (H, T, T)
        fig, axes = plt.subplots(1, len(heads), figsize=(4.2 * len(heads), 4.4))
        axes = np.atleast_1d(axes)

        for col, head in enumerate(heads):
            ax = axes[col]
            im = ax.imshow(a[head], cmap="viridis", aspect="auto",
                           vmin=0.0, vmax=float(a[head].max()))
            ax.set_title(f"layer {layer}, head {head}")
            ax.set_xlabel("key position (attended to)")
            if col == 0:
                ax.set_ylabel("query position (attending)")
            fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")

        position = ("early" if layer == 0
                    else "late" if layer == cfg.n_layers - 1 else "middle")
        fig.suptitle(f"{language.capitalize()} - attention weights, "
                     f"{position} layer {layer} "
                     f"(lower triangle only: causal masking)")
        fig.tight_layout()

        path = out_dir / f"phase2_attention_{language}_layer{layer}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        # relpath rather than Path.relative_to: the latter raises when out_dir
        # was passed as a relative path, and a crash after the figure is
        # already written would be an absurd way to lose the run.
        saved.append(os.path.relpath(path.resolve(), REPO_ROOT))
        print(f"  {path.name}")

    return saved, pieces


def summarise(stats: dict, language: str) -> str:
    """A readable per-layer table, for the report."""
    ent, dist = stats["entropy_bits"], stats["mean_distance"]
    norm = stats["entropy_normalised"]
    lines = [f"### {language.capitalize()}", "",
             "| layer | mean entropy (bits) | normalised | mean distance | "
             "most local head | most global head |",
             "|---|---:|---:|---:|---|---|"]
    for layer in range(ent.shape[0]):
        local = int(np.argmin(dist[layer]))
        glob = int(np.argmax(dist[layer]))
        lines.append(
            f"| {layer} | {ent[layer].mean():.3f} | {norm[layer].mean():.3f} | "
            f"{dist[layer].mean():.2f} | head {local} "
            f"({dist[layer][local]:.2f}) | head {glob} "
            f"({dist[layer][glob]:.2f}) |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=["marathi", "konkani"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sequences", type=int, default=32,
                    help="held-out sequences to average statistics over")
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    import sentencepiece as spm

    device = resolve_device(args.device)
    fig_dir = REPO_ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"ATTENTION ANALYSIS - {args.language}")
    print("=" * 74)

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["model_config"])
    model = DecoderLM(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    data = PackedTokens(args.language, args.split, REPO_ROOT, args.data_dir)

    print(f"  device         {device}")
    print(f"  layers x heads {cfg.n_layers} x {cfg.n_heads}")
    print(f"  data           {data.describe()}")

    print("\n--- heatmaps ---")
    figures, pieces = heatmaps(model, cfg, data, sp, device, args.language,
                               fig_dir)

    print("\n--- per-head statistics ---")
    stats = collect(model, cfg, data, device, args.sequences, args.seq_len)
    print(summarise(stats, args.language))

    payload = {
        "language": args.language,
        "checkpoint": str(args.checkpoint),
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "sequences_averaged": stats["sequences"],
        "seq_len": stats["seq_len"],
        "entropy_bits": stats["entropy_bits"].tolist(),
        "entropy_normalised": stats["entropy_normalised"].tolist(),
        "mean_distance": stats["mean_distance"].tolist(),
        "figures": figures,
        "heatmap_tokens": pieces,
        "heatmap_text": sp.decode([sp.piece_to_id(p) for p in pieces]),
    }
    out = REPO_ROOT / "report" / f"phase2_attention_{args.language}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  {out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
