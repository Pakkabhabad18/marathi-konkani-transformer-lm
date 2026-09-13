#!/usr/bin/env python3
"""
Supervised finetuning on the synthetic reasoning data.

Phase 3 asks for each pretrained model to be finetuned on reasoning tasks in its
own language, starting from that language's own checkpoint, with the tokenizer
and vocabulary held fixed and checkpoints saved in the same resume-capable format
as pretraining. This script does that.

    python3 tools/finetune.py --language konkani \\
        --pretrained ~/Desktop/phase2_checkpoints/konkani/pretrain_best.pt \\
        --epochs 3

WHAT MAKES THIS DIFFERENT FROM PRETRAINING
------------------------------------------
Pretraining computes loss at every position: every token is a training example.
That is wrong here. An example is

    प्रश्न: <facts> <question>   उत्तर: राम

and the model is not supposed to learn to produce the question - the question is
given to it at test time. Training on those positions spends capacity modelling
the template and dilutes the gradient from the part that matters.

So the loss is MASKED over the prompt. Targets for prompt positions are set to
-100, which `F.cross_entropy` ignores, and the mean is taken over answer
positions only. The boundary is exact because the generator stores the prompt and
the completion as separate strings rather than one blob that has to be re-split by
searching for a marker.

One subtlety worth stating out loud: the FIRST completion token is predicted from
the LAST prompt token, so the mask starts one position earlier than the naive
split. Getting that off by one would silently drop the single most important
prediction in every example - the one that decides the answer.

WHY A LOWER LEARNING RATE THAN PRETRAINING
------------------------------------------
3e-4 built these representations from noise over 3,814 steps. Finetuning runs a
few hundred steps on a narrow distribution, and at the pretraining rate it
overwrites the language model with the template - the model learns to emit
"उत्तर: राम" for anything. 1e-4 with warmup adapts without erasing. Whether that
is enough is not assumed: `tools/evaluate_reasoning.py` re-measures held-out
pretraining perplexity after finetuning, so forgetting is reported rather than
hoped about.

SEQUENCE LENGTH
---------------
Items are 38-96 tokens where pretraining used 512. Padding every example to 512
would waste roughly 80% of each batch on padding, so sequences are padded to the
longest item in the batch instead, and padded positions are masked out of the loss
along with the prompt. The model handles T < context_length without any change
because positional embeddings are looked up per position.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.data import resolve_device                            # noqa: E402
from common.model.config import ModelConfig                       # noqa: E402
from common.model.lm import DecoderLM                             # noqa: E402

IGNORE = -100          # F.cross_entropy skips these positions
PAD_ID = 3             # matches the tokenizer: unk 0, bos 1, eos 2, pad 3
EOS_ID = 2


@dataclass
class FinetuneConfig:
    """Everything that determines a finetuning run, saved into every checkpoint."""

    language: str = "konkani"
    epochs: int = 3
    batch_size: int = 32
    max_length: int = 160             # generous: longest observed item is 96

    # An order of magnitude below pretraining. See the module docstring.
    learning_rate: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05        # a larger fraction than pretraining, because
                                      # the whole run is only a few hundred steps
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    with_rationale: bool = False      # train on the chain-of-thought variant
    n_train: int = 0                  # 0 = all; set for the sample-count sweep

    eval_every: int = 50
    checkpoint_every: int = 200
    checkpoint_every_minutes: float = 10.0
    log_every: int = 10

    seed: int = 20260913
    amp: bool = True


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_items(language: str, split: str, data_dir: str | None) -> list[dict]:
    base = Path(data_dir) if data_dir else REPO_ROOT / language / "data" / "reasoning"
    path = base / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run tools/make_reasoning_data.py --language "
            f"{language} first, or pass --data-dir.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def encode(items: list[dict], sp, cfg: FinetuneConfig) -> list[dict]:
    """Tokenise each item into ids plus the index where the answer begins.

    The prompt and the completion are encoded SEPARATELY and then concatenated.
    Encoding the joined string and hunting for the marker afterwards would be
    fragile: BPE can merge the last prompt piece with the first completion piece,
    and the boundary would land in the middle of a token.
    """
    out, dropped = [], 0
    for it in items:
        completion = (f"{it['rationale']} {it['answer_marker']} {it['answer']}"
                      if cfg.with_rationale and it.get("rationale")
                      else f"{it['answer_marker']} {it['answer']}")
        p_ids = sp.encode(it["prompt"])
        c_ids = sp.encode(completion) + [EOS_ID]
        ids = p_ids + c_ids
        if len(ids) > cfg.max_length:
            dropped += 1
            continue
        out.append({"ids": ids, "n_prompt": len(p_ids),
                    "answer": it["answer"], "family": it["family"],
                    "pattern": it["pattern"], "id": it["id"]})
    if dropped:
        print(f"  dropped {dropped} items longer than max_length={cfg.max_length}")
    return out


def make_batch(rows: list[dict], device: torch.device):
    """Pad to the longest row, build inputs and masked targets.

    Position j of x predicts ids[j+1]. A target is kept only when ids[j+1] is a
    completion token, i.e. j + 1 >= n_prompt. So the first kept target index is
    n_prompt - 1, which is the position of the LAST PROMPT TOKEN - that is the
    step which predicts the first answer token.
    """
    width = max(len(r["ids"]) for r in rows)
    x = torch.full((len(rows), width - 1), PAD_ID, dtype=torch.long)
    y = torch.full((len(rows), width - 1), IGNORE, dtype=torch.long)

    for i, r in enumerate(rows):
        ids = r["ids"]
        n = len(ids)
        x[i, :n - 1] = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        keep = torch.zeros(n - 1, dtype=torch.bool)
        keep[max(0, r["n_prompt"] - 1):] = True      # answer span only
        y[i, :n - 1] = torch.where(keep, targets,
                                   torch.full_like(targets, IGNORE))
    return x.to(device), y.to(device)


def batches(rows: list[dict], cfg: FinetuneConfig, device: torch.device,
            shuffle: bool, rng: random.Random | None = None):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), cfg.batch_size):
        chunk = [rows[j] for j in order[i:i + cfg.batch_size]]
        yield make_batch(chunk, device)


# --------------------------------------------------------------------------
# Schedule, optimizer, checkpoints - same shapes as pretraining
# --------------------------------------------------------------------------
def lr_at(step: int, total: int, cfg: FinetuneConfig) -> float:
    warmup = max(1, int(total * cfg.warmup_ratio))
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    floor = cfg.learning_rate * cfg.min_lr_ratio
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (cfg.learning_rate - floor) * cosine


def build_optimizer(model: DecoderLM, cfg: FinetuneConfig) -> torch.optim.AdamW:
    """Identical grouping rule to pretraining: decay matrices, not biases,
    LayerNorm gains or embeddings."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if (param.dim() < 2 or "embedding" in name) else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=1e-8)


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


@torch.no_grad()
def evaluate(model, rows, cfg, device) -> dict:
    """Mean answer-span loss over the validation items.

    Batches differ in how many answer tokens they contain, so this sums the
    per-token loss and divides once at the end. Averaging per-batch means would
    weight a batch of short answers the same as a batch of long ones.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in batches(rows, cfg, device, shuffle=False):
        logits, _, _ = model(x)
        n = int((y != IGNORE).sum())
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), reduction="sum")
        total_loss += loss.float().item()
        total_tokens += n
    model.train()
    mean = total_loss / max(1, total_tokens)
    return {"val_loss": mean, "val_ppl": math.exp(min(mean, 20.0)),
            "val_answer_tokens": total_tokens}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=["marathi", "konkani"])
    ap.add_argument("--pretrained", required=True,
                    help="that language's own pretrain_best.pt")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=0,
                    help="use only the first N training items; 0 = all. This is "
                         "what the sample-count sweep varies.")
    ap.add_argument("--rationale", action="store_true",
                    help="train on the chain-of-thought variant")
    ap.add_argument("--tag", default=None,
                    help="suffix for checkpoint and log names, so sweep runs do "
                         "not overwrite each other")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    import sentencepiece as spm

    cfg = FinetuneConfig(language=args.language, epochs=args.epochs,
                         batch_size=args.batch_size, learning_rate=args.lr,
                         with_rationale=args.rationale, n_train=args.n_train,
                         amp=not args.no_amp)
    torch.manual_seed(cfg.seed)
    device = resolve_device(args.device)

    tok_path = args.tokenizer or (REPO_ROOT / args.language / "tokenizer" /
                                  f"{args.language}_bpe.model")
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / args.language / "model"
    tag = args.tag or ("rationale" if cfg.with_rationale else "plain")
    run = f"finetune_{tag}"
    ckpt_path, best_path = out_dir / f"{run}_latest.pt", out_dir / f"{run}_best.pt"
    log_path = out_dir / f"{run}_log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"FINETUNE - {args.language}  ({'with rationale' if cfg.with_rationale else 'answer only'})")
    print("=" * 74)
    print(f"  device            {device}")

    # ---- the pretrained model, rebuilt from the config inside its own checkpoint
    blob = torch.load(args.pretrained, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**blob["model_config"])
    if sp.get_piece_size() != model_cfg.vocab_size:
        raise SystemExit(
            f"Tokenizer vocabulary {sp.get_piece_size()} does not match the "
            f"checkpoint's {model_cfg.vocab_size}. Phase 3 requires the tokenizer "
            f"to stay fixed, so this is a wiring error, not something to override.")
    model = DecoderLM(model_cfg).to(device)
    model.load_state_dict(blob["model"])
    model.train()
    print(f"  pretrained from   {args.pretrained}")
    print(f"  pretrained steps  {blob['step']:,}  (val loss {blob.get('best_val', float('nan')):.4f})")
    print(f"  parameters        {model.count_parameters():,}")

    # A FRESH optimizer. The pretraining moment estimates describe a 3e-4 run on a
    # different distribution; carrying them in would make the first finetuning
    # steps badly scaled.
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(device.type,
                                  enabled=cfg.amp and device.type == "cuda")

    train_rows = encode(load_items(args.language, "train", args.data_dir), sp, cfg)
    val_rows = encode(load_items(args.language, "val", args.data_dir), sp, cfg)
    if cfg.n_train:
        train_rows = train_rows[:cfg.n_train]
    lengths = sorted(len(r["ids"]) for r in train_rows)
    print(f"  train items       {len(train_rows):,}")
    print(f"  val items         {len(val_rows):,}")
    print(f"  tokens/item       min {lengths[0]}  median {lengths[len(lengths)//2]}  max {lengths[-1]}")

    steps_per_epoch = math.ceil(len(train_rows) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    print(f"  steps/epoch       {steps_per_epoch:,}")
    print(f"  total steps       {total_steps:,}  ({cfg.epochs} epochs)")
    print(f"  peak lr           {cfg.learning_rate:g}")

    start_step, best_val = 0, float("inf")
    if ckpt_path.exists() and not args.no_resume:
        r = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(r["model"])
        optimizer.load_state_dict(r["optimizer"])
        if r.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(r["scaler"])
        start_step, best_val = r["step"], r.get("best_val", float("inf"))
        print(f"  RESUMED           step {start_step:,} of {total_steps:,}")
    if start_step >= total_steps:
        print("\n  Already complete; nothing to do.")
        return 0

    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "epoch", "train_loss", "val_loss", "val_ppl",
                      "lr", "grad_norm", "seconds"])

    def save(path: Path, step: int, best: float) -> None:
        """Same payload shape as pretraining, so the same loader reads both and a
        finetuned run resumes exactly like a pretraining run."""
        atomic_save({"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                     "step": step, "best_val": best,
                     "model_config": asdict(model_cfg),
                     "train_config": asdict(cfg),
                     "pretrained_from": str(args.pretrained),
                     "phase": "finetune"}, path)

    autocast = torch.autocast(device_type=device.type, dtype=torch.float16,
                              enabled=cfg.amp and device.type == "cuda")
    started = time.time()
    last_ckpt = time.time()
    step = start_step

    for epoch in range(cfg.epochs):
        # Seeded per epoch, so the shuffle for epoch k is identical whether the
        # run reaches it straight through or after a resume.
        epoch_rng = random.Random(cfg.seed + epoch)
        if (epoch + 1) * steps_per_epoch <= step:
            continue                          # this epoch finished before the interruption
        skip_in_epoch = max(0, step - epoch * steps_per_epoch)

        for batch_index, (x, y) in enumerate(
                batches(train_rows, cfg, device, shuffle=True, rng=epoch_rng)):
            # Replay past the batches this epoch already consumed, so a resumed
            # run does not retrain on the same items - the same reason
            # pretraining calls sampler.skip().
            if batch_index < skip_in_epoch:
                continue
            if step >= total_steps:
                break
            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with autocast:
                logits, _, _ = model(x)
                # Mean over answer tokens only; IGNORE positions are skipped.
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % cfg.log_every == 0:
                elapsed = time.time() - started
                print(f"  step {step:>5}/{total_steps}  epoch {epoch}  "
                      f"loss {loss.item():.4f}  lr {lr:.2e}  "
                      f"gn {grad_norm:.3f}  {elapsed:.0f}s", flush=True)
                log.writerow([step, epoch, f"{loss.item():.6f}", "", "",
                              f"{lr:.6e}", f"{grad_norm:.4f}", f"{elapsed:.1f}"])
                log_file.flush()

            if step % cfg.eval_every == 0 or step == total_steps:
                stats = evaluate(model, val_rows, cfg, device)
                print(f"    val loss {stats['val_loss']:.4f}  "
                      f"ppl {stats['val_ppl']:.3f}  "
                      f"over {stats['val_answer_tokens']:,} answer tokens",
                      flush=True)
                log.writerow([step, epoch, "", f"{stats['val_loss']:.6f}",
                              f"{stats['val_ppl']:.4f}", f"{lr:.6e}", "",
                              f"{time.time() - started:.1f}"])
                log_file.flush()
                if stats["val_loss"] < best_val:
                    best_val = stats["val_loss"]
                    save(best_path, step, best_val)

            due = (step % cfg.checkpoint_every == 0 or step == total_steps
                   or step == start_step + 1
                   or time.time() - last_ckpt >= cfg.checkpoint_every_minutes * 60)
            if due:
                save(ckpt_path, step, best_val)
                last_ckpt = time.time()

    save(ckpt_path, step, best_val)
    log_file.close()
    print(f"\n  best val loss     {best_val:.4f}")
    print(f"  best checkpoint   {best_path}")
    print(f"  log               {log_path}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
