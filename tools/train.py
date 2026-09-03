#!/usr/bin/env python3
"""
Pretrain one decoder-only language model.

Model H and Model L are trained by separate invocations of this script, on
separate data, into separate checkpoints. Nothing is shared at runtime and
neither model is ever initialised from the other.

CHECKPOINTING IS THE LOAD-BEARING PART
--------------------------------------
The specification requires every checkpoint to hold model weights, optimizer
state, scheduler state, the current step and the configuration, and requires that
a run resume after interruption. On free-tier compute that is not a formality:
Kaggle sessions have a hard ceiling and Colab ends sessions without warning, so a
run that cannot resume is a run that may never finish.

Two details make it actually work rather than nominally work:

  * Checkpoints are written to a temporary file and then renamed. Rename is
    atomic on POSIX, so a session killed mid-write leaves the previous checkpoint
    intact instead of a truncated file that loads as garbage.
  * Optimizer state is saved, not just weights. AdamW carries first and second
    moment estimates per parameter; resuming without them restarts the optimizer
    cold and produces a visible loss spike. If a resumed run spikes, that is the
    first thing to check.

Resume is tested by killing a run and restarting it (--smoke covers this in
seconds), because a checkpoint that has never been resumed from is a checkpoint
that does not work.

USAGE
-----
    # correctness, on CPU, in about a minute
    python3 tools/train.py --language konkani --smoke

    # the real run
    python3 tools/train.py --language konkani --max-tokens 500000000
    python3 tools/train.py --language marathi  --max-tokens 500000000

    # after an interrupted session - same command, it finds its own checkpoint
    python3 tools/train.py --language konkani --max-tokens 500000000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.data import (FixedWindowSampler, PackedTokens,      # noqa: E402
                         RandomWindowSampler, resolve_device)
from common.model.config import ModelConfig                      # noqa: E402
from common.model.lm import DecoderLM                            # noqa: E402


@dataclass
class TrainConfig:
    """Everything that determines a run, saved into every checkpoint."""

    language: str = "konkani"
    max_tokens: int = 500_000_000

    # Micro-batch: what one forward/backward actually holds. Chosen to fit GPU
    # memory, and it is the only value here that should change with hardware.
    batch_size: int = 32
    context_length: int = 512

    # Gradient accumulation makes the *effective* batch independent of what fits
    # in memory. 8 x 32 x 512 = 131,072 tokens per optimizer step. Keeping this
    # fixed is what makes a run on a T4 comparable with a run anywhere else -
    # otherwise the learning rate is being tuned against the hardware.
    grad_accum_steps: int = 8

    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1          # cosine floor, as a fraction of peak
    warmup_ratio: float = 0.02         # of total steps
    weight_decay: float = 0.1
    beta1: float = 0.9
    # 0.95 rather than the 0.999 default: a shorter second-moment memory tracks
    # the fast early drop in gradient scale during LM pretraining. 0.999 adapts
    # too slowly and the first few hundred steps take oversized steps.
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_every: int = 250              # optimizer steps
    eval_windows: int = 512
    checkpoint_every: int = 250        # optimizer steps
    # ...and also on a wall-clock timer. A step counter alone is not enough on
    # free-tier compute: if a session dies before the first step-based
    # checkpoint, the whole run is lost, and how many steps that is depends
    # entirely on the hardware. The timer bounds the loss to `checkpoint_every_
    # minutes` of work regardless of how fast the GPU happens to be. Found by
    # killing a run 15 seconds in and discovering there was nothing to resume.
    checkpoint_every_minutes: float = 10.0
    log_every: int = 10

    seed: int = 20260827
    amp: bool = True                   # mixed precision, roughly 2x on a T4

    def tokens_per_step(self) -> int:
        return self.batch_size * self.context_length * self.grad_accum_steps

    def total_steps(self) -> int:
        return max(1, self.max_tokens // self.tokens_per_step())


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to a floor.

    Warmup exists because AdamW's second-moment estimate is near zero at step 0,
    so its first updates are enormous relative to the weights. Ramping the rate
    from zero over the first 2% of steps lets those estimates stabilise before
    full-size steps are taken; without it a randomly initialised model can move
    somewhere it never recovers from in the first dozen updates.

    Cosine decay to 10% rather than to zero: the last steps still make progress,
    and a non-zero floor leaves the model in a usable state if the run is cut
    short.
    """
    total = cfg.total_steps()
    warmup = max(1, int(total * cfg.warmup_ratio))
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = cfg.learning_rate * cfg.min_lr_ratio
    return floor + (cfg.learning_rate - floor) * cosine


def build_optimizer(model: DecoderLM, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    Biases, LayerNorm gains and embeddings are excluded. Decay is a prior that
    smaller weights generalise better, which makes sense for a projection matrix
    and does not for an embedding row: decaying embeddings pulls rare tokens
    toward the origin simply because they appear rarely, which is the opposite of
    what is wanted.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=1e-8)


@torch.no_grad()
def evaluate(model: DecoderLM, sampler: FixedWindowSampler,
             device: torch.device, amp: bool) -> dict:
    """Mean cross-entropy over the fixed validation windows, plus perplexity.

    Model back to train() by the caller. Dropout must be off here or the reported
    loss is noisier than the model actually is.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    autocast = torch.autocast(device_type=device.type, dtype=torch.float16,
                              enabled=amp and device.type == "cuda")
    for x, y in sampler:
        with autocast:
            _, loss, _ = model(x, targets=y)
        total_loss += loss.float().item()
        n_batches += 1
    model.train()

    mean = total_loss / max(1, n_batches)
    return {
        "val_loss": mean,
        # PPL = e^loss. Comparable within a language; NOT comparable between
        # Model H and Model L, because their tokenizers segment differently.
        # Bits-per-byte in the evaluation script is the cross-language measure.
        "val_ppl": math.exp(min(mean, 20.0)),
    }


def atomic_save(payload: dict, path: Path) -> None:
    """Write, flush, fsync, then rename.

    torch.save straight to the destination leaves a truncated file if the process
    dies mid-write - and on Kaggle it will eventually die mid-write. The rename
    is atomic, so the destination is always either the old checkpoint or the
    complete new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", required=True, choices=["marathi", "konkani"])
    ap.add_argument("--max-tokens", type=int, default=500_000_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", default=None,
                    help="where the packed .bin files live; on Kaggle this is "
                         "the attached dataset path, e.g. /kaggle/input/...")
    ap.add_argument("--out-dir", default=None,
                    help="where checkpoints and logs go (default: "
                         "<language>/model). On Kaggle use /kaggle/working/...")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model, few steps, CPU-friendly: proves the loop "
                         "and the resume path work before spending GPU quota")
    args = ap.parse_args()

    cfg = TrainConfig(language=args.language, max_tokens=args.max_tokens,
                      batch_size=args.batch_size, grad_accum_steps=args.grad_accum,
                      learning_rate=args.lr, amp=not args.no_amp)

    model_cfg = ModelConfig(vocab_size=2500)
    if args.smoke:
        # Small enough to run through the whole loop, including a checkpoint and
        # an evaluation, in about a minute on a laptop CPU.
        model_cfg = ModelConfig(vocab_size=2500, d_model=128, n_layers=2,
                                n_heads=4, d_ff=512, context_length=64,
                                dropout=0.0)
        cfg.context_length = 64
        cfg.batch_size, cfg.grad_accum_steps = 8, 2
        cfg.max_tokens = 8 * 64 * 2 * 40          # 40 optimizer steps
        cfg.eval_every = cfg.checkpoint_every = 10
        cfg.checkpoint_every_minutes = 0.05      # 3 seconds, to exercise the timer
        cfg.log_every = 5
        cfg.eval_windows = 64
        cfg.amp = False

    torch.manual_seed(cfg.seed)
    device = resolve_device(args.device)

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / args.language / "model"
    run_name = "smoke" if args.smoke else "pretrain"
    ckpt_path = out_dir / f"{run_name}_latest.pt"
    best_path = out_dir / f"{run_name}_best.pt"
    log_path = out_dir / f"{run_name}_log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"PRETRAIN - {args.language}  ({'smoke test' if args.smoke else 'full run'})")
    print("=" * 74)
    print(f"  device            {device}")
    if device.type == "cuda":
        print(f"  gpu               {torch.cuda.get_device_name(0)}")

    train_data = PackedTokens(args.language, "train", REPO_ROOT, args.data_dir)
    val_data = PackedTokens(args.language, "val", REPO_ROOT, args.data_dir)
    print(f"  train             {train_data.describe()}")
    print(f"  val               {val_data.describe()}")

    if model_cfg.vocab_size != train_data.vocab_size:
        raise SystemExit(
            f"Config vocab {model_cfg.vocab_size} does not match the packed data "
            f"vocab {train_data.vocab_size}. The tokenizer and the model would "
            f"disagree about what every id means.")

    model = DecoderLM(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    # GradScaler keeps fp16 gradients from underflowing to zero by scaling the
    # loss up before backward and unscaling before the optimizer step.
    # torch.amp.GradScaler (not torch.cuda.amp) - the latter is deprecated and
    # warns on every run, which buries real warnings in the training log.
    scaler = torch.amp.GradScaler(device.type,
                                  enabled=cfg.amp and device.type == "cuda")

    total_steps = cfg.total_steps()
    print(f"  parameters        {model.count_parameters():,}")
    print(f"  tokens/step       {cfg.tokens_per_step():,} "
          f"({cfg.batch_size} x {cfg.context_length} x {cfg.grad_accum_steps})")
    print(f"  optimizer steps   {total_steps:,}")
    print(f"  token budget      {cfg.max_tokens:,}")
    print(f"  checkpoints       {ckpt_path.relative_to(Path.cwd()) if ckpt_path.is_relative_to(Path.cwd()) else ckpt_path}")

    start_step, best_val = 0, float("inf")
    if ckpt_path.exists() and not args.no_resume:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        if blob.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(blob["scaler"])
        start_step = blob["step"]
        best_val = blob.get("best_val", float("inf"))
        print(f"  RESUMED           step {start_step:,} of {total_steps:,}, "
              f"best val loss {best_val:.4f}")

    train_sampler = RandomWindowSampler(train_data, cfg.batch_size,
                                        cfg.context_length, device, cfg.seed)
    # Replay the sampler past completed steps so a resumed run does not retrain
    # on the same windows it already saw.
    if start_step:
        train_sampler.skip(start_step * cfg.grad_accum_steps)
    val_sampler = FixedWindowSampler(val_data, cfg.batch_size,
                                     cfg.context_length, device, cfg.eval_windows)

    if start_step >= total_steps:
        print("\n  Budget already reached; nothing to do.")
        return 0

    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "tokens", "train_loss", "val_loss", "val_ppl",
                      "lr", "grad_norm", "seconds", "tokens_per_sec"])

    autocast = lambda: torch.autocast(device_type=device.type,      # noqa: E731
                                      dtype=torch.float16,
                                      enabled=cfg.amp and device.type == "cuda")

    print("-" * 74)
    model.train()
    run_started = time.time()
    step_started = time.time()
    last_checkpoint = time.time()

    def save(path: Path, step: int) -> None:
        """Write a full checkpoint: weights, optimizer, scaler, step, configs."""
        atomic_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "step": step, "best_val": best_val,
            "model_config": model_cfg.to_dict(),
            "train_config": asdict(cfg),
        }, path)

    for step in range(start_step, total_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation: several forward/backward passes, one optimizer
        # step. Each micro-batch's loss is divided by the accumulation count so
        # the accumulated gradient equals the gradient of the mean loss over the
        # full effective batch - without the division the effective learning rate
        # would scale with grad_accum_steps.
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_sampler.batch()
            with autocast():
                _, loss, _ = model(x, targets=y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            accumulated += loss.float().item()

        # Unscale before clipping: clipping a scaled gradient clips the wrong
        # magnitude and the threshold would depend on the loss scale.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_seen = (step + 1) * cfg.tokens_per_step()

        if (step + 1) % cfg.log_every == 0 or step == start_step:
            elapsed = time.time() - step_started
            rate = cfg.tokens_per_step() * cfg.log_every / max(elapsed, 1e-9)
            pct = 100.0 * (step + 1) / total_steps
            eta = (total_steps - step - 1) * (elapsed / cfg.log_every) / 3600
            print(f"  step {step + 1:>6,}/{total_steps:,} ({pct:4.1f}%)  "
                  f"loss {accumulated:6.4f}  lr {lr:.2e}  "
                  f"|g| {grad_norm:5.2f}  {rate/1e3:6.1f}k tok/s  "
                  f"eta {eta:4.1f}h", flush=True)
            log.writerow([step + 1, tokens_seen, f"{accumulated:.6f}", "", "",
                          f"{lr:.6e}", f"{float(grad_norm):.4f}",
                          f"{time.time() - run_started:.1f}", f"{rate:.0f}"])
            log_file.flush()
            step_started = time.time()

        if (step + 1) % cfg.eval_every == 0 or step + 1 == total_steps:
            metrics = evaluate(model, val_sampler, device, cfg.amp)
            print(f"    val loss {metrics['val_loss']:.4f}  "
                  f"ppl {metrics['val_ppl']:.2f}", flush=True)
            log.writerow([step + 1, tokens_seen, "", f"{metrics['val_loss']:.6f}",
                          f"{metrics['val_ppl']:.4f}", f"{lr:.6e}", "",
                          f"{time.time() - run_started:.1f}", ""])
            log_file.flush()

            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                save(best_path, step + 1)

        # Checkpoint on whichever comes first: the step interval, the wall-clock
        # timer, or the end of the run. The first step also triggers one, so a
        # broken save path fails in the first seconds rather than after an hour.
        due = ((step + 1) % cfg.checkpoint_every == 0
               or step + 1 == total_steps
               or step == start_step
               or (time.time() - last_checkpoint)
               >= cfg.checkpoint_every_minutes * 60)
        if due:
            save(ckpt_path, step + 1)
            last_checkpoint = time.time()

    log_file.close()
    (out_dir / f"{run_name}_config.json").write_text(
        json.dumps({"model": model_cfg.to_dict(), "training": asdict(cfg)},
                   indent=2), encoding="utf-8")

    hours = (time.time() - run_started) / 3600
    print("-" * 74)
    print(f"  finished in {hours:.2f} h, best val loss {best_val:.4f} "
          f"(ppl {math.exp(min(best_val, 20.0)):.2f})")
    print(f"  {ckpt_path}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
