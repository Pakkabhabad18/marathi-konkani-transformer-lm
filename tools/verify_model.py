#!/usr/bin/env python3
"""
Correctness checks for the Transformer, run before anything touches a GPU.

WHY THIS EXISTS
---------------
Two of these checks are required by the specification: the exact parameter count
per model, and empirical proof that the causal mask works ("show that changing
token t+1 does not change the logits at position t"). The rest are here because
they catch the bugs that are otherwise invisible - a model with a leaking mask
still trains, still produces a falling loss curve, and still reports a
perplexity. It just reports one that is far too good, because it has been reading
the answer. That failure is silent, and it invalidates every number downstream.

Everything runs on CPU at a tiny configuration in a few seconds. Nothing goes to
Kaggle until this passes.

USAGE
-----
    python3 tools/verify_model.py
    python3 tools/verify_model.py --full     # also build the real 25M model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.model.config import KONKANI, MARATHI, TINY, ModelConfig  # noqa: E402
from common.model.lm import DecoderLM                                # noqa: E402

PASS, FAIL = "  PASS  ", "  FAIL  "
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL}{name}")
    if detail:
        print(f"          {detail}")
    return ok


def check_parameter_count(config: ModelConfig, label: str) -> None:
    """The analytic arithmetic and the real module tree must agree.

    ModelConfig.n_params() is used to choose the architecture before any tensor
    exists. If it disagrees with the built model, the config-based reasoning in
    the plan is wrong, and the report would quote a number the model does not
    have.
    """
    model = DecoderLM(config)
    analytic = config.n_params()["total"]
    actual = model.count_parameters()
    record(f"parameter count agrees with the arithmetic ({label})",
           analytic == actual,
           f"config says {analytic:,}, model has {actual:,}"
           + ("" if analytic == actual else f"  DIFFERENCE {actual - analytic:+,}"))
    if label != "tiny":
        within = 20_000_000 <= actual <= 30_000_000
        record(f"parameter count is near the ~25M target ({label})", within,
               f"{actual:,} trainable parameters")


def check_shapes(model: DecoderLM, config: ModelConfig) -> None:
    B, T = 3, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    logits, loss, attns = model(idx, targets=idx, return_attention=True)

    record("logits have shape (B, T, vocab_size)",
           tuple(logits.shape) == (B, T, config.vocab_size),
           f"got {tuple(logits.shape)}")
    record("loss is a finite scalar",
           loss.dim() == 0 and torch.isfinite(loss),
           f"loss {loss.item():.4f}")
    record("one attention tensor per layer",
           attns is not None and len(attns) == config.n_layers,
           f"got {len(attns) if attns else 0}, expected {config.n_layers}")
    record("attention has shape (B, heads, T, T)",
           tuple(attns[0].shape) == (B, config.n_heads, T, T),
           f"got {tuple(attns[0].shape)}")

    # An untrained model should sit near uniform over the vocabulary: the loss
    # should be about ln(vocab_size). Far below means information is leaking;
    # far above means initialisation is broken.
    import math
    expected = math.log(config.vocab_size)
    record("initial loss is near ln(vocab_size)",
           abs(loss.item() - expected) < 0.7,
           f"loss {loss.item():.4f}, ln({config.vocab_size}) = {expected:.4f}")


def check_attention_is_causal_by_construction(model: DecoderLM,
                                              config: ModelConfig) -> None:
    """Every attention row must be a distribution over the past only."""
    B, T = 2, 12
    idx = torch.randint(0, config.vocab_size, (B, T))
    model.eval()                       # dropout off, or rows will not sum to 1
    with torch.no_grad():
        _, _, attns = model(idx, return_attention=True)

    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    max_future = max(a[..., upper].abs().max().item() for a in attns)
    record("no attention weight on future positions", max_future == 0.0,
           f"largest weight above the diagonal: {max_future:.3e}")

    sums = torch.stack([a.sum(dim=-1) for a in attns])
    record("attention rows sum to 1",
           torch.allclose(sums, torch.ones_like(sums), atol=1e-5),
           f"max deviation {(sums - 1).abs().max().item():.3e}")


def check_future_tokens_cannot_change_the_past(model: DecoderLM,
                                               config: ModelConfig) -> None:
    """The test the specification asks for, stated in its own terms.

    Take a sequence, record the logits at every position, then change the token
    at position t+1 and run again. The logits at positions 0..t must be
    bit-identical, because none of them is allowed to have seen position t+1.
    Anything that leaks - a mask applied after the softmax, an off-by-one in the
    triangle, a mask built at the wrong length - shows up here as a non-zero
    difference.
    """
    B, T = 1, 24
    model.eval()
    idx = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        before, _, _ = model(idx)

    worst_leak, worst_position = 0.0, -1
    for t in range(T - 1):
        altered = idx.clone()
        # Change the token at t+1 to something definitely different.
        altered[0, t + 1] = (altered[0, t + 1] + 1) % config.vocab_size
        with torch.no_grad():
            after, _, _ = model(altered)
        # Positions 0..t must be untouched.
        delta = (before[0, :t + 1] - after[0, :t + 1]).abs().max().item()
        if delta > worst_leak:
            worst_leak, worst_position = delta, t

    record("changing token t+1 does not change logits at position t",
           worst_leak == 0.0,
           f"largest change across all t: {worst_leak:.3e}"
           + (f" at t={worst_position}" if worst_leak else ""))

    # The complement: changing a token at or before t MUST change the logits at
    # t. A model that passes the test above by ignoring its input entirely would
    # otherwise look correct.
    altered = idx.clone()
    altered[0, 0] = (altered[0, 0] + 1) % config.vocab_size
    with torch.no_grad():
        after, _, _ = model(altered)
    changed = (before[0, -1] - after[0, -1]).abs().max().item()
    record("changing token 0 does change logits at the last position",
           changed > 0.0,
           f"change {changed:.3e} (a zero here would mean the model ignores input)")


def check_can_overfit(config: ModelConfig, steps: int = 220) -> None:
    """Train the tiny model on one fixed batch until it memorises it.

    A model that cannot drive the loss to near zero on a batch it sees hundreds
    of times has a bug in the backward path - a detached tensor, a frozen
    parameter, an optimizer that is not stepping. This is the cheapest possible
    end-to-end test that learning works at all, and it runs in seconds on CPU.
    """
    torch.manual_seed(0)
    model = DecoderLM(config)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    B, T = 4, 32
    idx = torch.randint(0, config.vocab_size, (B, T + 1))
    x, y = idx[:, :-1], idx[:, 1:]

    started = time.time()
    first = last = None
    for step in range(steps):
        _, loss, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    record("tiny model can overfit one batch", last < 0.15,
           f"loss {first:.4f} -> {last:.4f} in {steps} steps "
           f"({time.time() - started:.1f}s)")


def check_gradients_reach_everything(config: ModelConfig) -> None:
    """Every trainable parameter must receive a gradient.

    A parameter with no gradient after backward is dead weight: it is counted in
    the parameter total, reported in the config, and never learns anything. This
    usually means a module was built but never called.
    """
    model = DecoderLM(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 16))
    _, loss, _ = model(idx, targets=idx)
    loss.backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    record("every parameter receives a gradient", not missing,
           "all reached" if not missing
           else f"{len(missing)} without gradient: {missing[:5]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also build the real ~25M model (slower, more memory)")
    args = ap.parse_args()

    torch.manual_seed(1234)

    print("=" * 70)
    print("MODEL VERIFICATION")
    print("=" * 70)
    print("\nTiny configuration used for the behavioural checks:")
    print(TINY.summary())
    print()

    print("-" * 70)
    check_parameter_count(TINY, "tiny")

    model = DecoderLM(TINY)
    check_shapes(model, TINY)
    check_attention_is_causal_by_construction(model, TINY)
    check_future_tokens_cannot_change_the_past(model, TINY)
    check_gradients_reach_everything(TINY)
    check_can_overfit(TINY)

    if args.full:
        print("-" * 70)
        print("Real configuration:")
        print(MARATHI.summary())
        print()
        check_parameter_count(MARATHI, "marathi")
        check_parameter_count(KONKANI, "konkani")
        big = DecoderLM(MARATHI)
        check_future_tokens_cannot_change_the_past(big, MARATHI)

    print("=" * 70)
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nFAILED:")
        for name in failed:
            print(f"  - {name}")
        print("\nDo not start a training run until these pass.")
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
