#!/usr/bin/env python3
"""
Batching for language-model pretraining, over the memory-mapped token arrays
produced by tools/pack_tokens.py.

WHY MEMORY-MAPPING RATHER THAN LOADING
--------------------------------------
Each train split is a 1.0 GB flat uint16 array. np.memmap leaves it on disk and
lets the operating system page in only the windows actually touched, so the
process never holds the corpus in RAM. On a Kaggle session with limited memory
that is the difference between running and being killed. It also makes start-up
instant: no parsing, no decoding, no per-document Python object.

WHY RANDOM WINDOWS FOR TRAINING AND FIXED WINDOWS FOR VALIDATION
-----------------------------------------------------------------
Training samples a uniformly random start offset for every sequence in every
batch. There is no epoch and no shuffle buffer - with 500M tokens and a 512-token
context there are effectively half a billion distinct windows, so sampling with
replacement covers the corpus without the bookkeeping. It also means a resumed
run does not need to know where in an epoch it stopped.

Validation must be the opposite: the same windows every time. If validation
sampled randomly, the val loss would move between evaluations because the text
changed, not because the model did, and the loss curve would be unreadable. So
validation windows are evenly spaced across the split and fixed for the whole
run.

WHY x AND y OVERLAP BY ALL BUT ONE TOKEN
----------------------------------------
The objective is next-token prediction at *every* position, so for a window of
length T+1 the inputs are tokens 0..T-1 and the targets are tokens 1..T. Position
i predicts token i+1. Every one of the T positions is a training example, which
is why a 512-token window yields 512 supervised predictions rather than one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class PackedTokens:
    """A memory-mapped split, with the metadata pack_tokens.py recorded."""

    def __init__(self, language: str, split: str, repo_root: Path | str,
                 data_dir: Path | str | None = None) -> None:
        """
        Args:
            data_dir: overrides where the packed files live. Needed on Kaggle,
                where the token arrays arrive as a read-only attached dataset at
                /kaggle/input/... rather than inside the repository.
        """
        root = Path(repo_root)
        base = Path(data_dir) if data_dir else root / language / "data" / "packed"
        self.bin_path = base / f"{split}.bin"
        self.meta_path = base / f"{split}.json"

        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"{self.bin_path} not found. Run:\n"
                f"    python3 tools/pack_tokens.py --language {language} "
                f"--split {split}")

        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.language, self.split = language, split

        # mode="r" is read-only: a bug in the training loop cannot corrupt the
        # corpus, and the same file can back several processes.
        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")

        if len(self.tokens) != self.meta["tokens"]:
            raise ValueError(
                f"{self.bin_path.name} holds {len(self.tokens):,} tokens but its "
                f"metadata claims {self.meta['tokens']:,}. The file is truncated "
                f"or the metadata is stale; re-pack this split.")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def vocab_size(self) -> int:
        return self.meta["vocab_size"]

    @property
    def eos_id(self) -> int:
        return self.meta["eos_id"]

    def describe(self) -> str:
        return (f"{self.language}/{self.split}: {len(self):,} tokens, "
                f"{self.meta['documents']:,} documents, "
                f"vocab {self.vocab_size:,}")


def _windows_to_tensors(tokens: np.memmap, starts: np.ndarray, context: int,
                        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice windows at the given offsets and split each into (input, target).

    The uint16 ids are widened to int64 because nn.Embedding and cross_entropy
    both index with long tensors. The copy is unavoidable and is why the slice is
    done in numpy first: one contiguous read per window, then a single conversion
    for the whole batch.
    """
    batch = np.stack([tokens[s:s + context + 1] for s in starts]).astype(np.int64)
    data = torch.from_numpy(batch)
    x = data[:, :-1].contiguous()
    y = data[:, 1:].contiguous()

    if device.type == "cuda":
        # pin_memory + non_blocking overlaps the host-to-device copy with
        # compute. Worth roughly a few percent of step time and costs nothing.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


class RandomWindowSampler:
    """Training batches: uniformly random windows, sampled with replacement."""

    def __init__(self, data: PackedTokens, batch_size: int, context: int,
                 device: torch.device, seed: int = 20260827) -> None:
        if len(data) < context + 1:
            raise ValueError(
                f"Split has {len(data):,} tokens, too few for a "
                f"{context}-token context.")
        self.data, self.batch_size, self.context = data, batch_size, context
        self.device = device
        # A dedicated generator, seeded explicitly, so a resumed run can be made
        # to draw the same sequence of batches if that is ever needed for
        # debugging. The training loop advances it past already-completed steps
        # on resume.
        self.rng = np.random.default_rng(seed)
        self.high = len(data) - context - 1

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, self.high, size=self.batch_size)
        return _windows_to_tensors(self.data.tokens, starts, self.context,
                                   self.device)

    def skip(self, n_batches: int) -> None:
        """Advance the generator without building tensors, used when resuming."""
        for _ in range(n_batches):
            self.rng.integers(0, self.high, size=self.batch_size)


class FixedWindowSampler:
    """Validation batches: the same evenly spaced windows on every evaluation.

    Deterministic by construction rather than by seeding, so the val loss is
    comparable across evaluations, across resumes, and between Model H and
    Model L.
    """

    def __init__(self, data: PackedTokens, batch_size: int, context: int,
                 device: torch.device, n_windows: int = 512) -> None:
        self.data, self.batch_size, self.context = data, batch_size, context
        self.device = device
        high = len(data) - context - 1
        n = min(n_windows, max(1, high))
        # linspace rather than arange so the windows span the whole split even
        # when n_windows is much smaller than the number of possible offsets.
        # Evaluating only the first N tokens would measure one part of the
        # corpus, and the splits are source-stratified, not shuffled by source.
        self.starts = np.linspace(0, high, num=n, dtype=np.int64)

    def __iter__(self):
        for i in range(0, len(self.starts), self.batch_size):
            chunk = self.starts[i:i + self.batch_size]
            yield _windows_to_tensors(self.data.tokens, chunk, self.context,
                                      self.device)

    def __len__(self) -> int:
        return (len(self.starts) + self.batch_size - 1) // self.batch_size

    @property
    def n_tokens(self) -> int:
        """Total predicted positions, used to weight the averaged loss."""
        return len(self.starts) * self.context


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a device, preferring CUDA and treating MPS as a last resort.

    MPS is deliberately not preferred. Phase 1 measured it running IndicTrans2 at
    0.006 sentences/second against 0.9-1.2 on CPU - a 45x slowdown from
    per-operation fallback (D-042). A hand-written transformer behaves better
    than that, but on this Mac neither MPS nor CPU is a training device; both are
    for correctness checks at the tiny configuration. Ask for it explicitly if
    you want it.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
