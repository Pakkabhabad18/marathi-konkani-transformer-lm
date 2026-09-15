#!/usr/bin/env python3
"""
Model configuration, and the parameter arithmetic that justifies it.

WHY A CONFIG OBJECT RATHER THAN CONSTRUCTOR ARGUMENTS
-----------------------------------------------------
The specification requires that every checkpoint carry the configuration that
produced it, and that hyperparameters be logged alongside checkpoints so runs are
reproducible from the README. A single serialisable object makes that one field
in the checkpoint rather than a dozen loose arguments that can drift apart from
the weights they describe. A checkpoint whose config is wrong is worse than no
checkpoint: the weights load into the wrong shapes, or - worse - into shapes that
happen to fit but mean something else.

THE PARAMETER BUDGET
--------------------
The target is approximately 25M trainable parameters per model. `n_params()`
computes the count analytically from the config, before any tensor is allocated,
so a configuration can be checked against the budget without building the model.
`DecoderLM.count_parameters()` counts the real tensors afterwards. The two are
asserted equal in tools/verify_model.py - if the arithmetic here and the module
tree disagree, one of them is wrong, and it is better to find out at startup than
to report a parameter count in the report that the model does not have.

Per transformer block, at model dimension d and feed-forward dimension f:

    attention   4 * d * d  weights + 4 * d biases      (W_Q, W_K, W_V, W_O)
    feed-forward    2 * d * f  weights + f + d biases
    layer norms     2 * (2 * d)                        (two per block, pre-norm)

At d = 512 and f = 2048 that is 3,152,896 parameters per block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture of one decoder-only Transformer language model.

    One config describes one model. Model H and Model L each get their own, and
    they are never shared - the specification requires the two models to be
    independent in data, tokenizer, vocabulary and weights.
    """

    # Set from the language's Phase 1 tokenizer. Both of ours are 2,500, so the
    # two models come out the same size; the specification allows them to differ.
    vocab_size: int = 2500

    # Width of the residual stream. Everything that flows between blocks has this
    # last dimension.
    d_model: int = 512

    # Depth. Seven is what lands nearest the 25M budget at d_model 512: six
    # leaves ~3M unused, eight overshoots to 28M.
    n_layers: int = 7

    # Attention heads. d_model must divide evenly by this, since each head works
    # in a d_model / n_heads subspace.
    n_heads: int = 8

    # Feed-forward inner dimension, conventionally 4 * d_model. This is where
    # roughly two thirds of each block's parameters live.
    d_ff: int = 2048

    # Maximum sequence length. With learned absolute positional embeddings this
    # is a hard architectural ceiling, not a runtime setting: there is no
    # embedding row for position 512 or beyond, so the model physically cannot
    # process a longer sequence.
    context_length: int = 512

    dropout: float = 0.1

    # Tying the output projection to the token embedding saves vocab * d_model
    # parameters. At vocabulary 2,500 that is 1.28M, about 5% of the budget - the
    # saving that motivates tying at vocabulary 50,000 is mostly not available
    # here, so we keep them separate and let the input and output representations
    # of a token differ. Recorded in D-043 and D-044.
    tie_embeddings: bool = False

    # Bonus ablation only (see report/bonus_decisions.md, B-001). When set, the
    # model is built with NO positional embedding at all: the residual stream
    # starts as the token embedding alone. Self-attention is permutation
    # invariant, so the only positional information left is whatever the causal
    # mask leaks - position t can attend to t+1 tokens and position 0 to one, so
    # the number of visible positions is itself a position signal. Whether that
    # is enough is precisely what the ablation measures. Default False: every
    # Phase 1-3 deliverable was trained with positional embeddings and must stay
    # reproducible from this file.
    no_positional_embedding: bool = False

    # Standard deviation for weight initialisation. 0.02 is the GPT-2 value and
    # is small enough that the residual stream does not blow up before the first
    # LayerNorm has anything to normalise.
    init_std: float = 0.02

    def __post_init__(self) -> None:
        # Fail at construction, not in the middle of a forward pass on a GPU.
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_heads "
                f"{self.n_heads}; each head needs an equal slice of the "
                f"residual stream.")
        if self.vocab_size < 1 or self.context_length < 1:
            raise ValueError("vocab_size and context_length must be positive.")

    @property
    def d_head(self) -> int:
        """Dimension each attention head works in."""
        return self.d_model // self.n_heads

    # ---- parameter arithmetic -------------------------------------------

    def n_params_per_block(self) -> int:
        d, f = self.d_model, self.d_ff
        attention = 4 * d * d + 4 * d          # W_Q, W_K, W_V, W_O with biases
        feed_forward = 2 * d * f + f + d       # two linears with biases
        layer_norms = 2 * (2 * d)              # gain and bias, two per block
        return attention + feed_forward + layer_norms

    def n_params(self) -> dict:
        """Parameter count by component, computed rather than measured.

        Returned as a breakdown rather than a single number because the report
        has to justify the shape, and "24.9M" on its own justifies nothing.
        """
        d, v = self.d_model, self.vocab_size
        blocks = self.n_layers * self.n_params_per_block()
        token_embedding = v * d
        positional_embedding = (0 if self.no_positional_embedding
                                else self.context_length * d)
        final_norm = 2 * d
        # A tied head reuses the embedding matrix, so it adds only the bias.
        output_head = v if self.tie_embeddings else v * d + v

        total = (blocks + token_embedding + positional_embedding
                 + final_norm + output_head)
        return {
            "blocks": blocks,
            "per_block": self.n_params_per_block(),
            "token_embedding": token_embedding,
            "positional_embedding": positional_embedding,
            "final_norm": final_norm,
            "output_head": output_head,
            "total": total,
            # Non-embedding parameters are what training FLOPs scale with
            # (6 * N * D), so this is the number to use for compute estimates,
            # not the total.
            "non_embedding": blocks,
        }

    # ---- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        p = self.n_params()
        return "\n".join([
            f"  vocab_size          {self.vocab_size:>12,}",
            f"  d_model             {self.d_model:>12,}",
            f"  n_layers            {self.n_layers:>12,}",
            f"  n_heads             {self.n_heads:>12,}  "
            f"(d_head {self.d_head})",
            f"  d_ff                {self.d_ff:>12,}",
            f"  context_length      {self.context_length:>12,}",
            f"  dropout             {self.dropout:>12}",
            f"  tie_embeddings      {str(self.tie_embeddings):>12}",
            "",
            f"  per block           {p['per_block']:>12,}",
            f"  {self.n_layers} blocks            {p['blocks']:>12,}",
            f"  token embedding     {p['token_embedding']:>12,}",
            f"  positional embedding{p['positional_embedding']:>12,}",
            f"  final norm          {p['final_norm']:>12,}",
            f"  output head         {p['output_head']:>12,}",
            f"  {'TOTAL':<20}{p['total']:>12,}",
            f"  non-embedding       {p['non_embedding']:>12,}  "
            f"(used for 6*N*D compute estimates)",
        ])


# The two Phase 2 models. Both are 2,500-vocabulary, so both come out at the same
# size; only the weights and the data differ. Kept here so the training script
# and the report read the same numbers from the same place.
MARATHI = ModelConfig(vocab_size=2500)
KONKANI = ModelConfig(vocab_size=2500)

# A deliberately tiny model for correctness testing on CPU. Every piece of code
# runs against this before anything touches a GPU: it trains in seconds, so a
# bug shows up in a coffee break rather than after an hour of Kaggle quota.
TINY = ModelConfig(vocab_size=2500, d_model=128, n_layers=2, n_heads=4,
                   d_ff=512, context_length=64, dropout=0.0)


if __name__ == "__main__":
    for name, cfg in (("MARATHI / KONKANI", MARATHI), ("TINY (CPU tests)", TINY)):
        print("=" * 62)
        print(name)
        print("=" * 62)
        print(cfg.summary())
        print()
