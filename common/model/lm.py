#!/usr/bin/env python3
"""
The decoder-only Transformer language model: feed-forward, block, and the full
stack with its output head.

Built from nn.Linear, nn.Embedding, nn.LayerNorm and nn.Dropout, plus the
attention module in attention.py. Nothing from nn.Transformer*, no HuggingFace
model classes.

WHY PRE-NORM
------------
Each sublayer computes  x + Sublayer(LayerNorm(x))  rather than the original
paper's  LayerNorm(x + Sublayer(x)).

The difference is what the gradient sees on the way back. With post-norm there is
a LayerNorm sitting between every residual addition and the next, so the gradient
reaching the embeddings has been rescaled once per layer; deep post-norm stacks
need careful warmup to train at all, and diverge without it. With pre-norm the
residual path from the loss to the embeddings passes through no normalisation and
no weight matrix - it is a clean identity - so gradients arrive at full strength
and training is markedly more forgiving of learning rate.

The cost is a slightly worse final loss in several published comparisons. At
seven layers that trade is worth taking: stability matters more than a fraction
of a nat when the whole run has to survive a free-tier session.

Because the residual stream is never normalised on its own path, its magnitude
grows with depth. That is why there is a final LayerNorm after the last block,
before the output projection: without it the logits inherit whatever scale the
stream happened to reach.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network: d_model -> d_ff -> d_model.

    Attention moves information between positions but applies no non-linear
    transformation to what it moved. This is where that happens, independently at
    each position - which is what "position-wise" means: the same two matrices are
    applied to every position, with no mixing across the sequence.

    The inner dimension is larger than the model dimension (4x here) so the layer
    projects up into a wider space, applies the non-linearity there, and projects
    back. Roughly two thirds of each block's parameters live here.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU rather than ReLU: it is smooth everywhere, so no position sits at
        # a hard zero-gradient kink, and it is the activation used by the
        # decoder-only models this architecture follows.
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """One decoder block: causal self-attention, then feed-forward, both residual."""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadCausalSelfAttention(config)
        self.norm_ffn = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Pre-norm: normalise a copy, transform it, add the result back to the
        # untouched residual stream. `x` itself is never overwritten, which is
        # what keeps the identity path intact.
        attn_out, attn_weights = self.attention(self.norm_attn(x),
                                                return_attention=return_attention)
        x = x + attn_out
        x = x + self.feed_forward(self.norm_ffn(x))
        return x, attn_weights


class DecoderLM(nn.Module):
    """Decoder-only Transformer trained with the causal language-modelling objective."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        # Token ids -> vectors. This is the only place discrete input enters.
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Position -> vector. Self-attention is permutation-invariant: permute the
        # input positions and the output permutes identically, so without this the
        # model literally cannot tell "dog bites man" from "man bites dog".
        # Learned absolute embeddings were chosen over sinusoidal because they
        # cost 0.26M parameters here and let the model learn whatever positional
        # structure the data actually has. The price is that context_length is a
        # hard ceiling - there is no row for position 512 - whereas sinusoidal
        # encodings can be evaluated at any position. That trade is acceptable
        # because nothing in this project needs to run past the trained context.
        # The bonus ablation removes this entirely rather than zeroing it: a
        # zeroed embedding is still a parameter that receives gradient and would
        # simply relearn itself. None means the module does not exist, so the
        # parameter count drops by context_length * d_model and
        # verify_model.py's "every parameter receives a gradient" check stays
        # meaningful. See report/bonus_decisions.md, B-002.
        self.position_embedding = (
            None if config.no_positional_embedding
            else nn.Embedding(config.context_length, config.d_model))
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)])

        # See the module docstring: pre-norm leaves the residual stream
        # un-normalised, so it is normalised once here before the output head.
        self.final_norm = nn.LayerNorm(config.d_model)

        # Projection to next-token logits, one score per vocabulary entry.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.tie_embeddings:
            # Sharing one matrix for both directions. Saves vocab * d_model
            # parameters; see D-044 for why this project does not use it.
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Scaled initialisation for the projections that write into the residual
        # stream. Every block adds to the same stream, so with n_layers blocks the
        # variance of the stream grows roughly n_layers-fold before training does
        # anything. Scaling these two by 1/sqrt(2 * n_layers) - two residual
        # writes per block - keeps the stream's variance roughly constant with
        # depth at initialisation. Without it deep stacks start with saturated
        # activations and the first few hundred steps are spent recovering.
        for name, param in self.named_parameters():
            if name.endswith("W_O.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(
                    param, mean=0.0,
                    std=config.init_std / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        """Normal(0, init_std) for weights, zeros for biases, standard for norms."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count real tensors in the module tree.

        Checked against ModelConfig.n_params() in tools/verify_model.py. If the
        analytic arithmetic and this disagree, one of them is wrong, and the
        number quoted in the report must be this one.
        """
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        # Tied weights appear twice in .parameters() under some PyTorch versions;
        # deduplicating by identity keeps the count honest either way.
        seen, total = set(), 0
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_attention: bool = False):
        """Run the model.

        Args:
            idx: token ids, shape (B, T).
            targets: next-token ids, shape (B, T). When given, the loss is
                returned as well. The caller is responsible for targets being
                idx shifted by one - doing the shift here would hide it.
            return_attention: collect per-layer attention weights for analysis.

        Returns:
            (logits, loss_or_None, attentions_or_None). logits shape
            (B, T, vocab_size); attentions is a list of (B, h, T, T) tensors,
            one per layer.
        """
        B, T = idx.shape
        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds context_length "
                f"{self.config.context_length}. Learned positional embeddings "
                f"make this a hard limit.")

        # Positions 0..T-1. Built on the input's device so this works unchanged
        # on CPU, MPS and CUDA.
        pos = torch.arange(T, device=idx.device)

        # Token meaning plus positional information. Added rather than
        # concatenated: concatenating would spend d_model on position and leave
        # less for content, whereas addition lets the model allocate the subspace
        # it needs for each.
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            x = x + self.position_embedding(pos)
        x = self.embed_dropout(x)

        attentions = [] if return_attention else None
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention:
                attentions.append(attn)

        x = self.final_norm(x)
        logits = self.lm_head(x)                         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten batch and time: cross_entropy wants (N, C) against (N,).
            # Averaged over every position in the batch, which is what "averaged
            # over all positions" in the objective means - every position is a
            # training example, not just the last one.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss, attentions

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 eos_id: int | None = None) -> torch.Tensor:
        """Autoregressive sampling, used by the generation-quality evaluation.

        Args:
            idx: prompt token ids, shape (B, T).
            temperature: divides the logits before softmax. Below 1 sharpens the
                distribution toward the argmax; above 1 flattens it. At exactly 0
                this switches to greedy decoding, because dividing by zero is not
                the same thing as taking the argmax.
            top_k: restrict sampling to the k most likely tokens.
            eos_id: stop early once every sequence in the batch has emitted this.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # The model can only see context_length positions, so once the
            # sequence is longer than that, condition on the most recent window.
            window = idx[:, -self.config.context_length:]
            logits, _, _ = self(window)
            # Only the last position predicts the next token; the rest are
            # predictions for tokens we already have.
            logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break
        return idx
