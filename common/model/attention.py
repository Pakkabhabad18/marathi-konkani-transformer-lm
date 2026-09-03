#!/usr/bin/env python3
"""
Multi-head causal self-attention, implemented from first principles.

Built from nn.Linear and nn.Dropout only. No nn.MultiheadAttention, no
F.scaled_dot_product_attention, no HuggingFace attention block - the point of
writing it out is to be able to say what every tensor operation in the forward
pass computes and what its shape is.

THE SHAPE JOURNEY, WHICH IS THE PART THAT IS EASY TO GET WRONG
--------------------------------------------------------------
    input                       (B, T, d)
    after W_Q / W_K / W_V       (B, T, d)
    view into heads             (B, T, h, d_head)      d_head = d / h
    transpose(1, 2)             (B, h, T, d_head)      <- heads become a batch dim
    scores = Q @ K^T            (B, h, T, T)
    weights = softmax(scores)   (B, h, T, T)
    context = weights @ V       (B, h, T, d_head)
    transpose(1, 2)             (B, T, h, d_head)
    reshape                     (B, T, d)              <- heads concatenated
    after W_O                   (B, T, d)

The transpose is what makes the heads independent. Once the head axis sits
beside the batch axis, the matmul treats each (B, h) pair as a separate T x T
attention problem, so head 3 of sequence 0 cannot see head 4 of sequence 0. The
final reshape is the concatenation step from the paper: the heads are laid back
out side by side along the feature axis, and W_O is what lets the model mix
information between them.

The `.contiguous()` before the final reshape is required, not cosmetic. After
`transpose` the tensor's memory layout no longer matches its shape, and `view`
on a non-contiguous tensor either errors or silently interleaves head outputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    """Causal self-attention over a sequence, split across independent heads."""

    def __init__(self, config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        # Separate projections rather than one fused Linear(d, 3d). A fused
        # projection is measurably faster because it is one matmul instead of
        # three, but keeping W_Q, W_K and W_V as named modules makes the mapping
        # to Q = X W_Q, K = X W_K, V = X W_V direct and inspectable. At this
        # model size the difference is a few percent of step time, which is not
        # worth the loss of clarity in a model that has to be explained.
        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)
        self.W_O = nn.Linear(self.d_model, self.d_model)

        # Dropout on the attention probabilities: randomly prevents a position
        # from attending to some other position during training, so the model
        # cannot rely on a single connection. Applied after softmax, so the rows
        # no longer sum to 1 during training - that is intended, and is what
        # makes it a regulariser rather than a reweighting.
        self.attn_dropout = nn.Dropout(config.dropout)
        # Dropout on the block's output, applied after W_O.
        self.resid_dropout = nn.Dropout(config.dropout)

        # 1 / sqrt(d_head). Q·K is a sum of d_head products of roughly
        # unit-variance terms, so its variance grows with d_head and its standard
        # deviation with sqrt(d_head). At d_head = 64 the raw logits have a
        # spread around 8, which pushes softmax into saturation: one weight goes
        # to ~1, the rest to ~0, and the gradient through softmax - which is
        # proportional to p(1-p) - vanishes. Dividing by sqrt(d_head) returns the
        # logits to unit variance, keeps the distribution soft, and keeps the
        # gradient alive. Precomputed because it is used on every forward pass.
        self.scale = 1.0 / math.sqrt(self.d_head)

        # The causal mask, as a boolean upper triangle above the diagonal:
        # True exactly where a query position would be looking into the future.
        # Registered as a non-persistent buffer, so it moves to the GPU with the
        # module but is not written into checkpoints - it is derived from
        # context_length, not learned, and storing it would make checkpoints
        # larger and would let a stale mask outlive a config change.
        causal = torch.triu(
            torch.ones(config.context_length, config.context_length,
                       dtype=torch.bool),
            diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Apply causal self-attention.

        Args:
            x: input activations, shape (B, T, d_model).
            return_attention: also return the post-softmax attention weights,
                shape (B, n_heads, T, T). Off by default because holding a
                T x T matrix per head per layer is expensive; switched on by the
                attention-analysis tooling, which needs the weights to draw
                heatmaps and to compute per-head entropy and mean attention
                distance.

        Returns:
            (output, attention_weights_or_None), output shape (B, T, d_model).
        """
        B, T, d = x.shape
        if T > self.causal_mask.size(0):
            # Learned absolute positional embeddings make context length a hard
            # ceiling. Failing here is much better than silently attending
            # outside the mask.
            raise ValueError(
                f"Sequence length {T} exceeds context_length "
                f"{self.causal_mask.size(0)}.")

        # Project the residual stream into queries, keys and values. Same input,
        # three different learned views of it: what this position is looking for,
        # what it offers as a match, and what it passes on when matched.
        q = self.W_Q(x)                                  # (B, T, d)
        k = self.W_K(x)
        v = self.W_V(x)

        # Split the feature axis into heads and move the head axis next to the
        # batch axis, so each head becomes an independent attention problem.
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        #                                                (B, h, T, d_head)

        # Scaled dot-product scores: how well each query matches each key.
        # (B, h, T, d_head) @ (B, h, d_head, T) -> (B, h, T, T)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Causal masking. Positions strictly above the diagonal are the future,
        # and are set to -inf BEFORE the softmax so they receive exactly zero
        # probability afterwards. Masking after the softmax would zero those
        # weights but leave the remaining ones summing to less than 1, which
        # silently scales down the attention output by an amount that varies with
        # position. -inf rather than a large negative constant so the result is
        # exact in both fp16 and fp32; softmax handles -inf cleanly, and
        # masked_fill on the sliced mask keeps this correct for any T <= context.
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        # Softmax along the key axis: for each query position, a distribution
        # over the positions it is allowed to see. Computed in fp32 even under
        # autocast, because a softmax over -inf-masked fp16 logits is where
        # numerical problems in attention usually start.
        weights = F.softmax(scores.float(), dim=-1).to(v.dtype)  # (B, h, T, T)
        weights_out = weights if return_attention else None
        weights = self.attn_dropout(weights)

        # Weighted sum of values. (B, h, T, T) @ (B, h, T, d_head)
        context = weights @ v                            # (B, h, T, d_head)

        # Concatenate the heads: move the head axis back beside the feature axis
        # and flatten the two. contiguous() is needed because transpose only
        # changes the view, not the underlying layout.
        context = context.transpose(1, 2).contiguous().view(B, T, d)

        # Output projection: lets the model mix across what the heads found.
        # Without W_O the heads would write into fixed, disjoint slices of the
        # residual stream and could never combine.
        out = self.resid_dropout(self.W_O(context))
        return out, weights_out
