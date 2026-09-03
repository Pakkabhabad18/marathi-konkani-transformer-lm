"""
Decoder-only Transformer language model, built from primitive PyTorch layers.

Shared between Model H (Marathi) and Model L (Konkani) as *code only*. Each model
gets its own ModelConfig, its own weights and its own checkpoint; nothing is
shared at runtime. Sharing the implementation is what makes the two models
comparable - a difference in results cannot be blamed on a difference in how the
architecture was written.

    common/model/
      config.py     ModelConfig, parameter arithmetic, the three configurations
      attention.py  multi-head causal self-attention, from first principles
      lm.py         FeedForward, TransformerBlock, DecoderLM

Correctness checks live in tools/verify_model.py and must pass before any
training run.
"""

from .attention import MultiHeadCausalSelfAttention
from .config import KONKANI, MARATHI, TINY, ModelConfig
from .lm import DecoderLM, FeedForward, TransformerBlock

__all__ = [
    "ModelConfig", "MARATHI", "KONKANI", "TINY",
    "DecoderLM", "TransformerBlock", "FeedForward",
    "MultiHeadCausalSelfAttention",
]
