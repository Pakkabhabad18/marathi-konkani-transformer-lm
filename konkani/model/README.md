# Konkani — model (Model L)

The two models are architecturally identical and differ only in vocabulary size
and weights, so the implementation is written once in `common/model/` rather than
copied per language:

| file | contents |
|---|---|
| `../../common/model/config.py` | `ModelConfig`, with the analytic parameter count and the `KONKANI` preset |
| `../../common/model/attention.py` | multi-head causal self-attention |
| `../../common/model/lm.py` | feed-forward block, transformer block, `DecoderLM`, `generate()` |

Duplicating the file per language would allow the two copies to drift, and a
drift between them would invalidate the central claim of this project — that the
only difference between Model H and Model L is the data.

What is specific to Konkani and lives here or nearby:

- `../configs/model_config.json` — the exact configuration used, written by the
  training run rather than by hand
- weights: `konkani_pretrain_best.pt` in `phase2_checkpoints/` on Google Drive
  (https://drive.google.com/drive/folders/1lUSriyp7_tltkmHCgINon175xnFp2lp-?usp=sharing)

Correctness checks for the shared implementation: `tools/verify_model.py`
(18 checks, including that a token at position t+1 cannot change the logits at
position t).
