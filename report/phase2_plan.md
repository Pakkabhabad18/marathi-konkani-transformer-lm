# Phase 2 — Plan: model, pretraining, evaluation

Model H: Marathi. Model L: Konkani (Devanagari). Written 27 August 2026,
9 days before the 5 September deadline.

This document says what will be built, why each piece is built that way, what it
will cost in wall-clock time, and where it will run. It is written to be read
before an oral examination as well as before the work.

## 1. What Phase 1 hands over

| | Marathi | Konkani |
|---|---:|---:|
| train tokens | 872,024,099 | 506,259,368 |
| val / test tokens | ~8.8M / ~8.7M | ~5.2M / ~5.1M |
| tokenizer | `marathi/tokenizer/marathi_bpe.model` | `konkani/tokenizer/konkani_bpe.model` |
| vocabulary | 2,500 | 2,500 |
| fertility (held-out) | 2.6301 | 2.5279 |
| unknown-token rate | 0.000000% | 0.000000% |

Both vocabularies are 2,500, so both models have identically sized embedding and
output layers. The specification allows them to differ; ours do not, which makes
the H-versus-L comparison cleaner, because any difference in results comes from
the data rather than from model capacity.

Splits live on Google Drive as `*_cleaned_splits.tar.gz`. The first job of
Phase 2 is to turn those text splits into a token stream on disk — a flat
`uint16` array of token ids per split, memory-mapped at training time. `uint16`
suffices because the vocabulary is 2,500, well under 65,536, so the whole
Marathi train split is 872M × 2 bytes ≈ 1.7 GB and Konkani ≈ 1.0 GB. Both fit in
a Kaggle working directory and both memory-map without loading into RAM.

## 2. Architecture

### 2.1 Chosen configuration

| | value |
|---|---|
| `d_model` | 512 |
| layers | 7 |
| heads | 8 (`d_head` = 64) |
| feed-forward inner dim | 2048 (4 × `d_model`) |
| context length | 512 tokens |
| positional encoding | learned absolute |
| normalization | pre-norm LayerNorm, plus a final LayerNorm |
| activation | GELU |
| dropout | 0.1 on attention probabilities, sublayer outputs and embeddings |
| output head | untied linear projection to vocabulary |

Parameter count, computed rather than estimated:

| component | count |
|---|---:|
| 7 transformer blocks | 22,070,272 |
| token embedding (2,500 × 512) | 1,280,000 |
| positional embedding (512 × 512) | 262,144 |
| final LayerNorm | 1,024 |
| output projection (untied) | 1,282,500 |
| **total** | **24,895,940** |

That is 24.90M against a ~25M target. `tools/count_params.py` will print this
from the config file so the number in the report is read from the model, not
copied from here.

### 2.2 Why this shape

**Depth over width.** At a fixed budget, `d_model` costs quadratically inside
each block (`12 × d²` per layer) while depth costs linearly. Seven layers at 512
gives more sequential composition steps than four layers at 768 for the same
parameters, and language modelling benefits from depth. Six layers would leave
~3M parameters unused; eight layers overshoots to 28.0M.

**Untied output head.** Tying saves 1.28M parameters, which at vocabulary 2,500
is only 5% of the budget — the saving that motivates tying at vocabulary 50,000
mostly is not available here. Keeping them untied lets the input and output
representations of a token differ, which is the usual argument for not tying.
D-043 flagged this as an open question from Phase 1 and this is the answer: at a
small vocabulary, tying buys little, so we spend the parameters.

**Context 512.** Learned absolute positional embeddings mean the model cannot
process a sequence longer than the positional table, so context length is a hard
architectural ceiling, not a runtime setting. 512 tokens is roughly 200 words at
our fertility — a paragraph. Attention cost is quadratic in context, so 1024
would roughly double attention FLOPs for material most documents do not have.

**Pre-norm.** LayerNorm before each sublayer rather than after. Post-norm puts a
normalization between every residual addition and the next, which attenuates the
gradient signal travelling down the residual path and makes deep stacks need
warmup to train at all. Pre-norm leaves a clean identity path from the loss to
the embeddings. The trade is slightly worse final loss in some published
comparisons, which at 7 layers is not the binding concern.

### 2.3 Components to implement, and what each does

Everything is written from `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and
`nn.Dropout`. No `nn.Transformer*`, no HuggingFace model classes, no pre-built
attention.

**Token embedding.** A lookup table of shape (vocab, d_model). Turns integer ids
into dense vectors. Without it there is nothing continuous to differentiate.

**Positional embedding.** Self-attention is permutation-invariant: shuffle the
input positions and the attention output shuffles identically, so position
carries no information unless it is injected. A learned table of shape
(context, d_model) is added to the token embeddings. This is why context length
is fixed at 512.

**Multi-head causal self-attention.** Project the input `X ∈ (B, T, d)` into Q, K
and V. Reshape each to `(B, T, h, d_head)` and transpose to `(B, h, T, d_head)`
so every head attends independently in its own 64-dimensional subspace. Compute
`softmax(QKᵀ/√d_head + M)V` per head, concatenate back to `(B, T, d)`, and apply
the output projection `W_O`.

The `1/√d_head` scaling matters: Q·K is a sum of `d_head` products, so with
unit-variance inputs its variance grows with `d_head`. Unscaled, logits at
`d_head = 64` have standard deviation around 8, softmax saturates, and gradients
through it approach zero. Dividing by `√d_head` returns the logits to unit
variance.

`M` is an additive mask, 0 where attention is allowed and a large negative number
on future positions, applied before the softmax so masked positions receive
essentially zero probability. Adding before softmax rather than zeroing after is
what keeps each row a proper distribution.

**Feed-forward network.** Two linear layers, 512 → 2048 → 512, with GELU
between. Attention mixes information across positions but applies no per-position
non-linear transformation; the FFN is where that happens. It holds roughly
two-thirds of each block's parameters.

**Residual connections and LayerNorm.** Each sublayer computes
`x + Sublayer(LayerNorm(x))`. The residual gives the gradient a path to the
embeddings that does not pass through any weight matrix. LayerNorm keeps
activation scale stable across depth.

**Output head and loss.** A linear projection from 512 to 2,500 producing
next-token logits, trained with cross-entropy between the logits at position `t`
and the token at `t+1`, averaged over all positions.

### 2.4 Verifying the causal mask

The specification asks for empirical proof rather than an assertion. The test:
run a batch, record the logits at position `t`, change the input token at
position `t+1`, run again, and assert the logits at position `t` are bit-identical.
If they change, the mask leaks and every perplexity number afterwards is
meaningless. This runs as a unit test on every training start, not as a one-off.

## 3. Training

### 3.1 Token budget

The specification targets ~500M training tokens per model. Konkani has
506,259,368, so it trains for one epoch. Marathi has 872,024,099.

**Decision (27 Aug): train both models on 500M tokens, and if the schedule
allows, run Marathi again on its full 872M as a documented extra.** The
controlled comparison is the primary result; the longer run, if it happens, is
reported separately and never substituted for it. Marathi's 500M pass sees about
57% of its corpus. This equalises the compute and the number of gradient
updates between the two models, so differences in perplexity and generation
quality come from *what the data is* rather than from one model simply having had
more training. The surplus Marathi data is not wasted — it is what makes Marathi's
500M tokens higher quality, since they are drawn from a larger, more diverse pool
and contain no synthetic text at all.

The alternative — one full epoch each, 872M against 506M — conflates data volume
with training budget and makes the Phase 3 question "how did data scale and
quality differ" harder to answer, not easier. Whichever is chosen, it is a
decision to record rather than a default to drift into.

### 3.2 Hyperparameters

| | value | reason |
|---|---|---|
| optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 | β₂ = 0.95 rather than 0.999 is standard for LM pretraining; shorter second-moment memory copes better with the loss dropping fast early |
| peak learning rate | 3e-4 | conventional for this parameter scale |
| schedule | linear warmup 2% of steps, then cosine decay to 10% of peak | warmup avoids the large early Adam steps that destabilise a randomly initialised model |
| batch | 32 sequences × 512 tokens = 16,384 tokens, gradient-accumulated to ~131k tokens | the accumulation target keeps the effective batch stable regardless of what fits in GPU memory |
| gradient clipping | global norm 1.0 | |
| precision | mixed (fp16/bf16 autocast) with fp32 master weights | roughly doubles throughput on a T4 |
| dropout | 0.1 | |
| steps | ~3,800 optimizer steps at 131k tokens each | |

Weight decay is applied to matrices but not to biases, LayerNorm parameters or
embeddings — decaying an embedding pulls rare tokens toward zero for no reason.

### 3.3 Checkpointing

Mandatory, and the free-tier compute makes it structural rather than defensive:
Colab sessions end unpredictably and Kaggle sessions have a hard ceiling. Every
checkpoint contains model weights, optimizer state, scheduler state, current step
and the config. Written atomically — to a temporary file, then renamed — so a
session killed mid-write leaves the previous checkpoint intact rather than a
truncated file.

Resume will be tested by killing a run and restarting it, and confirming the loss
curve continues rather than jumping. A checkpoint that has never been resumed
from is a checkpoint that does not work.

## 4. Where to run

### 4.1 Estimated cost

Training FLOPs ≈ `6 × N × D`, with N the non-embedding parameters (22.07M) and D
the training tokens. For 500M tokens that is **6.6 × 10¹⁶ FLOPs per model**.

Wall-clock depends on achieved utilisation. A hand-written attention without
fused kernels typically reaches 10–25% of peak on a T4:

| device | peak (fp16) | 25% util | 15% util | 10% util |
|---|---:|---:|---:|---:|
| T4 | 65 TFLOPS | 1.1 h | 1.9 h | 2.8 h |
| P100 | 18.7 TFLOPS | 3.9 h | 6.6 h | 10.0 h |

So **roughly 2–3 hours per model on a T4**, 4–6 hours for both, plus
evaluation. If Marathi trains on its full 872M instead, add about 75% to its run.

M1 Mac is not in the table because it is not a candidate. The M1 GPU is around
2.6 TFLOPS fp32 with no tensor cores, so the same run is 40+ hours — and Phase 1
already measured MPS running IndicTrans2 at 0.006 sentences/second against
0.9–1.2 on CPU, a 45× slowdown from per-operation fallback (D-042). A
hand-written transformer will behave better than that, but not well enough to
matter.

### 4.2 Platform comparison

| | Kaggle | Colab (free) | M1 Mac |
|---|---|---|---|
| GPU | T4 ×2, or P100 | T4 when available | none usable |
| published quota | 30 GPU-hours/week | none published | — |
| session ceiling | ~9–12 h | 12 h, often shorter | — |
| predictability | quota is stated and tracked | "neither a GPU nor a particular GPU model is guaranteed" | total |
| disk persistence | dataset + working dir persist | needs Drive mounting | native |

**Kaggle is the primary platform.** The published 30-hour weekly quota comfortably
covers a 4–6 hour job with room for restarts, the session ceiling is long enough
to finish a run in one sitting, and quota consumption is visible rather than
guessed at. Two T4s are available, though a single T4 is sufficient here and
avoids the complexity of multi-GPU.

Choose **T4 over P100** in the accelerator dropdown. The P100 has no fp16 tensor
cores, which is why it is 3–4× slower in the table above despite being the
nominally "bigger" card.

**Colab free is the fallback**, not the plan. Google publishes no hour allowance
and explicitly does not guarantee a GPU, so a run can be interrupted at any point
for reasons outside your control. That is survivable given the checkpointing
above, but it is not something to schedule against with 9 days remaining.

**The Mac's job is correctness, not throughput.** Every piece of code runs there
first on a tiny configuration — 2 layers, `d_model` 128, a few thousand tokens —
until the loss decreases, the causal-mask test passes, and a killed run resumes
cleanly. Only then does anything touch a GPU. Debugging on Kaggle spends quota on
problems a CPU would have found.

## 5. Evaluation

Every metric is reported for both models side by side.

**Intrinsic.** Validation cross-entropy and perplexity (`PPL = e^loss`), and
bits-per-byte on the same held-out text. BPB matters here specifically because
the two tokenizers differ in fertility — 2.6301 against 2.5279 — so perplexity
per *token* is not comparable between them, while bits per *byte* is. This is the
metric that makes the H-versus-L comparison honest, and it is worth being able to
explain why.

**Generation.** Continuations from held-out prefixes under greedy decoding and
temperatures 0.5, 1.0 and 1.5, scored with BLEU-4, chrF and ROUGE-L against the
reference continuations. All three are weak measures for open-ended generation —
a fluent continuation that differs from the reference scores badly — and chrF is
the least bad of them for Devanagari because it works on character n-grams and
so is not destroyed by morphological variation the way word-level BLEU is. Also:
repetition rate, Distinct-1 and Distinct-2, and qualitative notes.

**Attention analysis.** Heatmaps for at least one early and one late layer with
multiple heads, per-head attention entropy, and mean attention distance. The
expected pattern is early layers with low entropy and short distance (local,
positional heads) and later layers with higher entropy and longer distance
(content-based heads). Whether Model L shows the same structure on less data is
one of the more interesting things this project can report.

## 6. Schedule

Nine days. Compute is 4–6 hours of it; the rest is code and analysis.

| day | work | output |
|---|---|---|
| 27 Aug | phone-verify Kaggle; tokenize splits to `uint16` arrays; data loader | GPU access enabled, token arrays, loader with a test |
| 28 Aug | model: embeddings, attention, block, head | `common/model/`, parameter count printed |
| 29 Aug | causal-mask test; tiny-config overfit run on Mac | passing tests, loss to near zero on 100 sequences |
| 30 Aug | training loop, checkpointing, resume test | a killed run that resumes cleanly |
| 31 Aug | pretrain Model L (Konkani) on Kaggle | checkpoint, loss curve, training log |
| 1 Sep | pretrain Model H (Marathi) on Kaggle | checkpoint, loss curve, training log |
| 2 Sep | intrinsic eval: PPL, BPB, loss curves | tables and figures |
| 3 Sep | generation eval and diversity diagnostics | BLEU/chrF/ROUGE-L tables, samples |
| 4 Sep | attention analysis | heatmaps, entropy, distance |
| 5 Sep | report, README, Drive upload, push | submission, with hours to spare |

Two days of that are training runs that mostly babysit themselves, so evaluation
code can be written while they run. The schedule has no slack on 5 September by
design — the buffer is that training is expected to take 4–6 hours against a
budget of two full days.

## 7. Risks

**Achieved utilisation is much lower than 15%.** The estimate assumes a
reasonably efficient hand-written attention. If the first Kaggle run shows a
throughput implying 30+ hours, the response is to cut the token budget rather
than the model — a smaller D still trains a valid model, and the report states
the budget honestly.

**Kaggle quota exhausted by failed runs.** Mitigated by doing all debugging on
the Mac and never starting a GPU run that has not already completed a tiny
version on CPU.

**Loss diverges.** Usually warmup too short or learning rate too high at this
scale. Symptom is a loss spike that does not recover. Response is to resume from
the last good checkpoint with a lower peak rate, which is exactly what
checkpointing is for.

**The optional full-corpus Marathi run overruns the schedule.** It is explicitly
last: it starts only if both 500M runs, the full evaluation suite and the report
are already finished. `tools/pack_tokens.py` is resumable precisely so the extra
tokens can be packed later without re-encoding the first 500M.

## 8. Standing practices carried over from Phase 1

Inline comments explaining what each non-obvious block of code does and why,
written as ordinary engineering commentary. Docstrings on every method, which
the specification requires explicitly. Decisions appended to
`report/phase1_decisions.md` as D-044 onward, in the same form: what was
believed, what was measured, what changed. Superseded results kept and marked
rather than overwritten. A small verified step before any long run.
