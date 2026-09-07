# Phase 2 — Model, Pretraining and Evaluation

Model H: Marathi. Model L: Konkani (Devanagari). Branch `phase-2`, 7 September 2026.

Two decoder-only Transformers built from primitive PyTorch layers, pretrained
independently on equal token budgets, and evaluated on their own held-out test
splits. Every figure below is produced by a script in `tools/` and can be
regenerated with the commands in section 10.

## 1. Results

| | Marathi (H) | Konkani (L) |
|---|---:|---:|
| parameters | 24,892,356 | 24,892,356 |
| training tokens | 499,908,608 | 499,908,608 |
| optimizer steps | 3,814 | 3,814 |
| wall-clock (T4) | 3.57 h | 3.41 h |
| test cross-entropy | 2.1544 nats/token | 3.2794 nats/token |
| test perplexity | **8.62** | **26.56** |
| **test bits per byte** | **0.4728** | **0.7086** |
| BLEU-4 (best setting) | 8.03 | **0.00** |
| chrF (best setting) | 27.82 | 21.02 |
| ROUGE-L (best setting) | 13.93 | 5.53 |

The two models are identical in architecture, vocabulary size, hyperparameters
and token budget — equal to the token, because both used the same
131,072-token optimizer step and the same 3,814 steps. The only variable is the
data. Every difference below is therefore attributable to the corpus rather than
to one model having been given more of anything.

## 2. Architecture

| | value |
|---|---|
| `d_model` | 512 |
| layers | 7 |
| heads | 8 (`d_head` 64) |
| feed-forward inner dim | 2048 |
| context length | 512 |
| positional encoding | learned absolute |
| normalization | pre-norm LayerNorm, final LayerNorm |
| activation | GELU |
| dropout | 0.1 |
| output head | untied |

| component | parameters |
|---|---:|
| 7 transformer blocks | 22,066,688 |
| token embedding | 1,280,000 |
| positional embedding | 262,144 |
| final LayerNorm | 1,024 |
| output projection | 1,282,500 |
| **total** | **24,892,356** |

Configs in `marathi/configs/model_config.json` and
`konkani/configs/model_config.json`, written by the training run itself rather
than by hand, so they cannot drift from the weights they describe.

Depth over width: `d_model` costs quadratically inside a block (`12 × d²` per
layer) while depth costs linearly, so seven layers at 512 gives more sequential
composition than four at 768 for the same budget. Six layers would leave ~3M
parameters unused; eight overshoots to 28M.

The output head is untied. Tying saves `vocab × d_model` = 1.28M parameters,
which at vocabulary 2,500 is 5% of the budget — the saving that justifies tying
at vocabulary 50,000 is mostly unavailable here, so the parameters were spent on
letting the input and output representations of a token differ. This resolves
the question D-043 left open at the end of Phase 1.

Correctness was established before any GPU time was spent
(`tools/verify_model.py`, 18 checks, run again on the training machine):

- **Changing the token at position t+1 never changes the logits at position t** —
  tested at every position, largest difference exactly 0.000e+00. This is the
  check that matters most: a leaking causal mask still produces a falling loss
  curve and a plausible perplexity, just one that is far too good because the
  model has been reading the answer.
- Changing token 0 *does* change the last position — without this, a model that
  ignored its input entirely would pass the first test.
- Attention weights above the diagonal are exactly zero and rows sum to 1, which
  follows from masking with `-inf` *before* the softmax rather than zeroing
  after. Zeroing after leaves rows summing to less than 1 and silently scales the
  output by an amount that varies with position.
- Untrained loss 7.85 against ln(2500) = 7.82, the near-uniform prediction.

## 3. Pretraining

AdamW, β = (0.9, 0.95), weight decay 0.1 applied to matrices but not to biases,
LayerNorm gains or embeddings. Peak learning rate 3e-4, linear warmup over 2% of
steps then cosine decay to 10% of peak. Effective batch 131,072 tokens
(32 × 512 × 8 gradient accumulation steps). Gradient clipping at global norm 1.0.
Mixed precision with fp32 master weights.

Both models trained simultaneously, one per T4, on Kaggle.

Four signals the runs are healthy, from `report/training_logs/`:

- The learning-rate schedule completed: final `lr` 3.00e-05, exactly the 10%
  cosine floor. A truncated run would have stopped mid-decay.
- Gradient norms ended at 0.5–0.6 on both. Stable — not exploding, not vanishing.
- No loss spikes. A spike is the signature of a resume that lost optimizer state;
  every checkpoint carries optimizer, scheduler and scaler state precisely so
  that cannot happen, and the curves are the evidence it did not.
- No overfitting. Validation loss sits slightly *below* training loss throughout,
  which is correct rather than odd: training loss is measured per batch with
  dropout active, validation with dropout off.

Figures: `report/figures/phase2_loss_marathi.png`,
`phase2_loss_konkani.png`, `phase2_loss_comparison.png`.

## 4. Intrinsic evaluation

Measured on the **test** split — validation was used to select the best
checkpoint during training, so reporting on it would be optimistic. 256 windows
of 512 tokens each, evenly spaced across the split because the splits are
source-stratified and the first N tokens would be a single source.

| | Marathi | Konkani |
|---|---:|---:|
| cross-entropy | 2.1544 nats/token | 3.2794 nats/token |
| perplexity | 8.62 | 26.56 |
| bits per byte | 0.4728 | 0.7086 |
| bytes per token | 6.574 | 6.677 |
| tokens scored | 131,072 | 131,072 |
| bytes scored | 861,607 | 875,138 |

### Why bits-per-byte, and why the headline number changes

Perplexity is per *token*, and a model whose tokenizer splits text into more,
shorter pieces faces an easier per-token problem without being a better model of
the language. Bits per byte divides the same likelihood by UTF-8 bytes instead,
removing the tokenizer from the denominator: bytes are a property of the text,
not of how it was segmented.

The two framings give different-sounding answers to the same question:

- **Perplexity ratio: 3.08×** (26.56 / 8.62)
- **Bits-per-byte ratio: 1.50×** (0.7086 / 0.4728)

Both are correct. Perplexity is exponential in the loss, so a 1.125-nat gap
becomes a factor of e^1.125 = 3.08; bits per byte is linear in it. The honest
statement of the resource gap is that **Model L needs about 50% more bits to
encode a byte of its language than Model H does** — not that it is "three times
worse".

The tokenizers turn out to compress almost identically, 6.574 against 6.677 bytes
per token, a 1.6% difference. So perplexity happens to be more comparable here
than it usually would be. That is a property of these two tokenizers, discovered
by measuring rather than assumed, and it would not hold for an arbitrary pair.

## 5. Generation quality

Continuations from 24 fixed held-out prefixes of 64 tokens, generating 128 tokens
each, under greedy decoding and temperatures 0.5, 1.0 and 1.5. The same prompts
are used for every setting and for both models.

### Marathi

| setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | 4-gram repetition |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 7.35 | 25.84 | 13.14 | 0.145 | 0.283 | 0.629 |
| T = 0.5 | **8.03** | **27.82** | **13.93** | 0.216 | 0.529 | 0.278 |
| T = 1.0 | 3.57 | 27.34 | 9.80 | 0.328 | 0.830 | 0.047 |
| T = 1.5 | 1.13 | 22.32 | 4.33 | 0.403 | 0.955 | 0.001 |

### Konkani

| setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | 4-gram repetition |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 0.00 | 9.80 | 3.25 | 0.076 | 0.111 | **0.875** |
| T = 0.5 | 0.00 | 16.36 | **5.53** | 0.170 | 0.388 | 0.463 |
| T = 1.0 | 0.00 | 20.47 | 4.47 | 0.326 | 0.865 | 0.037 |
| T = 1.5 | 0.00 | **21.02** | 2.30 | 0.416 | 0.976 | 0.001 |

### Konkani's BLEU is exactly zero, and the reason is precise

Not a bug, and not rounding. Corpus BLEU is the geometric mean of modified
n-gram precisions for n = 1..4, and the per-order precisions are:

| | 1-gram | 2-gram | 3-gram | 4-gram |
|---|---:|---:|---:|---:|
| Marathi, T = 0.5 | 15.451 | 8.852 | 6.267 | 4.849 |
| Konkani, T = 0.5 | 6.894 | 0.624 | **0.000** | **0.000** |

Across 24 generations of 128 tokens, Model L produced **not one trigram** that
appears in its reference continuation. A single zero at any order makes the
geometric mean zero, and BLEU with it. Marathi manages 4.8% precision even at
4-gram order.

This is reported rather than smoothed away. Smoothed sentence-BLEU would return
a small positive number and hide the fact that the model reproduces no
three-word sequence of the reference at all.

### What the metrics are and are not telling us

**BLEU is the wrong instrument here and the zero proves it**, in the sense that
it has no resolution left: it cannot distinguish a Konkani model that is nearly
right from one that is nonsense, because both score 0. It is precision-oriented,
word-level, and requires contiguous matches — brittle in a morphologically rich
Devanagari language where a fluent continuation that inflects a stem differently
from the reference scores nothing.

**chrF is the most informative of the three.** It works on character n-grams, so
a correct stem with a different suffix still earns partial credit, and it keeps
discriminating where BLEU has bottomed out: Konkani moves 9.80 → 21.02 across
settings, which is real signal about output quality that BLEU reports as four
zeros.

**ROUGE-L is recall-oriented and subsequence-based**, so it rewards getting
content order right even with insertions between. It is the metric that most
disagrees with chrF on the best temperature, which is itself informative — see
below.

All three compare against a *single* reference continuation. For open-ended
generation there are many acceptable continuations, so absolute values are low
for every model and only the comparison carries information.

### The temperature trade-off, and that the two models disagree about it

Marathi peaks at T = 0.5 on all three metrics simultaneously. Konkani does not:
ROUGE-L peaks at 0.5 (5.53) while chrF keeps climbing to 1.5 (21.02). There is no
setting at which Model L is simultaneously best by both measures.

That disagreement is the finding. Higher temperature makes Konkani's output more
*character-plausible* — better n-gram statistics, more varied — while making it
less *content-faithful*. Model H has a genuine operating point; Model L trades
one failure mode for another.

## 6. Degeneration

The characteristic failure of a small language model is not incoherence but
looping, and perplexity cannot see it: a repeated high-probability phrase scores
*well*. This is why the diversity diagnostics are here.

Under greedy decoding, **Konkani's 4-gram repetition rate is 0.875** — seven of
every eight 4-gram occurrences are repeats — with Distinct-1 at 0.076, meaning
only 7.6% of generated tokens are distinct. Marathi degenerates too, at 0.629 and
0.145, but less severely.

The samples show exactly this. Konkani, greedy:

```
prompt:  रॉयल एअर फोर्स फिलिंगडेल्स (राफ फिलिंगडेल्स) हें इंग्लंडांतल्या उत्तर यॉर्क मूर हांगा …
output:  …ंड्सांतल्या बार्बरा हांगाच्या बार्बरा हांगाच्या बार्बरा हांगाच्या बार्बरा हांगाच्या …
```

Marathi, greedy — the same failure, arriving later:

```
prompt:  भाजपा हा राष्ट्रवादी काँग्रेसची नवी झेरॉक्स प्रत असल्याचा खळबळजनक आरोप त्यांनी केला …
output:  …नेही भाजपला पाठिंबा दिला. गोटे यांनीही गोटे यांना पाठिंबा दिला. गोटे यांनी गोटे यांना
         पाठिंबा दिला. गोटे यांना पाठिंबा देण्यासाठी …
```

The first clause is well-formed and topical before the loop closes.

At T = 0.5 Marathi produces genuinely fluent news prose:

```
…ने या मुद्द्यावरून हल्लाबोल केला. त्यानंतर त्यांनी त्यांच्यावर जोरदार टीका केली. या आरोपांना
उत्तर देताना त्यांनी भाजपला पाठिंबा दिला. रत्नागिरी : रत्नागिरी जिल…
```

Grammatical, idiomatic, topically coherent across several sentences, and it even
reproduces the dateline convention of Marathi news ("रत्नागिरी :"). Konkani at
the same temperature starts correctly — `ंडल आनी मूर हांगा आशिल्लें. ह्या
स्टेशनाचेर ऑस्ट्रेलियन युनियनाच्या` — and then locks onto `एअर फोर्साच्या`.

**Both models are locally fluent and globally incoherent**, which is the expected
result for 25M parameters on 500M tokens. Reporting it with samples is more
useful than presenting perplexity alone and implying more than the models can do.

## 7. Attention analysis

Per-head statistics averaged over 32 held-out sequences of 256 tokens. Entropy is
normalised by log2(t+1), the maximum achievable at query position t under causal
masking — without that normalisation a head at position 5 cannot be compared with
one at position 200. Position 0 is excluded, since it can only attend to itself.

| layer | Marathi entropy | Marathi distance | Konkani entropy | Konkani distance |
|---:|---:|---:|---:|---:|
| 0 | 0.773 | 30.61 | 0.870 | 37.75 |
| 1 | 0.778 | 31.55 | 0.857 | 45.48 |
| 2 | 0.607 | 11.50 | 0.655 | 23.63 |
| 3 | **0.462** | 7.95 | 0.524 | 5.94 |
| 4 | 0.599 | 18.08 | **0.368** | 5.66 |
| 5 | 0.591 | 51.07 | 0.612 | 45.88 |
| 6 | 0.761 | 43.84 | 0.772 | 54.57 |

### The expected pattern did not appear

The standard account is early layers doing local positional work with low entropy
and short attention distance, later layers doing content-based work with higher
entropy and longer range. **Neither model does this.** Both show a U-shape:
diffuse and moderately long-range at layers 0–1, sharply focused and local at
layers 3–4, diffuse and long-range again at 5–6.

The selective work happens in the *middle* of the stack. Layers 0–1 sit at 0.77
and 0.87 normalised entropy, close to uniform — they appear to be broadcasting
context rather than selecting from it. This is recorded as a measured result that
disagrees with the prediction, not adjusted after the fact.

### Head specialisation is strong

Within a single layer, heads learn very different jobs. Konkani layer 2 holds
head 3 at mean distance 4.13 and head 2 at 105.61 — a 25× spread. Marathi layer 2
spans 3.31 to 45.35, and layer 1 reaches 92.10 on head 7 against 10.71 on head 4.

That spread is the strongest evidence that multi-head attention is doing what it
is supposed to: if the heads had collapsed onto one behaviour, a single head with
`d_head` = 512 would have served identically and the implementation would be
suspect.

### Model H against Model L

Model L is more diffuse in the early layers — 0.870 against 0.773 at layer 0,
where 1.0 is exactly uniform. Its first two layers are closer to attending to
everything equally, which is to say less structured. Model L then goes *more*
extreme in the middle (0.368 at layer 4, the lowest value in either model) and
attends further at every depth.

A model trained on a noisier, partly synthetic corpus developing weaker early
structure and more extreme swings is consistent with the intrinsic results, but
it is one observation from one pair of models and is offered as such.

Figures: `report/figures/phase2_attention_{language}_layer{0,3,6}.png`, four heads
per layer, each panel with title, axis labels and colourbar.

## 8. Resource-level comparison

The controlled setup — same architecture, same 499,908,608 tokens, same
hyperparameters — means the following differences are attributable to the data.

**The gap is real but smaller than perplexity suggests.** 1.50× in bits per byte,
not 3.08×.

**The gap is much larger in generation than in likelihood.** Model L is 1.5× worse
at predicting the next token and *infinitely* worse at BLEU, because it produces
no matching trigram at all. Likelihood is a per-position average that a model can
do respectably at while still being unable to sustain a coherent sequence;
generation compounds errors over 128 steps. Any claim about model quality resting
on perplexity alone would miss this entirely.

**Model L degenerates harder.** 0.875 against 0.629 greedy 4-gram repetition. It
falls into loops sooner and more completely, which is what a model with a weaker
grasp of long-range structure does when forced to commit to its argmax.

**Where the difference comes from.** Both corpora carried the same token count,
but Marathi's 500M tokens were drawn from 872M available words with no synthetic
text, while Konkani's came from a 506M-token corpus of which **32.2% is
machine-translated or LLM-generated** (Phase 1, D-036 to D-039). Konkani spent
98.7% of its documents to reach the budget; Marathi spent 57%. The comparison is
therefore not "more data versus less data" — the budgets were equal — but
*shallower and partly synthetic* versus *deeper and entirely human-written*.

**A test this project has not run.** Every synthetic Konkani document is labelled
in the manifests, so perplexity could be measured separately on real and
synthetic held-out text. If the model scored markedly better on the synthetic
portion, that would indicate it had learned the translation system's output
distribution rather than Konkani itself. That is the single most valuable
follow-up available here and it is left for Phase 3.

## 9. Limitations

**24 prompts is a small generation sample.** Enough to establish the qualitative
result and the zero-trigram finding, not enough for tight confidence intervals on
BLEU. Chosen because generation has no KV cache — every new token re-runs the
full forward pass — and evaluation ran on a laptop CPU.

**Single-reference metrics.** BLEU, chrF and ROUGE-L all compare against one
continuation out of many acceptable ones, so absolute values understate quality
for both models.

**Attention statistics are from 32 sequences of 256 tokens**, not the full test
split, and cover four of eight heads in the heatmaps.

**The models are under-trained relative to their capacity in the usual sense.**
500M tokens for 25M parameters is close to compute-optimal, but "optimal" here
means best use of a fixed budget, not converged. Both would improve with more
tokens; Marathi has 372M more available and unused.

## 10. Reproduction

```bash
python3 tools/verify_model.py --full

python3 tools/train.py --language konkani --max-tokens 500000000 --device cuda \
  --data-dir <packed> --out-dir <checkpoints>
python3 tools/train.py --language marathi --max-tokens 500000000 --device cuda \
  --data-dir <packed> --out-dir <checkpoints>

python3 tools/plot_training.py \
  --log-marathi <ckpt>/marathi/pretrain_log.csv \
  --log-konkani <ckpt>/konkani/pretrain_log.csv

python3 tools/evaluate.py --language konkani \
  --checkpoint <ckpt>/konkani/pretrain_best.pt \
  --tokenizer konkani/tokenizer/konkani_bpe.model \
  --split test --windows 256 --prompts 24

python3 tools/attention_analysis.py --language konkani \
  --checkpoint <ckpt>/konkani/pretrain_best.pt \
  --tokenizer konkani/tokenizer/konkani_bpe.model
```

Kaggle notebook setup, including the two datasets and the cell sequence, is in
`report/phase2_kaggle_runbook.md`.

## 11. Deliverables

| deliverable | location |
|---|---|
| Transformer implementation | `common/model/{attention,lm,config}.py` |
| Model configuration files | `marathi/configs/model_config.json`, `konkani/configs/model_config.json` |
| Parameter counts | section 2; printed by `tools/verify_model.py` |
| Training scripts | `tools/train.py`, `tools/pack_tokens.py` |
| Pretrained checkpoints | Google Drive — see README |
| Training logs | `report/training_logs/*.csv` |
| Loss curves | `report/figures/phase2_loss_*.png` |
| Perplexity / BPB | section 4, `report/phase2_eval_*.json` |
| BLEU / chrF / ROUGE-L | section 5 |
| Generated samples, diversity stats | sections 5–6, `report/phase2_eval_*.json` |
| Attention heatmaps, entropy, distance | section 7, `report/phase2_attention_*.json` |
| Resource-level comparison | section 8 |

Metric implementations are in `common/metrics.py` — BLEU-4, chrF and ROUGE-L
written directly rather than imported, since the Kaggle notebook runs with
internet disabled and a metric whose behaviour has to be explained is easier to
explain when written. Each is verified against hand-computable values: identical
strings score 100, disjoint strings score 0 on BLEU and ROUGE-L, and
`LCS("abcde","ace")` returns 3.
