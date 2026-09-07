# Phase 2 — Model, Pretraining and Evaluation

Model H: Marathi. Model L: Konkani (Devanagari). Branch `phase-2`, 7 September 2026.

We trained two decoder-only Transformers from scratch, one per language, on equal
token budgets, and evaluated each on its own held-out test split.

## 1. Results

| | Marathi (H) | Konkani (L) |
|---|---:|---:|
| parameters | 24,892,356 | 24,892,356 |
| training tokens | 499,908,608 | 499,908,608 |
| optimizer steps | 3,814 | 3,814 |
| wall-clock (T4) | 3.57 h | 3.41 h |
| test cross-entropy | 2.1544 nats/token | 3.2794 nats/token |
| test perplexity | 8.62 | 26.56 |
| test bits per byte | 0.4728 | 0.7086 |
| BLEU-4 (best setting) | 8.03 | 0.00 |
| chrF (best setting) | 27.82 | 21.02 |
| ROUGE-L (best setting) | 13.93 | 5.53 |

Both models have the same architecture, vocabulary size and hyperparameters. The
token budgets are equal exactly, not approximately: 3,814 optimizer steps of
131,072 tokens each. The differences below come from the data.

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
| total | 24,892,356 |

The training run writes its own config to `marathi/configs/model_config.json` and
`konkani/configs/model_config.json`.

We took depth over width. Inside a block `d_model` costs `12 × d²` while depth
costs linearly, so seven layers at 512 buys more sequential composition than four
at 768 for the same parameters. Six layers leaves about 3M of the budget unused;
eight overshoots to 28M.

The output head is untied. Tying would save 1.28M parameters, 5% of the budget at
vocabulary 2,500. At vocabulary 50,000 that saving is worth having; here it is
not, so we spent it on letting a token's input and output representations differ.
D-043 left this open at the end of Phase 1.

`tools/verify_model.py` runs 18 checks. We ran it before spending any GPU time,
and again on the training machine. The ones that matter:

- Changing the token at position t+1 does not change the logits at position t.
  Tested at every position; largest difference 0.000e+00. A leaking mask still
  produces a falling loss curve and a plausible perplexity, so this cannot be
  caught from training alone.
- Changing token 0 *does* change the last position. Without this check, a model
  that ignored its input entirely would pass the one above.
- Attention weights above the diagonal are exactly zero and every row sums to 1.
  That follows from adding `-inf` before the softmax. Zeroing after the softmax
  leaves rows summing to less than 1.
- Untrained loss 7.85, against ln(2500) = 7.82.

## 3. Pretraining

AdamW, β = (0.9, 0.95). Weight decay 0.1 on matrices only, not on biases,
LayerNorm gains or embeddings. Peak learning rate 3e-4, linear warmup over 2% of
steps, cosine decay to 10% of peak. Effective batch 131,072 tokens
(32 × 512 × 8 gradient accumulation steps). Gradient clipping at global norm 1.0.
Mixed precision with fp32 master weights.

Both runs went on Kaggle, one model per T4, in parallel.

From `report/training_logs/`:

- Final `lr` 3.00e-05, the 10% cosine floor, so the schedule ran to completion.
- Gradient norms over the last five logged steps: 0.567–0.620 for Marathi,
  0.509–0.562 for Konkani.
- No loss spikes, which is what a resume that lost its optimizer state would look
  like.
- Validation loss sits slightly below training loss throughout. Training loss is
  per batch with dropout active; validation is measured with dropout off.

Figures: `report/figures/phase2_loss_marathi.png`, `phase2_loss_konkani.png`,
`phase2_loss_comparison.png`.

## 4. Intrinsic evaluation

Test split, 256 windows of 512 tokens, evenly spaced across the split. We report
on test rather than validation because validation chose the checkpoint. The
windows are spaced rather than taken from the front because the splits are
source-stratified.

| | Marathi | Konkani |
|---|---:|---:|
| cross-entropy | 2.1544 nats/token | 3.2794 nats/token |
| perplexity | 8.62 | 26.56 |
| bits per byte | 0.4728 | 0.7086 |
| bytes per token | 6.574 | 6.677 |
| tokens scored | 131,072 | 131,072 |
| bytes scored | 861,607 | 875,138 |

### Bits per byte

Perplexity is per token, so it is not comparable across two tokenizers. Bits per
byte divides the same likelihood by UTF-8 bytes instead.

The two give very different-looking gaps: 3.08× on perplexity (26.56 / 8.62),
1.50× on bits per byte (0.7086 / 0.4728). Perplexity is exponential in the loss
and BPB is linear, so the 1.125-nat gap becomes e^1.125 = 3.08 in one and a factor
of 1.5 in the other. We quote the second: Model L needs about 50% more bits to
encode a byte of its language.

The two tokenizers happen to compress almost identically here, 6.574 against
6.677 bytes per token, a 1.6% difference, so perplexity is more comparable in
this case than it usually would be.

## 5. Generation quality

24 held-out prefixes of 64 tokens, 128 tokens generated from each, under greedy
decoding and temperatures 0.5, 1.0 and 1.5. The same prompts for every setting
and for both models.

### Marathi

| setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | 4-gram repetition |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 7.35 | 25.84 | 13.14 | 0.145 | 0.283 | 0.629 |
| T = 0.5 | 8.03 | 27.82 | 13.93 | 0.216 | 0.529 | 0.278 |
| T = 1.0 | 3.57 | 27.34 | 9.80 | 0.328 | 0.830 | 0.047 |
| T = 1.5 | 1.13 | 22.32 | 4.33 | 0.403 | 0.955 | 0.003 |

### Konkani

| setting | BLEU-4 | chrF | ROUGE-L | Distinct-1 | Distinct-2 | 4-gram repetition |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 0.00 | 9.80 | 3.25 | 0.076 | 0.111 | 0.875 |
| T = 0.5 | 0.00 | 16.36 | 5.53 | 0.170 | 0.388 | 0.463 |
| T = 1.0 | 0.00 | 20.47 | 4.47 | 0.326 | 0.865 | 0.037 |
| T = 1.5 | 0.00 | 21.02 | 2.30 | 0.416 | 0.976 | 0.001 |

### Why Konkani BLEU is 0.00

Corpus BLEU is the geometric mean of the modified n-gram precisions for
n = 1..4. Ours:

| | 1-gram | 2-gram | 3-gram | 4-gram |
|---|---:|---:|---:|---:|
| Marathi, T = 0.5 | 15.451 | 8.852 | 6.267 | 4.849 |
| Konkani, T = 0.5 | 6.894 | 0.624 | 0.000 | 0.000 |

Across 24 generations of 128 tokens, Model L produced no trigram that appears in
its reference continuation. One zero order zeroes the geometric mean. Marathi
still reaches 4.849% at 4-gram order.

We did not smooth. Smoothed sentence-BLEU returns a small positive number and
hides the trigram result, which is the more useful thing to know.

### Reading the three metrics

BLEU has no resolution left at this quality level: a nearly-right Konkani model
and a nonsense one both score 0. It is also brittle in a morphologically rich
Devanagari language, where a fluent continuation that inflects a stem differently
from the reference earns nothing.

chrF is the most useful of the three here. Character n-grams give partial credit
for a correct stem with a different suffix, and chrF keeps moving where BLEU is
flat: Konkani goes 9.80 → 21.02 across the four settings.

ROUGE-L is recall-oriented and based on longest common subsequence, so it
tolerates insertions between matched content. It disagrees with chrF about the
best temperature for Konkani.

All three compare against a single reference continuation. Open-ended generation
has many acceptable continuations, so the absolute values are low for both models
and only the comparison carries information.

### Temperature

Marathi peaks at T = 0.5 on all three metrics at once. Konkani does not: ROUGE-L
peaks at 0.5 (5.53) while chrF keeps climbing to 1.5 (21.02). There is no setting
where Model L is best by both.

Raising the temperature makes Konkani's output more character-plausible and less
content-faithful at the same time.

## 6. Degeneration

Perplexity cannot see looping — a repeated high-probability phrase scores well —
so we measured Distinct-n and 4-gram repetition instead.

Under greedy decoding Konkani repeats 87.5% of its 4-grams, and only 7.6% of its
generated tokens are distinct. Marathi is at 0.629 and 0.145.

Konkani, greedy:

```
prompt:  रॉयल एअर फोर्स फिलिंगडेल्स (राफ फिलिंगडेल्स) हें इंग्लंडांतल्या उत्तर यॉर्क मूर हांगा …
output:  …ंड्सांतल्या बार्बरा हांगाच्या बार्बरा हांगाच्या बार्बरा हांगाच्या बार्बरा हांगाच्या …
```

Marathi, greedy — the same failure, later:

```
prompt:  भाजपा हा राष्ट्रवादी काँग्रेसची नवी झेरॉक्स प्रत असल्याचा खळबळजनक आरोप त्यांनी केला …
output:  …नेही भाजपला पाठिंबा दिला. गोटे यांनीही गोटे यांना पाठिंबा दिला. गोटे यांनी गोटे यांना
         पाठिंबा दिला. गोटे यांना पाठिंबा देण्यासाठी …
```

The first clause is well-formed and on topic before the loop closes.

Marathi at T = 0.5:

```
…ने या मुद्द्यावरून हल्लाबोल केला. त्यानंतर त्यांनी त्यांच्यावर जोरदार टीका केली. या आरोपांना
उत्तर देताना त्यांनी भाजपला पाठिंबा दिला. रत्नागिरी : रत्नागिरी जिल…
```

Grammatical, idiomatic, coherent across several sentences, and it reproduces the
dateline convention of Marathi news (`रत्नागिरी :`). Konkani at the same
temperature opens correctly — `ंडल आनी मूर हांगा आशिल्लें. ह्या स्टेशनाचेर
ऑस्ट्रेलियन युनियनाच्या` — then locks onto `एअर फोर्साच्या`.

Both models are locally fluent and globally incoherent, which is what 25M
parameters on 500M tokens gives.

## 7. Attention analysis

Per-head statistics over 32 held-out sequences of 256 tokens. Entropy is
normalised by log2(t+1), the causal maximum at query position t, so that heads at
different positions can be compared. Position 0 is excluded; it can only attend
to itself.

| layer | Marathi entropy | Marathi distance | Konkani entropy | Konkani distance |
|---:|---:|---:|---:|---:|
| 0 | 0.773 | 30.61 | 0.870 | 37.75 |
| 1 | 0.778 | 31.55 | 0.857 | 45.48 |
| 2 | 0.607 | 11.50 | 0.655 | 23.63 |
| 3 | 0.462 | 7.95 | 0.524 | 5.94 |
| 4 | 0.599 | 18.08 | 0.368 | 5.66 |
| 5 | 0.591 | 51.07 | 0.612 | 45.88 |
| 6 | 0.761 | 43.84 | 0.772 | 54.57 |

### Layer-wise pattern

We expected low-entropy local heads early and high-entropy long-range heads late.
Neither model does that. Both are U-shaped: diffuse and moderately long-range at
layers 0–1, focused and local at layers 3–4, diffuse and long-range again at 5–6.
The selective work happens in the middle of the stack.

Layers 0–1 sit at 0.77 and 0.87 normalised entropy, close to uniform, which looks
more like broadcasting context than selecting from it.

### Head specialisation

Heads inside a layer differ sharply. Konkani layer 2 holds head 3 at mean
distance 4.13 and head 2 at 105.61, a 25× spread. Marathi layer 2 spans 3.31 to
45.35; Marathi layer 1 spans 10.71 (head 4) to 92.10 (head 7).

Had the heads collapsed onto one behaviour, a single 512-dimensional head would
have done the same job. The spread is what says the multi-head implementation is
working.

### Model H against Model L

Model L is more diffuse early — 0.870 against 0.773 at layer 0, where 1.0 is
uniform — and then goes further the other way in the middle, to 0.368 at layer 4,
the lowest figure in either model. It attends further at every depth.

This is one observation from one pair of models.

Figures: `report/figures/phase2_attention_{language}_layer{0,3,6}.png`, four heads
per layer.

## 8. Resource-level comparison

Same architecture, same 499,908,608 tokens, same hyperparameters, so the
differences are attributable to the data.

The gap is 1.50× in bits per byte, not the 3.08× that perplexity suggests.

It is much wider in generation than in likelihood. Model L is 1.5× worse at
predicting the next token but produces no matching trigram at all. Likelihood is
a per-position average, and a model can do respectably on it while being unable
to sustain a sequence; generation compounds errors over 128 steps. Perplexity
alone would not show this.

Model L also degenerates harder: 0.875 against 0.629 on greedy 4-gram repetition.

Both budgets were 500M tokens, but Marathi drew them from 872M available words
with no synthetic text, and Konkani from a 506M-token corpus that is 32.2%
machine-translated or LLM-generated (D-036 to D-039). Konkani used 98.7% of its
documents to reach the budget; Marathi used 57%. So this is not more data against
less. It is shallower and partly synthetic against deeper and human-written.

One test we did not run: the synthetic Konkani documents are labelled in the
manifests, so perplexity could be measured separately on real and synthetic
held-out text. If the model scored markedly better on the synthetic half, it has
learned IndicTrans2's output distribution rather than Konkani. Left for Phase 3.

## 9. Limitations

24 prompts is a small sample — enough for the qualitative result and the
zero-trigram finding, not for confidence intervals on BLEU. We kept it small
because generation has no KV cache, so every new token re-runs the full forward
pass, and evaluation ran on a laptop CPU.

Single-reference metrics understate quality for both models.

The attention statistics come from 32 sequences of 256 tokens rather than the
full test split, and the heatmaps show four of the eight heads.

500M tokens for 25M parameters is close to compute-optimal, but that means best
use of a fixed budget, not converged. Marathi has 372M more tokens available and
unused.

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

BLEU-4, chrF and ROUGE-L are implemented in `common/metrics.py` rather than
imported, because the Kaggle notebook ran with internet disabled. Running
`python3 common/metrics.py` checks them against hand-computable cases: identical
strings score 100, disjoint strings score 0 on BLEU and ROUGE-L, and
`LCS("abcde","ace")` returns 3.
