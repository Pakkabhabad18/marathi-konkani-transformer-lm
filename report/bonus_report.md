# Bonus — Marathi Without Positional Embeddings

An ablation of Phase 2 decision D-046. Model H (Marathi) was retrained from
scratch with the learned absolute positional embedding removed, alongside a
control retrained in the same session with it intact. Identical configuration,
seed, data and 500M-token budget; the only difference is 262,144 parameters.

One unattended run of `report/bonus_nope.ipynb` on Kaggle, 2 × Tesla T4, both
arms in parallel, 3.45 hours. Decisions and the two mistakes made along the way
are in `report/bonus_decisions.md` (B-001 to B-006).

---

## 1. Results

| | control | ablated | change |
|---|---:|---:|---:|
| parameters | 24,892,356 | 24,630,212 | −262,144 |
| validation loss | 2.1356 | 2.2131 | +0.0775 |
| **validation perplexity** | **8.4621** | **9.1439** | **×1.081** |
| test perplexity | 9.0846 | 9.8451 | ×1.084 |
| test cross-entropy (nats/token) | 2.2066 | 2.2870 | +0.0804 |
| **test bits per byte** | **0.4767** | **0.4941** | **+0.0174** |
| optimizer steps | 3,814 | 3,814 | — |
| wall clock | 3.45 h | 3.41 h | — |

**The control reproduced Phase 2 exactly: 8.4621 against 8.4621, to four decimal
places.** The ablation comparison is therefore between two things that are
genuinely comparable, and the Phase 2 pretraining run is reproducible across
sessions weeks apart.

**Removing positional embeddings costs 8.1% perplexity.** That is far less than
we predicted, and §4 explains why.

Figure: `report/figures/bonus_loss.png`.

---

## 2. What we predicted, and why it was wrong

B-004 recorded the prediction before the run, precisely so it could not be
rationalised afterwards. In outline:

> Removing positional embeddings should hurt badly. Self-attention is
> permutation invariant […] It should not be total, because causal masking leaks
> position […] What would falsify our understanding: an ablated perplexity close
> to the control — say within 10% — would mean the learned embedding was
> contributing almost nothing over the causal mask's leak.

The measured cost is **8.1%**. The falsification condition fired.

The prediction had the mechanism right and the magnitude badly wrong. We
expected the causal mask's leak to be a weak residual signal and the learned
embedding to be doing most of the work. It is the other way round: the mask
carries almost all of the usable positional information at this scale, and the
262,144-parameter embedding is worth about 0.08 nats per token on top of it.

The initialisation probe that motivated the prediction — permuting a 16-token
input moves an untrained ablated model's last-position logits by 1.15 — was
evidence that the leak *exists*. We read it as evidence the leak was small. It
was not evidence about magnitude at all.

---

## 3. How the gap develops during training

Validation perplexity, ablated ÷ control, as tokens accumulate:

| tokens seen | control | ablated | ratio |
|---:|---:|---:|---:|
| 33M | 42.95 | 42.65 | 0.993 |
| 66M | 22.50 | 25.95 | **1.153** |
| 131M | 13.45 | 14.95 | 1.112 |
| 262M | 9.75 | 10.64 | 1.092 |
| 393M | 8.75 | 9.42 | 1.077 |
| 500M | 8.46 | 9.14 | 1.081 |

At 33M tokens the two are indistinguishable — neither model has learned anything
positional yet, and the ablated arm is marginally *ahead*. The gap opens to its
widest, 15.3%, at 66M tokens, then closes steadily to 8% and flattens.

That shape is the finding. The control gets positional information for free at
the input and can use it immediately. The ablated model has to construct it, and
the first third of training is largely spent doing so. Once it has, it recovers
about half the gap and then tracks the control at a constant offset.

---

## 4. Where the positional information comes from instead

Per-layer attention statistics, 32 sequences of 256 tokens from the test split.

| layer | entropy, control | ablated | change | mean distance, control | ablated | change |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.044 | 5.788 | **+0.744** | 30.61 | 59.26 | **+28.65** |
| 1 | 5.060 | 5.945 | **+0.885** | 31.55 | 57.66 | **+26.11** |
| 2 | 3.902 | 5.650 | **+1.749** | 11.50 | 34.80 | **+23.30** |
| 3 | 2.956 | 3.807 | +0.851 | 7.95 | 7.74 | −0.21 |
| 4 | 3.911 | 3.213 | **−0.698** | 18.08 | 11.13 | **−6.95** |
| 5 | 3.957 | 4.297 | +0.340 | 51.07 | 32.98 | **−18.09** |
| 6 | 5.039 | 4.863 | −0.176 | 43.84 | 45.73 | +1.89 |
| **all 56 heads** | **4.267** | **4.795** | **+0.528** | **27.80** | **35.62** | **+7.81** |

The reorganisation is systematic, and it runs in opposite directions at the two
ends of the stack.

**Layers 0–2 become far more diffuse.** Entropy rises by up to 1.75 bits and
mean attention distance roughly doubles or triples. Phase 2's Model H used its
early layers for local work — layer 2 averaged 11.5 positions. Without a
positional signal at the input, an early head cannot form a local window,
because "nearby" is not yet a thing the representation knows about. So the early
layers do a broad, undirected aggregation pass instead.

**Layers 4 and 5 become markedly more local.** Layer 4's entropy *falls* by 0.70
bits — the only layer in either model to sharpen — and its attention distance
drops from 18.1 to 11.1 positions. Layer 5 drops from 51.1 to 33.0.

**Layer 3 is the hinge.** Its attention distance is essentially unchanged, 7.95
against 7.74, in a stack where everything above and below it moved by 7 to 29
positions.

Read together: the ablated model reconstructs locality in the middle of the
network rather than receiving it at the input. The causal mask's leak is not a
signal that can be read off directly — position *t* attends over exactly *t+1*
tokens, and turning that count into something usable takes computation. The
model spends its first two or three layers doing that computation, and only then
can it do the local work that the control's layer 2 was already doing.

The cost is visible in the budget: three layers partly spent recovering what
262,144 parameters would have supplied for free, which is what an 8% perplexity
gap in a seven-layer model looks like.

Heatmaps for layers 0, 3 and 6 in both conditions:
`report/figures/bonus_attn_{control,ablated}_*`.

---

## 5. Generation

Sampling at four temperatures, 64 prompts each, against the Phase 2 protocol.

| | control | ablated |
|---|---:|---:|
| greedy — 4-gram repetition rate | 0.6717 | **0.7230** |
| greedy — distinct-2 | 0.2355 | 0.2026 |
| T = 0.5 — chrF | 25.42 | **21.88** |
| T = 0.5 — 4-gram repetition rate | 0.4116 | **0.5146** |
| T = 0.5 — distinct-2 | 0.4023 | 0.3310 |
| T = 1.0 — chrF | 26.05 | 25.67 |
| T = 1.0 — 4-gram repetition rate | 0.0975 | 0.1005 |

BLEU and ROUGE-L move by less than a point in both directions across
temperatures and are not informative here — Phase 2 §5 already recorded why
these metrics carry little signal for open-ended continuation at this scale.

The repetition numbers are informative. The ablated model loops more, and the
effect is strongest where sampling is least random: 4-gram repetition rises from
0.67 to 0.72 greedily and from 0.41 to 0.51 at T = 0.5, while at T = 1.0 the two
are indistinguishable. Unique bigrams at T = 0.5 fall from 3,270 to 2,690.

That is consistent with §4. A model with a weaker sense of position has more
trouble noticing that it has already produced a span, so it re-produces it.
Temperature masks the effect by injecting noise the model's own representation
is not supplying.

---

## 6. What this says about D-046

D-046 chose learned absolute embeddings over sinusoidal, and spent 262,144
parameters — 1% of the budget — on them. This ablation says that purchase bought
**8.1% perplexity and a measurable reduction in degenerate repetition**, and that
the alternative is not a broken model but a slower-learning, more repetitive one
that reorganises three of its seven layers to compensate.

Whether that is a good trade depends on what the parameters would otherwise buy.
262,144 parameters is about one twelfth of a transformer block, so the honest
comparison is 8.1% perplexity against a twelfth of a layer — and on that
comparison D-046 was clearly right. The decision is not vindicated by the
ablation being catastrophic; it is vindicated by being cheap.

The result also suggests the decision mattered less than the Phase 2 reasoning
implied. D-046 argued from permutation invariance: without positional
information the model "cannot distinguish dog bites man from man bites dog".
That is true of bidirectional self-attention and false of a causal decoder, and
the entropy table shows exactly how the causal version gets around it. The
argument was right about the conclusion and wrong about the reason.

---

## 7. Limitations

**One language, one seed.** Marathi only, one run per arm. Run-to-run variance is
not measured, so the 8.1% figure should be read as one measurement rather than an
estimate with an interval. The direction is not in doubt — the gap is present at
every evaluation point from 66M tokens onward — but the magnitude is a single
sample.

**One budget.** At 500M tokens the gap was still narrowing slightly. A longer run
might close it further; the ablated model was still learning positional structure
when the schedule ended.

**Attention statistics are averages over 32 sequences.** The per-layer story in
§4 is consistent and large relative to the Phase 3 finetuning comparison, where
the same measurement moved by 0.05 bits — but 32 sequences is a small sample for
a claim about individual heads, and §4 makes claims only about layer means.

**The smoke test did not test the ablation.** B-006. It was fixed after the run
started, and the ablation was confirmed instead from the parameter counts in the
two training logs and from `verify_model.py --full --no-positional-embeddings`
passing 24/24 before the run.

---

## 8. Reproduction

```
# Both arms, one per GPU, identical everything except the flag
python3 tools/train.py --language marathi --max-tokens 500000000 --device cuda \
  --data-dir marathi/data/packed --out-dir out/control

python3 tools/train.py --language marathi --max-tokens 500000000 --device cuda \
  --no-positional-embeddings \
  --data-dir marathi/data/packed --out-dir out/ablated

# Verify the ablated architecture before spending the compute
python3 tools/verify_model.py --full --no-positional-embeddings   # 24/24

# Evaluate and compare
python3 tools/evaluate.py --language marathi --split test \
  --checkpoint out/ablated/pretrain_nope_best.pt \
  --tokenizer marathi/tokenizer/marathi_bpe.model

python3 tools/attention_analysis.py --language marathi --split test \
  --checkpoint out/ablated/pretrain_nope_best.pt \
  --tokenizer marathi/tokenizer/marathi_bpe.model
```

The whole ablation runs unattended from `report/bonus_nope.ipynb` as a Kaggle
batch commit with `lma-bonus-code` and `lma-phase2-data` attached. It takes no
input while running.

---

## 9. Deliverables

| file | what |
|---|---|
| `common/model/config.py` | `no_positional_embedding`, default False |
| `common/model/lm.py` | the module is `None` when ablated, not zeroed |
| `tools/train.py` | `--no-positional-embeddings`, separate run name |
| `tools/verify_model.py` | three checks that the removal happened, plus every existing check against the ablated architecture |
| `report/bonus_nope.ipynb` | the run that produced every number here |
| `report/bonus_decisions.md` | B-001 to B-006, including what failed |
| `report/bonus_summary.json` | both arms, full validation curves |
| `report/bonus_eval_{control,ablated}.json` | test-set intrinsic and generation metrics |
| `report/bonus_{control,ablated}_log.csv` | per-step training logs |
| `report/figures/bonus_loss.png` | both curves, with the Phase 2 line |
| `report/figures/bonus_attn_*` | attention heatmaps, both conditions |

Both checkpoints are on Google Drive with the Phase 2 and Phase 3 ones; they are
not in the repository.

Everything above is on branch `bonus-no-positional`. Nothing on `phase-3` was
modified, so every Phase 1–3 deliverable is byte-identical to what was evaluated.
