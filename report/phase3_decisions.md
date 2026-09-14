# Phase 3 — Decisions Log

Same format as `phase1_decisions.md`, and the numbering continues from it: the
decision, why, and what would change our mind. Entries that record a mistake say
so plainly and keep the superseded numbers, because a result that was withdrawn
is only honest if it is still visible.

All figures below come from `report/phase3_final.ipynb`, run unattended on
Kaggle (2 × Tesla T4, 43.2 minutes, 26 training runs and 30 evaluations) on
14 September 2026. The calibration runs made earlier the same day, before the
notebook was made batch-runnable (D-052), reproduced identically — seeds are
fixed — so no figure here has two versions.

---

## D-044 — Five reasoning families over entities the corpus actually contains

**Decision.** The reasoning set has five families — `compare_two`,
`superlative_three`, `transitive_2hop`, `transitive_3hop`, `equality` — built
from names, attributes and numeral conventions verified against each language's
own training split before use. One surface pattern, `p_less`, is held out of
training entirely.

**Why.** A synthetic task is only a test of reasoning if the model is not also
being asked to read words it has never seen. Every lexical item, genitive form
and oblique stem was checked for attestation in a 400 MB sample of that
language's train split, so a wrong answer cannot be blamed on an unknown word.
The held-out pattern separates "learned this template" from "learned the
relation": a model that only memorised surface forms should collapse on
`p_less`.

**What would change it.** Nothing in this phase. If the models had shown real
lifts, the next step would be families whose answer is not one of the entities
named in the prompt, since selecting among visible options is the easiest
possible form of the task.

**Artifacts.** `report/phase3_lexicon_evidence.json`,
`report/phase3_reasoning_data_{marathi,konkani}.json`,
`report/phase3_reasoning_sample_{marathi,konkani}.jsonl`.

---

## D-045 — The rationale chain used `>`, which neither vocabulary contains

**What happened.** The first version of the chain-of-thought target wrote the
ordering as `अ > ब > क`. Neither 2,500-piece vocabulary contains `>`, so byte
fallback fired on it: 2.56% of the rationale tokens were `<0xNN>` pieces.

**Why that mattered.** Byte-fallback pieces are exactly the tokens the model has
the least evidence about — they appear in pretraining only where the corpus
contained characters the tokenizer could not otherwise represent. Putting one of
them at the centre of every reasoning chain meant the most load-bearing token in
the target was the one the model understood least.

**Fix.** The chain is now `"{attribute}: a, b, c"` — a Devanagari attribute noun,
a colon and comma-separated names, all of which are ordinary pieces. Measured
byte-fallback rate after the change: 0.000000%.

**What this shows.** The check that caught it — `token_check()` in
`tools/make_reasoning_data.py`, which reports the byte-fallback rate of generated
data — exists because of this. It runs on every generated set now.

---

## D-046 — The Konkani equality adjective agrees for gender; the Marathi one does not

**Decision.** Konkani `EQUAL` is keyed by the attribute's grammatical gender —
सारकी / सारकें — while Marathi uses the invariant समान.

**Why.** Corpus counts in the Konkani train split: सारकी 6,289, सारकें 5,046,
सारको 4,960, with agreeing noun–adjective pairs attested for each. Writing
`पिराय सारकें` would have been ungrammatical, and a model penalised for producing
the grammatical form instead would be measuring our error, not its reasoning.

**The side effect, which is a real limitation — see D-050.** Because Marathi's
adjective does not inflect, the Marathi `equality` family has exactly **one**
gold label across all 142 test items. Konkani's has two. That asymmetry turned
out to matter more than the grammar did.

---

## D-047 — The first sweep varied sample count at a learning rate that was destroying the model

**What happened.** The first Phase 3 run swept finetuning sample count over
500 / 1,000 / 2,000 / 4,000 / 8,000 while holding the learning rate at 1e-4 for
three epochs. Perplexity on the pretraining test split, as sample count rose:

| | 500 | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---:|---:|---:|---:|---:|
| Marathi (from 8.55) | 30.45 | 140.62 | 1,367.40 | 12,793.95 | **22,126.01** |
| Konkani (from 26.14) | — | — | — | — | **98,300.48** |

Reasoning accuracy was **14.10% at every one of the five sizes**, identical to
two decimal places, against a 25.95% chance floor. Format compliance was
99.6% / 99.9% throughout.

**What was wrong with it.** The recipe was destroying the pretrained model at
every point on the curve, so the sweep measured the same failure five times.
Accuracy identical to two decimals across five different training sets is not a
finding about sample count; it is the signature of a model that has stopped
reading its input. Learning rate had to be calibrated before anything else could
be varied, and it was not.

**Why it was not caught during training.** Validation loss looked healthy — 0.35
and falling. Under prompt-masked SFT the loss is computed over the answer span
only, and in the answer-only target that span is about five tokens of which most
are the answer marker. A model that emits the marker reliably and then guesses
scores well on that loss while learning nothing. This is the concrete reason the
evaluation carries a separate language-model check: the training objective could
not see the damage.

**Correction.** Order reversed. `report/phase3_final.ipynb` calibrates learning
rate and training target first (D-048, D-049), then varies epochs, then varies
sample count at a setting that leaves the model intact. The superseded notebook
is kept at `report/phase3.ipynb`.

---

## D-048 — Learning rate 5e-6, and the boundary past which the model is gone

**Decision.** 5e-6 for all reported runs.

**Measurement.** Learning rate × training target, one epoch, N = 8,000, both
languages. Perplexity relative to each model's own pretrained checkpoint:

| lr | Marathi, answer-only | Konkani, answer-only | Marathi, chain | Konkani, chain |
|---:|---:|---:|---:|---:|
| 5e-6 | ×1.09 | ×1.08 | ×1.13 | ×1.15 |
| 2e-5 | ×1.32 | ×1.36 | ×1.40 | ×1.51 |
| 1e-4 | **×470.82** | **×80.77** | ×16.85 | ×9.36 |

**Why.** A reasoning score taken from a model whose perplexity rose a
hundredfold is not a result about reasoning. We fixed a forgetting budget of
×1.5 — at most a 50% rise — before looking at any accuracy number, and treated
anything outside it as unusable regardless of how it scored. 1e-4 falls outside
by two orders of magnitude even at a single epoch; 2e-5 stays inside but buys no
accuracy over 5e-6.

**What would change it.** A budget is a judgement, not a law. At ×1.5 the
2e-5 chain-of-thought run for Konkani (×1.51) is excluded by a hair. It would not
change the conclusion — its accuracy is 19.00% against 25.95% — but a reader who
prefers ×2.0 should know which rows move.

---

## D-049 — Answer-only, not chain-of-thought, against our own expectation

**Decision.** The reported models are trained on the answer alone.

**What we expected.** The answer-only target gives roughly five tokens per
example, most of them the answer marker, so very little of the gradient is tied
to the reasoning itself. The chain-of-thought target produces 25–30 tokens that
genuinely depend on the input. We expected the chain to help, and put it in the
grid as a first-class factor rather than choosing by hand precisely so the
expectation could be tested.

**What happened.** It is worse at every learning rate, in both languages, and it
gets worse with training:

| epochs (lr 5e-6, N=8,000) | Marathi chain | Konkani chain | Marathi answer-only | Konkani answer-only |
|---:|---:|---:|---:|---:|
| 1 | 9.40% | 21.70% | 20.20% | 24.20% |
| 3 | 10.80% | 20.10% | 16.20% | 25.10% |
| 6 | **5.30%** | 20.00% | 19.60% | **28.20%** |

Perplexity also degrades faster under the chain (Marathi ×1.13 → ×1.29 over the
same range).

**Why, as far as the diagnostic shows.** Under the chain the model emits *more*
distinct predictions than there are gold answers — 11 or 12 against 6 for
Marathi — meaning it produces entity names that were never among the options in
that prompt. A longer target gives more opportunity to leave the answer set. At
this model scale the extra tokens bought drift, not reasoning.

---

## D-050 — The Marathi equality family admits a constant answer, and the model found it

**What the diagnostic shows.** Marathi, answer-only, 5e-6, one epoch:

| family | n | accuracy | uniform chance | majority | top prediction | share | distinct preds / golds |
|---|---:|---:|---:|---:|---|---:|---:|
| `compare_two` | 143 | **0.00%** | 33.33% | 21.68% | `समान` | 100.0% | **1 / 6** |
| `equality` | 142 | **99.30%** | 33.33% | 100.00% | `समान` | 100.0% | 1 / 1 |
| `superlative_three` | 143 | 10.49% | 25.00% | 20.28% | `गीता` | 32.2% | 5 / 6 |
| `transitive_2hop` | 286 | 11.89% | 25.00% | 19.58% | `गीता` | 33.2% | 6 / 6 |
| `transitive_3hop` | 286 | 4.20% | 20.00% | 18.53% | `विजय` | 57.3% | 3 / 6 |

One prediction, six possible golds, zero correct — and the same token scoring
99.30% one family over. The model learned a single rule, *say समान*, which is
perfectly right in one family and perfectly wrong in another.

**The limitation this exposes, which is ours.** Because Marathi समान does not
inflect (D-046), every one of the 142 Marathi `equality` items has the same
answer. One fifth of the training set is therefore solvable by a constant, and
that constant is the single most-rewarded token in the whole set. Konkani's
equality family has two labels and Konkani collapsed less far. We cannot prove
the asymmetry caused the difference from two languages, but it is the most
plausible mechanism available and it is a flaw in the data we generated, not in
the model.

**Corroboration.** At 1e-4 the collapse is total and unambiguous: accuracy goes
to exactly 0.00% on all four non-trivial families and 100.00% on `equality`, in
both languages.

**How much it distorts the headline figure — measured, after the final run.**
Model L's 28.20% clears both floors. Removing the 142 `equality` items removes
the result entirely:

| | all families | excluding `equality` |
|---|---:|---:|
| Marathi accuracy | 19.60% (uniform 25.95%) | **6.41%** (uniform 24.72%) |
| Marathi z | −4.61 | **−12.50** |
| Konkani accuracy | 28.20% (uniform 25.95%) | **16.90%** (uniform 24.72%) |
| Konkani z | +1.64, p ≈ 0.051 | **−5.34** |

The 142 equality items contribute 13.7 of Model L's 28.2 points against a uniform
expectation of 4.7 — a nine-point surplus that offsets a seven-point deficit
everywhere else. Neither model beats chance on any family that requires comparing
the entities. The degenerate family did not merely inflate a number; it created
the only positive result in the phase.

**What would change it.** Vary the equality answer some other way — a
grammatical hedge, an explicit "neither", or numeric equality stated as a value
rather than an adjective — and regenerate. We did not, because the finding was
established two days before the deadline and regenerating would have invalidated
every run above. It is recorded instead.

---

## D-051 — Two floors, and a per-item significance test rather than one chance number

**Decision.** Every accuracy is reported against both a uniform-chance floor and
a majority-class floor, with a one-tailed test that treats each item as its own
Bernoulli trial.

**Why uniform chance alone is not enough.** Chance is not one probability: it is
1/(k+1) per item, and k varies by family, so the mean is 25.95% but the variance
is the sum of p(1−p), not n·p(1−p) at a single p. More importantly, a model that
has stopped reading its input does not guess uniformly — it answers the
commonest label per family, which scores *better* than uniform. That majority
floor is 31.1% for Marathi and 23.3% for Konkani.

**What it changes.** Marathi's final model, 19.60%, is below both floors at
z = −4.61. Konkani's, 28.20%, clears both — but by 2.25 points on 1,000 items,
about 1.6 standard errors, one-tailed p = 0.051. We report it as marginal, and
D-050 shows it is an artifact of the `equality` family rather than a small real
effect. Without that family Konkani is at z = −5.34.

**The epoch trend does not rescue it either.** Twelve epochs was added to see
whether Konkani's rise (24.20 → 25.10 → 28.20) continued. It does not: 27.40% at
twelve, with perplexity up to ×1.28. Both languages peak at six and turn over, so
six is a plateau rather than a point on a climb.

**What would change it.** A larger test set. At n = 1,000 a lift has to reach
roughly 2.7 points to clear p < 0.05, and the lift we have is 2.25.

---

## D-052 — The notebook was rewritten to require no decisions while it runs

**What happened.** The calibration notebook asked the operator to read a table
and set three constants by hand between cells. That forced it to be run
interactively. The Kaggle session was lost partway through, the kernel restarted,
and `/kaggle/working` dropped from roughly 9 GB to 132 KiB — every intermediate
checkpoint and every evaluation JSON gone. The numbers survived only because
they had been printed to the notebook output and read off it.

**Fix.** `report/phase3_final.ipynb` contains no manual step. Every choice is
fixed in its first cell with the evidence beside it, so the whole thing runs as a
batch commit, server-side, and cannot be killed by a browser session. Results are
written to disk after each experiment rather than at the end. The checkpoints are
verified against the language recorded inside them before any GPU time is spent,
and the measured pretrained perplexity is asserted against the Phase 2 figures.

**The general point.** Any pipeline step that needs a human decision in the
middle cannot be re-run unattended, and a step that cannot be re-run unattended
will eventually be lost. The same reasoning produced the resumable checkpointing
in Phase 2.

---

## D-053 — What the phase concludes

**The claim.** Supervised finetuning on 8,000 templated reasoning items taught
both models the answer format completely — format compliance 0% → 99.60% and
99.90% — and taught them no measurable amount of the task. On the four families
that require comparing the entities, accuracy is 6.41% (Marathi) and 16.90%
(Konkani) against a 24.72% chance floor: both significantly *below* guessing,
which is the signature of a systematic wrong policy rather than of noise.

**The generalisation test confirms it independently.** `p_less`, the surface
pattern held out of training since the data was designed (D-044), scores 0.00%
and 4.55%, against 18.18% and 34.62% on the trained `p_more` pattern that
expresses the same relation inverted. Whatever was acquired is tied to specific
templates, not to the comparative relation underneath them.

**And the attention analysis agrees.** Mean entropy over all 56 heads moved
+0.050 bits in Marathi and +0.001 in Konkani. Finetuning changed the output
distribution and left the computation untouched — which is also why perplexity
held at ×1.14 and ×1.18. Three independent measurements, one conclusion.

**What supports it.** A pretrained baseline measured identically; a chance floor
and a majority floor; a forgetting budget fixed in advance; a learning-rate grid
that locates the boundary past which the base model is destroyed; a per-family
diagnostic showing the failure is a collapse onto one answer rather than noise;
and a training-target comparison that falsified our own expectation.

**What we would do with more time.** Fix the degenerate Marathi equality label
(D-050); enlarge the test set so a 2-point lift is testable (D-051); and try a
family whose answer is not among the entities named in the prompt, since
selecting from visible options is the easiest form of the task and the models
could not do even that.
