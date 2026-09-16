# Bonus — Decisions Log

The bonus is a single ablation: retrain one language with no positional
embeddings and measure what they were worth. It has its own numbering (B-001
onward) and its own branch, `bonus-no-positional`, because it modifies the model
code and nothing about it should touch the Phase 1–3 deliverables that were
already evaluated.

Format follows `phase1_decisions.md`: the decision, why, and what would change
our mind.

**Scope.** The thing under test is Phase 2 decision **D-046** — a learned
absolute `nn.Embedding(context_length, d_model)` added to the token embedding.
Everything else is held fixed.

---

## B-001 — One language, Marathi, and a retrained control rather than the Phase 2 checkpoint

**Decision.** Marathi (Model H). Two arms trained side by side in the same
session, one per T4: ablated on GPU 0, control on GPU 1. Identical config, seed,
data and 500M-token budget.

**Why retrain the control.** The Phase 2 Marathi checkpoint already exists and
was trained with exactly this configuration, so reusing it would have cost
nothing. We did not, because it was produced in a different Kaggle session weeks
earlier — a different machine allocation, possibly a different driver and
library stack. Any difference between the arms would then be confounded with
whatever else changed in between, and there would be no way to tell which. Both
arms in one session on one machine leaves exactly 262,144 parameters as the only
difference.

**What retraining buys beyond the control.** It is also a reproducibility test.
The control should land on the Phase 2 figure of validation perplexity 8.4621;
cell 7 of the notebook checks it and flags a deviation above 2%. If the control
does not reproduce, the ablation comparison is void and the report has to say so
rather than quoting a difference between two things that were never comparable.

**Why Marathi.** It is the better-resourced model and the stronger one —
validation perplexity 8.4621 against Konkani's 27.0704 — so it has more headroom
for a degradation to be visible against. On Konkani a large relative loss could
be confused with the noise of an already-weak model.

**What would change it.** Nothing about the design. Konkani is the obvious
extension if the quota and the calendar allow: the same notebook with
`LANGUAGE = 'konkani'` and the same two arms, one more four-hour commit.

---

## B-002 — The embedding is removed, not zeroed

**Decision.** `no_positional_embedding=True` sets `self.position_embedding =
None`. The module does not exist.

**Why not zero it.** A zeroed `nn.Embedding` is still a parameter tensor that
receives gradient on every step, and it would relearn a positional signal within
a few hundred steps. The run would then measure nothing at all while appearing
to work — the most expensive kind of failure, because there is no error to see.

**What makes the removal checkable.** Three things have to change together, and
`tools/verify_model.py --no-positional-embeddings` asserts all three:

| check | expected |
|---|---|
| parameter count drops by `context_length × d_model` | 24,892,356 → **24,630,212**, a drop of exactly 262,144 |
| the module is `None`, not a zeroed tensor | `position_embedding is None` |
| no positional key survives in the state dict | none |

The ablated arm then gets the same 18 checks the control does — 24 in total with
the three above and their `--full` repeats. Verified before the run: **24/24**
ablated, **18/18** control. Three and a half hours of compute is too much to
spend on a model nobody checked.

**A second consequence, deliberate.** "Every parameter receives a gradient" stays
a meaningful check. With a zeroed-but-present embedding it would pass while
describing a parameter that should not have been there.

---

## B-003 — The ablated arm writes to a different run name

**Decision.** `--no-positional-embeddings` changes the checkpoint basename from
`pretrain` to `pretrain_nope`.

**Why.** The two arms differ only in the flag. Sharing `pretrain_best.pt` would
let whichever finished second overwrite the first, and — worse — would let a
resume pick up the other arm's optimizer state and continue training a model
whose architecture does not match it. `train.py` resumes automatically when it
finds a checkpoint, so this is not hypothetical.

**What would change it.** Nothing. Separate output directories are also used, so
this is belt and braces, which is the right amount for a four-hour run that
cannot be repeated before the deadline.

---

## B-004 — What we expect, recorded before the run

Written down in advance so the result cannot be rationalised afterwards.

**The prediction.** Removing positional embeddings should hurt badly.
Self-attention is permutation invariant: permute the input positions and the
output permutes identically. Without positional information the model cannot
distinguish "dog bites man" from "man bites dog" except through whatever the
causal mask leaks.

**Why it should not be total.** The causal mask leaks a great deal. Position *t*
attends over exactly *t+1* tokens while position 0 attends over one, so the
number of visible positions is itself a positional signal, and a model can learn
to read it. This is measurable before any training: on a freshly initialised
ablated model, permuting a 16-token input changes the last position's logits by
**1.15** in absolute terms. The model is order-sensitive at initialisation.

**So the question is quantitative**, not whether the ablated model works at all.
How much of validation perplexity 8.4621 can implicit positional information
recover on its own?

**What would falsify our understanding.** An ablated perplexity close to the
control — say within 10% — would mean the learned embedding was contributing
almost nothing over the causal mask's leak, and D-046 bought 262,144 parameters
of very little. A catastrophic result — perplexity in the hundreds — would mean
the leak is much weaker in practice than the initialisation probe suggests.

---

## B-005 — Separate branch, separate documents

**Decision.** Branch `bonus-no-positional`, built on top of `phase-3` at commit
`69c2bf3`. The bonus writes only `report/bonus_*` files and its own decision
numbering. Nothing on `phase-3` moves.

The branch was rebased onto `phase-3` once Phase 3 was final, so it sits
directly on the submitted state rather than on an earlier snapshot of it. One
commit was dropped in that rebase: it carried the Phase 3 finetuning configs and
qualitative examples, which had been committed here by accident and belong to
Phase 3, and they now reach `phase-3` on their own.

**Why.** The bonus changes `common/model/config.py`, `common/model/lm.py`,
`tools/train.py` and `tools/verify_model.py` — four files at the centre of the
graded work. Keeping the flag on its own branch means every Phase 1–3 deliverable
stays byte-identical to what was evaluated, and the ablation is still a diff
anyone can read in one command.

**The one thing that crosses over.** `report/phase2_decisions.md` D-046 gained a
closing sentence pointing here, because a decision log that records a choice and
not the experiment that tested it is less useful than one that does. That edit
is on `phase-3`, is documentation only, and changes no result.

**What would change it.** Nothing before submission. Merging would be reasonable
afterwards, when there is no longer a graded tree to protect.

---

## B-006 — The smoke test was not testing the ablation

**What happened.** The notebook runs a two-minute smoke train of each arm before
committing to the real one. Both arms printed **identical** numbers: training
loss 7.2807 at step 25 and 7.1714 at step 40, validation perplexity 1344.47.
Two models differing by 262,144 parameters cannot produce identical losses.

**Why.** `tools/train.py` rebuilds the model config from scratch under
`--smoke`, to get a model small enough to run the whole loop in a minute:

```python
model_cfg = ModelConfig(vocab_size=2500, d_model=128, n_layers=2,
                        n_heads=4, d_ff=512, context_length=64, dropout=0.0)
```

That constructor call does not pass `no_positional_embedding`, so it takes the
default of `False`. Both smoke runs therefore built the same
with-positional-embedding tiny model, ran it with the same seed on the same
data, and got the same answer — correctly. The flag was dropped on the floor one
line after being read.

**What it did and did not affect.** The real run is unaffected: the non-smoke
path builds `ModelConfig(vocab_size=2500,
no_positional_embedding=args.no_positional_embeddings)` and passes the flag
through. The full architecture was also verified independently by
`verify_model.py --full --no-positional-embeddings`, which passed 24/24
including the checks on the real 25M config. So the ablation in the reported run
is real; what was lost was the smoke test's ability to catch it if it had not
been.

**Why it is worth recording anyway.** A check that silently tests the wrong
thing is worse than no check, because it produces confidence rather than doubt.
This one would have passed identically whether the ablation worked or not. It
was caught only because two numbers that should have differed were equal to four
decimal places — which is the same reading that caught the Phase 3 sweep
failure, where accuracy was 14.10% at every sample count (D-053). Identical
numbers where different ones are expected is the most reliable bug signal in
this project so far.

**Fix.** The smoke config now carries the flag. A rerun of cell 4 would show the
two arms diverging.

**The live check that replaced it.** Because the fix landed after the run had
started, the ablation was confirmed instead from the training logs: the two arms
must report different losses from the first logged step onward, and different
parameter counts in their headers — 24,630,212 against 24,892,356.
