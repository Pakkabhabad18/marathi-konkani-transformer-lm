# Phase 2 — Decisions Log

Continues the numbering of `phase1_decisions.md`, which ends at D-043. Same
format: the decision, why, and what would change our mind.

**On this file's date.** `phase2_plan.md` §8 committed to recording Phase 2
decisions here as D-044 onward. In practice the reasoning was written into
`phase2_report.md` §2 instead and the numbered entries were never created, which
left `common/model/config.py:82` and `common/model/lm.py:125` citing a D-044 that
did not exist. This file was written on 15 September 2026 to close that gap. It
adds no new reasoning and changes no result: every argument below is the one
already made in `phase2_report.md` §2, given the number the source code has been
citing since the model was written. Phase 3 entries were renumbered from D-044 to
D-050 at the same time so the two do not collide.

---

## D-044 — The output head is untied

**Decision.** `tie_embeddings = False`. The output projection and the token
embedding are separate matrices.

**Why.** Tying saves `vocab × d_model` = 1.28M parameters, about 5% of the 25M
budget at vocabulary 2,500. The saving that makes tying near-mandatory at
vocabulary 50,000 is mostly unavailable at this scale, so the parameters were
spent instead on letting a token's *input* representation differ from its
*output* representation — what it means when read and what predicts it when
written are not the same thing, and at 2,500 pieces we can afford to say so.

**Where it came from.** D-043 closed Phase 1 by noting that weight tying halves
the cost of a large vocabulary and therefore rules out an *untied* large
vocabulary rather than a large vocabulary as such — and it left the tying
question explicitly open as a Phase 2 decision. This is that decision.

**What it costs.** The parameter table in `phase2_report.md` §2 accounts for it:
token embedding 1,280,000 and output projection 1,282,500 as separate lines,
24,892,356 parameters in total. Had we tied, the budget would have allowed
roughly a third of an additional layer — not enough to change the architecture.

**What would change it.** A larger vocabulary. At 10,000 pieces the two lookup
tables cost 10.24M untied against 5.12M tied, and at that point the layer bought
by tying is worth more than the representational freedom given up.

---

## D-045 — Depth over width: seven layers at d_model 512

**Decision.** 7 layers, `d_model` 512, 8 heads of 64, feed-forward 2048.

**Why.** Inside a block, width costs `12 × d²` while depth costs linearly in the
number of blocks. Seven layers at 512 therefore buys more sequential composition
than four at 768 for the same parameter count, and composition is what a
transformer's depth is for. Six layers leaves about 3M of the budget unused;
eight overshoots to roughly 28M.

**What would change it.** A different budget. The choice is a consequence of
targeting ~25M, not a claim that 7 × 512 is optimal in general.

---

## D-046 — Learned absolute positional embeddings

**Decision.** A `nn.Embedding(context_length, d_model)` added to the token
embedding — not sinusoidal encodings, not rotary, not relative.

**Why.** Self-attention is permutation-invariant: permute the input positions and
the output permutes identically. Without positional information the model cannot
distinguish "dog bites man" from "man bites dog" except through whatever the
causal mask leaks. Learned absolute embeddings cost 262,144 parameters here — 1%
of the budget — and let the model learn whatever positional structure the data
actually has rather than the structure a fixed formula assumes. They are added
rather than concatenated: concatenating would spend part of `d_model` on position
and leave less for content, whereas addition lets the model allocate whatever
subspace it needs for each.

**What it costs.** `context_length` becomes a hard architectural ceiling rather
than a runtime setting. There is no embedding row for position 512, so the model
physically cannot process a longer sequence, and `DecoderLM.forward` raises
rather than silently indexing out of range. Sinusoidal encodings can be evaluated
at any position and would not have this limit. The trade was accepted because
nothing in this project runs past the trained context.

**What would change it.** Needing to extrapolate beyond 512 positions. It is also
the decision the bonus ablation tests directly — see `report/bonus_decisions.md`,
B-001 onward, which removes this embedding and measures what it was worth.

---

## D-047 — Pre-norm, and a scaled residual initialisation

**Decision.** LayerNorm before each sub-layer rather than after, a final
LayerNorm before the output head, and initialisation standard deviation scaled by
`1/√(2 · n_layers)` on the two projections that write into the residual stream
(`W_O.weight` and `fc_out.weight`).

**Why.** Pre-norm leaves the residual path free of normalisation, so gradients
reach early layers without passing through a LayerNorm at every block, which is
what makes deep stacks trainable without a long warmup. The consequence is that
the residual stream is never normalised on its way through, so it is normalised
once at the end — hence the final LayerNorm, which is not optional under
pre-norm. The scaled initialisation exists because with `n_layers` blocks each
adding to the same stream, unscaled initialisation lets its variance grow with
depth.

**Evidence it works.** Training ran to completion with no loss spikes and final
gradient norms of 0.567–0.620 (Marathi) and 0.509–0.562 (Konkani); see
`report/training_logs/`.

---

## D-048 — The causal mask adds `-inf` before the softmax

**Decision.** The mask is additive, applied to the attention scores before the
softmax, from a buffer registered with `persistent=False`.

**Why.** Zeroing the weights *after* the softmax leaves each row summing to less
than one, so every masked position silently rescales the remaining attention.
Adding `-inf` first makes the softmax itself produce exact zeros above the
diagonal and rows that sum to one. `tools/verify_model.py` checks both: the
largest weight above the diagonal is 0.000e+00 and the largest row-sum deviation
is 2.384e-07.

**Why `persistent=False`.** The mask is a constant derived from
`context_length`, not a learned parameter. Keeping it out of the state dict means
checkpoints carry weights only, and a checkpoint stays loadable if the mask's
construction ever changes.

**What this catches that training cannot.** A leaking mask still produces a
falling loss curve and a plausible perplexity — the model simply cheats. The two
checks that matter are that changing the token at position *t+1* does not change
the logits at position *t* (largest difference 0.000e+00, tested at every
position), and that changing token 0 *does* change the last position, without
which a model ignoring its input entirely would pass the first check.

---

## D-049 — Verify before spending GPU time

**Decision.** `tools/verify_model.py` runs 18 checks, and was run before any
training started and again on the training machine.

**Why.** The failures that matter in a transformer implementation are the silent
ones: a leaking causal mask, a parameter that receives no gradient, an
initialisation that makes the untrained loss wrong. All three produce a falling
loss curve. Checking that the untrained loss is 7.85 against `ln(2500) = 7.82`,
that every parameter receives a gradient, and that a tiny model can overfit a
single batch costs seconds and rules out a class of bug that would otherwise be
discovered after hours of compute — or not at all.

**What would change it.** Nothing. This practice is why the Phase 3 evaluation
carries a separate language-model check as well (D-053): a training objective
cannot be relied on to reveal damage it does not measure.
