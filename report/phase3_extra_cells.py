# =============================================================================
# THREE EXTRA CELLS.
#
# Paste these as new cells AFTER cell 7 (the one ending "BEST_N = 8000") and
# BEFORE cell 8 (the one starting "# Final models, and the full evaluation").
#
# Why they exist:
#
# Not one configuration in the grid beat the chance floor. The best was Konkani
# answer-only at lr=5e-6: 24.20% against 25.95% chance. Format compliance went
# from 0% to 99.9%, so the models certainly learned the output shape - they did
# not learn the task.
#
# Two objections a reader will raise, and these cells answer both:
#
#   "You only trained for one epoch."  -> cell 7c trains 3 and 6 epochs at the
#   one learning rate that leaves the language model intact.
#
#   "Below chance is just noise."  -> cell 7b shows what the model actually
#   emits. If it answers the same string regardless of the input, that is not
#   noise, it is a collapsed policy, and the correct floor to compare against is
#   the majority-class baseline rather than uniform guessing.
# =============================================================================


# =============================== CELL 7b =====================================
# What is the model actually saying? No GPU needed - this reads the evaluation
# JSONs already on disk.

from collections import Counter

def collapse_report(lang, tag, label):
    path = f'{OUT}/evals/phase3_reasoning_eval_{lang}_{tag}.json'
    if not os.path.exists(path):
        print(f'  (no eval for {lang} {tag})'); return None
    rows = json.load(open(path))['finetuned']['rows']
    answered = [r for r in rows if r['emitted_marker']]
    print(f'\n=== {lang} / {label} ===')
    print(f'    {len(answered)}/{len(rows)} items produced a parseable answer')
    print('    family              n   acc%   uniform%  majority%  '
          'top prediction (share)   distinct preds / golds')
    out = {}
    for fam in sorted({r['family'] for r in rows}):
        sub = [r for r in rows if r['family'] == fam]
        ans = [r for r in sub if r['emitted_marker']]
        if not ans:
            print(f'    {fam:<18}{len(sub):>4}   no parseable answers'); continue
        preds = Counter(r['predicted'] for r in ans)
        golds = Counter(r['gold'] for r in sub)
        top, cnt = preds.most_common(1)[0]
        # The floor a constant answerer would reach: always emit the single most
        # frequent gold for this family. This is the honest baseline for a model
        # that has stopped reading its input.
        majority = 100.0 * golds.most_common(1)[0][1] / len(sub)
        acc = 100.0 * sum(r['correct'] for r in sub) / len(sub)
        uniform = 100.0 * sum(r['chance'] for r in sub) / len(sub)
        print(f'    {fam:<18}{len(sub):>4}  {acc:5.2f}   {uniform:7.2f}   '
              f'{majority:8.2f}   {top!r:<22} {100*cnt/len(ans):5.1f}%   '
              f'{len(preds):>3} / {len(golds):>3}')
        out[fam] = {'n': len(sub), 'accuracy': acc, 'uniform_chance': uniform,
                    'majority_baseline': majority, 'top_prediction': top,
                    'top_prediction_share': round(100*cnt/len(ans), 2),
                    'distinct_predictions': len(preds),
                    'distinct_golds': len(golds)}
    return out

diagnostic = {}
for lang in LANGS:
    for tag, label in [('grid_plain_5e6', 'answer only, lr=5e-6'),
                       ('grid_rationale_5e6', 'chain of thought, lr=5e-6'),
                       ('grid_plain_1e4', 'answer only, lr=1e-4 (LM destroyed)')]:
        d = collapse_report(lang, tag, label)
        if d:
            diagnostic[f'{lang}/{tag}'] = d
json.dump(diagnostic, open(f'{OUT}/phase3_collapse_diagnostic.json', 'w'),
          indent=2, ensure_ascii=False)

print('\nRead it like this: if "distinct preds" is far below "distinct golds",')
print('and one prediction covers most of the answered items, the model is not')
print('reading the entities - it has settled on a constant answer per family.')


# =============================== CELL 7c =====================================
# Does it just need longer? 3 and 6 epochs at the only learning rate that keeps
# the language model intact. 8 runs, a few minutes each.

PROBE_LR, PROBE_N = 5e-6, 8000
PROBE_EPOCHS = [3, 6]

for ep in PROBE_EPOCHS:
    for vname, rat in GRID_VARIANTS:
        sub = f'probe_{vname}_e{ep}'
        print(f'===== {vname}, {ep} epochs =====')
        jobs = [finetune_job(lang, gpu, sub, sub, PROBE_LR, ep, PROBE_N, rat)
                for gpu, lang in enumerate(LANGS)]
        assert run_parallel(jobs), f'{sub} failed'
    prune_latest()

probe = []
for ep in PROBE_EPOCHS:
    for vname, rat in GRID_VARIANTS:
        sub = f'probe_{vname}_e{ep}'
        for lang in LANGS:
            s = score(lang, f'{OUT}/{lang}/{sub}/finetune_{sub}_best.pt', sub, rat)
            s.update(language=lang, variant=vname, epochs=ep,
                     ppl_ratio=s['lm_ppl'] / baseline[lang]['lm_ppl'])
            probe.append(s)

with open(f'{OUT}/phase3_epochs.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(probe[0])); w.writeheader(); w.writerows(probe)

print('\n language  variant     epochs   acc%  chance%   lift   format%   '
      'LM ppl  ppl x   verdict')
for lang in LANGS:
    b = baseline[lang]
    print(f"  {lang:<9}{'PRETRAINED':<12}{'-':<9}{b['accuracy']:6.2f} "
          f"{b['chance']:8.2f} {b['accuracy']-b['chance']:+6.2f}  {b['format']:7.2f} "
          f"{b['lm_ppl']:8.2f}  {'-':>5}")
    for r in [g for g in grid if g['language'] == lang
              and g['variant'] and g['lr'] == 5e-6]:
        print(f"  {lang:<9}{r['variant']:<12}{1:<9}{r['accuracy']:6.2f} "
              f"{r['chance']:8.2f} {r['accuracy']-r['chance']:+6.2f}  {r['format']:7.2f} "
              f"{r['lm_ppl']:8.2f}  x{r['ppl_ratio']:4.2f}")
    for r in [p for p in probe if p['language'] == lang]:
        lift = r['accuracy'] - r['chance']
        verdict = ('BEATS CHANCE' if lift > 0 and r['ppl_ratio'] <= FORGET_BUDGET
                   else 'lost LM' if r['ppl_ratio'] > FORGET_BUDGET else '')
        print(f"  {lang:<9}{r['variant']:<12}{r['epochs']:<9}{r['accuracy']:6.2f} "
              f"{r['chance']:8.2f} {lift:+6.2f}  {r['format']:7.2f} "
              f"{r['lm_ppl']:8.2f}  x{r['ppl_ratio']:4.2f}   {verdict}")

prune_dir(f'{OUT}/*/probe_*')


# =============================== CELL 7d =====================================
# Set the final configuration from the table above, then carry on to cell 8.
#
# Pick the row with the largest lift that is still inside the forgetting budget.
# If nothing beats chance - which is the likely outcome - pick the row with the
# largest lift anyway: the final models still have to exist for the report, and
# the claim being reported is that finetuning bought format compliance and not
# reasoning, which needs a model to have produced it.

BEST_LR      = 5e-6
BEST_VARIANT = 'plain'      # 'plain' or 'rationale'
BEST_EPOCHS  = 1            # set from the epochs table
BEST_N       = 8000
RAT = BEST_VARIANT == 'rationale'

print(f'final configuration: {BEST_VARIANT}, lr={BEST_LR:g}, '
      f'{BEST_EPOCHS} epoch(s), N={BEST_N}')
