# =============================================================================
# CELL 7e - paste AFTER cell 7c and BEFORE cell 7d (the one that sets BEST_*).
#
# Two jobs.
#
# 1. Konkani answer-only is monotone in epochs - 24.20, 25.10, 28.20 - and the
#    six-epoch run is the only configuration anywhere above the chance floor.
#    One more point decides whether that is a trend or a wobble. Twelve epochs,
#    answer-only, both languages: four runs.
#
# 2. Put a number on "above chance". Uniform chance is not one probability, it
#    is a different 1/(k+1) per item, so the right test sums the per-item
#    variance rather than assuming a single p. And uniform guessing is not the
#    only floor worth clearing: a model that has stopped reading its input and
#    answers the most frequent label per family does better than uniform, so
#    that majority-class baseline is reported alongside.
# =============================================================================

import math

PROBE2_EPOCHS = 12

for vname, rat in [('plain', False)]:
    sub = f'probe_{vname}_e{PROBE2_EPOCHS}'
    print(f'===== {vname}, {PROBE2_EPOCHS} epochs =====')
    jobs = [finetune_job(lang, gpu, sub, sub, PROBE_LR, PROBE2_EPOCHS,
                         PROBE_N, rat) for gpu, lang in enumerate(LANGS)]
    assert run_parallel(jobs), f'{sub} failed'
prune_latest()

for lang in LANGS:
    sub = f'probe_plain_e{PROBE2_EPOCHS}'
    s = score(lang, f'{OUT}/{lang}/{sub}/finetune_{sub}_best.pt', sub, False)
    s.update(language=lang, variant='plain', epochs=PROBE2_EPOCHS,
             ppl_ratio=s['lm_ppl'] / baseline[lang]['lm_ppl'])
    probe.append(s)
    print(f"  {lang:9} plain {PROBE2_EPOCHS} epochs  acc {s['accuracy']:5.2f}%  "
          f"format {s['format']:6.2f}%  LM ppl {s['lm_ppl']:8.2f}  "
          f"x{s['ppl_ratio']:.2f}")
prune_dir(f'{OUT}/*/probe_plain_e{PROBE2_EPOCHS}')


# ---- the two floors, and how far above them the model actually is -----------

def floors(lang, tag):
    """Accuracy against uniform chance and against a per-family constant.

    The z-score treats each item as its own Bernoulli trial with probability
    1/(k+1), so the variance is the sum of p(1-p) rather than n*p(1-p) at a
    single p. One-tailed, because the question is only whether the model is
    better than guessing.
    """
    path = f'{OUT}/evals/phase3_reasoning_eval_{lang}_{tag}.json'
    if not os.path.exists(path):
        return None
    rows = json.load(open(path))['finetuned']['rows']
    n = len(rows)
    k = sum(r['correct'] for r in rows)
    ps = [r['chance'] for r in rows]
    mu, var = sum(ps), sum(p * (1 - p) for p in ps)
    z = (k - mu) / math.sqrt(var) if var > 0 else 0.0
    pval = 0.5 * math.erfc(z / math.sqrt(2))
    # majority-class floor: per family, always answer that family's commonest gold
    maj = 0
    for fam in {r['family'] for r in rows}:
        sub = [r for r in rows if r['family'] == fam]
        counts = Counter(r['gold'] for r in sub)
        maj += counts.most_common(1)[0][1]
    return {'n': n, 'accuracy': 100 * k / n, 'uniform': 100 * mu / n,
            'majority': 100 * maj / n, 'z': z, 'p_one_tailed': pval}

print('\n language  config                acc%   uniform%   majority%     z     p'
      '      clears')
verdicts = {}
for lang in LANGS:
    for tag, label in [('grid_plain_5e6',       'plain, 1 epoch'),
                       ('probe_plain_e3',       'plain, 3 epochs'),
                       ('probe_plain_e6',       'plain, 6 epochs'),
                       (f'probe_plain_e{PROBE2_EPOCHS}',
                        f'plain, {PROBE2_EPOCHS} epochs'),
                       ('grid_rationale_5e6',   'rationale, 1 epoch'),
                       ('probe_rationale_e6',   'rationale, 6 epochs')]:
        f = floors(lang, tag)
        if not f:
            continue
        verdicts[f'{lang}/{tag}'] = f
        clears = []
        if f['accuracy'] > f['uniform']:
            clears.append('uniform')
        if f['accuracy'] > f['majority']:
            clears.append('majority')
        print(f"  {lang:<9}{label:<21}{f['accuracy']:6.2f}  {f['uniform']:8.2f}  "
              f"{f['majority']:9.2f}  {f['z']:+6.2f}  {f['p_one_tailed']:.3f}"
              f"   {', '.join(clears) if clears else 'neither'}")

json.dump(verdicts, open(f'{OUT}/phase3_floors.json', 'w'), indent=2)

print('\n  uniform%  = answer at random among the entities offered')
print('  majority% = answer each family\'s commonest label, ignoring the input')
print('  p < 0.05 is the bar for claiming the model is better than guessing.')
