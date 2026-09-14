# =============================================================================
# PHASE 3 ON KAGGLE  —  v2
#
# Accelerator: GPU T4 x2.  Internet: OFF.
# Attach: lma-phase3-code, lma-phase3-ckpt, lma-phase3-ckpt-mr
#
# WHAT CHANGED FROM v1, AND WHY
# -----------------------------
# v1 swept the number of finetuning samples while holding the learning rate at
# 1e-4 for 3 epochs. That recipe destroyed the language model at EVERY sample
# count - Marathi perplexity went 8.55 -> 22,126, Konkani 26.14 -> 98,300 - so
# the sweep measured the same failure five times. Accuracy sat at 14.10% against
# a 25.95% chance line, flat to two decimal places across all five sizes, which
# is the signature of a model that collapsed onto one input-independent answer.
#
# So the order is now: calibrate the learning rate FIRST, then sweep samples at
# a setting that leaves the model intact.
#
# The grid also varies the training target. In the answer-only variant the model
# emits about five tokens per example and most of them are the answer marker, so
# there is very little gradient tied to the reasoning itself. The
# chain-of-thought variant makes it produce the ordered chain - 25-30 tokens that
# genuinely depend on the input. That is a different learning problem, not a
# cosmetic option, so it belongs in the grid rather than being chosen by hand.
#
# Every run is 1 epoch. Three epochs at 8,000 samples is 750 optimizer steps on a
# distribution with five templates; that is where the overwriting happened.
# =============================================================================


# =============================== CELL 1 ======================================
# Locate the attached datasets. Each checkpoint is found by filename, because
# Marathi and Konkani ended up in separate datasets when the first upload failed
# partway.

import os, glob, json, shutil, subprocess, time, sys

def find(marker):
    for root, dirs, files in os.walk('/kaggle/input'):
        if marker in dirs or marker in files:
            return os.path.join(root, marker)
    return None

CODE = os.path.dirname(find('tools'))
PRETRAINED = {lang: find(f'{lang}_pretrain_best.pt')
              for lang in ['marathi', 'konkani']}
LANGS = ['marathi', 'konkani']

print('CODE =', CODE)
for lang, path in PRETRAINED.items():
    print(f'{lang:9} = {path}  ({os.path.getsize(path)/1e6 if path else 0:.0f} MB)')
assert CODE, 'lma-phase3-code is not attached'
assert all(PRETRAINED.values()), f'missing checkpoint: {PRETRAINED}'

print()
print(subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                      '--format=csv'], capture_output=True, text=True).stdout)


# =============================== CELL 2 ======================================
# Copy the code somewhere writable and sanity-check before spending GPU time.

REPO = '/kaggle/working/repo'
OUT  = '/kaggle/working/out'
if os.path.exists(REPO):
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.makedirs(OUT, exist_ok=True)
print('repo at', REPO)

import sentencepiece
print('sentencepiece', sentencepiece.__version__)

r = subprocess.run([sys.executable, 'tools/verify_model.py'],
                   cwd=REPO, capture_output=True, text=True)
print(r.stdout[-700:]); print(r.stderr[-400:])

for lang in LANGS:
    n = sum(1 for _ in open(f'{REPO}/{lang}/data/reasoning/train.jsonl'))
    t = sum(1 for _ in open(f'{REPO}/{lang}/data/reasoning/test.jsonl'))
    print(f'{lang}: {n} train, {t} test reasoning items')


# =============================== CELL 3 ======================================
# Helpers: parallel runner, evaluator wrapper, disk pruning, and the pretrained
# baseline that every later number is compared against.

def prune_latest():
    """Delete the resumable *_latest.pt files once a round is finished.

    Each run writes both a best and a latest checkpoint at ~285 MB. Kaggle gives
    19.5 GB of output space and v1 reached 9 GB partway through, so the latest
    copies are removed after each round. The best checkpoints - the ones every
    result is computed from - are kept.
    """
    freed = 0
    for p in glob.glob(f'{OUT}/**/*_latest.pt', recursive=True):
        freed += os.path.getsize(p); os.remove(p)
    if freed:
        print(f'  pruned {freed/1e9:.2f} GB of resumable checkpoints')

def prune_dir(pattern):
    """Delete finished run directories once their numbers are recorded.

    A grid or sweep checkpoint is only needed long enough to be scored; the
    numbers live in the CSV afterwards. Kaggle output is capped at 19.5 GB and
    every run writes ~285 MB, so the intermediate rounds are cleared and only
    the final models are kept for download.
    """
    freed = 0
    for d in glob.glob(pattern):
        for root, _, files in os.walk(d):
            for f in files:
                freed += os.path.getsize(os.path.join(root, f))
        shutil.rmtree(d)
    if freed:
        print(f'  cleared {freed/1e9:.2f} GB of scored checkpoints')

def run_parallel(jobs, poll=15):
    """jobs = [(name, argv, gpu_index)]. One job per GPU, polled together."""
    procs = []
    for name, cmd, gpu in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        log = open(f'/kaggle/working/{name}.log', 'w')
        procs.append((name, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                             stderr=subprocess.STDOUT), log))
        print(f'  launched {name} on GPU {gpu}')
    started = time.time()
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(poll)
        alive = [n for n, p, _ in procs if p.poll() is None]
        print(f'    [{time.time()-started:5.0f}s] {", ".join(alive)}', flush=True)
    ok = True
    for name, p, log in procs:
        log.close()
        if p.returncode != 0:
            ok = False
            print(f'  {name} FAILED ({p.returncode}):')
            print(open(f'/kaggle/working/{name}.log').read()[-1500:])
        else:
            tail = [l for l in open(f'/kaggle/working/{name}.log').read().splitlines()
                    if 'best val loss' in l]
            print(f'  {name} ok   {tail[-1].strip() if tail else ""}')
    return ok

def finetune_job(lang, gpu, tag, out_sub, lr, epochs, n_train, rationale):
    cmd = [sys.executable, 'tools/finetune.py',
           '--language', lang,
           '--pretrained', PRETRAINED[lang],
           '--data-dir', f'{REPO}/{lang}/data/reasoning',
           '--out-dir', f'{OUT}/{lang}/{out_sub}',
           '--n-train', str(n_train), '--epochs', str(epochs),
           '--lr', str(lr), '--tag', tag, '--device', 'cuda']
    if rationale:
        cmd.append('--rationale')
    return (f'{out_sub}_{lang}', cmd, gpu)

def score(lang, ckpt, tag, rationale, windows='128'):
    """Evaluate one checkpoint. Returns the parsed summary dict."""
    r = subprocess.run([
        sys.executable, 'tools/evaluate_reasoning.py',
        '--language', lang, '--checkpoint', ckpt,
        '--data-dir', f'{REPO}/{lang}/data/reasoning',
        '--lm-data-dir', f'{REPO}/{lang}/data/packed',
        '--lm-windows', windows,
        '--out-dir', f'{OUT}/evals', '--tag', tag,
        '--max-new-tokens', '64' if rationale else '24',
        '--batch-size', '64', '--device', 'cuda',
    ], cwd=REPO, capture_output=True, text=True)
    path = f'{OUT}/evals/phase3_reasoning_eval_{lang}_{tag}.json'
    if not os.path.exists(path):
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        raise RuntimeError(f'evaluation failed: {lang} {tag}')
    f = json.load(open(path))['finetuned']
    return {'accuracy': f['overall']['accuracy'],
            'chance': f['overall']['chance_accuracy'],
            'format': f['overall']['format_compliance'],
            'lm_ppl': f['language_model']['perplexity'],
            'lm_bpb': f['language_model']['bits_per_byte']}

# ---- the reference point: the pretrained models, measured identically -------
baseline = {}
for lang in LANGS:
    baseline[lang] = score(lang, PRETRAINED[lang], 'pretrained', rationale=False)
    b = baseline[lang]
    print(f"{lang:9} PRETRAINED  acc {b['accuracy']:5.2f}%  chance {b['chance']:5.2f}%  "
          f"format {b['format']:5.2f}%  LM ppl {b['lm_ppl']:8.2f}  bpb {b['lm_bpb']:.4f}")
json.dump(baseline, open(f'{OUT}/phase3_baseline.json', 'w'), indent=2)


# =============================== CELL 4 ======================================
# GRID: learning rate x training target, at 8,000 samples and 1 epoch.
# This is the calibration v1 skipped. ~12 runs, roughly 20 minutes.

GRID_LRS = [('5e6', 5e-6), ('2e5', 2e-5), ('1e4', 1e-4)]
GRID_VARIANTS = [('plain', False), ('rationale', True)]
GRID_N, GRID_EPOCHS = 8000, 1

for vname, rat in GRID_VARIANTS:
    for ltag, lr in GRID_LRS:
        sub = f'grid_{vname}_{ltag}'
        print(f'===== {vname}  lr={lr:g} =====')
        jobs = [finetune_job(lang, gpu, sub, sub, lr, GRID_EPOCHS, GRID_N, rat)
                for gpu, lang in enumerate(LANGS)]
        assert run_parallel(jobs), f'{sub} failed'
    prune_latest()

grid = []
for vname, rat in GRID_VARIANTS:
    for ltag, lr in GRID_LRS:
        sub = f'grid_{vname}_{ltag}'
        for lang in LANGS:
            s = score(lang, f'{OUT}/{lang}/{sub}/finetune_{sub}_best.pt', sub, rat)
            s.update(language=lang, variant=vname, lr=lr)
            grid.append(s)
            print(f"  {lang:9} {vname:10} lr={lr:<8g} acc {s['accuracy']:5.2f}%  "
                  f"format {s['format']:6.2f}%  LM ppl {s['lm_ppl']:9.2f}")

prune_dir(f'{OUT}/*/grid_*')

import csv
with open(f'{OUT}/phase3_grid.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(grid[0])); w.writeheader(); w.writerows(grid)

print('\n language  variant     lr         acc%  chance%  format%     LM ppl   LM bpb')
for lang in LANGS:
    b = baseline[lang]
    print(f"  {lang:<9}{'PRETRAINED':<12}{'-':<10}{b['accuracy']:6.2f} {b['chance']:8.2f} "
          f"{b['format']:8.2f} {b['lm_ppl']:10.2f}  {b['lm_bpb']:.4f}")
    for r in [g for g in grid if g['language'] == lang]:
        print(f"  {lang:<9}{r['variant']:<12}{r['lr']:<10.0e}{r['accuracy']:6.2f} "
              f"{r['chance']:8.2f} {r['format']:8.2f} {r['lm_ppl']:10.2f}  {r['lm_bpb']:.4f}")


# =============================== CELL 5 ======================================
# Choose the configuration. A run only counts as usable if it kept the language
# model close to where it started - the reasoning score of a model whose
# perplexity went up a hundredfold is not a result about reasoning.

FORGET_BUDGET = 1.5      # allow at most a 50% rise in pretraining perplexity

print('usable configurations (LM perplexity within budget):\n')
usable = []
for r in grid:
    ratio = r['lm_ppl'] / baseline[r['language']]['lm_ppl']
    r['ppl_ratio'] = ratio
    ok = ratio <= FORGET_BUDGET
    flag = 'ok  ' if ok else 'LOST'
    lift = r['accuracy'] - r['chance']
    print(f"  {flag} {r['language']:9} {r['variant']:10} lr={r['lr']:<8g} "
          f"ppl x{ratio:7.2f}  acc {r['accuracy']:5.2f}%  (chance {r['chance']:5.2f}%, "
          f"lift {lift:+6.2f})")
    if ok:
        usable.append(r)

# A configuration is judged on both languages at once: it is one recipe applied
# to two independently trained models, and a setting that only works for one of
# them is not the setting to carry into the sweep.
print('\n per configuration, both languages together:\n')
print('  variant     lr         worst ppl x   mean lift   both survive')
agg = []
for vname, _ in GRID_VARIANTS:
    for _, lr in GRID_LRS:
        rs = [r for r in grid if r['variant'] == vname and r['lr'] == lr]
        worst = max(r['ppl_ratio'] for r in rs)
        lift = sum(r['accuracy'] - r['chance'] for r in rs) / len(rs)
        both = all(r['ppl_ratio'] <= FORGET_BUDGET for r in rs)
        agg.append({'variant': vname, 'lr': lr, 'worst_ppl_ratio': worst,
                    'mean_lift': lift, 'survives': both})
        print(f"  {vname:<12}{lr:<10.0e} x{worst:9.2f}   {lift:+9.2f}   "
              f"{'yes' if both else 'NO'}")

survivors = [a for a in agg if a['survives']]
if survivors:
    pick = max(survivors, key=lambda a: a['mean_lift'])
    print(f"\n  -> highest lift among surviving configs: {pick['variant']} "
          f"at lr={pick['lr']:g}")
else:
    print('\n  NOTHING survived the forgetting budget. That is itself a finding '
          'worth reporting; rerun the grid one decade lower (1e-6, 2e-6) before '
          'going on.')

# >>> SET THESE FROM THE TABLE ABOVE <<<
BEST_LR       = 5e-6
BEST_VARIANT  = 'rationale'     # 'plain' or 'rationale'
BEST_EPOCHS   = 1


# =============================== CELL 6 ======================================
# NOW the sample-count sweep, at a setting that does not destroy the model.
# This is the figure the TA asked for: samples against perplexity on the normal
# language task, with reasoning accuracy beside it.

SIZES = [500, 1000, 2000, 4000, 8000]
RAT = BEST_VARIANT == 'rationale'

for n in SIZES:
    sub = f'sweep_{n}'
    print(f'===== N={n} =====')
    jobs = [finetune_job(lang, gpu, sub, sub, BEST_LR, BEST_EPOCHS, n, RAT)
            for gpu, lang in enumerate(LANGS)]
    assert run_parallel(jobs), f'N={n} failed'
prune_latest()

rows = []
for n in SIZES:
    for lang in LANGS:
        s = score(lang, f'{OUT}/{lang}/sweep_{n}/finetune_sweep_{n}_best.pt',
                  f'sweep{n}', RAT)
        s.update(language=lang, n_train=n)
        rows.append(s)
        print(f"  {lang:9} N={n:<6} acc {s['accuracy']:5.2f}%  "
              f"format {s['format']:6.2f}%  LM ppl {s['lm_ppl']:8.2f}")

prune_dir(f'{OUT}/*/sweep_*')

with open(f'{OUT}/phase3_sweep.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# =============================== CELL 7 ======================================
# The sweep figure. Left: reasoning accuracy against the chance floor and the
# pretrained starting point. Right: the language-model cost against the
# pretrained line. Title, axis labels and legend on both.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOURS = {'marathi': '#1f77b4', 'konkani': '#d62728'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for lang in LANGS:
    sub = [r for r in rows if r['language'] == lang]
    xs = [r['n_train'] for r in sub]
    c = COLOURS[lang]
    tag = f"{lang.capitalize()} (Model {'H' if lang == 'marathi' else 'L'})"
    ax1.plot(xs, [r['accuracy'] for r in sub], marker='o', color=c, label=tag)
    ax1.plot(xs, [r['chance'] for r in sub], ls=':', color=c, label=f'{tag} — chance')
    ax1.axhline(baseline[lang]['accuracy'], ls='--', lw=1, color=c, alpha=.55,
                label=f'{tag} — pretrained')
    ax2.plot(xs, [r['lm_ppl'] for r in sub], marker='o', color=c, label=tag)
    ax2.axhline(baseline[lang]['lm_ppl'], ls='--', lw=1, color=c, alpha=.55,
                label=f'{tag} — pretrained')

ax1.set_title('What finetuning bought: reasoning accuracy')
ax1.set_xlabel('finetuning samples'); ax1.set_ylabel('exact-match accuracy (%)')
ax1.set_xscale('log'); ax1.legend(fontsize=8); ax1.grid(alpha=.3, which='both')
ax2.set_title('What it cost: perplexity on the pretraining test split')
ax2.set_xlabel('finetuning samples'); ax2.set_ylabel('perplexity')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.legend(fontsize=8); ax2.grid(alpha=.3, which='both')
fig.suptitle(f'Finetuning sample count at lr={BEST_LR:g}, {BEST_VARIANT}, '
             f'{BEST_EPOCHS} epoch')
fig.tight_layout(); fig.savefig(f'{OUT}/phase3_sweep.png', dpi=140)
plt.show()

# >>> SET FROM THE CURVE <<<
BEST_N = 8000


# =============================== CELL 8 ======================================
# Final models, and the full evaluation the report needs: both variants, both
# languages, each against its own pretrained checkpoint, forgetting included.

for vname, rat in GRID_VARIANTS:
    print(f'===== final: {vname} =====')
    jobs = [finetune_job(lang, gpu, vname, f'final_{vname}', BEST_LR,
                         BEST_EPOCHS, BEST_N, rat)
            for gpu, lang in enumerate(LANGS)]
    assert run_parallel(jobs), f'final {vname} failed'
prune_latest()

for vname, rat in GRID_VARIANTS:
    for lang in LANGS:
        r = subprocess.run([
            sys.executable, 'tools/evaluate_reasoning.py',
            '--language', lang,
            '--checkpoint', f'{OUT}/{lang}/final_{vname}/finetune_{vname}_best.pt',
            '--compare-to', PRETRAINED[lang],
            '--data-dir', f'{REPO}/{lang}/data/reasoning',
            '--lm-data-dir', f'{REPO}/{lang}/data/packed', '--lm-windows', '256',
            '--out-dir', f'{OUT}/report', '--tag', f'final_{vname}',
            '--max-new-tokens', '64' if rat else '24',
            '--batch-size', '64', '--device', 'cuda',
        ], cwd=REPO, capture_output=True, text=True)
        print(f'########## {lang} / {vname} ##########')
        print(r.stdout[-2500:] or r.stderr[-1500:])


# =============================== CELL 9 ======================================
# Attention analysis, pretrained against finetuned. Specification 3.2.

os.makedirs(f'{OUT}/report', exist_ok=True)
for lang in LANGS:
    for label, ck in [('pretrained', PRETRAINED[lang]),
                      ('finetuned',
                       f'{OUT}/{lang}/final_{BEST_VARIANT}/finetune_{BEST_VARIANT}_best.pt')]:
        r = subprocess.run([
            sys.executable, 'tools/attention_analysis.py',
            '--language', lang, '--checkpoint', ck,
            '--tokenizer', f'{REPO}/{lang}/tokenizer/{lang}_bpe.model',
            '--data-dir', f'{REPO}/{lang}/data/packed',
            '--split', 'test', '--device', 'cuda',
        ], cwd=REPO, capture_output=True, text=True)
        print(f'--- {lang} / {label} ---')
        print(r.stdout[-1000:] or r.stderr[-800:])
        for pattern in (f'{REPO}/report/phase2_attention_{lang}*',
                        f'{REPO}/report/figures/phase2_attention_{lang}*'):
            for f in glob.glob(pattern):
                shutil.move(f, f'{OUT}/report/attn_{label}_{os.path.basename(f)}')


# =============================== CELL 10 =====================================
# Package for download.

import zipfile
for name in ['phase3_grid.csv', 'phase3_sweep.csv', 'phase3_sweep.png',
             'phase3_baseline.json']:
    if os.path.exists(f'{OUT}/{name}'):
        shutil.copy(f'{OUT}/{name}', f'{OUT}/report/')
for lang in LANGS:
    for vname, _ in GRID_VARIANTS:
        src = f'{OUT}/{lang}/final_{vname}/finetune_{vname}_log.csv'
        if os.path.exists(src):
            shutil.copy(src, f'{OUT}/report/{lang}_finetune_{vname}_log.csv')

with zipfile.ZipFile('/kaggle/working/phase3_results.zip', 'w',
                     zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(f'{OUT}/report'):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, f'{OUT}/report'))
    for f in glob.glob(f'{OUT}/evals/*.json'):
        z.write(f, 'evals/' + os.path.basename(f))
    for f in glob.glob('/kaggle/working/*.log'):
        z.write(f, 'logs/' + os.path.basename(f))
print('results  ', os.path.getsize('/kaggle/working/phase3_results.zip')/1e6, 'MB')

with zipfile.ZipFile('/kaggle/working/phase3_checkpoints.zip', 'w',
                     zipfile.ZIP_STORED) as z:
    for lang in LANGS:
        for vname, _ in GRID_VARIANTS:
            p = f'{OUT}/{lang}/final_{vname}/finetune_{vname}_best.pt'
            if os.path.exists(p):
                z.write(p, f'{lang}_finetune_{vname}_best.pt')
print('checkpoints', os.path.getsize('/kaggle/working/phase3_checkpoints.zip')/1e6, 'MB')
print('\nDownload both from the Output panel.')
