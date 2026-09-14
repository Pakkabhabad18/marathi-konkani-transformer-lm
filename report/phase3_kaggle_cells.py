# =============================================================================
# PHASE 3 ON KAGGLE — notebook cells
#
# Accelerator: GPU T4 x2.  Internet: OFF (nothing here needs it).
# Attach BOTH datasets:  lma-phase3-code   and   lma-phase3-ckpt
#
# Paste each CELL below into its own notebook cell, in order.
# Cells 4, 6, 7 and 8 run Marathi on GPU 0 and Konkani on GPU 1 at the same
# time, the same way Phase 2 did.
# =============================================================================


# =============================== CELL 1 ======================================
# Find the attached datasets. Kaggle has nested them under /kaggle/input/datasets/
# in the past, so this searches rather than assuming a path.

import os, glob, json, shutil, subprocess, time, sys

def find_dir(marker):
    for root, dirs, files in os.walk('/kaggle/input'):
        if marker in dirs or marker in files:
            return os.path.join(root, marker)
    return None

CODE = os.path.dirname(find_dir('tools'))
CKPT = os.path.dirname(find_dir('marathi_pretrain_best.pt'))
print('CODE =', CODE)
print('CKPT =', CKPT)
assert CODE and CKPT, 'one of the datasets is not attached'

for p in sorted(glob.glob(CODE + '/*'))[:12]:
    print('  ', p)
print()
print('GPUs:')
print(subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                      '--format=csv'], capture_output=True, text=True).stdout)


# =============================== CELL 2 ======================================
# Copy the code somewhere writable, and sanity-check before spending GPU time.

REPO = '/kaggle/working/repo'
OUT  = '/kaggle/working/out'
if os.path.exists(REPO):
    shutil.rmtree(REPO)
shutil.copytree(CODE, REPO)
os.makedirs(OUT, exist_ok=True)
print('repo at', REPO)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sentencepiece'])

# 18 correctness checks on the architecture. If this fails, stop.
r = subprocess.run([sys.executable, 'tools/verify_model.py'],
                   cwd=REPO, capture_output=True, text=True)
print(r.stdout[-1500:])
print(r.stderr[-800:])

for lang in ['marathi', 'konkani']:
    n = sum(1 for _ in open(f'{REPO}/{lang}/data/reasoning/train.jsonl'))
    t = sum(1 for _ in open(f'{REPO}/{lang}/data/reasoning/test.jsonl'))
    print(f'{lang}: {n} train, {t} test reasoning items')


# =============================== CELL 3 ======================================
# Runner. %%bash buffers its output and you see nothing for minutes, so this
# streams instead and reports the exit code. Jobs pinned to a GPU each run
# concurrently; the loop polls all of them.

def run_parallel(jobs, poll=20):
    """jobs = [(name, [argv...], gpu_index), ...]"""
    procs = []
    for name, cmd, gpu in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        log = open(f'/kaggle/working/{name}.log', 'w')
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        procs.append((name, p, log))
        print(f'launched {name} on GPU {gpu}: {" ".join(cmd[-6:])}')

    started = time.time()
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(poll)
        alive = [n for n, p, _ in procs if p.poll() is None]
        print(f'  [{time.time()-started:6.0f}s] running: {", ".join(alive)}',
              flush=True)
    for name, p, log in procs:
        log.close()
        print(f'{name} EXITED {p.returncode}')
        tail = open(f'/kaggle/working/{name}.log').read()[-900:]
        print(tail)
        print('-' * 70)
    return all(p.returncode == 0 for _, p, _ in procs)


# =============================== CELL 4 ======================================
# SAMPLE-COUNT SWEEP. The TA suggested plotting sample count against perplexity
# on the normal language task and choosing from the curve; this produces it.
# Only N varies - epochs, learning rate and seed are held fixed.
# Roughly 15 minutes for both languages.

SIZES = [500, 1000, 2000, 4000, 8000]

for n in SIZES:
    jobs = []
    for gpu, lang in enumerate(['marathi', 'konkani']):
        jobs.append((f'sweep_{lang}_{n}', [
            sys.executable, 'tools/finetune.py',
            '--language', lang,
            '--pretrained', f'{CKPT}/{lang}_pretrain_best.pt',
            '--data-dir', f'{REPO}/{lang}/data/reasoning',
            '--out-dir', f'{OUT}/{lang}/sweep_{n}',
            '--n-train', str(n), '--epochs', '3',
            '--tag', f'n{n}', '--device', 'cuda',
        ], gpu))
    print(f'===== sweep N={n} =====')
    ok = run_parallel(jobs, poll=15)
    assert ok, f'sweep N={n} failed'


# =============================== CELL 5 ======================================
# Score every sweep checkpoint: reasoning accuracy AND held-out pretraining
# perplexity, so the curve shows both what was gained and what it cost.

import csv
rows = []
for n in SIZES:
    for lang in ['marathi', 'konkani']:
        ck = f'{OUT}/{lang}/sweep_{n}/finetune_n{n}_best.pt'
        r = subprocess.run([
            sys.executable, 'tools/evaluate_reasoning.py',
            '--language', lang, '--checkpoint', ck,
            '--data-dir', f'{REPO}/{lang}/data/reasoning',
            '--lm-data-dir', f'{REPO}/{lang}/data/packed',
            '--lm-windows', '128',
            '--out-dir', f'{OUT}/sweep_eval', '--tag', f'n{n}',
            '--max-new-tokens', '24', '--batch-size', '64', '--device', 'cuda',
        ], cwd=REPO, capture_output=True, text=True)
        print(f'--- {lang} N={n} ---')
        print(r.stdout[-700:] or r.stderr[-700:])
        j = json.load(open(f'{OUT}/sweep_eval/phase3_reasoning_eval_{lang}_n{n}.json'))
        f = j['finetuned']
        rows.append({
            'language': lang, 'n_train': n,
            'accuracy': f['overall']['accuracy'],
            'chance': f['overall']['chance_accuracy'],
            'format_compliance': f['overall']['format_compliance'],
            'lm_perplexity': f['language_model']['perplexity'],
            'lm_bits_per_byte': f['language_model']['bits_per_byte'],
        })

with open(f'{OUT}/phase3_sweep.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

print('\n language   N     acc%   chance%  format%   LM ppl   LM bpb')
for r in rows:
    print(f"  {r['language']:<9}{r['n_train']:<6}{r['accuracy']:6.2f}  "
          f"{r['chance']:7.2f}  {r['format_compliance']:7.2f}  "
          f"{r['lm_perplexity']:8.2f}  {r['lm_bits_per_byte']:.4f}")


# =============================== CELL 6 ======================================
# The sweep figure. Two panels: what finetuning bought (reasoning accuracy) and
# what it cost (pretraining perplexity). Title, axis labels and legend on both,
# which the specification requires.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOURS = {'marathi': '#1f77b4', 'konkani': '#d62728'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for lang in ['marathi', 'konkani']:
    sub = [r for r in rows if r['language'] == lang]
    xs = [r['n_train'] for r in sub]
    tag = f"{lang.capitalize()} (Model {'H' if lang=='marathi' else 'L'})"
    ax1.plot(xs, [r['accuracy'] for r in sub], marker='o',
             color=COLOURS[lang], label=tag)
    ax1.plot(xs, [r['chance'] for r in sub], marker='', linestyle=':',
             color=COLOURS[lang], label=f'{tag} — chance')
    ax2.plot(xs, [r['lm_perplexity'] for r in sub], marker='o',
             color=COLOURS[lang], label=tag)

ax1.set_title('What finetuning bought: reasoning accuracy')
ax1.set_xlabel('finetuning samples'); ax1.set_ylabel('exact-match accuracy (%)')
ax1.set_xscale('log'); ax1.legend(); ax1.grid(alpha=.3, which='both')

ax2.set_title('What it cost: perplexity on the pretraining test split')
ax2.set_xlabel('finetuning samples'); ax2.set_ylabel('perplexity')
ax2.set_xscale('log'); ax2.legend(); ax2.grid(alpha=.3, which='both')

fig.suptitle('Finetuning sample count against reasoning gain and language-model cost')
fig.tight_layout()
fig.savefig(f'{OUT}/phase3_sweep.png', dpi=140)
print('wrote', f'{OUT}/phase3_sweep.png')
plt.show()

# >>> READ THE CURVE, THEN SET THIS <<<
# Pick the smallest N past which accuracy stops climbing, or where perplexity
# starts climbing sharply - whichever comes first.
BEST_N = 8000


# =============================== CELL 7 ======================================
# FINAL RUNS at the chosen size: answer-only and chain-of-thought, both
# languages. Four runs, two at a time.

for variant, flag in [('plain', []), ('rationale', ['--rationale'])]:
    jobs = []
    for gpu, lang in enumerate(['marathi', 'konkani']):
        jobs.append((f'final_{lang}_{variant}', [
            sys.executable, 'tools/finetune.py',
            '--language', lang,
            '--pretrained', f'{CKPT}/{lang}_pretrain_best.pt',
            '--data-dir', f'{REPO}/{lang}/data/reasoning',
            '--out-dir', f'{OUT}/{lang}/final',
            '--n-train', str(BEST_N), '--epochs', '3',
            '--tag', variant, '--device', 'cuda',
        ] + flag, gpu))
    print(f'===== final: {variant} =====')
    assert run_parallel(jobs, poll=15), f'{variant} failed'


# =============================== CELL 8 ======================================
# Evaluate the final models against their own pretrained checkpoints, with the
# catastrophic-forgetting check on.

for variant in ['plain', 'rationale']:
    mx = '24' if variant == 'plain' else '64'
    for lang in ['marathi', 'konkani']:
        r = subprocess.run([
            sys.executable, 'tools/evaluate_reasoning.py',
            '--language', lang,
            '--checkpoint', f'{OUT}/{lang}/final/finetune_{variant}_best.pt',
            '--compare-to', f'{CKPT}/{lang}_pretrain_best.pt',
            '--data-dir', f'{REPO}/{lang}/data/reasoning',
            '--lm-data-dir', f'{REPO}/{lang}/data/packed',
            '--lm-windows', '256',
            '--out-dir', f'{OUT}/report', '--tag', variant,
            '--max-new-tokens', mx, '--batch-size', '64', '--device', 'cuda',
        ], cwd=REPO, capture_output=True, text=True)
        print(f'########## {lang} / {variant} ##########')
        print(r.stdout[-2500:] or r.stderr[-1500:])


# =============================== CELL 9 ======================================
# Attention analysis on the finetuned checkpoints. Specification 3.2 asks for
# pretrained against finetuned heatmaps for an early and a late layer per model.

for gpu, lang in enumerate(['marathi', 'konkani']):
    for label, ck in [('pretrained', f'{CKPT}/{lang}_pretrain_best.pt'),
                      ('finetuned',  f'{OUT}/{lang}/final/finetune_plain_best.pt')]:
        r = subprocess.run([
            sys.executable, 'tools/attention_analysis.py',
            '--language', lang, '--checkpoint', ck,
            '--tokenizer', f'{REPO}/{lang}/tokenizer/{lang}_bpe.model',
            '--data-dir', f'{REPO}/{lang}/data/packed',
            '--split', 'test', '--device', 'cuda',
        ], cwd=REPO, capture_output=True, text=True)
        print(f'--- {lang} / {label} ---')
        print(r.stdout[-1200:] or r.stderr[-800:])
        # attention_analysis writes into REPO/report; move so the two don't collide
        for f in glob.glob(f'{REPO}/report/phase2_attention_{lang}*'):
            dst = f'{OUT}/report/attn_{label}_' + os.path.basename(f)
            os.makedirs(f'{OUT}/report', exist_ok=True)
            shutil.move(f, dst)
        for f in glob.glob(f'{REPO}/report/figures/phase2_attention_{lang}*'):
            dst = f'{OUT}/report/attn_{label}_' + os.path.basename(f)
            shutil.move(f, dst)


# =============================== CELL 10 =====================================
# Package everything for download. Checkpoints separately, because they are big.

import zipfile

os.makedirs(f'{OUT}/report', exist_ok=True)
shutil.copy(f'{OUT}/phase3_sweep.csv', f'{OUT}/report/')
shutil.copy(f'{OUT}/phase3_sweep.png', f'{OUT}/report/')
for lang in ['marathi', 'konkani']:
    for variant in ['plain', 'rationale']:
        src = f'{OUT}/{lang}/final/finetune_{variant}_log.csv'
        if os.path.exists(src):
            shutil.copy(src, f'{OUT}/report/{lang}_finetune_{variant}_log.csv')

with zipfile.ZipFile('/kaggle/working/phase3_results.zip', 'w',
                     zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(f'{OUT}/report'):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, f'{OUT}/report'))
    for f in glob.glob('/kaggle/working/*.log'):
        z.write(f, 'logs/' + os.path.basename(f))
print('results ->', os.path.getsize('/kaggle/working/phase3_results.zip') / 1e6, 'MB')

with zipfile.ZipFile('/kaggle/working/phase3_checkpoints.zip', 'w',
                     zipfile.ZIP_STORED) as z:   # already-compressed tensors
    for lang in ['marathi', 'konkani']:
        for variant in ['plain', 'rationale']:
            p = f'{OUT}/{lang}/final/finetune_{variant}_best.pt'
            if os.path.exists(p):
                z.write(p, f'{lang}_finetune_{variant}_best.pt')
print('checkpoints ->',
      os.path.getsize('/kaggle/working/phase3_checkpoints.zip') / 1e6, 'MB')
print('\nDownload both from the notebook Output panel.')
