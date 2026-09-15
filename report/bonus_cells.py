# =============================================================================
# BONUS - MARATHI WITHOUT POSITIONAL EMBEDDINGS
#
# Accelerator: GPU T4 x2.  Internet: OFF.
# Attach: lma-bonus-code, lma-phase2-data
#
# RUN THIS WITH "Save Version -> Save & Run All (Commit)", NOT INTERACTIVELY.
# Expected runtime about 4 hours. Nothing is set by hand.
#
# WHAT THIS MEASURES
# ------------------
# One variable: the learned absolute positional embedding (Phase 2 decision
# D-046). Two arms train side by side, one per GPU, from the same seed, on the
# same data, for the same 500M-token budget:
#
#   GPU 0   ablated   no_positional_embedding=True    24,630,212 parameters
#   GPU 1   control   no_positional_embedding=False   24,892,356 parameters
#
# The control is retrained rather than taken from Phase 2. The Phase 2
# checkpoint was produced weeks ago on a different Kaggle session; re-running it
# here means both arms share a driver, a GPU model and a wall clock, so the only
# thing that differs between them is the 262,144 parameters under test. It also
# tests reproducibility: the control should land on the Phase 2 figure of
# val perplexity 8.4621, and if it does not, something other than the ablation
# has changed and the comparison is void.
#
# WHAT TO EXPECT
# --------------
# Self-attention is permutation invariant, so removing positional embeddings
# should be severe. It will not be total: causal masking leaks position, because
# position t can attend to exactly t+1 tokens, and that count is itself a
# positional signal a model can learn to use. Measured on an untrained model,
# permuting the input already changes the last position's logits by 1.15, so the
# ablated model is order-sensitive before it has learned anything. The question
# is how much of the Phase 2 result that residual signal can recover.
# =============================================================================


# =============================== CELL 1 ======================================
# Inputs and configuration. Every choice is fixed here.

import os, sys, glob, json, csv, math, time, shutil, zipfile, pathlib, subprocess

T0 = time.time()

LANGUAGE   = 'marathi'          # one language, per the bonus statement
MAX_TOKENS = 500_000_000        # identical to Phase 2
SEED_NOTE  = 'train.py uses TrainConfig.seed; both arms share it'

CODE = '/kaggle/working/code'
DATA = f'/kaggle/working/data/{LANGUAGE}'
OUT  = '/kaggle/working/out'

def find(marker):
    for root, dirs, files in os.walk('/kaggle/input'):
        if marker in dirs or marker in files:
            return os.path.join(root, marker)
    return None

code_zip = find('lma_bonus_code.zip') or find('lma_phase2_code.zip')
assert code_zip, 'no code zip found - attach lma-bonus-code'
data_root = os.path.dirname(find(f'{LANGUAGE}_train.bin') or '')
assert data_root, 'packed data not found - attach lma-phase2-data'

print('code zip ', code_zip)
print('data root', data_root)
print(subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                      '--format=csv'], capture_output=True, text=True).stdout)
assert 'Tesla' in subprocess.run(['nvidia-smi', '-L'], capture_output=True,
                                 text=True).stdout, 'no GPU'


# =============================== CELL 2 ======================================
# Unpack the code and lay the data out the way the loader expects. The dataset
# stores files flat and prefixed; PackedTokens wants train/val/test in one
# directory. Symlinks, not copies - the train file is about 1 GB.

if os.path.exists(CODE):
    shutil.rmtree(CODE)
zipfile.ZipFile(code_zip).extractall(CODE)
sys.path.insert(0, CODE)
os.makedirs(OUT, exist_ok=True)

d = pathlib.Path(DATA); d.mkdir(parents=True, exist_ok=True)
for split in ('train', 'val', 'test'):
    for ext in ('bin', 'json'):
        dst = d / f'{split}.{ext}'
        src = f'{data_root}/{LANGUAGE}_{split}.{ext}'
        assert os.path.exists(src), f'missing {src}'
        if not dst.exists():
            dst.symlink_to(src)
for p in sorted(d.iterdir()):
    print(f'  {p.name:12} {os.path.getsize(p)/1e6:9.1f} MB')

# The ablated arm must be built from the same source that produced the control,
# so the branch actually carrying the no_positional_embedding flag has to be the
# one in the zip. If this import fails the wrong code was uploaded.
from common.model.config import ModelConfig
assert 'no_positional_embedding' in ModelConfig().__dataclass_fields__, (
    'the uploaded code has no no_positional_embedding field - you uploaded the '
    'phase-3 code, not the bonus-no-positional branch')
print('\ncode carries the ablation flag')


# =============================== CELL 3 ======================================
# Verify both architectures before spending four hours on them. The ablated arm
# gets the same checks as the control, plus three that the removal happened.

for label, extra in [('control (with positional embeddings)', []),
                     ('ablated (none)', ['--no-positional-embeddings'])]:
    r = subprocess.run([sys.executable, 'tools/verify_model.py', '--full'] + extra,
                       cwd=CODE, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-4:]
    print(f'--- {label} ---')
    print('\n'.join(tail))
    assert 'checks passed' in r.stdout and '\nFAILED' not in r.stdout, \
        f'verification failed for {label}:\n{r.stdout[-2000:]}'
    print()

print('parameter counts:')
print(f"  control {ModelConfig(vocab_size=2500).n_params()['total']:,}")
print(f"  ablated {ModelConfig(vocab_size=2500, no_positional_embedding=True).n_params()['total']:,}")


# =============================== CELL 4 ======================================
# A two-minute smoke run of each arm, on the GPU, before the real one.

for tag, extra in [('control', []), ('ablated', ['--no-positional-embeddings'])]:
    r = subprocess.run(
        [sys.executable, 'tools/train.py', '--language', LANGUAGE, '--smoke',
         '--device', 'cuda', '--data-dir', DATA,
         '--out-dir', f'{OUT}/smoke_{tag}'] + extra,
        cwd=CODE, capture_output=True, text=True)
    ok = r.returncode == 0
    print(f'--- smoke {tag}: {"ok" if ok else "FAILED"} ---')
    print(r.stdout[-700:] if ok else (r.stdout[-2500:] + r.stderr[-1500:]))
    assert ok, f'smoke run failed for {tag}'
shutil.rmtree(f'{OUT}/smoke_control', ignore_errors=True)
shutil.rmtree(f'{OUT}/smoke_ablated', ignore_errors=True)
print(f'\n[{time.time()-T0:.0f}s elapsed]')


# =============================== CELL 5 ======================================
# The real runs. Both arms at once, one per GPU, ~3.6 hours.
# Phase 2 took 12,846.9 s for this budget on one T4; an ETA far above that means
# something is wrong with batch size or precision.

def launch(tag, gpu, extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    log = open(f'/kaggle/working/{tag}.log', 'w')
    cmd = [sys.executable, 'tools/train.py', '--language', LANGUAGE,
           '--max-tokens', str(MAX_TOKENS), '--device', 'cuda',
           '--data-dir', DATA, '--out-dir', f'{OUT}/{tag}'] + extra
    return tag, subprocess.Popen(cmd, cwd=CODE, env=env, stdout=log,
                                 stderr=subprocess.STDOUT), log

procs = [launch('ablated', 0, ['--no-positional-embeddings']),
         launch('control', 1, [])]
started = time.time()
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(300)                       # five minutes; this runs for hours
    alive = [t for t, p, _ in procs if p.poll() is None]
    note = []
    for tag, _, _ in procs:
        lines = [l for l in open(f'/kaggle/working/{tag}.log').read().splitlines()
                 if l.strip().startswith(('step', '  step')) or 'loss' in l]
        if lines:
            note.append(f'{tag}: {lines[-1].strip()[:70]}')
    print(f'[{(time.time()-started)/60:6.1f} min] running {", ".join(alive)}',
          flush=True)
    for n in note:
        print(f'        {n}', flush=True)

for tag, p, log in procs:
    log.close()
    if p.returncode != 0:
        print(f'{tag} FAILED:'); print(open(f'/kaggle/working/{tag}.log').read()[-3000:])
    assert p.returncode == 0, f'{tag} training failed'
    print(f'{tag} finished ok')
print(f'\n[{(time.time()-T0)/60:.1f} min elapsed]')


# =============================== CELL 6 ======================================
# Intrinsic evaluation of both arms on the held-out test split, identical
# settings, using the Phase 2 evaluator.

CKPT = {tag: f'{OUT}/{tag}/pretrain{"_nope" if tag == "ablated" else ""}_best.pt'
        for tag in ('ablated', 'control')}
for tag, path in CKPT.items():
    assert os.path.exists(path), f'missing checkpoint for {tag}: {path}'
    print(f'  {tag:8} {os.path.getsize(path)/1e6:.0f} MB  {path}')

os.makedirs(f'{OUT}/report', exist_ok=True)
TOKENIZER = f'{CODE}/{LANGUAGE}/tokenizer/{LANGUAGE}_bpe.model'
assert os.path.exists(TOKENIZER), (
    f'{TOKENIZER} is missing - the code zip must include '
    f'{LANGUAGE}/tokenizer as well as common/ and tools/')

for tag, path in CKPT.items():
    r = subprocess.run(
        [sys.executable, 'tools/evaluate.py', '--language', LANGUAGE,
         '--checkpoint', path, '--tokenizer', TOKENIZER,
         '--data-dir', DATA, '--split', 'test',
         '--device', 'cuda', '--out-dir', f'{OUT}/report'],
        cwd=CODE, capture_output=True, text=True)
    print(f'########## {tag} ##########')
    print(r.stdout[-3000:] or r.stderr[-1500:], flush=True)
    # evaluate.py writes a fixed filename per language, so the second arm would
    # overwrite the first. Rename as soon as each one lands.
    src = f'{OUT}/report/phase2_eval_{LANGUAGE}.json'
    assert os.path.exists(src), f'evaluation produced nothing for {tag}'
    shutil.move(src, f'{OUT}/report/bonus_eval_{tag}.json')


# =============================== CELL 7 ======================================
# The comparison table, read back from whatever the evaluator wrote, plus the
# training curves. Phase 2's own figure is quoted as the reproducibility check.

PHASE2_VAL_PPL = 8.4621        # report/phase2_training_summary.json, marathi

def read_log(tag):
    name = f'pretrain{"_nope" if tag == "ablated" else ""}_log.csv'
    rows = list(csv.DictReader(open(f'{OUT}/{tag}/{name}')))
    return rows

summary = {}
for tag in ('control', 'ablated'):
    rows = read_log(tag)
    vals = [(int(r['step']), float(r['val_loss']), float(r['val_ppl']))
            for r in rows if r.get('val_loss')]
    last_step, last_loss, last_ppl = vals[-1]
    trains = [(int(r['step']), float(r['train_loss'])) for r in rows if r.get('train_loss')]
    summary[tag] = {
        'final_val_loss': last_loss, 'final_val_ppl': last_ppl,
        'best_val_ppl': min(v[2] for v in vals),
        'final_train_loss': trains[-1][1],
        'steps': last_step,
        'tokens': int(rows[-1]['tokens']),
        'seconds': float(rows[-1]['seconds']),
        'curve': vals,
    }

print(f"{'arm':10}{'params':>12}{'val loss':>10}{'val ppl':>10}{'hours':>8}")
for tag, params in (('control', 24_892_356), ('ablated', 24_630_212)):
    s = summary[tag]
    print(f"  {tag:8}{params:>12,}{s['final_val_loss']:>10.4f}"
          f"{s['final_val_ppl']:>10.4f}{s['seconds']/3600:>8.2f}")

drift = abs(summary['control']['final_val_ppl'] - PHASE2_VAL_PPL) / PHASE2_VAL_PPL
print(f"\nreproducibility: control {summary['control']['final_val_ppl']:.4f} vs "
      f"Phase 2 {PHASE2_VAL_PPL:.4f}  ({100*drift:+.2f}%)")
print('  the comparison is sound' if drift < 0.02 else
      '  WARNING: the control did not reproduce Phase 2; say so in the report')

r = summary['ablated']['final_val_ppl'] / summary['control']['final_val_ppl']
print(f"\ncost of removing positional embeddings: "
      f"perplexity x{r:.3f} "
      f"({summary['control']['final_val_ppl']:.4f} -> {summary['ablated']['final_val_ppl']:.4f})")
json.dump(summary, open(f'{OUT}/report/bonus_summary.json', 'w'), indent=2)


# =============================== CELL 8 ======================================
# One figure: both loss curves on the same axes, with the Phase 2 value marked.

import matplotlib.pyplot as plt

C = {'control': '#1f77b4', 'ablated': '#d62728'}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for tag in ('control', 'ablated'):
    rows = read_log(tag)
    ts = [(int(r['tokens']), float(r['train_loss'])) for r in rows if r.get('train_loss')]
    vs = [(int(r['tokens']), float(r['val_loss'])) for r in rows if r.get('val_loss')]
    lab = ('with positional embeddings' if tag == 'control'
           else 'without positional embeddings')
    a1.plot([t / 1e6 for t, _ in ts], [l for _, l in ts], color=C[tag], alpha=.45,
            lw=1, label=f'{lab} — train')
    a1.plot([t / 1e6 for t, _ in vs], [l for _, l in vs], color=C[tag], marker='o',
            ms=3, label=f'{lab} — validation')
    a2.plot([t / 1e6 for t, _ in vs],
            [math.exp(l) for _, l in vs], color=C[tag], marker='o', ms=3, label=lab)

a2.axhline(PHASE2_VAL_PPL, ls='--', lw=1, color='k', alpha=.6,
           label=f'Phase 2 reported ({PHASE2_VAL_PPL})')
a1.set_xlabel('training tokens (millions)'); a1.set_ylabel('cross-entropy loss')
a1.set_title('Loss'); a1.legend(fontsize=8); a1.grid(alpha=.3)
a2.set_xlabel('training tokens (millions)'); a2.set_ylabel('validation perplexity')
a2.set_yscale('log'); a2.set_title('Validation perplexity')
a2.legend(fontsize=8); a2.grid(alpha=.3, which='both')
fig.suptitle(f'{LANGUAGE.capitalize()} (Model H) — what the positional embedding '
             f'is worth\nidentical config, seed and data; 262,144 parameters '
             f'removed')
fig.tight_layout(); fig.savefig(f'{OUT}/report/bonus_loss.png', dpi=140)
plt.show()


# =============================== CELL 9 ======================================
# Attention, both arms. If the ablated model recovers any order sensitivity, the
# per-head distance statistics are where it should be visible.

for tag, path in CKPT.items():
    r = subprocess.run(
        [sys.executable, 'tools/attention_analysis.py', '--language', LANGUAGE,
         '--checkpoint', path,
         '--tokenizer', TOKENIZER,
         '--data-dir', DATA, '--split', 'test', '--device', 'cuda'],
        cwd=CODE, capture_output=True, text=True)
    print(f'--- {tag} ---')
    print(r.stdout[-1800:] or r.stderr[-900:], flush=True)
    for pat in (f'{CODE}/report/phase2_attention_{LANGUAGE}*',
                f'{CODE}/report/figures/phase2_attention_{LANGUAGE}*'):
        for f in glob.glob(pat):
            shutil.move(f, f'{OUT}/report/bonus_attn_{tag}_{os.path.basename(f)}')


# =============================== CELL 10 =====================================
# Package. Results are small; the two checkpoints go in their own zip.

for tag in ('control', 'ablated'):
    name = f'pretrain{"_nope" if tag == "ablated" else ""}_log.csv'
    shutil.copy(f'{OUT}/{tag}/{name}', f'{OUT}/report/bonus_{tag}_log.csv')

with zipfile.ZipFile('/kaggle/working/bonus_results.zip', 'w',
                     zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(f'{OUT}/report'):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, f'{OUT}/report'))
    for f in glob.glob('/kaggle/working/*.log'):
        z.write(f, 'logs/' + os.path.basename(f))
print('results    ', round(os.path.getsize('/kaggle/working/bonus_results.zip')/1e6, 1), 'MB')

with zipfile.ZipFile('/kaggle/working/bonus_checkpoints.zip', 'w',
                     zipfile.ZIP_STORED) as z:
    for tag, path in CKPT.items():
        z.write(path, f'{LANGUAGE}_bonus_{tag}_best.pt')
print('checkpoints', round(os.path.getsize('/kaggle/working/bonus_checkpoints.zip')/1e6, 1), 'MB')
print(f'\nTOTAL RUNTIME {(time.time()-T0)/3600:.2f} hours')
