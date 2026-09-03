# Running a pretraining job on Kaggle

Written for someone who has not used Kaggle before. Every step is literal.
Nothing here needs a credit card.

## 1. The mental model

Two Kaggle concepts matter:

- A **Notebook** is a Jupyter notebook on Kaggle's machine, with a GPU if you ask
  for one. **It starts empty every time and keeps nothing.** Anything created
  inside it disappears when the session ends.
- A **Dataset** is a folder of files uploaded once and attachable to any
  notebook, where it appears read-only under `/kaggle/input/`.

Everything else follows: code and data both have to arrive as Datasets, and
anything the run produces has to be downloaded before the session ends.

Free tier gives 30 GPU-hours per week, sessions up to roughly 9-12 hours, on a
T4 or a P100. The quota is published and tracked, which is the main reason to
prefer it over free Colab, where neither a GPU nor any hour allowance is
guaranteed.

**Choose T4, not P100.** The P100 is the nominally larger card but has no fp16
tensor cores, which makes it 3-4x slower for this workload.

## 2. Two datasets, not one

| dataset | contents | size | changes |
|---|---|---|---|
| `lma-phase2-data` | 12 packed `.bin` / `.json` files | ~2.0 GB | never |
| `lma-phase2-code` | `lma_phase2_code.zip` (`common/` + `tools/`) | ~130 KB | often |

Separate because a code fix should be a ten-second re-upload rather than 2 GB.

The code goes up as **a zip you make yourself**, unzipped by the notebook. The
`--dir-mode zip` option uploads each directory as its own archive and whether
Kaggle re-extracts them is not documented clearly enough to depend on; a zip you
control behaves the same way every time.

## 3. Credentials

Kaggle's Settings page has two token mechanisms. The one labelled **API Tokens
(Recommended)** displays a token string once and downloads nothing. The `kaggle`
CLI's standard flow wants the *other* one:

1. kaggle.com -> profile picture -> **Settings** -> **API Tokens** tab.
2. Scroll to **Legacy API Credentials** and use its button. `kaggle.json`
   downloads.

```bash
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list --max-size 1000 | head -3      # prints rows if it worked
```

`chmod 600` is required - the tool refuses to run on a world-readable
credentials file.

## 4. Build the upload folders

```bash
cd ~/Desktop/individual-project-Pakkabhabad18
rm -rf ~/Desktop/kaggle_upload
mkdir -p ~/Desktop/kaggle_upload/data ~/Desktop/kaggle_upload/code

for L in marathi konkani; do
  for S in train val test; do
    cp "$L/data/packed/$S.bin"  ~/Desktop/kaggle_upload/data/"${L}_${S}.bin"
    cp "$L/data/packed/$S.json" ~/Desktop/kaggle_upload/data/"${L}_${S}.json"
  done
done

find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
zip -qr ~/Desktop/kaggle_upload/code/lma_phase2_code.zip common tools \
    -x '*__pycache__*'

du -sh ~/Desktop/kaggle_upload/data ~/Desktop/kaggle_upload/code
```

Expect ~1.9 GiB and ~130 KB. The data files are flattened and prefixed because
the two languages would otherwise collide in one folder.

## 5. Metadata, then upload

Reading the username from `kaggle.json` avoids the usual typo:

```bash
python3 - <<'EOF'
import json, pathlib
user = json.loads((pathlib.Path.home()/".kaggle"/"kaggle.json").read_text())["username"]
for folder, slug in (("data", "lma-phase2-data"), ("code", "lma-phase2-code")):
    p = pathlib.Path.home()/"Desktop"/"kaggle_upload"/folder/"dataset-metadata.json"
    p.write_text(json.dumps({"title": slug, "id": f"{user}/{slug}",
                             "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    print("wrote", p)
EOF

cd ~/Desktop/kaggle_upload/code && kaggle datasets create -p .   # seconds
cd ~/Desktop/kaggle_upload/data && kaggle datasets create -p .   # 10-30 min
```

Datasets are private by default, which is correct here.

**Updating the code** afterwards - used every time something is fixed:

```bash
cd ~/Desktop/individual-project-Pakkabhabad18 && \
  rm -f ~/Desktop/kaggle_upload/code/lma_phase2_code.zip && \
  zip -qr ~/Desktop/kaggle_upload/code/lma_phase2_code.zip common tools \
      -x '*__pycache__*' && \
  cd ~/Desktop/kaggle_upload/code && \
  kaggle datasets version -p . -m "update code"
```

Notebooks pick up the newest version automatically.

## 6. Create the notebook

1. kaggle.com/code -> **New Notebook**.
2. Right panel -> **Input** -> **Add Input** -> add `lma-phase2-code` and
   `lma-phase2-data`. If they do not appear, switch the search filter to
   **Your Datasets**.
3. Right panel -> **Session options** -> **Accelerator** -> **GPU T4 x2**.
   This restarts the session, so do it before running anything.
4. **Internet: Off.** Nothing here downloads anything.

## 7. The cells

**Cell 1 - confirm a GPU is really attached.** A session that quietly fell back
to CPU looks the same for the first few minutes and then takes about three weeks.

```python
import torch, subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU. Accelerator -> GPU T4 x2, then rerun."
```

**Cell 2 - unpack the code, lay out the data.**

```python
import sys, pathlib, zipfile

zipfile.ZipFile("/kaggle/input/lma-phase2-code/lma_phase2_code.zip"
                ).extractall("/kaggle/working/code")
sys.path.insert(0, "/kaggle/working/code")

# The dataset stores files flat and prefixed; the loader wants train/val/test per
# language. Symlinks rather than copies - each train file is 1 GB and
# /kaggle/working is not large.
for lang in ("marathi", "konkani"):
    d = pathlib.Path(f"/kaggle/working/data/{lang}")
    d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for ext in ("bin", "json"):
            dst = d / f"{split}.{ext}"
            if not dst.exists():
                dst.symlink_to(f"/kaggle/input/lma-phase2-data/{lang}_{split}.{ext}")

print(sorted(p.name for p in pathlib.Path("/kaggle/working/data/konkani").iterdir()))
```

**Cell 3 - the same checks that passed on the Mac.**

```python
!cd /kaggle/working/code && python3 tools/verify_model.py --full
```

**Cell 4 - smoke run on the GPU, about two minutes.**

```python
!cd /kaggle/working/code && python3 tools/train.py \
  --language konkani --smoke --device cuda \
  --data-dir /kaggle/working/data/konkani \
  --out-dir /kaggle/working/checkpoints/konkani
```

**Cell 5 - the real run.** Konkani first: it is the model this project is really
about, and the one least worth losing to a mistake.

```python
!cd /kaggle/working/code && python3 tools/train.py \
  --language konkani --max-tokens 500000000 --device cuda \
  --data-dir /kaggle/working/data/konkani \
  --out-dir /kaggle/working/checkpoints/konkani
```

The first log lines print tokens/sec and an ETA. An ETA above ~6 hours means
something is wrong with batch size or precision; stop and diagnose rather than
waiting it out.

**Cell 6 - save the results.** `/kaggle/working` is wiped when the session ends.

```python
import shutil
shutil.make_archive("/kaggle/working/konkani_pretrain", "zip",
                    "/kaggle/working/checkpoints/konkani")
```

Then download the zip from the **Output** panel. It holds the checkpoint, the CSV
loss log and the config.

## 8. Running unattended

For a multi-hour job use **Save Version -> Save & Run All (Commit)** rather than
an interactive session. It runs the notebook top to bottom on Kaggle's side and
keeps the outputs, so the laptop can sleep.

If a session dies mid-run, re-run the same command: the training script finds its
own checkpoint and resumes. That only helps if the checkpoint still exists, which
is another argument for committing rather than sitting on an interactive tab.

Remaining quota is under your profile -> **Accelerator** usage.

## 9. Then Marathi

Identical cells with `--language marathi` and
`--out-dir /kaggle/working/checkpoints/marathi`. Same token budget, so expect a
similar runtime.

## 10. After both runs

Checkpoints go to Google Drive with the links in the README - the specification
requires that, and a checkpoint of this size must not be committed to git.
Training logs and loss curves *do* belong in the repository under `report/`,
because they are graded material and the specification says graders will not open
external dashboards to find figures.
