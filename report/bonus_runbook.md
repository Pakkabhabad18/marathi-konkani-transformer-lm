# Bonus — Runbook

Everything needed to run the ablation, in order. Follows
`phase2_kaggle_runbook.md`; only the differences are spelled out.

**Before you start, check two things on Kaggle:**

1. The dataset **`lma-phase2-data`** still exists (Datasets → Your Work). It is
   the ~2 GB of packed `.bin` / `.json` files from Phase 2 and it is what makes
   this a four-hour job rather than a re-upload. If it is gone, section 1 below
   rebuilds it — but the upload alone takes 10–30 minutes.
2. Your GPU quota has at least **8 hours** left (two T4s × ~4 hours). The quota
   line is under the accelerator selector.

---

## 1. Build and upload the bonus code dataset

The model code changed, so Phase 2's `lma-phase2-code` is the wrong source. This
uploads a new one from the bonus branch. It is ~1 MB and takes seconds.

```bash
cd ~/Desktop/individual-project-Pakkabhabad18
git checkout bonus-no-positional

rm -rf ~/Desktop/kaggle_bonus
mkdir -p ~/Desktop/kaggle_bonus/code

find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
zip -qr ~/Desktop/kaggle_bonus/code/lma_bonus_code.zip \
    common tools marathi/tokenizer -x '*__pycache__*'

du -sh ~/Desktop/kaggle_bonus/code
unzip -l ~/Desktop/kaggle_bonus/code/lma_bonus_code.zip | tail -3
```

`marathi/tokenizer` is included because Phase 2's zip was not, and both
`evaluate.py` and `attention_analysis.py` need the SentencePiece model. Cell 6
of the notebook asserts it is present rather than failing three and a half hours
in.

Then the metadata and the upload:

```bash
python3 - <<'EOF'
import json, pathlib
user = json.loads((pathlib.Path.home()/".kaggle"/"kaggle.json").read_text())["username"]
p = pathlib.Path.home()/"Desktop"/"kaggle_bonus"/"code"/"dataset-metadata.json"
p.write_text(json.dumps({"title": "lma-bonus-code",
                         "id": f"{user}/lma-bonus-code",
                         "licenses": [{"name": "CC0-1.0"}]}, indent=2))
print("wrote", p)
EOF

cd ~/Desktop/kaggle_bonus/code && kaggle datasets create -p .
```

If the dataset already exists from an earlier attempt, push a new version
instead:

```bash
cd ~/Desktop/kaggle_bonus/code && kaggle datasets version -p . -m "bonus code"
```

**If `lma-phase2-data` is missing**, rebuild it exactly as
`phase2_kaggle_runbook.md` §4–5 describes. Only Marathi is needed for this run,
so the loop can be shortened to `for L in marathi`.

---

## 2. Set the notebook up

1. Kaggle → **Code → New Notebook** → **File → Import Notebook** → upload
   `report/bonus_nope.ipynb`.
2. Settings: **Accelerator `GPU T4 x2`**, **Internet `Off`**.
3. **Add Input** → **Your Work** → attach **`lma-bonus-code`** and
   **`lma-phase2-data`**.

---

## 3. Run it

**Save Version → Save & Run All (Commit) → Save.** Not Run All.

This is a four-hour job. A browser session will not survive it and an
interactive run will lose everything if it drops — which is what happened in
Phase 3 and is recorded as D-058. A committed version runs server-side and
persists its output.

Close the tab. The version appears under the notebook's **Versions** tab when it
finishes.

### What the first fifteen minutes should look like

| cell | what it does | how long |
|---|---|---|
| 1–2 | find inputs, unpack code, symlink data | seconds |
| 3 | verify both architectures | ~1 min |
| 4 | smoke-train both arms on the GPU | ~2 min |
| 5 | the real runs, progress every 5 minutes | ~3.6 h |

Cell 3 must print **18/18** for the control and **24/24** for the ablated arm.
Cell 5 prints a progress line every five minutes with the last logged step of
each arm.

If cell 5's first progress line implies an ETA far above four hours, stop the
version: something is wrong with batch size or precision, and it is cheaper to
diagnose than to wait out.

---

## 4. Collect the results

From the finished version's **Output** panel:

- **`bonus_results.zip`** — a few MB. Evaluation JSONs for both arms, both
  training logs, the loss figure, the attention analysis. This is the one to
  hand back for the report.
- **`bonus_checkpoints.zip`** — ~600 MB, the two trained models. Goes to Google
  Drive alongside the Phase 2 and Phase 3 checkpoints. Not to git.

---

## 5. Afterwards

1. `report/bonus_report.md` is written from `bonus_results.zip`.
2. Commit the results to `bonus-no-positional`, not to `phase-3`.
3. Add the Drive link for `bonus_checkpoints.zip` to the README's bonus section.

---

## Notes

**Why both arms rather than reusing the Phase 2 checkpoint.** B-001. Briefly:
the Phase 2 model was trained in a different session weeks ago, so any
difference would be confounded with whatever else changed. Retraining the
control also tests that Phase 2 reproduces — cell 7 checks it against 8.4621 and
flags a deviation above 2%.

**Why the checkpoints have different names.** B-003. `train.py` resumes
automatically when it finds a checkpoint, so two arms sharing
`pretrain_best.pt` could silently continue each other's training. The ablated
arm writes `pretrain_nope_best.pt`.

**If the run dies partway.** `train.py` is resumable: re-committing the same
notebook picks up each arm from its own `*_latest.pt`. But Kaggle wipes
`/kaggle/working` between versions, so a resume only works within a version —
across versions it starts over. There is no time for two attempts before the
deadline, which is why cells 3 and 4 verify and smoke-test first.
