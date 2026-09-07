# Konkani — pretraining (Model L)

Training is driven by `tools/train.py`, shared between the two languages and
selected with `--language konkani`. The command used:

```
python3 tools/train.py --language konkani --max-tokens 500000000 --device cuda \
  --data-dir <packed> --out-dir <checkpoints>
```

Run on a Kaggle T4, both languages simultaneously on separate GPUs. Notebook
setup is in `../../report/phase2_kaggle_runbook.md`.

| artifact | location |
|---|---|
| training log (per-step loss, lr, grad norm, tokens) | `../../report/training_logs/konkani_pretrain_log.csv` |
| loss curves | `../../report/figures/phase2_loss_konkani.png` |
| model configuration | `../configs/model_config.json` |
| training configuration and checkpoint | `konkani_pretrain_config.json`, `konkani_pretrain_best.pt` in `phase2_checkpoints/` on Drive (https://drive.google.com/drive/folders/1lUSriyp7_tltkmHCgINon175xnFp2lp-?usp=sharing) |

Token packing (splits to a flat `uint16` array) is `tools/pack_tokens.py`.
