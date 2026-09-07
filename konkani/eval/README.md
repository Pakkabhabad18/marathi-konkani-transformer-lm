# Konkani — evaluation (Model L)

Two scripts, both shared between languages:

```
python3 tools/evaluate.py --language konkani \
  --checkpoint <ckpt>/konkani/pretrain_best.pt \
  --tokenizer konkani/tokenizer/konkani_bpe.model \
  --split test --windows 256 --prompts 24

python3 tools/attention_analysis.py --language konkani \
  --checkpoint <ckpt>/konkani/pretrain_best.pt \
  --tokenizer konkani/tokenizer/konkani_bpe.model
```

| result | location |
|---|---|
| perplexity, bits-per-byte, BLEU / chrF / ROUGE-L, diversity, sample generations | `../../report/phase2_eval_konkani.json` |
| per-head attention entropy and mean distance | `../../report/phase2_attention_konkani.json` |
| attention heatmaps (layers 0, 3, 6) | `../../report/figures/phase2_attention_konkani_layer*.png` |
| written analysis | `../../report/phase2_report.md` |

Metrics are implemented in `common/metrics.py` rather than imported, because the
Kaggle notebook ran with internet disabled. `python3 common/metrics.py` runs the
self-test.

Reported on the **test** split. Validation was used to select the best
checkpoint during training, so reporting on it would be optimistic.
