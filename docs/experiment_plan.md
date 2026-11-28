# Experiment Plan

## 1. Models
- **Baseline Model:** TF-IDF feature extraction combined with a Logistic Regression classifier.
- **Deep Learning Model:** BERTurk pretrained model equipped with 7 distinct output heads for multi-aspect classification.

## 2. Evaluation Metrics
- Macro-F1 score per aspect.
- Overall Macro-F1 score across all aspects.
- Confusion matrix for each aspect to analyze misclassifications.

## 3. Data Split
- Split `split_v1` (see `docs/data_split.md`, #53), used by every model:
  - **test:** the 200 reviews of the adjudicated kappa set (independent human labels, #51)
  - **train / val:** the remaining 1,604 batch reviews, 85% / 15%, near-duplicates kept in one split
- The originally planned random 70% / 15% / 15% split was replaced because the batch labels are
  in practice LLM labels (#52); testing on independent human labels measures agreement with humans.
- A fixed seed (`random_state=42`) is used for the split and for the models.
- Hyperparameters are chosen on **val** only; **test** is used once per final model.

## 4. Experiment Tracking
For every experimental run, the following information must be recorded:
- The model used (architecture and baseline details).
- Data version and split configuration.
- Hyperparameter settings (learning rate, batch size, etc.).
- Evaluation scores (as defined in the metrics section).

## 5. Experiment Log

| # | Date | Model | Data | Split | Settings | val macro-F1 | test macro-F1 | Notes |
|---|---|---|---|---|---|---:|---:|---|
| 0 | – | TF-IDF + LR | first 295 checked reviews | random 80/20 (236 / 59) | as run 1 | – | (0.470) | pipeline check (#33), **not a result**, not comparable |
| 1 | 2026-11-27 | TF-IDF (1–2 gram, sublinear tf) + LogisticRegression (`class_weight="balanced"`, `max_iter=1000`), one per aspect | `final_labels.csv` (1,804) | `split_v1` (1,363 / 241 / 200) | `random_state=42` | 0.547 | **0.501** | reference for BERTurk; always-`yok`: 0.241 / 0.233; details in `docs/baseline_results.md` (#54) |
| 2 | 2026-11-28 | BERTurk (`dbmdz/bert-base-turkish-cased`) + 7 linear heads | `final_labels.csv` (1,804) | `split_v1` | sweep: lr 2e-5 / 3e-5 / 5e-5 × class weights off / on; batch 16, 5 epochs, seed 42 | 0.548 – 0.680 | – | best: lr 5e-5, class weights off (0.680); all runs peaked in the last epoch → epoch limit raised to 8 |
| 3 | 2026-11-28 | same | same | `split_v1` | lr 5e-5, no class weights, batch 16, ≤ 8 epochs, early stopping (patience 2), seeds 42 / 43 / 44 | **0.690 ± 0.011** (42: 0.689, 43: 0.704, 44: 0.676) | **0.625** (seed 43) | selected model: seed 43 → Drive `absa-data/models/berturk_v1/`; test evaluated once in #66, details in `docs/berturk_results.md` |

Reproduce run 1:

```bash
uv run python -m absa.models.baseline_full \
    --labels data/interim/labeling/final_labels.csv \
    --split data/interim/splits/split_v1.csv
```

Reproduce runs 2 and 3 (GPU, e.g. Google Colab T4; ~10 s per epoch):

```bash
python -m absa.models.berturk --labels final_labels.csv --split split_v1.csv \
    --out outputs/sweep --lr 2e-5 3e-5 5e-5 --class-weights off on --seeds 42
python -m absa.models.berturk --labels final_labels.csv --split split_v1.csv \
    --out outputs/final --lr 5e-5 --class-weights off --seeds 42 43 44 --epochs 8
```
