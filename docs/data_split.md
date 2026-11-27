# Train / validation / test split

The labeled data (`final_labels.csv`, 1,804 reviews, see #50) is split once with
`src/absa/data/split.py`. Every model (TF-IDF baseline, BERTurk) uses the same split file, so
their scores are comparable. **The test set is frozen and is not changed again.**

## Rule

| Split | Reviews | Labels |
|---|---|---|
| test | the 200 reviews of the adjudicated kappa set (`kappa_final.csv`) | independent human labels: two annotators labeled them without LLM suggestions, disagreements were decided jointly (#51) |
| train / val | the remaining batch reviews, 85 % / 15 %, `random_state=42` | LLM pre-labels checked by an annotator |

Why the kappa set is the test set: #52 showed that annotators left 99.5 % of the LLM
pre-labels in the batches unchanged, so the batch labels are in practice LLM labels. Testing on
them would measure agreement with the LLM, not with humans. The models are therefore trained on
the batch labels (weak / "silver" labels) and evaluated on independent human labels ("gold").

The original plan in #53 was a random 70 / 15 / 15 split; it was changed for the reason above.

## Near-duplicates

Reviews whose `dedup_key` (Turkish lowercase, punctuation and whitespace ignored, see
`src/absa/data/cleaner.py`) is the same form a group. Train and val are split by group, so a
group never appears in both. A batch review that is a near-duplicate of a test review is removed
from train/val (listed in the report), so no test text is seen during training.

## Reproduce

```bash
uv run python -m absa.data.split \
    --labels data/interim/labeling/final_labels.csv \
    --test-ids data/interim/labeling/kappa_final.csv \
    --out data/interim/splits/split_v1.csv
```

Outputs (never committed; Drive: `absa-data/splits/`): `split_v1.csv` (`review_id`, `split`)
and `split_v1_report.md`.

## Result (split_v1)

<!-- Paste the content of data/interim/splits/split_v1_report.md below. -->
