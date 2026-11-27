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

### Split sizes

| Split | reviews | share | labels |
|---|---:|---:|---|
| train | 1363 | 75.6% | batches (LLM pre-labels, checked) |
| val | 241 | 13.4% | batches (LLM pre-labels, checked) |
| test | 200 | 11.1% | adjudicated kappa set (independent human labels) |
| **total** | **1804** | | |

Near-duplicates of test reviews removed from train/val: 0

### Label distribution per split (olumlu / olumsuz / notr; the rest is `yok`)

| Aspect | train | val | test |
|---|---|---|---|
| `kalite` | 168/263/16 | 31/49/3 | 24/33/10 |
| `fiyat` | 124/56/5 | 30/10/0 | 17/1/2 |
| `kargo_paketleme` | 123/81/18 | 21/16/3 | 22/10/3 |
| `satici_hizmet` | 52/117/1 | 11/10/0 | 10/14/2 |
| `islev_performans` | 264/278/17 | 46/52/7 | 29/33/17 |
| `gorunum_tasarim` | 65/93/4 | 7/26/2 | 7/15/0 |
| `beden_uyum` | 54/162/10 | 5/31/1 | 4/22/3 |

### Classes missing from val or test

- `fiyat` / `notr`: val içinde örnek yok
- `satici_hizmet` / `notr`: val içinde örnek yok
- `gorunum_tasarim` / `notr`: test içinde örnek yok
