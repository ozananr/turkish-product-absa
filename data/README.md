# Data

The contents of this directory are **never committed**. Only the folder structure is tracked.

| Folder | Contents |
|---|---|
| `raw/` | Original data as collected. Never edited by hand. |
| `interim/` | Intermediate files during cleaning |
| `processed/` | Final labeled data split into train/val/test |

Shared data location: [reviews_clean.csv](https://github.com/ozananr/turkish-product-absa/releases/download/data-v1/reviews_clean.csv) (GitHub Release `data-v1`, cleaned and masked data, 220,985 reviews). Download it into `data/interim/`.

To regenerate: `uv run python -m absa.data.pipeline --trsa <TRSAv1.csv> --trendyol <trendyol.csv>`
