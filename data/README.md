# Data

The contents of this directory are **never committed**. Only the folder structure is tracked.

| Folder | Contents |
|---|---|
| `raw/` | Original data as collected. Never edited by hand. |
| `interim/` | Intermediate files during cleaning |
| `processed/` | Final labeled data split into train/val/test |

Shared data location: _(link will be added)_

To regenerate: `uv run python -m absa.data.pipeline --trsa <TRSAv1.csv> --trendyol <trendyol.csv>`
