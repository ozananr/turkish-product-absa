# Data

The contents of this directory are **never committed**. Only the folder structure is tracked.

| Folder | Contents |
|---|---|
| `raw/` | Original data as collected. Never edited by hand. |
| `interim/` | Intermediate files during cleaning |
| `processed/` | Final labeled data split into train/val/test |

Shared data location: [reviews_clean.csv (Google Drive)](https://drive.google.com/file/d/1kbEh5o8Jb7CrbVngM_zFfci5-quw6E39/view?usp=drive_link) (cleaned and masked data, 220,985 reviews). Access is restricted because the source datasets are licensed: ask a team member for access. Download it into `data/interim/`.

To regenerate: `uv run python -m absa.data.pipeline --trsa <TRSAv1.csv> --trendyol <trendyol.csv>`
