# Data

The contents of this directory are **never committed**. Only the folder structure is tracked. The review datasets are licensed (TRSAv1: CC BY-SA 4.0, Trendyol: MIT, see [`docs/data_sources.md`](../docs/data_sources.md)), so all data and the trained model are shared only through a **restricted** Google Drive folder.

**Shared folder:** [`absa-data` (Google Drive, restricted)](https://drive.google.com/drive/folders/1ejuDxnYZ26be-paumcfRYs7cSfYUx3cM). Ask a team member for access.

## Drive structure (`absa-data/`)

| Folder | Contents | Created by | Local path |
|---|---|---|---|
| `raw/` | `TRSAv1.csv`, `trendyol.csv` (original sources) | downloaded from the sources | `data/raw/` |
| `clean/` | `reviews_clean.csv` – 220,985 cleaned and masked reviews | `absa.data.pipeline` | `data/interim/` |
| `sample/` | `labeling_sample.csv` – 2,500 reviews selected for labeling | `absa.data.sampling` | `data/interim/` |
| `prelabels/` | `prelabels.csv` (LLM pre-labels, prompt v2), `prelabels_stats`, `prelabels_cache.jsonl`, `prelabels_pilot.csv`, `pilot_checked.csv` | `absa.labeling.run` | `data/interim/` |
| `labeling/` | Google Sheets used for labeling (`kappa_ozan`, `kappa_salih`, `batch_ozan`, `batch_salih`, `batch_yusuf`), `assignment.md` | `absa.labeling.prepare` | – |
| `labeling/private/` | `kappa_prelabels.csv` – LLM labels of the kappa set, kept away from the kappa annotators | `absa.labeling.prepare` | `data/interim/` |
| `labeling/exports/` | Annotator exports (`export_<name>.csv`, `export_kappa_<name>.csv`); only `checked` rows count | Google Sheets → CSV | `data/interim/labeling/` |
| `final/` | `final_labels.csv` – 1,804 verified reviews; `kappa_final.csv` – the 200 adjudicated kappa-set reviews | `absa.data.labels.load_labels`, `absa.labeling.adjudicate` | `data/interim/` |
| `splits/` | `split_v1.csv` (`review_id` → train / val / test), `split_v1_report.md` | `absa.data.split` | `data/interim/splits/` |
| `models/` | `berturk_v1/` – the trained BERTurk model used by the app | `absa.models.berturk` | `models/berturk_v1/` (repository root) |

## Local folders

| Folder | Contents |
|---|---|
| `raw/` | Original data as collected. Never edited by hand. |
| `interim/` | Cleaned data, samples, labels, splits and reports |
| `processed/` | Not used (the split is a `review_id` → split file in `interim/splits/`) |

The model is not stored under `data/`: download `absa-data/models/berturk_v1/` into `models/berturk_v1/` at the repository root (see the main README, "Run the app"). `models/` is ignored by git.

## Regenerating

| Step | Command | Documentation |
|---|---|---|
| Clean and mask | `uv run python -m absa.data.pipeline --trsa <TRSAv1.csv> --trendyol <trendyol.csv>` | `docs/data_sources.md` |
| Labeling sample | `uv run python -m absa.data.sampling` | `docs/labeling_sample.md` |
| LLM pre-labels | `uv run python -m absa.labeling.run --out data/interim/prelabels.csv` (needs an API key, `.env.example`) | `docs/prelabeling.md` |
| Labeling sheets | `uv run python -m absa.labeling.prepare --annotators ozan salih yusuf --kappa-annotators ozan salih` | `docs/annotation/labeling_guide.md` |
| Kappa, adjudication, LLM report | `absa.labeling.compare`, `absa.labeling.adjudicate`, `absa.labeling.report` | `docs/annotation/labeling_guide.md`, section 7 |
| Split | `uv run python -m absa.data.split --labels ... --test-ids ... --out ...` | `docs/data_split.md` |

Labeling itself is manual and cannot be regenerated; `final_labels.csv` and `kappa_final.csv` are the source of truth for training and evaluation.
