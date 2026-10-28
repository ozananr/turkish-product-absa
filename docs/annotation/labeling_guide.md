# Labeling Guide

How the team labels the 2,500-review sample. The rules for *which label to choose* are in [`aspects.md`](aspects.md) and [`boundary_rules.md`](boundary_rules.md); this guide covers the tool, the files and the workflow.

## 1. Tool

**Google Sheets**, one sheet per file below, in the restricted Drive folder `absa-data/labeling/`. A shared spreadsheet is enough for a 3-person team and needs no installation.

## 2. Files and assignment

The files are created by `src/absa/labeling/prepare.py` from the LLM pre-labels (`prelabels.csv`, #31):

```bash
uv run python -m absa.labeling.prepare --annotators ozan salih yusuf --kappa-annotators ozan salih
```

| File | Annotator | Reviews | Pre-labels |
|---|---|---:|---|
| `kappa_ozan.csv` | ozan | 200 | no (kappa set) |
| `kappa_salih.csv` | salih | 200 | no (kappa set) |
| `batch_ozan.csv` | ozan | 700 | yes |
| `batch_salih.csv` | salih | 700 | yes |
| `batch_yusuf.csv` | yusuf | 900 | yes |

- **Batches** contain the LLM pre-labels as starting values. Batches do not overlap.
- **Kappa set:** the same 200 reviews for two annotators, **without pre-labels**, so the LLM does not influence agreement (#32). They are not in any batch. The two kappa annotators get smaller batches, so everyone labels 900 reviews in total.
- The 50 pilot reviews from #31 (`pilot_checked.csv`) are never in the kappa set, because they were checked with the LLM suggestions visible.
- `kappa_prelabels.csv` holds the LLM labels of the kappa set. It is **not** shared with annotators; it is kept to compare the LLM with humans later.
- Same input and seed (42) always give the same files.

## 3. Setting up a sheet (once per file)

1. Drive → **New → File upload**, upload the CSV, open it with Google Sheets.
2. **View → Freeze → 1 row**, so the header stays visible.
3. Select the 7 aspect columns (`kalite` … `beden_uyum`) → **Data → Data validation → Dropdown** with the items `yok`, `olumlu`, `olumsuz`, `notr`, and **Reject the input** for invalid data. This prevents typos.
4. Select the `checked` column → **Insert → Checkbox**.
5. Optional: **Data → Protect sheets and ranges** for `review_id`, `text` and `annotator`, so they are not changed by accident.

## 4. How to label

- Label from top to bottom. The rows are in random order on purpose, so a batch that is only partly finished is still a representative sample.
- Read the review and set one label for **each** of the 7 aspects, following `aspects.md` and `boundary_rules.md`. Labels are based **only on the text**; there is no star rating in the sheet on purpose.
- In batches the LLM suggestion is already filled in. Treat it as a suggestion: check every aspect, not only the ones the LLM marked. Known LLM weaknesses (#31): it suggests `kalite` too often and almost never suggests `notr`.
- When all 7 labels of a row are final, tick **`checked`**. Only checked rows are exported, so unticked rows count as not labeled yet, even if they have labels.
- If a case is not covered by the rules, label it as well as you can and write the question in **`note`**. Notes are collected and decided by the team; new decisions go into `boundary_rules.md`.
- **Kappa set:** label it on your own. Do not look at the other annotator's sheet or at the LLM labels, and do not discuss these reviews until both are finished.
- Do not add, delete or reorder rows or columns.

## 5. Export format

Each annotator downloads their sheet with **File → Download → Comma-separated values (.csv)**.

| Column | Content |
|---|---|
| `review_id` | unchanged from the sample |
| `text` | unchanged |
| `kalite` … `beden_uyum` | one of `yok`, `olumlu`, `olumsuz`, `notr` (keys and labels from `src/absa/schema.py`) |
| `annotator` | annotator name |
| `checked` | `TRUE` when the row is final |
| `note` | optional |

Only rows with `checked = TRUE` are used. The same `review_id` appears twice only in the kappa set (two annotators), never twice for one annotator.

Check an export before sharing it:

```bash
uv run python -c "import pandas as pd; from absa.labeling.validate import validate_labels; df = pd.read_csv('data/interim/labeling/export_ozan.csv', dtype=str, keep_default_na=False); print(validate_labels(df[df['checked'] == 'TRUE']) or 'OK')"
```

`validate_labels` reports missing columns, invalid or empty labels, empty texts, empty `annotator` and duplicate ids for one annotator. An empty list (`OK`) means the export is valid.

## 6. Data policy

All sheets and exports stay in the restricted Drive folder. They are never committed (the data guard blocks `.csv` files).
