# Test Plan

What is tested, how, and what is out of scope. The traceability table in section 4 maps every functional (FR) and non-functional (NFR) requirement of [`requirements.md`](requirements.md) to the test or evidence that checks it.

Related documents: [`requirements.md`](requirements.md) (requirements and `predict()` contract), [`design/`](design/README.md) (use case, architecture and sequence diagrams), [`berturk_results.md`](berturk_results.md) (model evaluation).

## 1. Test strategy

| Level | What | How | When |
|---|---|---|---|
| Automated unit and contract tests | Schema, `predict()` contract and model backend, UI logic, evaluation, models, data pipeline, labeling tools | `uv run pytest` (27 test files under `tests/`) | Locally and in CI on every pull request |
| Static checks | Code style and bugs; no data files or model weights committed | `ruff check`, `ruff format --check`, `scripts/check_no_data.py` | Pre-commit hooks and CI (`lint`, `data-guard` jobs) |
| Measurements | Model loading and prediction time | `scripts/measure_latency.py` | After a model change, on an idle machine (section 3.3) |
| Model evaluation | Quality of the trained models on the frozen test set | `absa.models.baseline_full`, `absa.models.compare_models` | Once per final model (`experiment_plan.md`, `berturk_results.md`) |
| Manual UI checks | Web interface behaviour | Checklist in section 6 | Before merging UI changes and before the demo |
| Human annotation checks | Label quality and agreement | Annotation rules, kappa on an overlap set, adjudication | During labeling (section 7) |

Automated tests run on Ubuntu in CI (`.github/workflows/ci.yml`, job `test`) with the dependencies pinned in `uv.lock` (NFR-5). The pytest settings are in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["src"]`).

### Mock and real model

The trained model (440 MB) is never committed, so CI has no model. `tests/conftest.py` then sets `ABSA_BACKEND=mock`, and the contract tests run against the deterministic mock `predict()`. When `models/berturk_v1/` exists locally, the same tests run against the real BERTurk model. The real loading path (`inference.load()`, tokenizer, model heads) is still tested in CI: `tests/test_inference.py` and `tests/test_compare_models.py` train and save a tiny random BERT in a temporary folder, without any download.

Run the whole suite against both backends locally:

```bash
uv run pytest                       # real model if models/berturk_v1/ exists, otherwise mock
ABSA_BACKEND=mock uv run pytest     # always mock (as in CI)
```

## 2. Test files

| File | Covers |
|---|---|
| **Contract, inference and UI** | |
| `test_contract.py` | Schema names, ASCII keys, all 7 aspects in order, `label` and `confidence` only, valid labels, confidence in [0, 1], determinism, Turkish characters and emoji, edge cases, 10,000-character input, `ValueError` / `TypeError` |
| `test_inference.py` | Real loading path with a tiny saved model: contract, model loaded once and reused, same input gives same output, truncation detection (`is_truncated`), clear error when the model folder is missing, mock backend, validation before loading, masking of personal data before prediction |
| `test_app.py` | `analyze()` without a browser: truncation note for long reviews (FR-7), no note at the 128-token limit, empty input gives no result, `yok` aspects separated from the table (FR-4) |
| `test_smoke.py` | Package imports |
| **Models and evaluation** | |
| `test_evaluate.py` | Shared `evaluate()`: perfect score, hand-computed example, confusion-matrix label order, length mismatch, report lists every aspect |
| `test_baseline.py` | TF-IDF baseline: valid labels, single-label aspects, error before `fit`, length mismatch |
| `test_baseline_full.py` | Baseline on the frozen split: split file use, unknown review id error, better than always-`yok`, review types, error table, report |
| `test_first_run.py` | First end-to-end run (#42): seeded split, pipeline runs, warnings for unseen test labels, report |
| `test_berturk.py` | BERTurk training code with a tiny random model: output shape, label encoding and class weights, loss, best epoch on val, save/load gives the same predictions, seed summary, CLI sweep |
| `test_compare_models.py` | BERTurk vs baseline comparison: error types, cell changes, CLI |
| **Data pipeline** | |
| `test_data_prep.py` | Loaders map both sources to the common schema; end-to-end pipeline |
| `test_cleaner.py` | Word count, short / English / meaningless / template filters, dedup key (Turkish lowercase), step counts, masking before deduplication, edge cases |
| `test_masker.py` | Phone, landline, e-mail, URL, IBAN, card, valid TCKN, names; text without personal data unchanged; long texts; non-string input |
| `test_keywords.py` | Rare-aspect keyword hits, whole-word matching, Turkish lowercase |
| `test_categories.py` | Category rules, rule order, unknown category |
| `test_sampling.py` | Same seed gives the same sample, label shares, category cap, per-product cap, rare-aspect quota, statistics |
| `test_split.py` | Same seed gives the same split, test = kappa set, val share, near-duplicates in one part, test copies removed from train/val, missing-class warning, CLI |
| **Labeling** | |
| `test_labels.py` | Reading verified labels: safe fixes, invalid rows reported, BOM, `checked` rows only, kappa-set handling, class distribution and rare classes |
| `test_labeling.py` | Export validation (`validate_labels`) |
| `test_labeling_validate.py` | Parsing and validating LLM answers against the schema |
| `test_labeling_prompt.py` | Prompt versions contain every aspect and label; examples follow the schema |
| `test_labeling_client.py` | LLM client settings from the environment |
| `test_labeling_labeler.py` | LLM labeling with retries, invalid and failed answers, cache |
| `test_labeling_run.py` | Pre-labeling run: seeded pilot, resume from cache, cost estimate |
| `test_labeling_prepare.py` | Labeling sheets: batches and kappa set do not overlap, kappa set without pre-labels, equal workload, seeded files |
| `test_labeling_compare.py` | LLM / annotator comparison: accuracy, kappa, missed / extra / wrong sentiment, mean kappa |
| `test_labeling_adjudicate.py` | Adjudication of the kappa set: disagreements, final file, missing or invalid decisions |
| `test_labeling_report.py` | #52 report: batches vs kappa set, distribution, rare classes, per-annotator change rate |

All tests use small synthetic data and never read real review files (NFR-4).

## 3. What is verified, and how

### 3.1 `predict()` contract

| Contract rule (`requirements.md` section 5) | Test |
|---|---|
| Rule 1: all 7 aspects, in `ASPECTS` order | `test_contract.py::test_all_aspects_present_in_order`, `test_inference.py::test_model_yolu_ve_sozlesme` |
| Rule 2: `label` is one of `LABELS` | `test_contract.py::test_every_label_is_valid` |
| Rule 3: `confidence` is a float in [0, 1] | `test_contract.py::test_confidence_is_float_between_0_and_1` |
| Rule 4: same input, same output | `test_contract.py::test_same_input_returns_same_output`, `test_inference.py::test_ayni_girdi_ayni_cikti` |
| Rule 5: input longer than 128 tokens is truncated | `test_inference.py::test_kesme_bilgisi`, `test_contract.py::test_predict_very_long_text` |
| Rule 6: model loaded on the first call and reused | `test_inference.py::test_model_bir_kez_yuklenir` |
| Errors: not a `str` → `TypeError`, empty → `ValueError` | `test_contract.py::test_non_str_input_raises_type_error`, `test_empty_input_raises_value_error`, `test_inference.py::test_bos_ve_yanlis_tip_model_yuklemeden_hata` |
| Model folder missing → clear error | `test_inference.py::test_model_yoksa_acik_hata` |

### 3.2 Model quality

Model quality is not a pass/fail test. Both models are trained on the same split (`docs/data_split.md`) and evaluated with the same `evaluate()`; the test set (the 200 adjudicated kappa-set reviews with independent human labels) was used once for the final model. Results: [`baseline_results.md`](baseline_results.md), [`berturk_results.md`](berturk_results.md).

| Model | test macro-F1 |
|---|---:|
| always `yok` | 0.233 |
| TF-IDF + LR | 0.501 |
| **BERTurk** (used in the app) | **0.625** |

### 3.3 Performance measurement (NFR-1, NFR-2)

`scripts/measure_latency.py` on the 200 test reviews, model `berturk_v1`, CPU only:

| Date | Machine | Load (import + `load()`) | `predict()` median / p95 / max | 10,000+ character review |
|---|---|---:|---:|---:|
| 2026-11-28 | Windows 11, AMD x86-64 CPU, 8 threads, torch 2.15 (CPU) | 5.9 s + 12.9–13.7 s = **about 19–20 s** | **0.05 / 0.07 / 0.13 s** | 0.14 s |

Both limits are met (60 s and 2 s). The very first start right after copying the model took 27.7 s + 89.8 s, while another process was running and the weight file was not yet in the disk cache. Later starts are in the range above. Of the 1,804 labeled reviews, 5 are longer than 128 tokens, so truncation (FR-7) is rare.

## 4. Traceability

Status: **met** = checked by an automated test or a measurement; **manual** = checked with the UI checklist; **partly** = see the note.

### 4.1 Functional requirements

| ID | Requirement | Test / evidence | Status |
|---|---|---|---|
| FR-1 | Enter a review as free text | Manual check M-1; `test_app.py` calls `analyze()` with free text | met + manual |
| FR-2 | Start the analysis with the "Analiz et" button | Manual check M-1 (button and Enter are bound to `analyze()` in `app.py`) | manual |
| FR-3 | Label and confidence for each of the 7 aspects | `test_contract.py` (all aspects, labels, confidence); manual check M-2 | met + manual |
| FR-4 | `yok` aspects shown separately | `test_app.py::test_yok_olanlar_tablo_disinda`; manual check M-3 | met + manual |
| FR-5 | 3–4 example reviews run with one click | Manual check M-4 (4 examples in `app.py`) | manual |
| FR-6 | Empty input shows an error message instead of a result | `test_app.py::test_bos_girdide_sonuc_ve_not_yok`, `test_contract.py::test_empty_input_raises_value_error`; manual check M-5 | met + manual |
| FR-7 | Long reviews are truncated and the user is informed | Truncation: `test_inference.py::test_kesme_bilgisi`; note in the UI: `test_app.py::test_uzun_yorumda_kesme_notu_gosterilir`, `test_sinirda_kesme_yok`; manual check M-8 | met + manual |

### 4.2 Non-functional requirements

| ID | Requirement | Test / evidence | Status |
|---|---|---|---|
| NFR-1 | A prediction takes at most 2 s on CPU after loading | Measurement in section 3.3: max 0.13 s | met |
| NFR-2 | Model loaded once at startup, within 60 s | Once: `app.py` calls `load()` at startup, `test_inference.py::test_model_bir_kez_yuklenir`. Time: section 3.3, about 19–20 s | met |
| NFR-3 | Turkish characters, emoji and punctuation; UTF-8 | `test_contract.py::test_turkish_characters_and_emoji_do_not_fail`, `test_predict_edge_cases`; `test_cleaner.py::test_dedup_key_turkish_lowercase`; manual check M-6 | met |
| NFR-4 | User input is not stored or logged; no personal data in training data | `test_masker.py` (all masking rules), `test_cleaner.py::test_clean_reviews_maskeler`, `test_inference.py::test_kisisel_veri_maskelenir`; code review: `app.py` and `inference.py` neither write files nor log input; `scripts/check_no_data.py` blocks data files | met |
| NFR-5 | Dependencies pinned; fixed seeds | `uv sync --locked` in CI; seeds: `test_sampling.py::test_ayni_seed_ayni_ornek`, `test_split.py::test_ayni_seed_ayni_bolme`, `test_labeling_prepare.py::test_ayni_seed_ayni_dosyalar_satir_sirasi_onemsiz`; training seeds in `experiment_plan.md` | met |
| NFR-6 | Runs on Windows and Linux with Python 3.12 | Linux: every CI run. Windows: full test suite (262 tests, real model and mock) and the measurement in section 3.3 on Windows 11 | met (Windows checked locally, not in CI) |
| NFR-7 | `predict()` contract covered by automated tests in CI | Section 3.1; CI job `test` | met |

## 5. Known limitations

- **No browser-level UI test.** `analyze()` is tested directly; the Gradio layout and the button binding are checked manually (section 6).
- **CI runs on Linux only.** Windows is verified by running the suite locally.
- **The real model is not tested in CI.** CI uses the mock and a tiny random model for the loading path; the real model is tested locally and evaluated in `berturk_results.md`.
- **Performance depends on the machine.** Section 3.3 is one measurement on one computer; re-run the script on the demo machine.
- **`notr` is never predicted** by the final model (`berturk_results.md`, 3.3). This is a model-quality limitation, not a test failure.

## 6. Manual UI checklist

Start the app with `uv run python app.py` (model in `models/berturk_v1/`) and check each item. Record the date, the tester and the result.

| ID | Steps | Expected result |
|---|---|---|
| M-1 | Type a review, click **Analiz et**; repeat by pressing Enter | A result table appears (FR-1, FR-2) |
| M-2 | Analyze a normal review | Table has columns Özellik, Duygu, Güven; each shown row has a label and a percentage (FR-3) |
| M-3 | Open the "Bahsedilmeyen özellikler (yok)" section | Aspects labeled `yok` are listed there and not in the table (FR-4) |
| M-4 | Click each of the example reviews | The text is filled in and a result is shown (FR-5) |
| M-5 | Click **Analiz et** with an empty box, then with only spaces | A warning message appears and no result is shown (FR-6) |
| M-6 | Analyze a review with Turkish characters and an emoji | No error; a result is shown (NFR-3) |
| M-7 | Analyze the same review twice | Identical results (contract rule 4) |
| M-8 | Paste a review longer than 128 tokens (about 100 words or more) | A result is shown with the note that only the first part was analyzed (FR-7) |
| M-9 | Start the app without the model folder | The app stops with a message that names the missing folder and `ABSA_BACKEND=mock` |

## 7. Human annotation

Labels are not tested by code but checked by process (details: [`annotation/labeling_guide.md`](annotation/labeling_guide.md)).

- **Rules:** [`annotation/aspects.md`](annotation/aspects.md) and [`annotation/boundary_rules.md`](annotation/boundary_rules.md).
- **Tool:** Google Sheets with drop-down lists and a `checked` box per row, in the restricted Drive folder `absa-data/labeling/`. Only `checked` rows are used; every export is validated (`validate_labels`).
- **Agreement:** two annotators labeled the same 200 reviews without LLM suggestions. Mean Cohen's kappa 0.64 (target 0.60); four aspects below 0.60 got a written cause and a rule clarification (#51, #60). All disagreements were adjudicated into `kappa_final.csv`.
- **Labeled data:** 1,804 checked reviews (Salih 404, Ozan 300, Yusuf 900 in the pre-labeled batches, plus the 200-review kappa set). In the batches, annotators changed only 0.5% of the LLM labels (#52), so the batch labels are used for training only and the kappa set is the test set (`data_split.md`).

## 8. Out of scope

- Model quality targets. Section 3.2 reports the scores; there is no pass/fail threshold.
- Load testing with many users. The app is a single-user demo.
- Testing Gradio itself or browser compatibility.
- Security testing. User input is not stored (NFR-4), and the app is not exposed publicly.
- Batch analysis, free-text aspect term extraction and languages other than Turkish (`requirements.md` section 1).
