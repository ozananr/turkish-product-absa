# Test Plan (Draft)

This document describes what is tested, how, and what is out of scope. It is a draft: the real model arrives in Week 6 and sections marked *planned* will be updated then.

Related documents: [`requirements.md`](requirements.md) (requirements and `predict()` contract), [`design/`](design/README.md) (use case, architecture and sequence diagrams).

## 1. Test strategy

| Level | What | How | When |
|---|---|---|---|
| Automated unit and contract tests | Schema, `predict()` contract, evaluation functions, baseline, data pipeline | `uv run pytest` | Locally and in CI on every pull request |
| Static checks | Code style and bugs; no data files committed | `ruff check`, `ruff format --check`, `scripts/check_no_data.py` | Pre-commit hooks and CI (`lint`, `data-guard` jobs) |
| Manual UI checks | Web interface behaviour | Checklist in section 5 | Before merging UI changes and before the demo |
| Human annotation checks | Label quality | Annotation rules in `annotation/aspects.md` and `annotation/boundary_rules.md` | During labeling (section 6) |

Automated tests run on Ubuntu in CI (`.github/workflows/ci.yml`, job `test`). Dependencies are pinned in `uv.lock` (NFR-5). The pytest settings are in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["src"]`).

## 2. What is tested

### 2.1 Schema and `predict()` contract

Contract tests live in `tests/test_contract.py` and do not depend on Gradio. They run against the mock `predict()` today and must run unchanged against the real model (`requirements.md` section 6).

| Contract rule or requirement | Test |
|---|---|
| Aspect and label names are fixed (`schema.py`) | `test_schema_contents` |
| Keys use ASCII only (`requirements.md` section 2) | `test_keys_are_ascii_only` |
| Output contains all 7 aspects in `ASPECTS` order (rule 1) | `test_all_aspects_present_in_order` |
| Each result has exactly `label` and `confidence` | `test_each_result_has_label_and_confidence_only` |
| `label` is always one of `LABELS` (rule 2) | `test_every_label_is_valid` |
| `confidence` is a float between 0 and 1 (rule 3) | `test_confidence_is_float_between_0_and_1` |
| Same input gives same output (rule 4) | `test_same_input_returns_same_output` |
| Turkish characters and emojis do not fail (NFR-3) | `test_turkish_characters_and_emoji_do_not_fail` |
| Empty or whitespace-only input raises `ValueError` | `test_empty_input_raises_value_error` |
| Non-`str` input raises `TypeError` | `test_non_str_input_raises_type_error` |
| Package imports | `tests/test_smoke.py::test_package_imports` |

### 2.2 Evaluation functions

`src/absa/models/evaluate.py` is shared by all models, so its correctness decides whether model results are comparable. Tests: `tests/test_evaluate.py`.

| Behaviour | Test |
|---|---|
| Perfect predictions give score 1 | `test_perfect_predictions_score_one` |
| Scores match a hand-computed example | `test_known_example` |
| Confusion matrix follows the label order | `test_confusion_matrix_uses_label_order` |
| Length mismatch raises an error | `test_length_mismatch_raises` |
| Report lists every aspect | `test_format_report_lists_every_aspect` |

The TF-IDF baseline (`src/absa/models/baseline.py`) has its own tests in `tests/test_baseline.py`: valid labels for every aspect, single-label aspects, error before `fit`, and length mismatch.

### 2.3 Data pipeline

Cleaning, masking and sampling code in `src/absa/data/` is covered by `tests/test_cleaner.py`, `tests/test_masker.py`, `tests/test_data_prep.py` (loaders and end-to-end pipeline), `tests/test_keywords.py`, `tests/test_categories.py` and `tests/test_sampling.py` (same seed gives the same sample, category cap, label shares, stable `review_id`). Tests use small synthetic data, never real review files (NFR-4).

### 2.4 UI behaviour

There is **no automated UI test yet**. UI behaviour is checked manually (section 5). The functional requirements map to checks as follows.

| Requirement | Verified by |
|---|---|
| FR-1 Enter a review as free text | Manual check M-1 |
| FR-2 Start analysis with the "Analiz et" button | Manual check M-1 |
| FR-3 Label and confidence for each of the 7 aspects | Manual check M-2; contract tests cover the data it displays |
| FR-4 `yok` aspects shown separately | Manual check M-3 |
| FR-5 Example reviews run with one click | Manual check M-4 |
| FR-6 Empty input shows an error message | Manual check M-5; `ValueError` and `TypeError` from `predict()` are covered by contract tests |
| FR-7 Long reviews truncated and user informed | *Planned*: not implemented yet (see section 4) |

## 3. Non-functional requirements

| Requirement | Verification |
|---|---|
| NFR-1 Prediction takes at most 2 seconds on CPU | *Planned*: measured when the real model exists. The mock is not representative. |
| NFR-2 Model loads once at startup, within 60 seconds | *Planned*: with the real model. |
| NFR-3 Turkish characters, emojis, UTF-8 | `test_turkish_characters_and_emoji_do_not_fail`, manual check M-6 |
| NFR-4 No storage of user input; no personal data in training data | Masking tests (`tests/test_masker.py`), code review (no logging or saving in `app.py` and `inference.py`) |
| NFR-5 Reproducibility | `uv sync --locked` in CI, seeded sampling and baseline tests |
| NFR-6 Windows and Linux, Python 3.12 | CI runs on Ubuntu; Windows is verified by team members locally |
| NFR-7 Contract covered by tests in CI | Section 2.1, job `test` in CI |

## 4. Known gaps

- No automated UI test. `analyze()` in `app.py` could be tested directly later (it needs no browser), but that is not planned yet.
- FR-7 (truncate input longer than 128 tokens and inform the user) is not implemented. The mock `predict()` does not truncate, and the contract does not yet define how the UI learns that truncation happened. A test is added together with the implementation.
- NFR-1 and NFR-2 cannot be checked until the real model exists.
- Contract rule 6 (model loaded on first call and reused) is not testable with the mock, which has no model.
- The CI runs on Linux only.

## 5. Manual UI checklist

Start the app with `uv run python app.py` and check each item. Note the date, the tester and the result when the checklist is run.

| ID | Steps | Expected result |
|---|---|---|
| M-1 | Type a review, click **Analiz et**; repeat by pressing Enter | A result table appears (FR-1, FR-2) |
| M-2 | Analyze a normal review | Table has columns Özellik, Duygu, Güven; each shown row has a label and a percentage (FR-3) |
| M-3 | Open the "Bahsedilmeyen özellikler (yok)" section | Aspects labeled `yok` are listed there and not in the table (FR-4) |
| M-4 | Click each of the example reviews | The text is filled in and a result is shown (FR-5) |
| M-5 | Click **Analiz et** with an empty box, then with only spaces | A warning message appears and no result is shown (FR-6) |
| M-6 | Analyze a review with Turkish characters and an emoji | No error; a result is shown (NFR-3) |
| M-7 | Analyze the same review twice | Identical results (contract rule 4) |
| M-8 | *Planned*: paste a review longer than 128 tokens | A result is shown and the user is told that the text was truncated (FR-7) |

## 6. Human annotation

Training labels come from human annotators. Annotators correct or assign labels for the 7 aspects using the definitions in `annotation/aspects.md` and the rules in `annotation/boundary_rules.md`. The labeling sample is described in [`labeling_sample.md`](labeling_sample.md).

- Labeling tool and number of reviews per team member: *to be decided by the team and recorded here*.
- Disagreements are resolved by the boundary rules; unresolved cases are discussed within the team.

## 7. Out of scope

- Model quality targets and experiments (planned in `experiment_plan.md`). This plan covers the correctness of the evaluation code, not the scores of any model.
- Performance and load testing beyond NFR-1.
- Testing Gradio itself or browser compatibility.
- Security testing. User input is not stored (NFR-4), and the app is not exposed publicly.
- Batch analysis, free-text aspect term extraction and languages other than Turkish (out of scope in `requirements.md` section 1).
