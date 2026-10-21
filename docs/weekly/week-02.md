# Week 2 Report

* **Captain:** @MSalih2756
* **Dates:** 08.10.2026 – 15.10.2026
* **Milestone:** Week 2 - Data Collection & Mock Demo

## Goals and status

Goal of the week: Combine, clean, and mask review datasets, establish evaluation metrics with a baseline model skeleton, and build a mock Gradio demo interface.

| Issue | Assignee | PR | Status |
| :--- | :--- | :--- | :--- |
| #21 Combine, clean and mask review datasets | @MSalih2756 | #25 | Done |
| #22 Evaluation metrics and baseline model skeleton | @ozananr | #26 | Done |
| #23 Schema, mock predict and Gradio demo interface | @yusufacik26 | #29 | Done |

## Decisions made

**Data & Tooling**
* Data cleaning and masking code was placed directly under `src/absa/data/` instead of keeping it only in notebooks.
* Added required packages using `uv add` and committed both `pyproject.toml` and `uv.lock`.

**Model & Evaluation**
* Added scikit-learn and placed Transformers in a separate group (`--group ml`) to prevent PyTorch from installing in CI pipelines.
* Built a TF-IDF + Logistic Regression baseline skeleton with separate classifiers for the 7 aspects.

**Interface**
* Implemented a deterministic mock `predict()` function so UI development can proceed in parallel until the real model is ready.
* Tests were designed to only check `schema.py` and `predict()` without relying on Gradio dependencies.

## Problems and risks

**Problems encountered**
* *Nothing*

**Risks**
* Reaching the target of 10,000 clean reviews might be challenging after filtering and deduplication.
* The 128 token limit might not be sufficient based on the token length analysis.

## Lessons learned

* Strict `.gitignore` rules and CI checks successfully prevented raw/interim data files from being committed to the repository. Data sharing via the external folder worked well.

## Notes for the next captain

**Open items to carry over**
* Add `.env.example` with empty placeholders for API keys (`HF_TOKEN`, `OPENAI_API_KEY`).

**Suggested Week 3 work (Labeling & System Design)**
* Select a balanced labeling sample (2,000–2,500 reviews) with oversampled negative/neutral reviews and rare aspects.
* Set up the LLM pre-labeling pipeline to assist annotators and validate outputs against the schema.
* Prepare the labeling tool, split batches, and create a 200-review kappa set for inter-annotator agreement.
* Run the baseline model end-to-end on the first 300–500 verified labels to check the pipeline flow.
* Draft UML diagrams (use case, component, sequence) and a test plan to document system design.
