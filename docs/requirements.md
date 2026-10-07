# Requirements and `predict()` Contract

This document defines what the system does and the interface between the model and the web application. Model and interface work proceed in parallel from Week 2 based on this contract.

## 1. Scope

The system takes a single Turkish e-commerce review and predicts the sentiment for each of 7 aspects. Results are shown in a single-page web interface.

**In scope**

- Aspect-level sentiment for 7 fixed aspects
- 4 labels per aspect: not mentioned, positive, negative, neutral
- A single-page web interface (Gradio)

**Out of scope (for now)**

- Extracting aspect terms as free text (sequence labeling)
- Languages other than Turkish
- Storing user input or analysis history
- Batch analysis of many reviews at once

## 2. Aspects and labels

These names are fixed for code and must match `docs/annotation/aspects.md`.

| Key (code) | Display name (UI) | Description |
| --- | --- | --- |
| `kalite` | Kalite / Dayanıklılık | Quality and durability |
| `fiyat` | Fiyat | Price |
| `kargo_paketleme` | Kargo / Paketleme | Shipping and packaging |
| `satici_hizmet` | Satıcı / Hizmet | Seller and service |
| `islev_performans` | İşlev / Performans | Function and performance |
| `gorunum_tasarim` | Görünüm / Tasarım | Appearance and design |
| `beden_uyum` | Beden / Uyum | Size and fit |

| Key (code) | Display name (UI) | Meaning |
| --- | --- | --- |
| `yok` | Yok | The aspect is not mentioned |
| `olumlu` | Olumlu | Positive sentiment |
| `olumsuz` | Olumsuz | Negative sentiment |
| `notr` | Nötr | Mentioned, but neither positive nor negative |

Keys use ASCII characters only (no `ç`, `ş`, `ö`, `ü`, `ı`, `ğ`) to avoid encoding issues in code, file names and JSON.

## 3. Functional requirements

| ID | Requirement |
| --- | --- |
| FR-1 | The user can enter a review as free text. |
| FR-2 | The user starts the analysis with an "Analyze" button. |
| FR-3 | The system shows a label and a confidence score for each of the 7 aspects. |
| FR-4 | Aspects labeled `yok` are shown separately or visually de-emphasized, so mentioned aspects stand out. |
| FR-5 | The interface offers 3–4 example reviews that the user can run with one click. |
| FR-6 | Empty or whitespace-only input shows an error message instead of a result. |
| FR-7 | Reviews longer than the model limit are truncated, and the user is informed. |

## 4. Non-functional requirements

| ID | Category | Requirement |
| --- | --- | --- |
| NFR-1 | Performance | After the model is loaded, a single prediction takes at most 2 seconds on CPU. |
| NFR-2 | Performance | The model is loaded once at startup; loading takes at most 60 seconds. |
| NFR-3 | Language | Turkish characters (ç, ğ, ı, ö, ş, ü), emojis and punctuation are handled without errors. All text is UTF-8. |
| NFR-4 | Privacy | User input is not stored or logged. The training data contains no personal data (names, phone numbers and e-mail addresses are masked). |
| NFR-5 | Reproducibility | Dependencies are pinned in `uv.lock`; data splits and training use a fixed random seed. |
| NFR-6 | Portability | The application runs on Windows and Linux with Python 3.12. |
| NFR-7 | Maintainability | The `predict()` contract is covered by automated tests that run in CI. |

## 5. `predict()` contract

### Signature

```python
# src/absa/inference.py
def predict(text: str) -> dict[str, dict[str, str | float]]: ...
```

Aspect and label names are defined once in `src/absa/schema.py` and imported everywhere else:

```python
ASPECTS = [
    "kalite",
    "fiyat",
    "kargo_paketleme",
    "satici_hizmet",
    "islev_performans",
    "gorunum_tasarim",
    "beden_uyum",
]
LABELS = ["yok", "olumlu", "olumsuz", "notr"]
```

### Output

```json
{
  "kalite":           {"label": "olumlu",  "confidence": 0.91},
  "fiyat":            {"label": "yok",     "confidence": 0.97},
  "kargo_paketleme":  {"label": "olumsuz", "confidence": 0.88},
  "satici_hizmet":    {"label": "yok",     "confidence": 0.95},
  "islev_performans": {"label": "yok",     "confidence": 0.93},
  "gorunum_tasarim":  {"label": "yok",     "confidence": 0.96},
  "beden_uyum":       {"label": "yok",     "confidence": 0.99}
}
```

Example input for the output above: *"Ürün çok kaliteli ama kargo çok geç geldi."*

### Rules

1. The output always contains all 7 aspects, in the order of `ASPECTS`. `yok` is a valid result, not a missing value.
2. `label` is always one of `LABELS`.
3. `confidence` is the model's probability for the chosen label, a float between 0 and 1.
4. The same input always returns the same output (no randomness at inference time).
5. Input longer than 128 tokens is truncated.
6. The model is loaded on the first call and reused for later calls.

### Errors

| Input | Behavior |
| --- | --- |
| Not a `str` | Raises `TypeError` |
| Empty or whitespace-only | Raises `ValueError` |

The web interface catches these errors and shows a user-friendly message.

### Mock implementation

Until the real model is ready (Week 6), `predict()` returns random results that follow this contract exactly. Until the real model is ready (Week 6), `predict()` returns placeholder results that follow this contract exactly. The placeholder is deterministic: results are derived from the input text (e.g. by seeding a random generator with a hash of the text), so the same input always returns the same output and rule 4 holds for the mock as well. The interface is built and tested against the mock, so replacing it with the real model requires no interface changes.

## 6. Verification

In Week 2, a contract test is added to `tests/` that checks:

- all 7 aspects are present, in the correct order
- every `label` is in `LABELS`
- every `confidence` is between 0 and 1
- empty input raises `ValueError`

The same test runs against both the mock and the real model.

## 7. Open questions

- Aspect names may change after `docs/annotation/aspects.md` is finalized; this document will be updated in the same week.
- The 2-second limit (NFR-1) will be checked once the model is trained; if it is not met, a smaller model or a shorter input limit will be considered.
