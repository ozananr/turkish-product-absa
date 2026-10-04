# Turkish Product ABSA

[![CI](https://github.com/ozananr/turkish-product-absa/actions/workflows/ci.yml/badge.svg)](https://github.com/ozananr/turkish-product-absa/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Aspect-based sentiment analysis (ABSA) for Turkish product reviews: detecting **which product aspects** a review talks about and **how the reviewer feels** about each one.

> **Status:** Week 1 – Setup & Preparation

## Example

> *"Kamerası harika ama bataryası akşamı çıkarmıyor, kargo da geç geldi."*
> ("The camera is great but the battery doesn't last until evening, and shipping was late.")

| Aspect | Sentiment |
|---|---|
| Camera | Positive |
| Battery | Negative |
| Shipping | Negative |

A standard sentiment model gives this review a single label. ABSA separates opinions per aspect.

## Team

| Member | GitHub |
|---|---|
| Ozan Anar | [@ozananr](https://github.com/ozananr) |
| Mehmet Salih Kendirkıran | [@MSalih2756](https://github.com/MSalih2756) |
| Yusuf Açık | [@yusufacik26](https://github.com/yusufacik26) |

The team lead (captain) rotates weekly. See [milestones](https://github.com/ozananr/turkish-product-absa/milestones) for the current week.

## Tech stack

- **Environment:** [uv](https://docs.astral.sh/uv/), Python 3.12
- **Code quality:** [Ruff](https://docs.astral.sh/ruff/), pre-commit, pytest
- **CI:** GitHub Actions
- **Modeling (planned):** Hugging Face Transformers, BERTurk

## Getting started

Requirements: [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/ozananr/turkish-product-absa.git
cd turkish-product-absa
git config core.autocrlf false   # Windows only
uv sync
uv run pre-commit install
```

Run the checks:

```bash
uv run ruff check .
uv run pytest
```

## Project structure

```
├── .github/          # CI workflows, issue and PR templates
├── data/             # local data only, never committed (see data/README.md)
├── docs/             # guides and weekly reports
├── scripts/          # helper scripts
├── src/absa/         # project source code
└── tests/            # tests
```

## Contributing

Issue → branch → pull request → 1 approval + passing CI → squash merge.
See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, commit conventions and captain duties.

## Data policy

Raw data, labeled data, model weights and secrets are **never** committed. This is enforced by `.gitignore`, a pre-commit hook and a CI check.

## License

[MIT](LICENSE)
