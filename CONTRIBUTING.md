# Contributing Guide

## Ground rules

1. **Every piece of work starts with an issue.** Use a template, set labels and the weekly milestone, and assign someone.
2. **No direct pushes to `main`.** All changes go through pull requests.
3. **Every PR needs 1 approval** from another member and passing CI.
4. **Squash merge only.** After approval, the **PR author** merges.
5. **Never commit data, model weights or `.env` files.**

## Setup

```bash
git clone https://github.com/ozananr/turkish-product-absa.git
cd turkish-product-absa
git config core.autocrlf false   # Windows only
uv sync
uv run pre-commit install
```

## Branch naming

`<type>/<issue-number>-<short-description>`

| Type | Use for | Example |
|---|---|---|
| `feat` | new code or functionality | `feat/12-review-cleaning` |
| `fix` | bug fixes | `fix/15-test-set-leak` |
| `exp` | experiments | `exp/21-xlmr-large` |
| `docs` | documentation | `docs/3-data-sources` |
| `chore` | setup, tooling, config | `chore/1-project-setup` |

## Commit messages and PR titles

With squash merge, the **PR title** becomes the commit message on `main`, so PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/). CI checks this.

```
feat: add review cleaning step
fix: remove products leaking into the test set
docs: add irony examples to annotation guidelines
chore: update ruff version
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Write `Closes #<issue-number>` in the PR description so the issue closes automatically on merge.

## Typical workflow

```bash
git switch main
git pull
git switch -c feat/12-review-cleaning
# ... work ...
git add .
git commit -m "feat: add cleaning function"
git push -u origin feat/12-review-cleaning
# Open a PR on GitHub and request a review
```

If `main` has moved on while you were working, use the **Update branch** button on the PR.

## Reviews

- Try to respond to review requests within 24 hours.
- Comment on the code, not the person. Prefix optional suggestions with `nit:`.
- All conversations must be resolved before merging.

## Labels

| Group | Labels |
|---|---|
| Type | `type: feature`, `type: bug`, `type: experiment`, `type: docs`, `type: chore`, `type: dependencies` |
| Area | `area: data`, `area: model`, `area: demo` |
| Priority | `priority: high`, `priority: medium`, `priority: low` |
| Status | `status: blocked` |

Every issue needs at least one **type**, one **area** and one **priority** label.

## Weekly captain duties

**Monday**
- Create the week's milestone.
- Open issues under it, label them and assign them.

**Midweek**
- Check for issues labeled `status: blocked` and help unblock them.

**Sunday**
- Copy `docs/weekly/_template.md` to `docs/weekly/week-XX.md`, fill it in and open a PR.
- Move unfinished issues to the next milestone.
- Close the milestone and hand over to the next captain.

## Secrets

API keys (`HF_TOKEN`, `WANDB_API_KEY`, etc.) live only in your local `.env` file or in Colab/Kaggle Secrets. Never paste keys into notebooks. If a key is exposed, revoke it immediately and create a new one.
