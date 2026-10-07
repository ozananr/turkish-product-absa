# Week 1 Report

- **Captain:** @ozananr
- **Dates:** 01.10.2026 – 08.10.2026
- **Milestone:** Week 1 – Setup & Preparation

## Goals and status

Goal of the week: set up the repository and team workflow, and prepare the groundwork for data collection and annotation.

| Issue | Assignee | PR | Status |
|---|---|---|---|
| #1 Project setup: repository structure, conventions and CI | @ozananr | #2 | Done |
| #7 Aspect definitions | @MSalih2756 | #14 | Done |
| #8 Boundary rules | @yusufacik26, @ozananr | #16 | Done |
| #9 Dataset review | @yusufacik26 | #12 | Done |
| #10 Pretrained model trial and experiment plan | @MSalih2756 | #15 | Done |
| #11 Requirements analysis and predict() contract | @ozananr | #13 | Done |
| #17 Add code of conduct | @ozananr | #18 | Done |
| #19 Week 1 report | @ozananr | #20 | Done |

Infrastructure follow-ups merged during the week: Dependabot updates for GitHub Actions (#3, #4, #5) and team usernames in the README (#6).

## Decisions made

**Tooling**
- Python 3.12, pinned with `.python-version` to match Google Colab, where models will be trained.
- uv for environment and dependency management; `uv.lock` ensures every member has an identical environment.
- Ruff for linting and formatting, pytest for tests, pre-commit for local checks.
- GitHub Actions runs four required checks on every PR: `lint`, `data-guard`, `test`, `pr-title`.

**Workflow**
- Every piece of work starts with an issue; every change reaches `main` through a PR with 1 approval and passing checks.
- Squash merge only, so each PR becomes one clear commit on `main`. The PR author merges after approval.
- `main` is protected by a ruleset with no bypass: no direct pushes, no force pushes, branches must be up to date.
- Branch names follow `<type>/<issue-no>-<description>`; PR titles follow Conventional Commits.
- Issues, PRs, commits, branch names and documents are written in English; issue closing comments are written in Turkish.
- One milestone per week; tasks are due Tuesday 23:59. The captain writes a weekly report.
- Added an `area: project` label for project management tasks such as weekly reports.

**Data**
- Data files, model weights and secrets are never committed. This is enforced by `.gitignore`, a pre-commit hook and a CI check.
- We will use existing Turkish review datasets instead of scraping.
- Existing labels are for whole reviews, not per aspect, so they are used only for sampling (prioritizing negative and neutral reviews), not for training.
- Planned raw pool: about 38,000 reviews from TRSAv1 and two Trendyol datasets, expected to yield about 10,000 usable reviews after filtering and deduplication.

**Annotation**
- 7 aspects with fixed code keys: `kalite`, `fiyat`, `kargo_paketleme`, `satici_hizmet`, `islev_performans`, `gorunum_tasarim`, `beden_uyum`.
- 4 labels per aspect: `yok`, `olumlu`, `olumsuz`, `notr`.
- Broken or stopped working → quality; works but works poorly → function.
- Shipping fees belong to shipping/packaging, not price.
- `yok` means the aspect is not mentioned; `notr` means it is mentioned without a positive or negative judgment.

**System**
- `predict(text: str)` always returns all 7 aspects in a fixed order, each with a label and a confidence score; empty input raises `ValueError`.
- Until the real model is ready, a deterministic mock `predict()` will be used so model and interface work can proceed in parallel.

## Problems and risks

**Problems encountered**
- A member cloned the repository to the Desktop (OneDrive and a non-ASCII path), which broke `import absa` on Windows. Fixed by re-cloning to `C:\projects`. A `PYTHONPATH` workaround was avoided because it hid the real cause.
- The `pr-title` check rejected a PR title starting with `exp:`. `exp` is a valid branch prefix in our rules but not a Conventional Commits type. The title was changed to `docs:`.
- Some PRs had incomplete descriptions or a wrong issue reference (`Closes 9#` instead of `Closes #9`), so the issue was not linked. One branch did not follow the naming rule.
- Two Dependabot PRs conflicted on the same workflow file; resolved with `@dependabot rebase`.
- Branches fell behind `main` after other merges. "Update branch" fixes this, but it adds a commit and dismisses earlier approvals.
- Ruff also formats Python code blocks inside Markdown files, which made a commit fail once; re-adding the formatted file fixed it.

**Risks**
- Dataset quality notes (duplicates, template-like reviews) are estimates and have not been measured yet.
- The 10,000-review target is met with little margin after filtering.
- Neutral reviews are rare in the existing datasets, which may make the `notr` label hard to learn.
- Licenses of some datasets are not specified.
- Annotation in Weeks 3–4 is the largest workload of the project; delays there affect every later week.
- The aspect definitions and boundary rules must stay consistent (e.g. "missing items in the package" must be labeled the same way in both documents), otherwise annotator agreement will drop.

## Lessons learned

- Clear acceptance criteria in issues made reviews faster and more objective; most review comments pointed back to them.
- Small, focused PRs were reviewed and merged quickly.
- Documents that depend on each other (aspect definitions, boundary rules, predict() contract) should be cross-checked before merging.
- Repositories should be cloned to simple paths without OneDrive or Turkish characters.
- The PR template and the "Closes #" line need extra attention; reviewers should check the Development panel on every PR.

## Notes for the next captain

**Open items to carry over**
- Allow `exp` as a PR title type in `.github/workflows/pr-title.yml` and update `CONTRIBUTING.md`.
- Add the issue closing comment rule and the `area: project` label to `CONTRIBUTING.md`.
- Add `.env.example` (listed in #1, not yet created).
- Confirm that "missing items in the package" is labeled as shipping/packaging in both `aspects.md` and `boundary_rules.md`.
- Move any Week 1 issues that are still open to the Week 2 milestone.

**Suggested Week 2 work (data collection and EDA)**
- Download the selected datasets, combine them and deduplicate across datasets.
- Measure the real quality numbers (duplicates, short and template-like reviews, class balance) and update `docs/data_sources.md`.
- Implement cleaning and personal data masking.
- Create `src/absa/schema.py` and the mock `predict()` with a contract test, based on `docs/requirements.md`.
- Start the Gradio interface skeleton using the mock `predict()`.
