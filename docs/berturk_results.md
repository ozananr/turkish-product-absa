# BERTurk results and comparison with the baseline (#66)

The BERTurk model chosen on val in #65 (lr 5e-5, no class weights, seed 43, Drive
`absa-data/models/berturk_v1/`) was evaluated **once** on the test set and compared with the
TF-IDF + LR baseline (#54) under the same conditions: split `split_v1` (train 1,363 / val 241 /
test 200; test = adjudicated kappa set with independent human labels), the shared `evaluate()` and
the same error analysis (`src/absa/models/compare_models.py`).

## 1. Overall

| Model | val macro-F1 | test macro-F1 |
|---|---:|---:|
| always `yok` | 0.241 | 0.233 |
| TF-IDF + LR (#54) | 0.547 | 0.501 |
| **BERTurk** | **0.704** | **0.625** |

BERTurk improves test macro-F1 by **+0.124** (+25% relative). On val the selected setting reached
0.690 ± 0.011 over three seeds (#65), so the chosen model is not an outlier.

The val–test gap is larger for BERTurk (0.079) than for the baseline (0.046). Val labels come from
the same source as train (LLM pre-labels checked by an annotator, #52), test labels are
independent human labels. A stronger model fits the training label source better, so part of its
val score does not transfer to human labels.

## 2. Per aspect (test)

| Aspect | TF-IDF | BERTurk | Δ | F1 olumlu T / B | F1 olumsuz T / B | F1 notr T / B |
|---|---:|---:|---:|---|---|---|
| `kalite` | 0.541 | 0.608 | +0.067 | 0.59 / 0.84 | 0.46 / 0.69 | 0.29 / 0.00 |
| `fiyat` | 0.435 | 0.590 | +0.155 | 0.78 / 0.97 | 0.00 / 0.40 | 0.00 / 0.00 |
| `kargo_paketleme` | 0.465 | 0.623 | +0.158 | 0.67 / 0.91 | 0.25 / 0.61 | 0.00 / 0.00 |
| `satici_hizmet` | 0.484 | 0.605 | +0.121 | 0.38 / 0.89 | 0.62 / 0.57 | 0.00 / 0.00 |
| `islev_performans` | 0.515 | 0.604 | +0.089 | 0.63 / 0.81 | 0.61 / 0.69 | 0.00 / 0.00 |
| `gorunum_tasarim` | 0.499 | 0.747 | +0.248 | 0.17 / 0.62 | 0.38 / 0.64 | - / - |
| `beden_uyum` | 0.569 | 0.595 | +0.026 | 0.33 / 0.80 | 0.62 / 0.64 | 0.40 / 0.00 |

- BERTurk is better on **every aspect**. The largest gains are `gorunum_tasarim` (+0.248),
  `kargo_paketleme` (+0.158) and `fiyat` (+0.155).
- `olumlu` F1 rises to 0.80–0.97 on six aspects; `olumsuz` improves on six aspects (only `satici_hizmet` drops slightly, 0.62 → 0.57).
- Per-aspect macro-F1 is held down by `notr` (F1 0 everywhere, see 3.3). For the other three labels
  BERTurk's F1 is mostly between 0.57 and 0.99 (lowest: `fiyat`/olumsuz 0.40, a single test example).

## 3. Error analysis (test)

### 3.1 Fewer errors, mostly in mention detection

| | TF-IDF | BERTurk |
|---|---:|---:|
| wrong cells (of 1,400) | 224 | **138** (−38%) |
| missed mention (gold mentioned → `yok`) | 99 | 49 |
| extra mention (gold `yok` → mentioned) | 90 | 56 |
| wrong sentiment | 35 | 33 |
| reviews with at least one error | 72.0% | **45.5%** |

BERTurk fixed 136 of the baseline's wrong cells and made 50 new errors; 88 cells are wrong in both.
The gain comes almost entirely from **deciding whether an aspect is mentioned** (missed mentions
halved). Wrong sentiment on correctly detected aspects stays at the same level (35 → 33).

### 3.2 Contrast, negation and short reviews

| Review type | reviews | TF-IDF error rate | BERTurk error rate |
|---|---:|---:|---:|
| all | 200 | 72.0% | 45.5% |
| short (≤ 5 words) | 35 | 60.0% | 25.7% |
| contrast (ama / fakat / ancak ...) | 56 | 89.3% | 62.5% |
| negation word (değil / hiç / asla) | 35 | 85.7% | 62.9% |
| mixed sentiment (olumlu and olumsuz) | 13 | 100.0% | 84.6% |
| contains a `notr` label | 32 | 96.9% | 100.0% |

The contextual model clearly helps where the bag-of-words baseline failed (#54, section 2.1):
contrast and negation error rates drop by about 23–27 points and short reviews by 34 points.
Reviews with mixed sentiment remain hard (85%).

### 3.3 `notr` is never predicted

BERTurk predicts `notr` for **no** test review (the `notr` column of every confusion matrix is 0),
so `notr` F1 is 0 for all six aspects that have `notr` examples; the baseline had 0.29 (`kalite`)
and 0.40 (`beden_uyum`). Gold `notr` reviews are labeled with a sentiment instead, most often
`islev_performans`: notr → olumsuz (9 of 17).

Causes:
- The training labels contain very few `notr` examples (1–18 per aspect), because the LLM
  pre-labels almost never use `notr` and annotators rarely changed them (#52). The test labels
  (independent humans) contain 37 `notr` cells.
- Val has only 16 `notr` cells, so the class-weighted setting, which targets rare classes, could
  not show its effect on val (0.677 vs 0.680 in #65) and was not selected. Choosing it now on the
  basis of the test set would not be allowed.

### 3.4 Remaining frequent errors

- `kalite`: yok → olumsuz (13) – the same pattern as in the baseline, where general negative
  judgments were linked to quality although the rules say they stay `yok` (#54, 2.3).
- `beden_uyum`: yok → olumsuz (13) – the model assigns size statements to `beden_uyum`. Some test
  reviews about size ("smaller than it looks") have the adjudicated label `gorunum_tasarim`,
  although the clarified rule (#60) says `beden_uyum` (`docs/baseline_results.md`, 2.6), which
  likely contributes to these errors and to the small gain on `beden_uyum` (+0.026).
- `islev_performans` / `satici_hizmet`: olumsuz → yok (7 and 6) – complaints that are expressed
  indirectly are missed.

## 4. Conclusion and limitations

- BERTurk is the model used in the application (#67): test macro-F1 0.625 vs 0.501, 38% fewer wrong
  cells, and large gains on contrast, negation and short reviews.
- `notr` is the main weakness. It needs more `notr` training labels from independent human
  annotation; a class-weighted loss can be revisited only with a validation set that contains
  enough `notr` examples.
- The test set is small (200 reviews); scores of rare classes (1–3 test examples, e.g.
  `fiyat`/olumsuz) are very noisy.
- The test set was used once for the final model; no setting was chosen on it.

Reproduce:

```bash
uv run python -m absa.models.compare_models \
    --labels data/interim/labeling/final_labels.csv \
    --split data/interim/splits/split_v1.csv \
    --model models/berturk_v1
```
