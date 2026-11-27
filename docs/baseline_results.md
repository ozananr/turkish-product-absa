# Baseline results and error analysis (#54)

Reference results for BERTurk. Model: TF-IDF (1–2 gram, sublinear tf) + Logistic Regression
(`class_weight="balanced"`), one classifier per aspect, `random_state=42`
(`src/absa/models/baseline_full.py`). Split `split_v1`: train 1,363 / val 241 / test 200; the test
set is the adjudicated kappa set with independent human labels (`docs/data_split.md`).

## 1. Results

| | val | test |
|---|---:|---:|
| **TF-IDF + LR** | **0.547** | **0.501** |
| always `yok` | 0.241 | 0.233 |

Per aspect (test):

| Aspect | macro-F1 test | macro-F1 val | F1 olumlu | F1 olumsuz | F1 notr | test support (yok/olumlu/olumsuz/notr) |
|---|---:|---:|---:|---:|---:|---|
| `kalite` | 0.541 | 0.490 | 0.59 | 0.46 | 0.29 | 133/24/33/10 |
| `fiyat` | 0.435 | 0.665 | 0.78 | 0.00 | 0.00 | 180/17/1/2 |
| `kargo_paketleme` | 0.465 | 0.479 | 0.67 | 0.25 | 0.00 | 165/22/10/3 |
| `satici_hizmet` | 0.484 | 0.635 | 0.38 | 0.62 | 0.00 | 174/10/14/2 |
| `islev_performans` | 0.515 | 0.507 | 0.63 | 0.61 | 0.00 | 121/29/33/17 |
| `gorunum_tasarim` | 0.499 | 0.509 | 0.17 | 0.38 | - | 178/7/15/0 |
| `beden_uyum` | 0.569 | 0.547 | 0.33 | 0.62 | 0.40 | 171/4/22/3 |

`-`: no test example. Full precision / recall per label and confusion matrices: #54.

**Reading the numbers**

- The baseline more than doubles the always-`yok` reference on val and test.
- Test is 0.046 lower than val. Val labels come from the same source as train (LLM pre-labels
  checked by an annotator); test labels are independent human labels, so part of the gap is a
  label-source shift, not only harder reviews.
- Macro-F1 of `fiyat` (0.435) is pulled down by classes with 1–2 test examples (`olumsuz`,
  `notr`); its `olumlu` F1 (0.78) is the best positive class of all aspects. With 200 test
  reviews, per-class scores of rare classes are very noisy.

## 2. Error analysis (test)

144 of 200 reviews (72%) have at least one wrong aspect; 224 of 1,400 cells are wrong.

**Most errors are about whether an aspect is mentioned, not about the sentiment.**

| Error | cells | share |
|---|---:|---:|
| missed mention (gold mentioned → predicted `yok`) | 99 | 44% |
| extra mention (gold `yok` → predicted mentioned) | 90 | 40% |
| wrong sentiment (both mentioned, different label) | 35 | 16% |

**Error rate by review type** (a review can have several types; detected automatically):

| Review type | reviews | with an error | error rate |
|---|---:|---:|---:|
| all | 200 | 144 | 72.0% |
| short (≤ 5 words) | 35 | 21 | 60.0% |
| contrast (ama / fakat / ancak ...) | 56 | 50 | 89.3% |
| negation word (değil / hiç / asla) | 35 | 30 | 85.7% |
| mixed sentiment (olumlu and olumsuz in one review) | 13 | 13 | 100.0% |
| contains a `notr` label | 32 | 31 | 96.9% |

### 2.1 Negation and contrast

A bag-of-words model sees the positive word and misses that it is negated or outweighed:

- "kesinlikle söylendiği gibi kokmuyor kalıcı değil kokusu ve yetersiz" → `kalite`: olumsuz predicted as **olumlu**
- "Fena değil sadece leke bırakıyor" → `kalite`: notr predicted as **olumlu**
- "Ürün çok güzel ama defolu maalese iade" → `kalite`: olumsuz predicted as **olumlu**
- "Çok güzel fakat 3 tane çıt var boynunda 2 tanesi tutmuyor" → `kalite`: olumsuz predicted as **olumlu**

### 2.2 `notr` is almost never predicted

`notr` F1 is 0 for five aspects; all 17 `islev_performans`/notr test reviews are missed. The class
has very few training examples (e.g. 17 for `islev_performans`, 1 for `satici_hizmet`), because the
LLM pre-labels almost never use `notr` (#52), while the human test labels do (37 cells).

- "Kokusu gayet güzel ve fiyatına göre kalıcılığı normal" → `islev_performans`: notr predicted as olumlu
- "Yani 1 kere kullandım çok wow bir ürün değil fiyatına göre idare eder alınır" → `islev_performans`: notr predicted as yok

### 2.3 General judgments spread over aspects

General statements must stay `yok` for every aspect (`boundary_rules.md`), but the model links
strong sentiment words to aspects:

- "Bir kez kullanıp çöpe attım ." → `kalite`, `fiyat`, `islev_performans` all predicted **olumsuz**
- "Hiç beğenmedim no" → `kalite`, `islev_performans` predicted **olumsuz**
- "Almanızı tavsiye etmiyorum önermiyorum" → `kalite` predicted **olumsuz**

### 2.4 Ambiguous keywords

Single words decide the aspect even when they mean something else:

- "çok rahat günlük kullanıma **uygun**" → `fiyat` predicted olumlu ("uygun" as in "uygun fiyat")
- "Uygun fiyata aldım ürün güzel" → `kargo_paketleme` predicted olumlu
- "Hasarlı ürün gönderildi" → gold `kargo_paketleme`/olumsuz, predicted `satici_hizmet`/olumsuz

### 2.5 Spelling and dialect

Reviews written without Turkish characters or in Azerbaijani Turkish share no tokens with the
training vocabulary:

- "Urun hızlıca sorunsuz bır sekılde elıme ulastı cok ucuz ve kalıtelı ..." → 4 of 4 mentioned aspects missed
- "sağlam gəldi çox gözəl paketlənmiş satıcıya təşəkkür" → `kargo_paketleme` and `satici_hizmet` missed

### 2.6 Size vs. appearance in the gold labels

In several test reviews the model predicts `beden_uyum`/olumsuz for "smaller than it looks",
while the adjudicated label is `gorunum_tasarim`/olumsuz, e.g. "Boy olarak çok küçük ama güzel" and
"göründüğü kadar büyük değil küçük kutusu". The clarified
rule from #51 (PR #60: product size counts as `beden_uyum` for every product) supports the model's
prediction, so these are likely inconsistencies in the gold labels rather than model errors. They
are kept as they are (the test set is frozen) and noted as a limitation.

## 3. What BERTurk should improve, and how it is compared

- **Contrast and negation (2.1)** are the clearest case for a contextual model.
- **Mention detection (84% of errors)** should improve with context; the per-error-type table above
  is reproduced for BERTurk with the same script so the two can be compared directly.
- **`notr`** will stay hard: the training labels contain very few `notr` examples. A class-weighted
  loss is planned; results are reported per class so the effect is visible.
- **Spelling / dialect (2.5)**: subword tokenization should be more robust than word n-grams.
- Same split (`split_v1`), same `evaluate()`, test used once for the final model.
