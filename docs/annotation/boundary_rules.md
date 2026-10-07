# Boundary Rules

Rules for labeling cases where aspects or sentiments overlap. These rules complement the aspect definitions in `docs/annotation/aspects.md`. When a case is not covered here, annotators add a note and the team decides together.

## 1. Aspect overlaps

Sometimes a review can refer to multiple aspects or it is unclear which aspect is being discussed. Below are 10 boundary cases with the agreed-upon decisions to ensure annotator consistency.

| Review (Ambiguous Case) | Conflicting Aspects | Decision & Reason |
| :--- | :--- | :--- |
| "Bozuk geldi" | `kargo_paketleme` vs `kalite` | If the outer box is crushed/damaged → `kargo_paketleme` (olumsuz). If the box is fine but the item doesn't work → `kalite` (olumsuz). |
| "İki günde bozuldu" | `kalite` vs `islev` | `kalite` (olumsuz). Durability and quick breakdown fall under overall product quality. |
| "Pakette eksik parça vardı" | `kargo_paketleme` vs `satici` | `satici` (olumsuz). Missing items inside an intact package is a preparation error by the seller, not a shipping issue. |
| "Yanlış beden/renk göndermişler" | `beden_uyum` vs `satici` | `satici` (olumsuz). The product itself isn't necessarily the wrong fit; the seller shipped the wrong item entirely. |
| "Fiyatına göre çok iyi" | `fiyat` vs `kalite` | `fiyat` (olumlu). The primary sentiment is about value for money. `kalite` remains `yok` or `notr` unless explicitly praised. |
| "Resimdekiyle alakası yok, rengi farklı" | `kalite` vs `gorsel_uyum` | `gorsel_uyum` (olumsuz). The issue is specifically about a mismatch with the advertised image/design, not necessarily poor material quality. |
| "Kurulumu çok zor, kılavuz yetersiz" | `kalite` vs `kurulum` | `kurulum` (olumsuz). The struggle is explicitly with the assembly and setup process. |
| "İade ettim paramı haftalarca yatırmadılar" | `satici` vs `fiyat` | `satici` (olumsuz). Refund delays are a customer service/seller behavior issue, independent of the product's price. |
| "Çalışırken çok fazla ses çıkarıyor" | `kalite` vs `islev` | `islev` (olumsuz). If the product works as intended but is annoyingly loud, it is a functional trait rather than a broken/quality defect. |
| "Hediye paketi istemiştim yapılmamış" | `kargo_paketleme` vs `satici` | `satici` (olumsuz). Failure to fulfill a special request is a seller omission, not standard shipping damage. |

## 2. Label rules

Each of the 7 aspects gets exactly one label per review: `yok`, `olumlu`, `olumsuz` or `notr`.

### 2.1 `yok` vs `notr`

This is the most important distinction.

- **`yok`**: the aspect is not mentioned at all.
- **`notr`**: the aspect is mentioned, but without a positive or negative judgment.

| Review | Aspect | Label | Reason |
| --- | --- | --- | --- |
| "Kargo 3 günde geldi." | `kargo_paketleme` | `notr` | Mentioned as a fact, no judgment |
| "Fiyatı 500 TL." | `fiyat` | `notr` | Mentioned as a fact, no judgment |
| "Ürün çok kaliteli." | `fiyat` | `yok` | Price is not mentioned |

**General statements** that cannot be tied to a specific aspect ("Güzel ürün", "Berbat", "Herkese tavsiye ederim") are labeled `yok` for all aspects. We only label sentiment that clearly refers to one of the 7 aspects.

### 2.2 Mixed sentiment within the same aspect

When a review contains both positive and negative opinions about the **same** aspect:

1. Label the side the reviewer emphasizes. In Turkish, the part after "ama", "fakat" or "ancak" usually carries the final opinion.
2. If both sides are truly balanced, label `notr`.

| Review | Aspect | Label |
| --- | --- | --- |
| "Kargo hızlıydı ama kutu ezik geldi." | `kargo_paketleme` | `olumsuz` |
| "Kutu biraz ezikti ama ürün sağlam ve kargo çok hızlıydı." | `kargo_paketleme` | `olumlu` |
| "Kargo hızlıydı, paketleme ise özensizdi." | `kargo_paketleme` | `notr` |

Different aspects in the same review are labeled independently ("Ürün kaliteli ama kargo geç geldi" → `kalite`: `olumlu`, `kargo_paketleme`: `olumsuz`).

### 2.3 Negation

Label the meaning, not the words.

| Review | Label |
| --- | --- |
| "Kalitesi kötü değil." | `olumlu` |
| "Fena değil." (about a specific aspect) | `olumlu` |
| "Pek iyi değil." | `olumsuz` |
| "Beklediğim gibi değil." | `olumsuz` |

### 2.4 Irony and sarcasm

Label the intended meaning.

| Review | Aspect | Label |
| --- | --- | --- |
| "3 haftada geldi, harika hız 👏" | `kargo_paketleme` | `olumsuz` |
| "Bu fiyata bu kalite, gerçekten çok teşekkürler satıcıya 🙃" | `fiyat`, `kalite` | `olumsuz` |

If the intended meaning is unclear, the annotator adds a note and the case is discussed by the team.

### 2.5 Other rules

- **Text only:** labels are based only on the review text, never on the star rating. Ratings and text sometimes contradict each other.
- **Emojis** count as part of the text and can change the meaning (e.g. 👏 or 🙃 in ironic reviews).
- **Comparisons** ("Önceki telefonumdan daha hızlı") are labeled by their direction: better → `olumlu`, worse → `olumsuz`.
- **Expectations and advice** are labeled by what they imply about the product: "Bir beden büyük alın" means the size runs small → `beden_uyum`: `olumsuz`.
