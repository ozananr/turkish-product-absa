# Dataset Review

## 1. Trendyol Product Reviews and Ratings Dataset
* **Source/Link:** https://www.kaggle.com/datasets/alpsencerzdemir/trendyol-product-reviews-and-ratings-dataset
* **Size:** 970,073
* **Columns:** product_name, review_rating, review_body
* **Label Distribution:** Not labeled
* **Product Category:** Mixed categories (can be filtered via product_name column).
* **License:** MIT
* **Quality:**
  * Duplicates: Expected to be high.
  * Very short reviews: Present (needs filtering).
  * Template-like reviews: Present (e.g., "hızlı kargo").
  * Class imbalance: N/A (unlabeled).

## 2. Trendyol Urun Yorumlari Duygu Analizi
* **Source/Link:** https://www.kaggle.com/datasets/sedayazici/trendyol-urun-yorumlari-duygu-analizi
* **Size:** 73,392
* **Columns:** Comments, Emotions
* **Label Distribution:** Positive: 59,956 (81.7%), Negative: 11,861 (16.2%), Neutral: 1,363 (1.9%), Other: 212 (0.3%)
* **Product Category:** Not specified explicitly (mixed e-commerce).
* **License:** CC0: Public Domain
* **Quality:**
  * Duplicates: Moderate.
  * Very short reviews: Present.
  * Template-like reviews: Present.
  * Class imbalance: Highly imbalanced (82% positive).

## 3. Turkish Product Reviews Sentiment
* **Source/Link:** https://huggingface.co/datasets/anilguven/turkish_product_reviews_sentiment
* **Size:** 243,000
* **Columns:** Label, Review
* **Label Distribution:** LABEL_0 (Negative), LABEL_1 (Positive). Exact counts/percentages not specified on source.
* **Product Category:** Not specified explicitly.
* **License:** Not specified.
* **Quality:**
  * Duplicates: Unknown.
  * Very short reviews: Present.
  * Template-like reviews: Present.
  * Class imbalance: Unknown.

## 4. SentiTurca
* **Source/Link:** https://huggingface.co/datasets/turkish-nlp-suite/SentiTurca
* **Size:** 234,345
* **Columns:** Text, Label
* **Label Distribution:** 1 Star to 5 Star. Exact counts/percentages not specified on source.
* **Product Category:** Mixed categories.
* **License:** CC-BY-SA-4.0
* **Quality:**
  * Duplicates: Unknown.
  * Very short reviews: Expected.
  * Template-like reviews: Expected.
  * Class imbalance: Unknown.

## 5. Turkish_SentimentAnalysis_TRSAv1
* **Source/Link:** https://huggingface.co/datasets/maydogan/Turkish_SentimentAnalysis_TRSAv1
* **Size:** 150,000
* **Columns:** Id, Score, Review
* **Label Distribution:** Positive: 49,950 (33.3%), Negative: 49,950 (33.3%), Neutral: 49,950 (33.3%)
* **Product Category:** Mixed categories.
* **License:** CC-BY-SA-4.0
* **Quality:**
  * Duplicates: Moderate.
  * Very short reviews: Present.
  * Template-like reviews: Present.
  * Class imbalance: Perfectly balanced.

## 6. E-Ticaret Ürün Yorumları (Etiketsiz)
* **Source/Link:** https://www.kaggle.com/datasets/mujdatcabuk/e-ticaret-rn-yorumlar-etiketsiz-200k
* **Size:** 178,972
* **Columns:** Metin
* **Label Distribution:** Not labeled
* **Product Category:** Mixed e-commerce.
* **License:** CC0: Public Domain
* **Quality:**
  * Duplicates: High.
  * Very short reviews: Present.
  * Template-like reviews: Present.
  * Class imbalance: N/A (unlabeled).

## Team Recommendation
* **Recommended Strategy:** Combine datasets to reach the 10,000+ review target and fix class imbalances. Existing labels are for the whole review, not aspect-based, so we cannot train on them directly. We will label aspects ourselves. The existing labels will only be used for smart sampling (prioritizing negative and neutral reviews).
* **Sampling Plan (Reaching 10k+):**
  * **Dataset 5 (TRSAv1):** Extract ~15,000 negative and neutral reviews (since it is perfectly balanced).
  * **Dataset 2:** Extract the available 11,861 negative and 1,363 neutral reviews.
  * **Dataset 1:** Use `product_name` to sample ~10,000 reviews specifically targeting missing categories (e.g., clothing for "size/fit" aspects, electronics).
* **Yield Expectation:** After aggressive filtering (removing duplicates, very short reviews under 3-4 words, and template reviews like "hızlı kargo", "güzel ürün"), we estimate a 25-30% retention rate. Starting with a combined raw pool of ~38,000 filtered samples will comfortably leave us with a high-quality, balanced set of 10,000+ reviews ready for aspect manual annotation.
