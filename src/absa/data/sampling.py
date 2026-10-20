"""Etiketleme için dengeli örnek seçimi.

Kullanım:
    uv run python -m absa.data.sampling

Temiz havuzdan (data/interim/reviews_clean.csv) sabit seed ile örnek seçer ve
data/interim/labeling_sample.csv dosyasına yazar. Aynı seed ve aynı havuz aynı örneği verir.

Strateji:
1. Hücre kotaları: kaynak (trendyol/trsa) x etiket (negatif/nötr/pozitif).
   Negatif ve nötr yorumlar havuzdaki oranlarından fazla örneklenir.
2. Her hücrede önce nadir aspect kotası, sonra kalan kota rastgele doldurulur.
3. Her hücrede bilinen bir kategori en fazla `max_category_share` payı alır
   (kategori yalnızca ürün adı olan Trendyol yorumlarında bilinir).
4. Aynı üründen en fazla `max_per_product` yorum seçilir.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from absa.data.categories import UNKNOWN, assign_category
from absa.data.keywords import RARE_ASPECTS, add_hit_columns, hit_column

DEFAULT_N = 2500
DEFAULT_SEED = 42
LABEL_SHARE = {"negative": 0.40, "neutral": 0.30, "positive": 0.30}
SOURCE_SHARE = {"trendyol": 0.40, "trsa": 0.60}
ASPECT_SHARE = 0.15  # her hücrede her nadir aspect için hedef isabet payı
MAX_CATEGORY_SHARE = 0.30
MAX_PER_PRODUCT = 10

OUTPUT_COLUMNS = [
    "review_id",
    "text",
    "source",
    "label",
    "rating",
    "product_name",
    "category",
    *[hit_column(a) for a in RARE_ASPECTS],
]


def _split(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Toplamı paylara böler (en büyük kalan yöntemi), toplam korunur."""
    raw = {k: total * v for k, v in shares.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    leftover = total - sum(out.values())
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:leftover]:
        out[k] += 1
    return out


def prepare_pool(pool: pd.DataFrame) -> pd.DataFrame:
    """Kategori ve aspect isabet kolonlarını ekler. review_id'ye göre sıralar (determinizm)."""
    df = pool.sort_values("review_id").reset_index(drop=True)
    df["category"] = [
        assign_category(p, s) for p, s in zip(df["product_name"], df["source"], strict=True)
    ]
    return add_hit_columns(df)


def _draw_cell(
    cand: pd.DataFrame,
    quota: int,
    rng: np.random.Generator,
    product_counts: Counter,
    aspect_share: float,
    max_category_share: float,
    max_per_product: int,
) -> list[int]:
    """Tek hücreden (kaynak x etiket) `quota` satır seçer. Satır konumlarını döner."""
    order = rng.permutation(len(cand))
    categories = cand["category"].to_numpy()
    products = cand["product_name"].to_numpy()
    hits = {a: cand[hit_column(a)].to_numpy() for a in RARE_ASPECTS}
    cat_cap = max(1, math.floor(quota * max_category_share))

    chosen: list[int] = []
    chosen_set: set[int] = set()
    cat_counts: Counter = Counter()

    def try_add(pos: int) -> bool:
        if pos in chosen_set:
            return False
        cat, prod = categories[pos], products[pos]
        if cat != UNKNOWN and cat_counts[cat] >= cat_cap:
            return False
        has_product = isinstance(prod, str)
        if has_product and product_counts[prod] >= max_per_product:
            return False
        chosen.append(pos)
        chosen_set.add(pos)
        cat_counts[cat] += 1
        if has_product:
            product_counts[prod] += 1
        return True

    need = math.ceil(quota * aspect_share)
    for aspect in RARE_ASPECTS:
        have = sum(hits[aspect][p] for p in chosen)
        for pos in order:
            if have >= need or len(chosen) >= quota:
                break
            if hits[aspect][pos] and try_add(pos):
                have += 1
    for pos in order:
        if len(chosen) >= quota:
            break
        try_add(pos)
    return chosen


def draw_sample(
    pool: pd.DataFrame,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    label_share: dict[str, float] | None = None,
    source_share: dict[str, float] | None = None,
    aspect_share: float = ASPECT_SHARE,
    max_category_share: float = MAX_CATEGORY_SHARE,
    max_per_product: int = MAX_PER_PRODUCT,
) -> pd.DataFrame:
    """Temiz havuzdan dengeli etiketleme örneği seçer."""
    label_share = label_share or LABEL_SHARE
    source_share = source_share or SOURCE_SHARE
    df = prepare_pool(pool)
    rng = np.random.default_rng(seed)
    product_counts: Counter = Counter()

    parts = []
    for source, n_source in _split(n, source_share).items():
        for label, quota in _split(n_source, label_share).items():
            cand = df[(df["source"] == source) & (df["label"] == label)]
            pos = _draw_cell(
                cand, quota, rng, product_counts, aspect_share, max_category_share, max_per_product
            )
            parts.append(cand.iloc[pos])
    sample = pd.concat(parts).sort_values("review_id").reset_index(drop=True)
    return sample[OUTPUT_COLUMNS]


def compute_stats(
    sample: pd.DataFrame, target_n: int = DEFAULT_N, seed: int = DEFAULT_SEED
) -> dict:
    """Örnek istatistikleri."""
    known = sample[sample["category"] != UNKNOWN]
    cat_counts = known["category"].value_counts()
    return {
        "size": len(sample),
        "target_size": target_n,
        "seed": seed,
        "unique_review_ids": int(sample["review_id"].nunique()),
        "by_source": sample["source"].value_counts().to_dict(),
        "label_distribution": sample["label"].value_counts().to_dict(),
        "label_distribution_pct": (sample["label"].value_counts(normalize=True) * 100)
        .round(1)
        .to_dict(),
        "label_by_source": {
            s: g["label"].value_counts().to_dict() for s, g in sample.groupby("source")
        },
        "rating_distribution": sample["rating"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict(),
        "category_distribution": sample["category"].value_counts().to_dict(),
        "known_category_distribution_pct": (cat_counts / max(len(known), 1) * 100)
        .round(1)
        .to_dict(),
        "max_known_category_pct": round(float(cat_counts.max() / len(known) * 100), 1)
        if len(known)
        else 0.0,
        "unique_products": int(known["product_name"].nunique()),
        "keyword_hits": {a: int(sample[hit_column(a)].sum()) for a in RARE_ASPECTS},
        "keyword_hits_pct": {
            a: round(float(sample[hit_column(a)].mean() * 100), 1) for a in RARE_ASPECTS
        },
        "any_rare_hit": int(sample[[hit_column(a) for a in RARE_ASPECTS]].any(axis=1).sum()),
    }


def render_markdown(s: dict) -> str:
    """İstatistikleri markdown'a çevirir (issue/rapor için)."""
    lines = [
        f"- **Örnek boyutu:** {s['size']:,} (hedef {s['target_size']:,}, seed={s['seed']}, "
        f"benzersiz review_id: {s['unique_review_ids']:,})",
        f"- **Kaynak dağılımı:** {s['by_source']}",
        "- **Etiket dağılımı:** "
        + ", ".join(
            f"{k}: {v:,} (%{s['label_distribution_pct'][k]})"
            for k, v in s["label_distribution"].items()
        ),
        f"- **Kaynak bazında etiket:** {s['label_by_source']}",
        f"- **Trendyol puan dağılımı:** {s['rating_distribution']}",
        f"- **Kategori dağılımı (sayı):** {s['category_distribution']}",
        f"- **Bilinen kategorilerde pay (%):** {s['known_category_distribution_pct']} "
        f"(en büyük: %{s['max_known_category_pct']}, benzersiz ürün: {s['unique_products']})",
        "- **Nadir aspect anahtar kelime isabetleri:**",
    ]
    for aspect, count in s["keyword_hits"].items():
        lines.append(f"  - `{aspect}`: {count:,} (%{s['keyword_hits_pct'][aspect]})")
    lines.append(f"- **En az bir nadir aspect isabeti olan yorum:** {s['any_rare_hit']:,}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/interim/reviews_clean.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/interim/labeling_sample.csv"))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    pool = pd.read_csv(args.pool, low_memory=False, dtype={"product_name": "string"})
    pool["product_name"] = (
        pool["product_name"].astype(object).where(pool["product_name"].notna(), None)
    )
    sample = draw_sample(pool, n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)

    stats = compute_stats(sample, args.n, args.seed)
    stem = args.out.with_suffix("")
    Path(f"{stem}_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=int), encoding="utf-8"
    )
    Path(f"{stem}_stats.md").write_text(render_markdown(stats), encoding="utf-8")
    if len(sample) < args.n:
        print(f"UYARI: hedeflenen {args.n} yerine {len(sample)} yorum seçilebildi.")
    print(render_markdown(stats))


if __name__ == "__main__":
    main()
