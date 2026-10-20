import numpy as np
import pandas as pd

from absa.data.categories import UNKNOWN
from absa.data.keywords import RARE_ASPECTS, hit_column
from absa.data.sampling import _split, compute_stats, draw_sample, render_markdown


def _pool(seed: int = 0) -> pd.DataFrame:
    """Sentetik havuz: pozitif ağırlıklı, 2 kaynak, birkaç kategori."""
    rng = np.random.default_rng(seed)
    rows = []
    products = {
        "giyim": [f"Kadın Tayt {i}" for i in range(40)],
        "kozmetik_bakim": [f"Şampuan {i}" for i in range(40)],
        "ayakkabi": [f"Spor Ayakkabı {i}" for i in range(40)],
        "aksesuar": [f"Güneş Gözlüğü {i}" for i in range(40)],
    }
    texts = [
        "Kargo çok geç geldi ama ürün iyi",
        "Satıcı ilgisizdi iade istedim",
        "Beden dar geldi kalıbı küçük",
        "Ürünün kalitesi gayet iyi bence",
        "Kokusu güzel ama kalıcı değil",
    ]
    i = 0
    for source in ("trendyol", "trsa"):
        for label, count in (("positive", 1500), ("negative", 400), ("neutral", 300)):
            for _ in range(count):
                if source == "trendyol":
                    cat = rng.choice(list(products))
                    name = rng.choice(products[cat])
                else:
                    name = None
                rows.append(
                    {
                        "review_id": f"{source}_{i}",
                        "source": source,
                        "text": f"{texts[rng.integers(len(texts))]} {i}",
                        "rating": 5.0 if source == "trendyol" else np.nan,
                        "label": label,
                        "product_name": name,
                    }
                )
                i += 1
    return pd.DataFrame(rows)


def test_split_toplami_korur():
    out = _split(2500, {"a": 0.4, "b": 0.3, "c": 0.3})
    assert out == {"a": 1000, "b": 750, "c": 750}
    assert sum(_split(101, {"a": 0.4, "b": 0.3, "c": 0.3}).values()) == 101


def test_ayni_seed_ayni_ornek():
    pool = _pool()
    a = draw_sample(pool, n=300, seed=7, max_per_product=50)
    b = draw_sample(pool.sample(frac=1, random_state=1), n=300, seed=7, max_per_product=50)
    pd.testing.assert_frame_equal(a, b)


def test_farkli_seed_farkli_ornek():
    pool = _pool()
    a = draw_sample(pool, n=300, seed=1, max_per_product=50)
    b = draw_sample(pool, n=300, seed=2, max_per_product=50)
    assert set(a["review_id"]) != set(b["review_id"])


def test_boyut_ve_kolonlar():
    sample = draw_sample(_pool(), n=300, seed=3, max_per_product=50)
    assert len(sample) == 300
    assert sample["review_id"].is_unique
    for col in ("review_id", "text", "source", "label", "category"):
        assert col in sample.columns
    assert sample["review_id"].is_monotonic_increasing  # kararlı sıralama


def test_negatif_ve_notr_fazla_orneklenir():
    pool = _pool()
    sample = draw_sample(pool, n=300, seed=3, max_per_product=50)
    pool_pos = (pool["label"] == "positive").mean()
    assert pool_pos > 0.6
    shares = sample["label"].value_counts(normalize=True)
    assert shares["negative"] == 0.4 and shares["positive"] == 0.3 and shares["neutral"] == 0.3


def test_kategori_siniri():
    sample = draw_sample(_pool(), n=300, seed=3, max_per_product=50)
    known = sample[sample["category"] != UNKNOWN]
    assert len(known) > 0
    assert known["category"].value_counts(normalize=True).max() <= 0.30 + 1e-9


def test_trsa_kategorisi_bilinmiyor():
    sample = draw_sample(_pool(), n=300, seed=3, max_per_product=50)
    assert (sample.loc[sample["source"] == "trsa", "category"] == UNKNOWN).all()


def test_urun_basina_sinir():
    sample = draw_sample(_pool(), n=300, seed=3, max_per_product=2)
    counts = sample["product_name"].dropna().value_counts()
    assert counts.max() <= 2


def test_nadir_aspect_temsili():
    sample = draw_sample(_pool(), n=300, seed=3, aspect_share=0.15, max_per_product=50)
    for aspect in RARE_ASPECTS:
        assert sample[hit_column(aspect)].sum() >= 0.15 * len(sample) * 0.9


def test_istatistikler():
    sample = draw_sample(_pool(), n=300, seed=3, max_per_product=50)
    stats = compute_stats(sample, 300, 3)
    assert stats["size"] == 300
    assert stats["unique_review_ids"] == 300
    assert set(stats["keyword_hits"]) == set(RARE_ASPECTS)
    assert stats["max_known_category_pct"] <= 30.0
    assert "Örnek boyutu" in render_markdown(stats)
