import pandas as pd

from absa.data.loader import COLUMNS, load_trendyol, load_trsa, rating_to_label
from absa.data.pipeline import run


def test_rating_to_label():
    assert rating_to_label(1) == "negative"
    assert rating_to_label(2) == "negative"
    assert rating_to_label(3) == "neutral"
    assert rating_to_label(5) == "positive"
    assert rating_to_label(float("nan")) is None


def _write_inputs(tmp_path):
    trsa = tmp_path / "trsa.csv"
    trendyol = tmp_path / "trendyol.csv"
    pd.DataFrame(
        {
            "id": [0, 1],
            "score": ["Positive", "Negative"],
            "review": ["Kokusu çok güzel etkisi de efsane", "Hiç bir fark olmadı siyah nokta"],
        }
    ).to_csv(trsa, index=False)
    pd.DataFrame(
        {
            "product_name": ["Şampuan", "Şampuan"],
            "review_rating": [5, 1],
            "review_body": [
                "Kokusu çok güzel etkisi de efsane",  # TRSA ile tekrar
                "Saçlarım döküldü, ara 0555 555 55 55 lütfen",
            ],
        }
    ).to_csv(trendyol, index=False)
    return trsa, trendyol


def test_loaders_ortak_sema(tmp_path):
    trsa, trendyol = _write_inputs(tmp_path)
    a, b = load_trsa(trsa), load_trendyol(trendyol)
    assert list(a.columns) == list(b.columns) == COLUMNS
    assert a["label"].tolist() == ["positive", "negative"]
    assert b["label"].tolist() == ["positive", "negative"]
    assert a["review_id"].is_unique and b["review_id"].is_unique


def test_pipeline_uctan_uca(tmp_path):
    trsa, trendyol = _write_inputs(tmp_path)
    raw, interim = tmp_path / "raw", tmp_path / "interim"
    metrics = run(trsa, trendyol, raw, interim)

    assert (raw / "TRSAv1.csv").exists() and (raw / "trendyol.csv").exists()
    assert (interim / "reviews_combined.csv").exists()
    cleaned = pd.read_csv(interim / "reviews_clean.csv")
    assert metrics["initial"] == 4
    assert len(cleaned) == metrics["final"] == 3
    assert metrics["target_reached"] is False
    assert any("[TELEFON]" in t for t in cleaned["text"])
