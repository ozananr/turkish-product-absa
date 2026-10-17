"""Ham veri setlerini okuyup ortak şemaya çeviren fonksiyonlar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["review_id", "source", "text", "rating", "label", "product_name"]

_TRSA_LABELS = {"positive": "positive", "negative": "negative", "neutral": "neutral"}


def rating_to_label(rating: float) -> str | None:
    """1-2 yıldız negatif, 3 nötr, 4-5 pozitif."""
    if pd.isna(rating):
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def load_trsa(path: str | Path) -> pd.DataFrame:
    """TRSAv1 (id, score, review) -> ortak şema. Puan yok, yalnızca etiket var."""
    raw = pd.read_csv(path, encoding="utf-8")
    return pd.DataFrame(
        {
            "review_id": "trsa_" + raw["id"].astype(str),
            "source": "trsa",
            "text": raw["review"],
            "rating": float("nan"),
            "label": raw["score"].astype(str).str.lower().map(_TRSA_LABELS),
            "product_name": pd.NA,
        }
    )[COLUMNS]


def load_trendyol(path: str | Path) -> pd.DataFrame:
    """Trendyol (product_name, review_rating, review_body) -> ortak şema."""
    raw = pd.read_csv(path, encoding="utf-8")
    rating = pd.to_numeric(raw["review_rating"], errors="coerce")
    return pd.DataFrame(
        {
            "review_id": "trendyol_" + raw.index.astype(str),
            "source": "trendyol",
            "text": raw["review_body"],
            "rating": rating,
            "label": rating.map(rating_to_label),
            "product_name": raw["product_name"],
        }
    )[COLUMNS]


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Ortak şemadaki tabloları tek tabloda birleştirir."""
    return pd.concat(frames, ignore_index=True)[COLUMNS]
