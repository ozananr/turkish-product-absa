import pandas as pd

from absa.schema import ASPECTS, LABELS


def validate_labels(df: pd.DataFrame) -> list[str]:
    """Hata mesajlarının listesini döndürür. Boş liste = geçerli."""
    required = ["review_id", "text", *ASPECTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"Eksik sütun: {missing}"]

    errors = []
    if df["review_id"].duplicated().any():
        errors.append("Tekrar eden review_id var.")
    for a in ASPECTS:
        bad = df[~df[a].isin(LABELS)]
        if not bad.empty:
            errors.append(f"{a}: geçersiz değer, satırlar {list(bad.index[:5])}")
    return errors
