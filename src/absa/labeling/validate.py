"""LLM cevabını ayrıştırır ve schema.py'ye göre doğrular.

Geçerli cevap: 7 aspect'in hepsini içeren, başka anahtar içermeyen ve her değeri
LABELS içinde olan bir JSON nesnesi. Küçük biçim farkları (büyük harf, boşluk,
```json ... ``` bloğu) düzeltilir; bunun dışındaki her sapma InvalidResponse fırlatır.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from absa.schema import ASPECTS, LABELS

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class InvalidResponse(ValueError):
    """LLM cevabı şemaya uymuyorsa"""


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    match = _FENCE.match(text)
    return match.group(1) if match else text


def parse_labels(raw: str | None) -> dict[str, str]:
    """Ham cevabı {aspect: label} sözlüğüne çevirir (ASPECTS sırasında).

    Raises:
        InvalidResponse: JSON değilse, eksik/fazla aspect varsa ya da etiket geçersizse.
    """
    if not raw or not raw.strip():
        raise InvalidResponse("boş cevap")
    try:
        data = json.loads(_strip_fence(raw))
    except json.JSONDecodeError as exc:
        raise InvalidResponse(f"JSON değil: {exc.msg}") from exc
    # Bazı modeller (ör. Gemini) nesneyi tek elemanlı bir listeye sarıyor: [{...}]
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        raise InvalidResponse("JSON nesnesi değil")

    normalized: dict[str, object] = {}
    for key, value in data.items():
        k = str(key).strip().lower()
        normalized[k] = value.strip().lower() if isinstance(value, str) else value

    missing = [a for a in ASPECTS if a not in normalized]
    extra = sorted(k for k in normalized if k not in ASPECTS)
    if missing:
        raise InvalidResponse(f"eksik aspect: {missing}")
    if extra:
        raise InvalidResponse(f"bilinmeyen aspect: {extra}")

    bad = {a: normalized[a] for a in ASPECTS if normalized[a] not in LABELS}
    if bad:
        raise InvalidResponse(f"geçersiz etiket: {bad}")
    return {a: str(normalized[a]) for a in ASPECTS}


def is_valid_row(labels: dict[str, str]) -> bool:
    """Bir satırın 7 aspect'inin de geçerli etiket taşıyıp taşımadığını söyler."""
    return list(labels) == ASPECTS and all(labels[a] in LABELS for a in ASPECTS)


EXPORT_COLUMNS = ["review_id", "text", *ASPECTS, "annotator"]


def validate_labels(df: pd.DataFrame) -> list[str]:
    """Etiketleyici export'unu kontrol eder. Hata mesajlarının listesini döner; boş liste = geçerli.

    Aynı review_id iki farklı etiketleyicide olabilir (kappa seti), aynı etiketleyicide olamaz.
    """
    missing = [c for c in EXPORT_COLUMNS if c not in df.columns]
    if missing:
        return [f"Eksik sütun: {missing}"]
    errors = []
    df = df.fillna("").astype(str)
    duplicated = df.duplicated(subset=["review_id", "annotator"])
    if duplicated.any():
        errors.append(f"Aynı etiketleyicide tekrar eden review_id: {_ids(df[duplicated])}")
    empty_text = df["text"].str.strip() == ""
    if empty_text.any():
        errors.append(f"Boş metin: {_ids(df[empty_text])}")
    empty_annotator = df["annotator"].str.strip() == ""
    if empty_annotator.any():
        errors.append(f"annotator boş: {_ids(df[empty_annotator])}")
    for a in ASPECTS:
        bad = ~df[a].isin(LABELS)
        if bad.any():
            values = sorted(set(df.loc[bad, a]))
            errors.append(f"{a}: geçersiz değer {values}, satırlar {_ids(df[bad])}")
    return errors


def _ids(rows: pd.DataFrame, limit: int = 5) -> list[str]:
    return rows["review_id"].head(limit).tolist()
