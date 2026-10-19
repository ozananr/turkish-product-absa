"""Nadir aspect'leri yakalamak için anahtar kelime listeleri.

Etiketleme örneğinde nadir aspect'lerin (kargo_paketleme, satici_hizmet, beden_uyum) yeterli
sayıda temsil edilmesi için kullanılır. Liste `docs/labeling_sample.md` içinde de dokümante edilir.

Eşleşme kuralları:
- Metin Türkçe kurallarına göre küçük harfe çevrilir (İ -> i, I -> ı).
- ``PREFIX``: kelime başında eşleşir, ek alabilir ("kargo" -> "kargoyu", "kargosu").
- ``EXACT``: yalnızca tam kelime olarak eşleşir ("dar" -> "darbe" eşleşmez).
- Bu listeler yalnızca ADAY seçmek içindir. Aspect etiketi insan tarafından verilir,
  bir anahtar kelimenin geçmesi aspect'in yorumda olduğunu garanti etmez.
"""

from __future__ import annotations

import re

import pandas as pd

RARE_ASPECTS = ("kargo_paketleme", "satici_hizmet", "beden_uyum")

_TR_TABLE = str.maketrans({"İ": "i", "I": "ı"})

ASPECT_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "kargo_paketleme": {
        "prefix": [
            "kargo",
            "paket",
            "ambalaj",
            "kutu",
            "koli",
            "teslimat",
            "teslim",
            "kurye",
            "gönderi",
            "gonderi",
            "sevkiyat",
            "yurtiçi",
            "yurtici",
            "aras ",
            "mng",
            "ptt",
            "hasarlı",
            "hasarli",
            "ezik",
            "ezilmiş",
            "ezilmis",
            "balonlu",
            "poşet",
        ],
        "exact": ["geç geldi", "gec geldi", "geç ulaştı", "geç teslim"],
    },
    "satici_hizmet": {
        "prefix": [
            "satıcı",
            "saticı",
            "satici",
            "mağaza",
            "magaza",
            "iade",
            "değişim",
            "degisim",
            "müşteri hizmet",
            "musteri hizmet",
            "müşteri temsilci",
            "musteri temsilci",
            "iletişim",
            "iletisim",
            "garanti",
            "servis",
            "geri ödeme",
            "şikayet",
            "sikayet",
            "destek",
            "cevap",
            "ilgilen",
            "ilgili satıcı",
            "yanıt",
            "yanit",
            "mesaj",
        ],
        "exact": ["firma", "şikâyet"],
    },
    "beden_uyum": {
        "prefix": [
            "beden",
            "kalıp",
            "kalip",
            "numara",
            "ölçü",
            "olcu",
            "büyük geldi",
            "buyuk geldi",
            "küçük geldi",
            "kucuk geldi",
            "büyük aldım",
            "küçük aldım",
            "bir beden",
            "sıkıyor",
            "sıktı",
            "sikiyor",
            "boyu",
            "uydu",
            "uyuyor",
            "oturdu",
            "oturuyor",
            "rahat",
            "bedenim",
            "yakıştı",
            "tam oldu",
        ],
        "exact": ["dar", "bol", "xl", "xxl", "xs", "s beden", "boy"],
    },
}


def tr_lower(text: str) -> str:
    """Türkçe küçük harf dönüşümü."""
    return text.translate(_TR_TABLE).lower()


def _compile(keywords: dict[str, list[str]]) -> re.Pattern[str]:
    parts = [rf"(?<!\w){re.escape(tr_lower(k))}" for k in keywords["prefix"]]
    parts += [rf"(?<!\w){re.escape(tr_lower(k))}(?!\w)" for k in keywords["exact"]]
    return re.compile("|".join(parts))


_PATTERNS = {aspect: _compile(kw) for aspect, kw in ASPECT_KEYWORDS.items()}


def hit_column(aspect: str) -> str:
    """Aspect için isabet kolonunun adı."""
    return f"hit_{aspect}"


def find_hits(text: object) -> dict[str, bool]:
    """Metinde her nadir aspect için anahtar kelime isabeti var mı?"""
    lowered = tr_lower(text) if isinstance(text, str) else ""
    return {aspect: bool(pat.search(lowered)) for aspect, pat in _PATTERNS.items()}


def add_hit_columns(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Her nadir aspect için bool `hit_<aspect>` kolonu ekler."""
    out = df.copy()
    lowered = out[text_col].map(lambda t: tr_lower(t) if isinstance(t, str) else "")
    for aspect, pat in _PATTERNS.items():
        out[hit_column(aspect)] = lowered.map(lambda t, p=pat: bool(p.search(t)))
    return out
