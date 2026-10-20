"""Ürün adından basit kural tabanlı kategori çıkarımı.

Yalnızca ürün adı olan (Trendyol) yorumlar için kategori atanır. Ürün adı olmayan
(TRSAv1) yorumlar `bilinmiyor` olur ve kategori dengesi sınırına dahil edilmez.

Kurallar sırayla denenir, ilk eşleşen kategori seçilir. Sıra önemlidir:
örneğin "Emici Külot Lohusa Pedi" giyim değil `hijyen_ped` olmalıdır.
"""

from __future__ import annotations

import re

from absa.data.keywords import tr_lower

UNKNOWN = "bilinmiyor"
OTHER = "diger"

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("hijyen_ped", ["ped", "pedi", "hijyenik"]),
    (
        "ayakkabi",
        ["ayakkab", "bot", "terlik", "sandalet", "sneaker", "çizme", "bootie", "babet", "patik"],
    ),
    ("spor", ["fitness", "crossfit", "kaleci", "yüzücü", "dambıl", "yoga", "halter"]),
    (
        "aksesuar",
        [
            "gözlük",
            "gözlüğ",
            "çanta",
            "eldiven",
            "bileklik",
            "şemsiye",
            "takı",
            "saat",
            "cüzdan",
            "kemer",
            "şapka",
            "atkı",
            "bere",
            "halhal",
            "küpe",
            "kolye",
            "yüzük",
        ],
    ),
    (
        "kozmetik_bakim",
        [
            "şampuan",
            "şampun",
            "parfüm",
            "deodorant",
            "deo",
            "roll on",
            "roll-on",
            "krem",
            "jel",
            "makyaj",
            "serum",
            "losyon",
            "sprey",
            "bakım",
            "maske",
            "ruj",
            "rimel",
            "kolonya",
            "sabun",
            "boya",
            "dolgunlaştır",
        ],
    ),
    (
        "giyim",
        [
            "elbise",
            "tayt",
            "sweat",
            "külot",
            "korse",
            "atlet",
            "çorap",
            "etek",
            "şort",
            "bluz",
            "t-shirt",
            "tişört",
            "sütyen",
            "jüpon",
            "pantolon",
            "gömlek",
            "ceket",
            "mont",
            "kazak",
            "hırka",
            "pijama",
            "tulum",
            "body",
            "tunik",
            "eşofman",
            "mayo",
            "bikini",
            "çamaşır",
            "kaşkorse",
            "triko",
            "yelek",
            "hamile",
            "emzirme",
            "bone",
            "fular",
            "şal",
            "bandana",
            "kaban",
            "yağmurluk",
            "badi",
            "tarlatan",
            "boyunluk",
            "tesettür",
        ],
    ),
]

# Kelime başında eşleşir ("bot" -> "botlar"). Çok kısa kelimeler tam kelime olmalıdır
# ("ped" -> "pedal" eşleşmesin).
_EXACT_ONLY = {"ped", "bot", "jel", "saat", "bere"}
_COMPILED = [
    (
        name,
        re.compile(
            "|".join(
                rf"(?<!\w){re.escape(tr_lower(w))}" + (r"(?!\w)" if w in _EXACT_ONLY else "")
                for w in words
            )
        ),
    )
    for name, words in CATEGORY_RULES
]


def assign_category(product_name: object, source: str | None = None) -> str:
    """Ürün adından kategori. Ürün adı yoksa: Trendyol için `diger`, diğerleri için `bilinmiyor`."""
    if not isinstance(product_name, str) or not product_name.strip():
        return OTHER if source == "trendyol" else UNKNOWN
    lowered = tr_lower(product_name)
    for name, pattern in _COMPILED:
        if pattern.search(lowered):
            return name
    return OTHER
