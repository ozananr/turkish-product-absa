"""Sahte (mock) predict(): docs/requirements.md sözleşmesine birebir uyar.

Sonuçlar girdi metninin hash'inden türetilen bir tohumla üretilir, bu yüzden
aynı girdi her zaman aynı çıktıyı verir (kural 4). Gerçek model hazır olunca
yalnızca bu dosyanın içi değişir, imza aynı kalır.
"""

import hashlib
import random

from .schema import ASPECTS, LABELS

# LABELS sırasıyla: yok, olumlu, olumsuz, notr. "yok" en sık, "notr" en nadir.
_LABEL_WEIGHTS = [4, 3, 2, 1]


def predict(text: str) -> dict[str, dict[str, str | float]]:
    """Yorum için her aspect'in etiketini ve güven skorunu döndürür.

    Returns:
        {"kalite": {"label": "olumlu", "confidence": 0.91}, ...}
        Çıktı her zaman 7 aspect'i ASPECTS sırasında içerir.

    Raises:
        TypeError: Girdi str değilse.
        ValueError: Girdi boş ya da yalnızca boşluktan oluşuyorsa.
    """
    if not isinstance(text, str):
        raise TypeError("Girdi str olmalıdır.")
    if not text.strip():
        raise ValueError("Yorum metni boş olamaz.")

    normalized = " ".join(text.split()).casefold()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))

    result: dict[str, dict[str, str | float]] = {}
    for aspect in ASPECTS:
        label = rng.choices(LABELS, weights=_LABEL_WEIGHTS, k=1)[0]
        confidence = round(rng.uniform(0.50, 0.99), 2)
        result[aspect] = {"label": label, "confidence": confidence}
    return result
