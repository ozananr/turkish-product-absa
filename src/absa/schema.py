"""Aspect ve etiket şeması (docs/requirements.md sözleşmesinin kod karşılığı)."""

from typing import Final

ASPECTS: Final[tuple[str, ...]] = (
    "kalite",
    "fiyat",
    "kargo",
    "satici",
    "islev",
    "gorunum",
    "beden",
)

# yok = aspect yorumda hiç geçmiyor şeklinde ayarlanmıştır


LABELS: Final[tuple[str, ...]] = ("yok", "olumlu", "olumsuz", "notr")

NO_MENTION: Final[str] = "yok"
