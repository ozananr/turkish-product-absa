"""Aspect ve etiket şeması (docs/requirements.md sözleşmesinin kod karşılığı).

Anahtarlar yalnızca ASCII karakter içerir. Görünen adlar yalnızca arayüz içindir.
"""

ASPECTS = [
    "kalite",
    "fiyat",
    "kargo_paketleme",
    "satici_hizmet",
    "islev_performans",
    "gorunum_tasarim",
    "beden_uyum",
]

LABELS = ["yok", "olumlu", "olumsuz", "notr"]

# "yok" = aspect yorumda bahsedilmiyor.
NO_MENTION = "yok"

# Arayüzde gösterilecek adlar (sözleşme bölüm 2).
ASPECT_DISPLAY = {
    "kalite": "Kalite / Dayanıklılık",
    "fiyat": "Fiyat",
    "kargo_paketleme": "Kargo / Paketleme",
    "satici_hizmet": "Satıcı / Hizmet",
    "islev_performans": "İşlev / Performans",
    "gorunum_tasarim": "Görünüm / Tasarım",
    "beden_uyum": "Beden / Uyum",
}

LABEL_DISPLAY = {
    "yok": "Yok",
    "olumlu": "Olumlu",
    "olumsuz": "Olumsuz",
    "notr": "Nötr",
}
