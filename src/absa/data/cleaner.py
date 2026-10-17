"""Yorum temizleme: tekrar, kısa, İngilizce, anlamsız ve şablon yorumları eler.

Her adım elenen yorum sayısını `steps` listesine kaydeder (rapor için).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from absa.data.masker import mask_personal_data

MIN_WORDS = 3

_TR_TABLE = str.maketrans({"İ": "i", "I": "ı"})
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
_VOWELS = set("aeıioöuüAEIİOÖUÜ")

_ENGLISH_WORDS = {
    "the", "and", "is", "are", "was", "were", "this", "that", "it", "very", "good", "great",
    "product", "quality", "love", "nice", "not", "but", "for", "with", "have", "has", "i",
    "my", "you", "your", "to", "of", "in", "on", "so", "bad", "perfect", "fast", "delivery",
    "recommend", "best", "worth", "money", "like", "just", "its", "me", "we", "they", "a",
    "an", "be", "as", "at", "from", "will", "would", "can", "buy", "bought", "item", "works",
    "amazing", "excellent", "thanks", "thank", "happy", "size", "fit", "color", "price",
}  # fmt: skip
_TURKISH_WORDS = {
    "bir", "ve", "çok", "bu", "için", "ama", "ürün", "gibi", "da", "de", "ile", "daha", "hiç",
    "güzel", "ben", "var", "yok", "oldu", "olarak", "kadar", "değil", "tavsiye", "kargo",
    "cok", "icin", "urun", "guzel", "ederim", "memnun", "kaldım", "aldım", "ne", "mi", "bence",
    "çünkü", "sonra", "en", "her", "kullanıyorum", "teşekkürler", "fiyat", "kalite",
}  # fmt: skip

# Kendi başına bilgi taşımayan genel kelimeler. Yorum yalnızca bunlardan oluşuyorsa şablondur.
_GENERIC_WORDS = {
    "hızlı", "hizli", "kargo", "kargoya", "güzel", "guzel", "ürün", "urun", "ürünü", "urunu",
    "teşekkürler", "tesekkurler", "teşekkür", "tesekkur", "ederim", "tavsiye", "çok", "cok",
    "süper", "super", "harika", "mükemmel", "mukemmel", "memnun", "kaldım", "kaldim", "iyi",
    "gayet", "bir", "ve", "ama", "de", "da", "ile", "satıcı", "satici", "sağ", "olun", "olsun",
    "beğendim", "begendim", "aldım", "aldim", "tam", "istediğim", "gibi", "geldi", "elime",
    "ulaştı", "ulasti", "paketleme", "fiyat", "performans", "kaliteli", "bayıldım", "bayildim",
    "tşk", "tsk", "eyvallah", "hepsi", "her", "şey", "sey", "hızlıydı", "hizliydi", "gerçekten",
    "gercekten", "kesinlikle", "bence", "bu", "şu", "o", "bende", "sorunsuz", "ayrıca",
    "oldu", "olmuş", "olmus", "hayırlı", "hayirli", "yorum", "yapıyorum", "5", "yıldız",
}  # fmt: skip

TEMPLATE_PHRASES = {
    "hızlı kargo", "güzel ürün", "teşekkürler", "tavsiye ederim", "çok güzel", "çok iyi",
    "ürün güzel", "kargo hızlı", "hızlı kargo teşekkürler", "güzel ürün tavsiye ederim",
    "memnun kaldım", "harika ürün", "mükemmel ürün", "süper ürün", "beğendim", "çok beğendim",
    "tavsiye ederim teşekkürler", "ürün güzel kargo hızlı", "herkese tavsiye ederim",
    "kesinlikle tavsiye ederim", "çok memnun kaldım", "gayet güzel", "sorunsuz ürün",
}  # fmt: skip


@dataclass
class CleaningReport:
    """Adım adım elenen yorum sayıları."""

    initial: int = 0
    steps: list[dict] = field(default_factory=list)

    def add(self, name: str, before: int, after: int) -> None:
        self.steps.append(
            {"step": name, "before": before, "dropped": before - after, "after": after}
        )

    @property
    def final(self) -> int:
        return self.steps[-1]["after"] if self.steps else self.initial


def normalize_text(text: object) -> str:
    """Boşlukları sadeleştirir. Metin değilse boş string döner."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dedup_key(text: str) -> str:
    """Tekrar tespiti için anahtar: Türkçe küçük harf, noktalama ve boşluk yok sayılır."""
    lowered = text.translate(_TR_TABLE).lower()
    return " ".join(_TOKEN.findall(lowered)) or lowered


def word_count(text: str) -> int:
    """Maske etiketleri ve noktalama sayılmadan kelime sayısı."""
    return len(_TOKEN.findall(re.sub(r"\[[A-Z]+\]", " ", text)))


def is_too_short(text: str, min_words: int = MIN_WORDS) -> bool:
    return word_count(text) < min_words


def is_english(text: str) -> bool:
    """Tamamen İngilizce mi? Bağımlılık gerektirmeyen kelime listesi sezgisi."""
    if any(c in _TURKISH_CHARS for c in text):
        return False
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if len(tokens) < 2:
        return False
    en = sum(t in _ENGLISH_WORDS for t in tokens)
    tr = sum(t in _TURKISH_WORDS for t in tokens)
    return en >= 2 and en / len(tokens) >= 0.4 and en > tr


def is_meaningless(text: str) -> bool:
    """Harf oranı düşük, ünsüz yığını, aynı kelime/harf tekrarı gibi anlamsız metinler."""
    body = re.sub(r"\[[A-Z]+\]", "", text)
    chars = [c for c in body if not c.isspace()]
    if not chars:
        return True
    if sum(c.isalpha() for c in chars) / len(chars) < 0.5:
        return True
    tokens = _TOKEN.findall(body)
    if not tokens:
        return True
    if len(tokens) >= MIN_WORDS and len({t.lower() for t in tokens}) == 1:
        return True
    no_vowel = sum(1 for t in tokens if len(t) >= 3 and not any(c in _VOWELS for c in t))
    if no_vowel / len(tokens) >= 0.5:
        return True
    if re.search(r"(.)\1{5,}", body):
        return True
    return False


def is_template(text: str) -> bool:
    """Şablon yorum: bilinen kalıp ya da yalnızca genel kelimelerden oluşan kısa metin."""
    tokens = [t.translate(_TR_TABLE).lower() for t in _TOKEN.findall(text)]
    if not tokens:
        return False
    phrase = " ".join(tokens)
    if phrase in {p.translate(_TR_TABLE).lower() for p in TEMPLATE_PHRASES}:
        return True
    generic = {w.translate(_TR_TABLE).lower() for w in _GENERIC_WORDS}
    return len(tokens) <= 6 and all(t in generic for t in tokens)


def clean_with_report(
    df: pd.DataFrame, text_col: str = "text", min_words: int = MIN_WORDS
) -> tuple[pd.DataFrame, CleaningReport]:
    """Tabloyu temizler ve adım adım rapor döner.

    Sıra: normalize + maskele -> boş -> tekrar (kaynak içi) -> tekrar (kaynaklar arası)
    -> kısa -> İngilizce -> anlamsız -> şablon. Maskeleme tekrar tespitinden önce yapılır.
    """
    report = CleaningReport(initial=len(df))
    out = df.copy()
    out[text_col] = out[text_col].map(normalize_text).map(mask_personal_data)

    def step(name: str, keep: pd.Series) -> None:
        nonlocal out
        before = len(out)
        out = out[keep]
        report.add(name, before, len(out))

    step("Boş metin", out[text_col].str.len() > 0)
    out = out.assign(_key=out[text_col].map(dedup_key))
    if "source" in out.columns:
        step("Tekrar (veri seti içi)", ~out.duplicated(subset=["source", "_key"]))
    step("Tekrar (veri setleri arası)", ~out.duplicated(subset=["_key"]))
    step(
        f"Çok kısa (<{min_words} kelime)", ~out[text_col].map(lambda t: is_too_short(t, min_words))
    )
    step("Tamamen İngilizce", ~out[text_col].map(is_english))
    step("Anlamsız", ~out[text_col].map(is_meaningless))
    step("Şablon", ~out[text_col].map(is_template))
    return out.drop(columns="_key").reset_index(drop=True), report


def clean_reviews(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Yalnızca temizlenmiş tabloyu döner (rapor gerekmiyorsa)."""
    return clean_with_report(df, text_col)[0]
