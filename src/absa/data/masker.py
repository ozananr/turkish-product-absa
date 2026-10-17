"""Yorumlardaki kişisel verileri regex ile etiketlerle maskeleyen fonksiyonlar.

Etiketler: [EPOSTA], [URL], [IBAN], [KART], [TELEFON], [TCKN], [ISIM].

Not: İsim tespiti regex ile yapıldığı için yalnızca belirgin kalıpları yakalar
("adım Ayşe", "Sayın Ahmet Yılmaz", "Mehmet Bey" gibi). Her ismi yakalamayı garanti etmez.
"""

from __future__ import annotations

import re

TAGS = ("EPOSTA", "URL", "IBAN", "KART", "TELEFON", "TCKN", "ISIM")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)
_URL = re.compile(r"(?:https?://|www\.)[^\s'’]+", re.IGNORECASE)
_IBAN = re.compile(r"(?<![\w])TR\d{2}(?:[\s-]?\d){22}(?!\d)", re.IGNORECASE)
_CARD = re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)")
# Cep: +90 / 90 / 0 ön eki opsiyonel, 5XX XXX XX XX. Sabit hat: 0 ile başlar, 2XX/3XX/4XX.
_PHONE = re.compile(
    r"(?<![\d\w])(?:"
    r"(?:\+?90[\s\-.]*)?\(?0?5\d{2}\)?[\s\-.]*\d{3}[\s\-.]*\d{2}[\s\-.]*\d{2}"
    r"|0[\s\-.]*\(?[234]\d{2}\)?[\s\-.]*\d{3}[\s\-.]*\d{2}[\s\-.]*\d{2}"
    r")(?!\d)"
)
_TCKN = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")

_NAME = r"[A-ZÇĞİÖŞÜ][a-zçğıöşü]{1,}"
# "adım Ayşe", "ismim Ahmet Yılmaz", "benim adım Ali"
_NAME_INTRO = re.compile(
    rf"(?i:\b(?:ad[ıi]m|ism[iı]m|ad[ıi]m[ıi]z))\s*[:\-]?\s+{_NAME}(?:\s+{_NAME})?"
)
# "Sayın Ahmet Yılmaz", "Sn. Ayşe"
_NAME_TITLE = re.compile(rf"\b(?:Say[ıi]n|Sn\.?)\s+{_NAME}(?:\s+{_NAME})?")
# "Mehmet Bey", "Ayşe Hanım"
_NAME_HONORIFIC = re.compile(rf"\b{_NAME}(?:\s+{_NAME})?\s+(?:Bey|Han[ıi]m)\b")

_INTRO_PREFIX = re.compile(r"(?i:^\b(?:ad[ıi]m|ism[iı]m|ad[ıi]m[ıi]z))\s*[:\-]?\s+")
_TITLE_PREFIX = re.compile(r"^(?:Say[ıi]n|Sn\.?)\s+")
_HONORIFIC_SUFFIX = re.compile(r"\s+(?:Bey|Han[ıi]m)$")


def _valid_tckn(number: str) -> bool:
    """T.C. kimlik numarası sağlama algoritması."""
    digits = [int(c) for c in number]
    if len(digits) != 11 or digits[0] == 0:
        return False
    odd = sum(digits[0:9:2])
    even = sum(digits[1:8:2])
    if (odd * 7 - even) % 10 != digits[9]:
        return False
    return sum(digits[:10]) % 10 == digits[10]


def _tckn_sub(match: re.Match[str]) -> str:
    return "[TCKN]" if _valid_tckn(match.group()) else match.group()


def _name_sub(prefix: re.Pattern[str] | None, suffix: re.Pattern[str] | None):
    def repl(match: re.Match[str]) -> str:
        found = match.group()
        keep_prefix = ""
        keep_suffix = ""
        if prefix is not None and (m := prefix.search(found)):
            keep_prefix = found[: m.end()]
        if suffix is not None and (m := suffix.search(found)):
            keep_suffix = found[m.start() :]
        return f"{keep_prefix}[ISIM]{keep_suffix}"

    return repl


def mask_personal_data(text: str) -> str:
    """Metindeki kişisel verileri etiketlerle değiştirir. Metin değilse boş string döner."""
    if not isinstance(text, str):
        return ""
    text = _EMAIL.sub("[EPOSTA]", text)
    text = _URL.sub("[URL]", text)
    text = _IBAN.sub("[IBAN]", text)
    text = _CARD.sub("[KART]", text)
    text = _PHONE.sub("[TELEFON]", text)
    text = _TCKN.sub(_tckn_sub, text)
    text = _NAME_INTRO.sub(_name_sub(_INTRO_PREFIX, None), text)
    text = _NAME_TITLE.sub(_name_sub(_TITLE_PREFIX, None), text)
    text = _NAME_HONORIFIC.sub(_name_sub(None, _HONORIFIC_SUFFIX), text)
    return text


def count_masks(text: str) -> dict[str, int]:
    """Metindeki maske etiketlerinin sayısını döner."""
    return {tag: text.count(f"[{tag}]") for tag in TAGS if f"[{tag}]" in text}
