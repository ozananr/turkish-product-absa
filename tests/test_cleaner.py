import pandas as pd
import pytest

from absa.data.cleaner import (
    clean_reviews,
    clean_with_report,
    dedup_key,
    is_english,
    is_meaningless,
    is_template,
    is_too_short,
    word_count,
)


def test_word_count_maske_ve_noktalama_saymaz():
    assert word_count("Ürün çok güzel!!!") == 3
    assert word_count("[TELEFON] ara") == 1


@pytest.mark.parametrize(
    "text,expected", [("iyi", True), ("çok iyi", True), ("çok iyi ürün", False)]
)
def test_is_too_short(text, expected):
    assert is_too_short(text) is expected


def test_dedup_key_buyuk_kucuk_ve_noktalama_yok_sayar():
    assert dedup_key("Kargo ÇOK hızlı!!") == dedup_key("kargo çok hızlı")
    assert dedup_key("IŞIK iyi") == dedup_key("ışık İYİ")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This product is very good and I love it", True),
        ("great quality, fast delivery", True),
        ("Ürün çok güzel ama kargo geç geldi", False),
        ("urun cok guzel tavsiye ederim", False),  # aksansız Türkçe
        ("Telefon good ama şarj bitiyor", False),
    ],
)
def test_is_english(text, expected):
    assert is_english(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("asdfghjkl qwrtp zxcvb", True),
        ("aaaaaaaaaa güzel", True),
        ("111 222 333 !!!", True),
        ("iyi iyi iyi iyi", True),
        ("Kumaşı kaliteli ama bedeni küçük geldi", False),
    ],
)
def test_is_meaningless(text, expected):
    assert is_meaningless(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hızlı kargo", True),
        ("güzel ürün tavsiye ederim", True),
        ("Çok güzel ürün, teşekkürler!", True),
        ("HIZLI KARGO GÜZEL ÜRÜN", True),
        ("Güzel ürün ama beden küçük geldi", False),
        ("Şarj çok çabuk bitiyor", False),
    ],
)
def test_is_template(text, expected):
    assert is_template(text) is expected


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": range(9),
            "source": ["a", "a", "a", "b", "b", "a", "b", "a", "b"],
            "text": [
                "Kamera muazzam ama şarj hemen bitiyor",  # kalır
                "kamera MUAZZAM ama şarj hemen bitiyor!",  # veri seti içi tekrar
                "iyi",  # kısa
                "Kamera muazzam ama şarj hemen bitiyor",  # veri setleri arası tekrar
                "Hızlı kargo güzel ürün",  # şablon
                "Mail test@example.com adresine yazın lütfen",  # kalır, maskelenir
                "This product is very good and I love it",  # İngilizce
                "asdfg hjklş zxcvb qwrt",  # anlamsız
                None,  # boş
            ],
        }
    )


def test_clean_with_report_adim_sayilari():
    cleaned, report = clean_with_report(_sample())
    dropped = {s["step"]: s["dropped"] for s in report.steps}
    assert dropped["Boş metin"] == 1
    assert dropped["Tekrar (veri seti içi)"] == 1
    assert dropped["Tekrar (veri setleri arası)"] == 1
    assert dropped["Tamamen İngilizce"] == 1
    assert dropped["Anlamsız"] == 1
    assert dropped["Şablon"] == 1
    assert len(cleaned) == report.final == 2
    assert report.initial == 9
    # Her adımda önce - elenen = sonra
    assert all(s["before"] - s["dropped"] == s["after"] for s in report.steps)


def test_clean_reviews_maskeler():
    cleaned = clean_reviews(_sample())
    assert any("[EPOSTA]" in t for t in cleaned["text"])
    assert not any("test@example.com" in t for t in cleaned["text"])


def test_maskeleme_tekrar_tespitinden_once_yapilir():
    df = pd.DataFrame(
        {
            "source": ["a", "a"],
            "text": [
                "Ara 0555 111 22 33 hemen sorunu çözün",
                "Ara 0555 999 88 77 hemen sorunu çözün",
            ],
        }
    )
    assert len(clean_reviews(df)) == 1
