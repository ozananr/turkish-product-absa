import pytest

from absa.data.masker import count_masks, mask_personal_data


@pytest.mark.parametrize(
    "text",
    [
        "Bana 0555 555 55 55 numarasından ulaşın",
        "Bana 05555555555 numarasından ulaşın",
        "Bana +90 555 555 55 55 numarasından ulaşın",
        "Bana 0 (555) 555-55-55 numarasından ulaşın",
        "Bana 555 555 55 55 numarasından ulaşın",
    ],
)
def test_telefon_maskelenir(text):
    masked = mask_personal_data(text)
    assert "[TELEFON]" in masked
    assert "555" not in masked


def test_sabit_hat_maskelenir():
    assert mask_personal_data("Ara 0212 345 67 89 hemen") == "Ara [TELEFON] hemen"


def test_eposta_maskelenir():
    assert mask_personal_data("Mail adresim test.user@example.com") == "Mail adresim [EPOSTA]"


def test_url_maskelenir():
    assert mask_personal_data("Bak https://site.com/urun?id=3 güzel") == "Bak [URL] güzel"
    assert mask_personal_data("www.site.com'dan aldım") == "[URL]'dan aldım"


def test_iban_ve_kart_maskelenir():
    assert "[IBAN]" in mask_personal_data("IBAN TR33 0006 1005 1978 6457 8413 26 yazdım")
    assert mask_personal_data("Kart 4111 1111 1111 1111 çekildi") == "Kart [KART] çekildi"


def test_tckn_yalnizca_gecerliyse_maskelenir():
    assert mask_personal_data("TC 10000000146 yazdım") == "TC [TCKN] yazdım"
    assert mask_personal_data("Sipariş 12345678901 geldi") == "Sipariş 12345678901 geldi"


def test_isim_maskelenir():
    assert mask_personal_data("Merhaba adım Ayşe Yılmaz, ürün güzel") == (
        "Merhaba adım [ISIM], ürün güzel"
    )
    assert mask_personal_data("Sayın Ahmet Kaya teşekkürler") == "Sayın [ISIM] teşekkürler"
    assert mask_personal_data("Mehmet Bey çok ilgiliydi") == "[ISIM] Bey çok ilgiliydi"


def test_normal_metin_degismez():
    text = "Kumaşı çok kaliteli, 3 gün önce aldım, 250 TL."
    assert mask_personal_data(text) == text


def test_metin_olmayan_girdi():
    assert mask_personal_data(None) == ""
    assert mask_personal_data(float("nan")) == ""


def test_count_masks():
    masked = mask_personal_data("a@b.com ve 0555 555 55 55 ve c@d.com")
    assert count_masks(masked) == {"EPOSTA": 2, "TELEFON": 1}
