import pytest

from absa.data.keywords import RARE_ASPECTS, add_hit_columns, find_hits, hit_column, tr_lower


def test_tr_lower():
    assert tr_lower("KARGO İADE ISIK") == "kargo iade ısık"


@pytest.mark.parametrize(
    "text,aspect",
    [
        ("Kargo çok hızlı geldi", "kargo_paketleme"),
        ("Paketleme berbattı, kutu ezilmiş", "kargo_paketleme"),
        ("KARGOYU geç teslim ettiler", "kargo_paketleme"),
        ("Satıcı çok ilgiliydi", "satici_hizmet"),
        ("İade sürecinde müşteri hizmetleri rezalet", "satici_hizmet"),
        ("Beden tam oldu", "beden_uyum"),
        ("Kalıbı dar", "beden_uyum"),
        ("Bir numara büyük geldi", "beden_uyum"),
    ],
)
def test_nadir_aspect_isabeti(text, aspect):
    assert find_hits(text)[aspect] is True


def test_tam_kelime_eslesmesi():
    # "dar" yalnızca tam kelime: "darbe" eşleşmemeli
    assert find_hits("Darbeye dayanıklı bir ürün")["beden_uyum"] is False
    assert find_hits("Ürün çok dar geldi")["beden_uyum"] is True


def test_isabet_yok():
    hits = find_hits("Şarj çok çabuk bitiyor ve kamerası kötü")
    assert not any(hits.values())


def test_metin_olmayan_girdi():
    assert not any(find_hits(None).values())
    assert not any(find_hits(float("nan")).values())


def test_add_hit_columns():
    import pandas as pd

    df = add_hit_columns(pd.DataFrame({"text": ["Kargo geç geldi", "Ürün güzel"]}))
    for aspect in RARE_ASPECTS:
        assert hit_column(aspect) in df.columns
    assert df[hit_column("kargo_paketleme")].tolist() == [True, False]
