import json

import pytest

from absa.labeling.validate import InvalidResponse, is_valid_row, parse_labels
from absa.schema import ASPECTS

VALID = {a: "yok" for a in ASPECTS} | {"kalite": "olumlu", "kargo_paketleme": "olumsuz"}


def test_gecerli_cevap_aspect_sirasinda_doner():
    out = parse_labels(json.dumps(VALID))
    assert list(out) == ASPECTS
    assert out["kalite"] == "olumlu"
    assert is_valid_row(out)


def test_kod_blogu_ve_buyuk_harf_duzeltilir():
    shouty = {k.upper(): f" {v.upper()} " for k, v in VALID.items()}
    raw = "```json\n" + json.dumps(shouty) + "\n```"
    assert parse_labels(raw) == parse_labels(json.dumps(VALID))


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        None,
        "bu bir json değil",
        "[1, 2, 3]",
        json.dumps({a: "yok" for a in ASPECTS[:-1]}),  # eksik aspect
        json.dumps(VALID | {"kurulum": "olumsuz"}),  # fazla aspect
        json.dumps(VALID | {"fiyat": "pozitif"}),  # geçersiz etiket
        json.dumps(VALID | {"fiyat": "nötr"}),  # ASCII olmayan etiket
        json.dumps(VALID | {"fiyat": None}),
    ],
)
def test_gecersiz_cevaplar_reddedilir(raw):
    with pytest.raises(InvalidResponse):
        parse_labels(raw)


def test_is_valid_row_yanlis_sirayi_reddeder():
    assert not is_valid_row(dict(reversed(list(VALID.items()))))


def test_tek_elemanli_listeye_sarili_nesne_kabul_edilir():
    assert parse_labels(json.dumps([VALID])) == parse_labels(json.dumps(VALID))


def test_birden_fazla_nesneli_liste_reddedilir():
    with pytest.raises(InvalidResponse):
        parse_labels(json.dumps([VALID, VALID]))
