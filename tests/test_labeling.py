import pandas as pd

from absa.labeling.validate import validate_labels
from absa.schema import ASPECTS


def _row(review_id="r1", label="yok", annotator="ozan", text="metin"):
    return {"review_id": review_id, "text": text, **{a: label for a in ASPECTS},
            "annotator": annotator}  # fmt: skip


def test_gecerli_export():
    assert validate_labels(pd.DataFrame([_row(label="olumlu"), _row("r2")])) == []


def test_kappa_setinde_ayni_yorum_iki_etiketleyicide_olabilir():
    df = pd.DataFrame([_row("r1", annotator="ozan"), _row("r1", annotator="salih")])
    assert validate_labels(df) == []


def test_ayni_etiketleyicide_tekrar_eden_id():
    assert validate_labels(pd.DataFrame([_row("r1"), _row("r1")]))


def test_gecersiz_bos_ve_buyuk_harfli_etiket():
    for label in ["Olumlu", "", "nötr", None]:
        errors = validate_labels(pd.DataFrame([_row(label=label)]))
        assert errors and "geçersiz değer" in errors[0]


def test_bos_metin_ve_bos_annotator():
    assert any("Boş metin" in e for e in validate_labels(pd.DataFrame([_row(text=" ")])))
    assert any("annotator" in e for e in validate_labels(pd.DataFrame([_row(annotator="")])))


def test_eksik_sutun():
    assert validate_labels(pd.DataFrame([{"review_id": "r1", "text": "x"}]))
