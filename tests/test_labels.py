import pandas as pd

from absa.data.labels import check_labels, class_distribution, load_labels, rare_classes
from absa.schema import ASPECTS, LABELS


def _row(rid, text="metin", **labels):
    return {"review_id": rid, "text": text} | {a: "yok" for a in ASPECTS} | labels


def test_temiz_veri_degismeden_gecer():
    df = pd.DataFrame([_row("r1", kalite="olumlu"), _row("r2", fiyat="olumsuz")])
    clean, report = check_labels(df)
    assert len(clean) == 2 and not report.has_problems and report.dropped_ids == []


def test_guvenli_duzeltmeler_otomatik_yapilir():
    df = pd.DataFrame([_row("r1", kalite=" Olumlu ", fiyat="nötr", kargo_paketleme="YOK")])
    clean, report = check_labels(df)
    assert clean.loc[0, "kalite"] == "olumlu"
    assert clean.loc[0, "fiyat"] == "notr"
    assert report.auto_fixed == 3 and not report.has_problems


def test_sorunlu_satirlar_ayiklanir_ve_raporlanir():
    df = pd.DataFrame(
        [
            _row("r1", kalite="olumlu"),
            _row("r2", text="   "),  # boş metin
            _row("r1"),  # tekrar eden id
            _row("r3", fiyat="olumluu"),  # yazım hatası
            _row("r4", beden_uyum=""),  # boş etiket
            _row("r5"),
        ]
    )
    clean, report = check_labels(df)
    assert clean["review_id"].tolist() == ["r1", "r5"]
    assert report.empty_text == ["r2"]
    assert report.duplicate_ids == ["r1"]
    assert report.invalid_labels == {"fiyat": {"olumluu": 1}}
    assert report.empty_labels == {"beden_uyum": ["r4"]}
    assert set(report.dropped_ids) == {"r2", "r1", "r3", "r4"}
    assert report.has_problems


def test_eksik_sutun_raporlanir():
    df = pd.DataFrame([_row("r1")]).drop(columns=["kalite"])
    clean, report = check_labels(df)
    assert report.missing_columns == ["kalite"] and clean.empty


def test_csv_bom_ve_fazla_sutunla_okunur(tmp_path):
    path = tmp_path / "labels.csv"
    df = pd.DataFrame([_row("r1", kalite="olumlu") | {"not": "emin değilim"}])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    clean, report = load_labels(path)
    assert list(clean.columns) == ["review_id", "text", *ASPECTS]
    assert clean.loc[0, "kalite"] == "olumlu" and not report.has_problems


def test_sinif_dagilimi_ve_nadir_siniflar():
    df = pd.DataFrame([_row("r1", kalite="olumlu"), _row("r2"), _row("r3")])
    dist = class_distribution(df)
    assert list(dist.index) == ASPECTS and list(dist.columns) == LABELS
    assert dist.loc["kalite", "olumlu"] == 1 and dist.loc["kalite", "yok"] == 2
    rare = rare_classes(dist, min_count=2)
    assert ("kalite", "olumlu", 1) in rare and ("fiyat", "notr", 0) in rare
    assert ("fiyat", "yok", 3) not in rare


def _export(rows, annotator, checked):
    df = pd.DataFrame(rows)
    df["annotator"] = annotator
    df["checked"] = checked
    df["note"] = ""
    return df


def test_yalnizca_checked_satirlar_ve_birden_fazla_dosya(tmp_path):
    a = _export([_row("r1", kalite="olumlu"), _row("r2")], "ozan", ["TRUE", "FALSE"])
    b = _export([_row("r3", fiyat="olumsuz")], "salih", ["true"])
    pilot = pd.DataFrame([_row("r4", beden_uyum="olumsuz")])  # checked sütunu yok
    paths = []
    for name, df in [("a", a), ("b", b), ("pilot", pilot)]:
        paths.append(tmp_path / f"{name}.csv")
        df.to_csv(paths[-1], index=False)
    clean, report = load_labels(paths)
    assert sorted(clean["review_id"]) == ["r1", "r3", "r4"]
    assert report.files == 3 and report.unchecked_skipped == 1
    assert list(clean.columns) == ["review_id", "text", *ASPECTS]


def test_kappa_seti_ayni_etiket_bir_kez_farkli_etiket_kullanilmaz():
    df = pd.concat(
        [
            _export([_row("k1", kalite="olumlu"), _row("k2", fiyat="olumlu")], "ozan", "TRUE"),
            _export([_row("k1", kalite="olumlu"), _row("k2", fiyat="notr")], "salih", "TRUE"),
        ]
    )
    clean, report = check_labels(df)
    assert clean["review_id"].tolist() == ["k1"]
    assert report.kappa_agreed == 1 and report.conflicting_ids == ["k2"]
    assert not report.has_problems  # kappa anlaşmazlığı format hatası değildir


def test_ayni_etiketleyicide_tekrar_hala_hata():
    df = _export([_row("r1"), _row("r1")], "ozan", "TRUE")
    clean, report = check_labels(df)
    assert report.duplicate_ids == ["r1"] and len(clean) == 1
