import pandas as pd
import pytest

from absa.data.split import assign_splits, main, missing_classes, render_report, split_counts
from absa.schema import ASPECTS

WORDS = [
    "elma",
    "kitap",
    "masa",
    "kalem",
    "bulut",
    "deniz",
    "orman",
    "yol",
    "ses",
    "renk",
    "kapı",
    "taş",
    "çay",
    "ışık",
    "gece",
    "sabah",
    "kum",
    "kar",
    "rüzgar",
    "dağ",
]


def _text(i):
    """dedup_key rakamları saymadığı için her yorum farklı kelimelerden oluşur."""
    return f"{WORDS[i % 20]} {WORDS[(i // 20) % 20]} {WORDS[(i // 400) % 20]} güzel"


def _labels(n=60):
    rows = []
    for i in range(n):
        labels = {a: "yok" for a in ASPECTS}
        labels["kalite"] = ["olumlu", "olumsuz", "yok"][i % 3]
        rows.append({"review_id": f"r{i:03d}", "text": _text(i)} | labels)
    return pd.DataFrame(rows)


TEST_IDS = {f"r{i:03d}" for i in range(10)}


def test_ayni_seed_ayni_bolme():
    df = _labels()
    a, _ = assign_splits(df, TEST_IDS)
    b, _ = assign_splits(df, TEST_IDS)
    assert a.equals(b)


def test_her_yorum_tek_parcada_ve_test_kappa_seti():
    splits, dropped = assign_splits(_labels(), TEST_IDS)
    assert splits["review_id"].is_unique
    assert set(splits.loc[splits["split"] == "test", "review_id"]) == TEST_IDS
    assert len(splits) == 60 and dropped == []


def test_val_orani_yaklasik():
    splits, _ = assign_splits(_labels(200), TEST_IDS, val_size=0.15)
    pool = (splits["split"] != "test").sum()
    assert abs((splits["split"] == "val").sum() / pool - 0.15) < 0.02


def test_neredeyse_ayni_yorumlar_ayni_parcada():
    df = _labels()
    df.loc[20, "text"] = "Güzel ürün, çok beğendim!"
    df.loc[21, "text"] = "güzel ürün çok beğendim"
    splits, _ = assign_splits(df, TEST_IDS)
    s = splits.set_index("review_id")["split"]
    assert s["r020"] == s["r021"]


def test_test_yorumunun_kopyasi_havuzdan_cikarilir():
    df = _labels()
    df.loc[30, "text"] = df.loc[0, "text"].upper() + "!!"
    splits, dropped = assign_splits(df, TEST_IDS)
    assert dropped == ["r030"]
    assert "r030" not in set(splits["review_id"])


def test_olmayan_test_id_hata_verir():
    with pytest.raises(ValueError, match="etiket dosyasında yok"):
        assign_splits(_labels(), TEST_IDS | {"yok_boyle_bir_id"})


def test_eksik_sinif_uyarisi_ve_rapor():
    df = _labels()
    df["fiyat"] = "yok"
    df.loc[40, "fiyat"] = "notr"  # yalnızca bir örnek: val ya da test'te olamaz
    splits, dropped = assign_splits(df, TEST_IDS)
    counts = split_counts(df, splits)
    warnings = missing_classes(counts)
    assert any("`fiyat` / `notr`" in w for w in warnings)
    report = render_report(splits, dropped, counts)
    assert "| test | 10 |" in report and "`kalite`" in report


def test_cli(tmp_path):
    df = _labels()
    labels = tmp_path / "final_labels.csv"
    kappa = tmp_path / "kappa_final.csv"
    out = tmp_path / "splits" / "split_v1.csv"
    df.to_csv(labels, index=False)
    df[df["review_id"].isin(TEST_IDS)].to_csv(kappa, index=False)
    main(["--labels", str(labels), "--test-ids", str(kappa), "--out", str(out)])
    written = pd.read_csv(out)
    assert list(written.columns) == ["review_id", "split"]
    assert set(written["split"]) == {"train", "val", "test"}
    assert (tmp_path / "splits" / "split_v1_report.md").exists()
