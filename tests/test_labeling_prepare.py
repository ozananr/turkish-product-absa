import pandas as pd
import pytest

from absa.labeling.prepare import SHEET_COLUMNS, assignment_markdown, main, prepare
from absa.labeling.validate import validate_labels
from absa.schema import ASPECTS

NAMES = ["ozan", "salih", "yusuf"]
KAPPA = ["ozan", "salih"]


def _prelabels(n=500):
    rows = []
    for i in range(n):
        labels = {a: "yok" for a in ASPECTS} | {"kalite": "olumlu" if i % 3 else "olumsuz"}
        rows.append({"review_id": f"r{i:04d}", "text": f"yorum {i}"} | labels)
    return pd.DataFrame(rows)


def _files(**kw):
    return prepare(_prelabels(), NAMES, KAPPA, kappa_size=50, **kw)


def _ids(df):
    return set(df["review_id"])


def test_batchler_ve_kappa_seti_cakismaz_ve_hepsini_kapsar():
    files = _files()
    batches = [files[f"batch_{n}.csv"] for n in NAMES]
    kappa = _ids(files["kappa_ozan.csv"])
    seen = set()
    for batch in batches:
        assert not _ids(batch) & seen and not _ids(batch) & kappa
        seen |= _ids(batch)
    assert seen | kappa == _ids(_prelabels())
    # 450 kalan + 2x50 kappa = 550 iş -> kişi başı 184/183/183; kappa yapanlardan 50 düşer
    assert [len(b) for b in batches] == [134, 133, 183]


def test_kappa_seti_iki_kisiye_ayni_ve_on_etiketsiz():
    files = _files()
    a, b = files["kappa_ozan.csv"], files["kappa_salih.csv"]
    assert len(a) == 50 and _ids(a) == _ids(b)
    assert (a[ASPECTS] == "").all().all()
    assert set(a["annotator"]) == {"ozan"} and set(b["annotator"]) == {"salih"}
    assert _ids(files["kappa_prelabels.csv"]) == _ids(a)


def test_batchler_on_etiketli_ve_yalnizca_izin_verilen_sutunlar():
    files = _files()
    batch = files["batch_yusuf.csv"]
    assert list(batch.columns) == SHEET_COLUMNS
    assert "rating" not in batch.columns and "label" not in batch.columns
    assert (batch["kalite"] != "").all()
    assert validate_labels(batch) == []


def test_pilot_yorumlari_kappa_setine_girmez():
    exclude = {f"r{i:04d}" for i in range(0, 500, 2)}  # yorumların yarısı
    files = _files(exclude_from_kappa=exclude)
    assert not _ids(files["kappa_ozan.csv"]) & exclude


def test_ayni_seed_ayni_dosyalar_satir_sirasi_onemsiz():
    a = prepare(_prelabels(), NAMES, KAPPA, kappa_size=50)
    shuffled = _prelabels().sample(frac=1, random_state=1).sort_values("review_id")
    b = prepare(shuffled, NAMES, KAPPA, kappa_size=50)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_kappa_etiketleyicileri_dogrulanir():
    with pytest.raises(ValueError):
        prepare(_prelabels(), NAMES, ["ozan", "ahmet"], kappa_size=50)
    with pytest.raises(ValueError):
        prepare(_prelabels(), NAMES, ["ozan"], kappa_size=50)


def test_main_dosyalari_yazar(tmp_path):
    pre = tmp_path / "prelabels.csv"
    _prelabels().to_csv(pre, index=False)
    pilot = tmp_path / "pilot_checked.csv"
    _prelabels().head(20).to_csv(pilot, index=False)
    out = tmp_path / "labeling"
    main(["--prelabels", str(pre), "--exclude-from-kappa", str(pilot), "--out", str(out),
          "--annotators", *NAMES, "--kappa-annotators", *KAPPA, "--kappa-size", "50"])  # fmt: skip
    written = {p.name for p in out.iterdir()}
    assert written == {
        "kappa_ozan.csv", "kappa_salih.csv", "kappa_prelabels.csv", "assignment.md",
        "batch_ozan.csv", "batch_salih.csv", "batch_yusuf.csv",
    }  # fmt: skip
    kappa = pd.read_csv(out / "kappa_ozan.csv", dtype=str)
    assert not set(kappa["review_id"]) & {f"r{i:04d}" for i in range(20)}
    assert "| `batch_yusuf.csv` | yusuf | 183 | yes |" in (out / "assignment.md").read_text()


def test_atama_tablosu():
    md = assignment_markdown(_files(), 42)
    assert "| `kappa_salih.csv` | salih | 50 | no (kappa set) |" in md


def test_toplam_is_herkese_esit():
    from absa.labeling.prepare import batch_sizes

    sizes = batch_sizes(2300, NAMES, KAPPA, 200)
    assert sizes == {"ozan": 700, "salih": 700, "yusuf": 900}
    assert sum(sizes.values()) == 2300


def test_batchler_karisik_sirada_kaynaga_gore_sirali_degil():
    # ID'ler kaynağa göre başlar; sıralı batch'in ilk yarısı tek kaynaktan olurdu.
    pre = _prelabels(600)
    pre["review_id"] = [("trendyol_" if i % 2 else "trsa_") + f"{i:04d}" for i in range(600)]
    files = prepare(pre, NAMES, KAPPA, kappa_size=50)
    for name in NAMES:
        batch = files[f"batch_{name}.csv"]
        first_half = batch.head(len(batch) // 2)["review_id"]
        assert first_half.str.startswith("trendyol_").mean() < 0.7
        assert first_half.str.startswith("trsa_").mean() < 0.7
