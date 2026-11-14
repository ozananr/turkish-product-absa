import pandas as pd
import pytest

from absa.labeling.compare import compare, render_markdown
from absa.schema import ASPECTS


def _df(rows):
    base = {a: "yok" for a in ASPECTS}
    return pd.DataFrame(
        [{"review_id": rid, "text": f"metin {rid}"} | base | labels for rid, labels in rows]
    )


def test_karsilastirma_metrikleri():
    pred = _df(
        [
            ("r1", {"kalite": "olumlu"}),
            ("r2", {"kalite": "olumlu", "fiyat": "olumlu"}),  # fiyat fazladan
            ("r3", {}),  # kargo kaçırıldı
            ("r4", {"kalite": "olumsuz"}),  # duygu yanlış
        ]
    )
    gold = _df(
        [
            ("r1", {"kalite": "olumlu"}),
            ("r2", {"kalite": "olumlu"}),
            ("r3", {"kargo_paketleme": "olumsuz"}),
            ("r4", {"kalite": "olumlu"}),
            ("r5", {}),  # pred'de yok, karşılaştırılmaz
        ]
    )
    summary, diff = compare(pred, gold)
    assert summary["reviews"] == 4
    assert summary["review_exact_match"] == 0.25
    assert summary["per_aspect"]["fiyat"]["extra"] == 1
    assert summary["per_aspect"]["kargo_paketleme"]["missed"] == 1
    assert summary["per_aspect"]["kalite"]["wrong_sentiment"] == 1
    assert len(diff) == 3 and set(diff["aspect"]) == {"fiyat", "kargo_paketleme", "kalite"}
    assert "kargo_paketleme" in render_markdown(summary)


def test_ortak_id_yoksa_hata():
    with pytest.raises(ValueError):
        compare(_df([("a", {})]), _df([("b", {})]))


def test_genel_kappa_aspect_ortalamasidir():
    # kalite: tam uyum (kappa 1.0). fiyat: tam uyumsuz (kappa < 0). Diğerleri hep yok (-).
    pred = _df(
        [
            ("r1", {"kalite": "olumlu", "fiyat": "olumlu"}),
            ("r2", {"kalite": "olumsuz", "fiyat": "olumsuz"}),
            ("r3", {"kalite": "yok", "fiyat": "olumlu"}),
            ("r4", {"kalite": "olumlu", "fiyat": "olumsuz"}),
        ]
    )
    gold = _df(
        [
            ("r1", {"kalite": "olumlu", "fiyat": "olumsuz"}),
            ("r2", {"kalite": "olumsuz", "fiyat": "olumlu"}),
            ("r3", {"kalite": "yok", "fiyat": "olumsuz"}),
            ("r4", {"kalite": "olumlu", "fiyat": "olumlu"}),
        ]
    )
    summary, _ = compare(pred, gold)
    per = summary["per_aspect"]
    assert per["kalite"]["kappa"] == 1.0
    assert per["fiyat"]["kappa"] < 0
    assert per["beden_uyum"]["kappa"] is None  # iki dosyada da hep yok
    assert summary["kappa_defined"] == 2
    assert summary["mean_kappa"] == round((per["kalite"]["kappa"] + per["fiyat"]["kappa"]) / 2, 3)
    assert "Mean kappa" in render_markdown(summary)


def test_genel_kappa_hic_yoksa_tire():
    same = _df([("r1", {}), ("r2", {})])
    summary, _ = compare(same, same)
    assert summary["mean_kappa"] is None and summary["kappa_defined"] == 0
    assert "Mean kappa (mean over aspects with a defined kappa):** -" in render_markdown(summary)
