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
