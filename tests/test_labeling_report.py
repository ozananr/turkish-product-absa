import pandas as pd

from absa.labeling.report import RARE_THRESHOLD, build_report, render_distribution, split_final
from absa.schema import ASPECTS


def _df(rows):
    base = {a: "yok" for a in ASPECTS}
    return pd.DataFrame(
        [{"review_id": rid, "text": f"metin {rid}"} | base | labels for rid, labels in rows]
    )


def _data():
    # b1..b3: batch yorumları, k1..k2: kappa seti
    prelabels = _df(
        [
            ("b1", {"kalite": "olumlu"}),
            ("b2", {"kalite": "olumlu", "fiyat": "olumlu"}),
            ("b3", {"kargo_paketleme": "olumsuz"}),
            ("k1", {"kalite": "olumlu"}),
            ("k2", {"fiyat": "olumlu"}),
        ]
    )
    final = _df(
        [
            ("b1", {"kalite": "olumlu"}),
            ("b2", {"kalite": "olumlu"}),  # fiyat LLM fazladan
            ("b3", {"kargo_paketleme": "olumsuz"}),
            ("k1", {"kalite": "olumsuz"}),  # kappa seti
            ("k2", {"fiyat": "olumlu"}),
        ]
    )
    kappa_final = final[final["review_id"].isin(["k1", "k2"])].reset_index(drop=True)
    kappa_prelabels = prelabels[prelabels["review_id"].isin(["k1", "k2"])].reset_index(drop=True)
    return prelabels, final, kappa_prelabels, kappa_final


def test_split_final_kappa_yorumlarini_cikarir():
    _, final, _, kappa_final = _data()
    batches = split_final(final, kappa_final)
    assert batches["review_id"].tolist() == ["b1", "b2", "b3"]


def test_rapor_iki_ayri_karsilastirma_yapar():
    prelabels, final, kappa_prelabels, kappa_final = _data()
    text, diff_batch, diff_kappa = build_report(prelabels, final, kappa_prelabels, kappa_final)

    # Batch: yalnızca b1..b3 karşılaştırılır; tek fark b2/fiyat (LLM fazladan)
    assert "Reviews compared:** 3" in text
    assert len(diff_batch) == 1 and diff_batch.iloc[0]["aspect"] == "fiyat"
    assert set(diff_batch["review_id"]) <= {"b1", "b2", "b3"}
    # Kappa seti: yalnızca k1, k2; tek fark k1/kalite (duygu)
    assert "Reviews compared:** 2" in text
    assert len(diff_kappa) == 1 and diff_kappa.iloc[0]["review_id"] == "k1"
    assert "anchoring" in text
    assert "## 3. Final label distribution" in text


def test_dagilim_ve_nadir_siniflar():
    _, final, _, _ = _data()
    text = render_distribution(final)
    assert "- **Reviews:** 5" in text
    assert f"fewer than {RARE_THRESHOLD} examples" in text
    # 5 yorumda hiçbir sınıf 30'a ulaşmaz: her aspect için en az bir nadir sınıf listelenir
    assert "`kalite` / `olumlu`: 2" in text
    assert "`beden_uyum` / `olumlu`: 0" in text
