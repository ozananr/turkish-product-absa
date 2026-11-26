import pandas as pd

from absa.labeling.report import (
    RARE_THRESHOLD,
    annotator_change_rates,
    build_report,
    load_batch_export,
    main,
    render_distribution,
    split_final,
)
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


def test_etiketleyici_degisiklik_orani():
    prelabels, _, _, _ = _data()
    exports = {
        "a": _df([("b1", {"kalite": "olumlu"}), ("b2", {"kalite": "olumlu"})]),  # b2/fiyat değişti
        "b": _df([("b3", {"kargo_paketleme": "olumsuz"})]),  # değişiklik yok
    }
    table = annotator_change_rates(prelabels, exports).set_index("annotator")
    assert table.loc["a", "reviews"] == 2
    assert table.loc["a", "cells"] == 2 * len(ASPECTS)
    assert table.loc["a", "cells_changed"] == 1
    assert table.loc["a", "reviews_changed"] == 1
    assert table.loc["b", "cells_changed"] == 0
    assert table.loc["b", "pct_cells_changed"] == 0.0
    assert table.loc["total", "reviews"] == 3
    assert table.loc["total", "cells_changed"] == 1
    assert abs(table.loc["total", "pct_cells_changed"] - 1 / (3 * len(ASPECTS))) < 1e-9


def test_prelabels_disindaki_yorumlar_sayilmaz():
    prelabels, _, _, _ = _data()
    table = annotator_change_rates(prelabels, {"a": _df([("b1", {}), ("yok_boyle", {})])})
    assert table.set_index("annotator").loc["a", "reviews"] == 1


def test_rapor_dorduncu_bolum_yalnizca_exports_ile():
    prelabels, final, kappa_prelabels, kappa_final = _data()
    text, _, _ = build_report(prelabels, final, kappa_prelabels, kappa_final)
    assert "## 4." not in text
    exports = {"a": _df([("b2", {"kalite": "olumlu"})])}
    text, _, _ = build_report(prelabels, final, kappa_prelabels, kappa_final, exports)
    assert "## 4. Changes per annotator" in text
    assert "| a | 1 | 1 | 1 / 7 |" in text
    assert "| **total** |" in text


def _export_csv(path, rows, annotator, checked="TRUE"):
    df = _df(rows)
    df["annotator"] = annotator
    df["checked"] = checked
    df.to_csv(path, index=False)


def test_load_batch_export_ad_ve_checked(tmp_path):
    p = tmp_path / "export_xyz.csv"
    _export_csv(p, [("b1", {}), ("b2", {})], "salih")
    name, df = load_batch_export(p)
    assert name == "salih" and len(df) == 2

    q = tmp_path / "export_yusuf.csv"
    _export_csv(q, [("b1", {})], "", checked="FALSE")
    name, df = load_batch_export(q)
    assert name == "yusuf" and len(df) == 0  # ad dosya adından, işaretsiz satır atılır


def test_cli_exports_ile(tmp_path):
    prelabels, final, kappa_prelabels, kappa_final = _data()
    paths = {}
    for name, df in {
        "prelabels": prelabels,
        "final": final,
        "kappa_prelabels": kappa_prelabels,
        "kappa_final": kappa_final,
    }.items():
        paths[name] = tmp_path / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    exp = tmp_path / "export_a.csv"
    _export_csv(exp, [("b1", {"kalite": "olumlu"}), ("b2", {"kalite": "olumlu"})], "a")
    out = tmp_path / "out"
    main(
        ["--prelabels", str(paths["prelabels"]), "--final", str(paths["final"]),
         "--kappa-prelabels", str(paths["kappa_prelabels"]),
         "--kappa-final", str(paths["kappa_final"]), "--exports", str(exp), "--out", str(out)]
    )  # fmt: skip
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## 4. Changes per annotator" in report
    assert "| a | 2 | 1 | 1 / 14 |" in report
