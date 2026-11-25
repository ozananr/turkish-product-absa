import pandas as pd
import pytest

from absa.labeling.adjudicate import (
    FINAL_COLUMNS,
    build_conflicts,
    finalize,
    load_export,
    main,
    prepare_export,
)
from absa.schema import ASPECTS, LABELS


def _export(rows, annotator="x", checked="TRUE"):
    base = {a: "yok" for a in ASPECTS}
    return pd.DataFrame(
        [
            {"review_id": rid, "text": f"metin {rid}"}
            | base
            | labels
            | {"annotator": annotator, "checked": checked, "note": ""}
            for rid, labels in rows
        ]
    )


def _pair():
    a = _export(
        [
            ("r1", {"kalite": "olumlu"}),
            ("r2", {"kalite": "olumlu", "fiyat": "olumsuz"}),
            ("r3", {}),
        ],
        "ozan",
    )
    b = _export(
        [
            ("r1", {"kalite": "olumlu"}),  # tamamen aynı
            ("r2", {"kalite": "olumsuz", "fiyat": "olumsuz"}),  # kalite anlaşmazlık
            ("r3", {"kargo_paketleme": "notr"}),  # kargo anlaşmazlık
        ],
        "salih",
    )
    return prepare_export(a), prepare_export(b)


def test_anlasmazliklar_bulunur():
    a, b = _pair()
    conflicts, summary = build_conflicts(a, b, "ozan", "salih")
    assert list(conflicts.columns) == ["review_id", "aspect", "ozan", "salih", "final", "text"]
    assert set(zip(conflicts["review_id"], conflicts["aspect"], strict=True)) == {
        ("r2", "kalite"),
        ("r3", "kargo_paketleme"),
    }
    assert (conflicts["final"] == "").all()
    assert summary["reviews"] == 3
    assert summary["reviews_fully_agreed"] == 1
    assert summary["conflict_cells"] == 2
    assert summary["by_aspect"]["kalite"] == 1


def test_finalize_uzlasilanlari_ve_kararlari_birlestirir():
    a, b = _pair()
    conflicts, _ = build_conflicts(a, b, "ozan", "salih")
    conflicts["final"] = ["olumsuz", "notr"]
    final = finalize(a, b, conflicts)

    assert list(final.columns) == FINAL_COLUMNS
    assert final["review_id"].is_unique and len(final) == 3
    assert (final["annotator"] == "final").all() and (final["checked"] == "TRUE").all()
    by_id = final.set_index("review_id")
    assert by_id.loc["r1", "kalite"] == "olumlu"  # otomatik uzlaşı
    assert by_id.loc["r2", "kalite"] == "olumsuz"  # karar
    assert by_id.loc["r2", "fiyat"] == "olumsuz"  # otomatik uzlaşı
    assert by_id.loc["r3", "kargo_paketleme"] == "notr"  # karar
    for aspect in ASPECTS:
        assert set(final[aspect]) <= set(LABELS)


def test_finalize_eksik_karar_hata_verir():
    a, b = _pair()
    conflicts, _ = build_conflicts(a, b, "ozan", "salih")
    conflicts["final"] = ["olumsuz", ""]  # ikincisi boş
    with pytest.raises(ValueError, match="r3 / kargo_paketleme"):
        finalize(a, b, conflicts)


def test_finalize_gecersiz_karar_hata_verir():
    a, b = _pair()
    conflicts, _ = build_conflicts(a, b, "ozan", "salih")
    conflicts["final"] = ["olumsuz", "belki"]
    with pytest.raises(ValueError, match="geçersiz"):
        finalize(a, b, conflicts)


def test_finalize_notr_yazimi_duzeltilir():
    a, b = _pair()
    conflicts, _ = build_conflicts(a, b, "ozan", "salih")
    conflicts["final"] = [" Olumsuz ", "nötr"]
    final = finalize(a, b, conflicts).set_index("review_id")
    assert final.loc["r2", "kalite"] == "olumsuz"
    assert final.loc["r3", "kargo_paketleme"] == "notr"


def test_finalize_anlasmazlik_olmayan_hucre_icin_karar_hata_verir():
    a, b = _pair()
    conflicts, _ = build_conflicts(a, b, "ozan", "salih")
    conflicts["final"] = ["olumsuz", "notr"]
    extra = pd.DataFrame([{"review_id": "r1", "aspect": "kalite", "final": "olumlu"}])
    decisions = pd.concat([conflicts, extra], ignore_index=True)
    with pytest.raises(ValueError, match="Anlaşmazlık olmayan"):
        finalize(a, b, decisions)


def test_farkli_yorum_kumeleri_hata_verir():
    a, b = _pair()
    with pytest.raises(ValueError, match="aynı yorumlar"):
        build_conflicts(a, b.iloc[:2], "ozan", "salih")


def test_isaretlenmemis_satirlar_atilir():
    df = pd.concat(
        [_export([("r1", {})]), _export([("r2", {})], checked="FALSE")], ignore_index=True
    )
    assert prepare_export(df)["review_id"].tolist() == ["r1"]


def test_gecersiz_etiket_ve_tekrar_hata_verir():
    with pytest.raises(ValueError, match="geçersiz etiket"):
        prepare_export(_export([("r1", {"kalite": "iyi"})]))
    with pytest.raises(ValueError, match="Tekrar"):
        prepare_export(_export([("r1", {}), ("r1", {})]))
    with pytest.raises(ValueError, match="Eksik sütun"):
        prepare_export(_export([("r1", {})]).drop(columns=["fiyat"]))


def test_cli_uctan_uca(tmp_path):
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    _export([("r1", {"kalite": "olumlu"}), ("r2", {"kalite": "olumlu"})], "ozan").to_csv(
        pa, index=False
    )
    _export([("r1", {"kalite": "olumlu"}), ("r2", {"kalite": "olumsuz"})], "salih").to_csv(
        pb, index=False
    )
    conflicts_path = tmp_path / "conflicts.csv"
    main(["conflicts", "--a", str(pa), "--b", str(pb), "--names", "ozan", "salih",
          "--out", str(conflicts_path)])  # fmt: skip
    conflicts = pd.read_csv(conflicts_path, dtype=str, keep_default_na=False)
    assert len(conflicts) == 1

    conflicts["final"] = "olumsuz"
    filled = tmp_path / "filled.csv"
    conflicts.to_csv(filled, index=False)
    out = tmp_path / "kappa_final.csv"
    main(["finalize", "--a", str(pa), "--b", str(pb), "--decisions", str(filled),
          "--out", str(out)])  # fmt: skip

    final = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(final) == 2 and set(final["annotator"]) == {"final"}
    assert final.set_index("review_id").loc["r2", "kalite"] == "olumsuz"
    assert load_export(out).shape[0] == 2  # tekrar okunabilir (checked=TRUE)
