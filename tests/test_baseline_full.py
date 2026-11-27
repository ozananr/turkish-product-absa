import json

import numpy as np
import pandas as pd
import pytest

from absa.models.baseline_full import (
    error_table,
    load_split_data,
    main,
    render_report,
    review_types,
    run,
)
from absa.schema import ASPECTS

WORDS = {
    ("kalite", "olumlu"): "malzemesi çok kaliteli sağlam",
    ("kalite", "olumsuz"): "kalitesiz iki günde bozuldu",
    ("kargo_paketleme", "olumlu"): "kargo çok hızlı geldi",
    ("kargo_paketleme", "olumsuz"): "kargo çok geç geldi kutu ezik",
    ("fiyat", "olumlu"): "fiyatı çok uygun",
}


def _labels(n=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        labels = {a: "yok" for a in ASPECTS}
        parts = []
        for (aspect, label), words in WORDS.items():
            if rng.random() < 0.3 and labels[aspect] == "yok":
                labels[aspect] = label
                parts.append(words)
        text = " ".join(parts) or "güzel ürün"
        rows.append({"review_id": f"r{i:03d}", "text": text, "annotator": "x"} | labels)
    return pd.DataFrame(rows)


def _split(df):
    names = ["train"] * 100 + ["val"] * 25 + ["test"] * 25
    return pd.DataFrame({"review_id": df["review_id"], "split": names[: len(df)]})


def _files(tmp_path, df=None):
    df = _labels() if df is None else df
    labels, split = tmp_path / "final_labels.csv", tmp_path / "split_v1.csv"
    df.to_csv(labels, index=False)
    _split(df).to_csv(split, index=False)
    return labels, split


def test_bolme_dosyasi_ile_parcalar(tmp_path):
    parts = load_split_data(*_files(tmp_path))
    assert {k: len(v) for k, v in parts.items()} == {"train": 100, "val": 25, "test": 25}
    assert not set(parts["train"]["review_id"]) & set(parts["test"]["review_id"])


def test_etiket_dosyasinda_olmayan_yorum_hata_verir(tmp_path):
    labels, split = _files(tmp_path)
    s = pd.read_csv(split)
    s.loc[0, "review_id"] = "olmayan"
    s.to_csv(split, index=False)
    with pytest.raises(ValueError, match="etiket dosyasında yok"):
        load_split_data(labels, split)


def test_baseline_hep_yoktan_iyi_val_ve_test(tmp_path):
    results, errors = run(load_split_data(*_files(tmp_path)))
    for name in ("val", "test"):
        assert results[name]["baseline"]["macro_f1"] > results[name]["always_yok"]["macro_f1"]
    assert len(errors) == 25 and {"types", "n_errors", "errors"} <= set(errors.columns)


def test_yorum_turleri():
    gold = {a: "yok" for a in ASPECTS} | {"kalite": "olumlu", "kargo_paketleme": "olumsuz"}
    assert review_types("güzel ama kargo geç", gold) == ["short", "contrast", "mixed"]
    long_text = "bu ürün hiç beklediğim gibi değil maalesef iade edeceğim çok üzgünüm"
    assert review_types(long_text, {a: "yok" for a in ASPECTS} | {"fiyat": "notr"}) == [
        "negation",
        "notr",
    ]


def test_hata_tablosu():
    test = pd.DataFrame([{"review_id": "r1", "text": "kargo geç"} | {a: "yok" for a in ASPECTS}])
    pred = {a: ["yok"] for a in ASPECTS} | {"kargo_paketleme": ["olumsuz"]}
    row = error_table(test, pred).iloc[0]
    assert row["n_errors"] == 1 and row["errors"] == "kargo_paketleme: yok→olumsuz"


def test_rapor_ve_main(tmp_path):
    labels, split = _files(tmp_path)
    results, errors = run(load_split_data(labels, split))
    md = render_report(results, errors)
    assert "### Overall macro-F1" in md and "Confusion matrices" in md and "Review type" in md
    out = tmp_path / "out"
    main(["--labels", str(labels), "--split", str(split), "--out", str(out)])
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["sizes"] == {"train": 100, "val": 25, "test": 25}
    assert (out / "report.md").exists() and (out / "test_errors.csv").exists()
