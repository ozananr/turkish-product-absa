import json

import numpy as np
import pandas as pd

from absa.data.labels import check_labels, class_distribution
from absa.models.first_run import main, render_report, run, split, split_warnings
from absa.schema import ASPECTS

WORDS = {
    ("kalite", "olumlu"): "malzemesi çok kaliteli sağlam",
    ("kalite", "olumsuz"): "kalitesiz iki günde bozuldu",
    ("kargo_paketleme", "olumlu"): "kargo çok hızlı geldi",
    ("kargo_paketleme", "olumsuz"): "kargo çok geç geldi kutu ezik",
    ("fiyat", "olumlu"): "fiyatı çok uygun",
}


def _labels(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        labels = {a: "yok" for a in ASPECTS}
        parts = []
        for (aspect, label), words in WORDS.items():
            if rng.random() < 0.3 and labels[aspect] == "yok":
                labels[aspect] = label
                parts.append(words)
        rows.append({"review_id": f"r{i:03d}", "text": " ".join(parts) or "güzel ürün"} | labels)
    return pd.DataFrame(rows)


def test_bolme_sabit_seed_ile_tekrarlanabilir():
    df = _labels()
    a_train, a_test = split(df, 0.2, 42)
    b_train, b_test = split(df, 0.2, 42)
    assert len(a_test) == 24 and len(a_train) == 96
    assert a_test["review_id"].tolist() == b_test["review_id"].tolist()
    assert not set(a_train["review_id"]) & set(a_test["review_id"])


def test_uctan_uca_calisir_ve_baseline_hep_yoktan_iyi():
    results = run(_labels())
    assert set(results["baseline"]["per_aspect"]) == set(ASPECTS)
    assert results["baseline"]["macro_f1"] > results["always_yok"]["macro_f1"]
    # her zaman 'yok' olan aspect için uyarı verilir
    assert any("`beden_uyum`" in w for w in results["warnings"])


def test_train_de_olmayan_test_etiketi_uyarilir():
    train = pd.DataFrame({a: ["yok", "olumlu"] for a in ASPECTS})
    test = pd.DataFrame({a: ["notr"] for a in ASPECTS})
    warnings = split_warnings(train, test)
    assert any("test'te var, train'de yok: ['notr']" in w for w in warnings)


def test_rapor_problemleri_ve_tabloyu_icerir():
    df = _labels()
    df.loc[0, "fiyat"] = "olumluu"
    clean, report = check_labels(df)
    md = render_report(report, class_distribution(clean), len(clean), run(clean))
    assert "olumluu" in md and "| **overall** |" in md and "not results" in md


def test_main_dosyalari_yazar(tmp_path):
    path = tmp_path / "labels.csv"
    _labels().to_csv(path, index=False)
    out = tmp_path / "out"
    main(["--labels", str(path), "--out", str(out)])
    assert (out / "report.md").read_text(encoding="utf-8").startswith("> Pipeline check")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_test"] == 24 and "always_yok" in metrics
