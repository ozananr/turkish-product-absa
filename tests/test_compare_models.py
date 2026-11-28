"""Karşılaştırma kodu testleri: küçük, rastgele bir BERT kaydedilip yüklenir (indirmesiz)."""

import json

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_berturk import _labels, tiny_factory  # noqa: E402

from absa.models.berturk import TrainConfig, save_model, train  # noqa: E402
from absa.models.compare_models import (  # noqa: E402
    cell_changes,
    comparison_table,
    error_kinds,
    main,
)
from absa.schema import ASPECTS  # noqa: E402


def _errors(rows):
    return pd.DataFrame(
        [
            {"review_id": rid, "text": "t", "types": "", "n_errors": len(e), "errors": "; ".join(e)}
            for rid, e in rows
        ]
    )


def test_hata_turleri():
    err = _errors(
        [
            ("r1", ["kalite: yok→olumlu", "fiyat: olumsuz→yok"]),
            ("r2", ["kargo_paketleme: olumlu→olumsuz"]),
            ("r3", []),
        ]
    )
    assert error_kinds(err) == {"missed": 1, "extra": 1, "sentiment": 1}


def test_hucre_degisimleri_ve_tablo():
    base = _errors([("r1", ["kalite: yok→olumlu", "fiyat: olumsuz→yok"]), ("r2", [])])
    bert = _errors([("r1", ["fiyat: olumsuz→yok"]), ("r2", ["beden_uyum: yok→olumsuz"])])
    assert cell_changes(base, bert) == {"fixed": 1, "new": 1, "both": 1}
    table = comparison_table(base, bert)
    assert list(table["review_id"]) == ["r1", "r2"]
    assert {"errors_tfidf", "errors_berturk", "delta"} <= set(table.columns)


def test_cli_karsilastirma(tmp_path):
    df = _labels(60)
    labels, split = tmp_path / "labels.csv", tmp_path / "split.csv"
    df.to_csv(labels, index=False)
    pd.DataFrame(
        {"review_id": df["review_id"], "split": ["train"] * 40 + ["val"] * 10 + ["test"] * 10}
    ).to_csv(split, index=False)
    parts = {
        "train": df.iloc[:40].reset_index(drop=True),
        "val": df.iloc[40:50].reset_index(drop=True),
    }
    model, tokenizer, info = train(
        parts,
        TrainConfig(lr=1e-3, epochs=1, batch_size=8),
        factory=tiny_factory,
        log=lambda _: None,
    )
    save_model(model, tokenizer, tmp_path / "model", info)

    out = tmp_path / "out"
    main(
        [
            "--labels",
            str(labels),
            "--split",
            str(split),
            "--model",
            str(tmp_path / "model"),
            "--out",
            str(out),
        ]
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "**BERTurk**" in report and "TF-IDF + LR (#54)" in report and "Review type" in report
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"tfidf", "berturk"}
    assert set(metrics["berturk"]["test"]["per_aspect"]) == set(ASPECTS)
    assert len(pd.read_csv(out / "test_comparison.csv")) == 10
