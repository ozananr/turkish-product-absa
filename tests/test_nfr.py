"""NFR ölçüm komutu ve gizlilik (NFR-4) testleri. CI'da sahte, yerelde gerçek modelle çalışır."""

import logging

import pandas as pd
import pytest

pytest.importorskip("torch")

from absa import inference  # noqa: E402
from absa.nfr import load_texts, main, summarize, verdict  # noqa: E402

MARKER = "Gizli test yorumu 7f3a9c: kargo geç geldi"


def test_ozet_ve_karar():
    stats = summarize([0.1] * 90 + [3.0] * 10)
    assert stats["median_s"] == 0.1 and stats["max_s"] == 3.0
    assert verdict(10.0, stats) == {"NFR-1": False, "NFR-2": True}
    assert verdict(61.0, summarize([0.1] * 100)) == {"NFR-1": True, "NFR-2": False}


def test_metin_kaynagi(tmp_path):
    assert len(load_texts(None, 25)) == 25
    path = tmp_path / "labels.csv"
    pd.DataFrame({"text": ["bir", "iki", "üç"]}).to_csv(path, index=False)
    assert sorted(set(load_texts(path, 5))) == ["bir", "iki", "üç"]


def test_olcum_komutu(tmp_path):
    res = main(["--n", "5", "--warmup", "1", "--out", str(tmp_path)])
    assert res["predict"]["n"] == 5
    assert set(res["verdict"]) == {"NFR-1", "NFR-2"}
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "NFR-1" in report and "NFR-2" in report


def test_girdi_kaydedilmez_ve_loglanmaz(tmp_path, monkeypatch, caplog, capsys):
    """NFR-4: tahmin sırasında girdi dosyaya yazılmaz, loglara ve çıktıya düşmez."""
    inference.load()  # model yüklemesi (önbellek dosyaları) ölçümün dışında kalsın
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    with caplog.at_level(logging.DEBUG):
        inference.predict(MARKER)
    assert set(tmp_path.rglob("*")) == before
    assert MARKER not in caplog.text
    out = capsys.readouterr()
    assert MARKER not in out.out and MARKER not in out.err
