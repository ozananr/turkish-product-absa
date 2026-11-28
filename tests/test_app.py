"""Arayüz (app.py) testleri: analyze() tarayıcı olmadan doğrudan çağrılır (FR-4, FR-6, FR-7).

Sahte (mock) arka uç kullanılır; mock'ta token sayısı = kelime sayısı + 2, sınır 128.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("gradio")

from absa import inference  # noqa: E402
from absa.schema import ASPECT_DISPLAY  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("ABSA_BACKEND", "mock")
    inference.reset()
    spec = importlib.util.spec_from_file_location("absa_app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    inference.reset()


def test_kisa_yorumda_kesme_notu_yok(app):
    table, hidden, note = app.analyze("Ürün çok kaliteli ama kargo geç geldi.")
    assert note == ""
    assert list(table.columns) == app.HEADERS


def test_uzun_yorumda_kesme_notu_gosterilir(app):
    long_text = "kargo " * 200  # 202 mock token > 128
    table, hidden, note = app.analyze(long_text)
    assert note == app.TRUNCATION_NOTE
    assert "128" in note
    assert list(table.columns) == app.HEADERS  # sonuç yine gösterilir


def test_sinirda_kesme_yok(app):
    assert app.truncation_note("a " * 126) == ""  # 126 + 2 = 128 token, kesilmez
    assert app.truncation_note("a " * 127) == app.TRUNCATION_NOTE  # 129 token


@pytest.mark.filterwarnings("ignore:Lütfen analiz etmek")  # gr.Warning arayüz dışında uyarı verir
@pytest.mark.parametrize("bad", ["", "   ", None])
def test_bos_girdide_sonuc_ve_not_yok(app, bad):
    table, hidden, note = app.analyze(bad)
    assert table.empty and hidden == "" and note == ""


def test_yok_olanlar_tablo_disinda(app):
    table, hidden, _ = app.analyze("Fiyatına göre gayet iyi, görüntüsü de resimdeki gibi.")
    shown = set(table["Özellik"])
    hidden_names = set() if hidden == "Yok" else set(hidden.split(", "))
    assert shown.isdisjoint(hidden_names)
    assert shown | hidden_names == set(ASPECT_DISPLAY.values())
