"""predict() arka uç testleri: küçük, rastgele bir BERT kaydedilip gerçek yükleme yolu denenir."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_berturk import _labels, tiny_factory  # noqa: E402

from absa import inference  # noqa: E402
from absa.models.berturk import TrainConfig, save_model, train  # noqa: E402
from absa.schema import ASPECTS, LABELS  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_backend():
    """Bu dosyadaki testler arka ucu değiştirir; önce ve sonra yüklü modeli bırak."""
    inference.reset()
    yield
    inference.reset()


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    df = _labels(30)
    parts = {
        "train": df.iloc[:24].reset_index(drop=True),
        "val": df.iloc[24:].reset_index(drop=True),
    }
    model, tokenizer, info = train(
        parts,
        TrainConfig(lr=1e-3, epochs=1, batch_size=8, max_length=32),  # küçük modelde 64 pozisyon
        factory=tiny_factory,
        log=lambda _: None,
    )
    path = tmp_path_factory.mktemp("model")
    save_model(model, tokenizer, path, info)
    return path


@pytest.fixture
def real_backend(tiny_model_dir, monkeypatch):
    monkeypatch.setenv("ABSA_MODEL_DIR", str(tiny_model_dir))
    monkeypatch.delenv("ABSA_BACKEND", raising=False)
    inference.reset()
    return tiny_model_dir


def test_model_yolu_ve_sozlesme(real_backend):
    assert inference.load() == "berturk"
    result = inference.predict("çok kaliteli ürün kargo hızlı")
    assert list(result) == ASPECTS
    for value in result.values():
        assert set(value) == {"label", "confidence"}
        assert value["label"] in LABELS
        assert isinstance(value["confidence"], float) and 0.0 <= value["confidence"] <= 1.0


def test_model_bir_kez_yuklenir(real_backend, monkeypatch):
    inference.predict("ürün bozuldu")
    calls = []
    monkeypatch.setattr(inference, "_load_backend", lambda: calls.append(1))
    inference.predict("çok kaliteli")
    inference.predict("kargo geç")
    assert calls == []


def test_ayni_girdi_ayni_cikti(real_backend):
    assert inference.predict("kargo geç geldi") == inference.predict("kargo geç geldi")


def test_kesme_bilgisi(real_backend):
    assert not inference.is_truncated("çok kaliteli ürün")
    assert inference.is_truncated("çok " * 300)
    assert inference.predict("çok " * 300).keys() == set(ASPECTS)


def test_model_yoksa_acik_hata(tmp_path, monkeypatch):
    monkeypatch.setenv("ABSA_MODEL_DIR", str(tmp_path / "yok"))
    monkeypatch.delenv("ABSA_BACKEND", raising=False)
    inference.reset()
    with pytest.raises(inference.ModelNotFoundError, match="ABSA_BACKEND=mock"):
        inference.predict("kargo geç")


def test_mock_arka_uc(monkeypatch):
    monkeypatch.setenv("ABSA_BACKEND", "mock")
    inference.reset()
    assert inference.load() == "mock"
    assert list(inference.predict("güzel ürün")) == ASPECTS
    assert not inference.is_truncated("güzel ürün")


def test_bos_ve_yanlis_tip_model_yuklemeden_hata(tmp_path, monkeypatch):
    monkeypatch.setenv("ABSA_MODEL_DIR", str(tmp_path / "yok"))
    monkeypatch.delenv("ABSA_BACKEND", raising=False)
    inference.reset()
    with pytest.raises(ValueError):
        inference.predict("   ")
    with pytest.raises(TypeError):
        inference.predict(None)


def test_kisisel_veri_maskelenir():
    assert "[EPOSTA]" in inference.preprocess("bana ali@ornek.com adresinden yazın")
