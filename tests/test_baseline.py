import pytest

from absa.models.baseline import TfidfBaseline
from absa.schema import ASPECTS, LABELS

TEXTS = [
    "ürün çok kaliteli malzemesi sağlam",
    "kalitesi berbat iki günde bozuldu",
    "kargo çok hızlı geldi paketleme özenli",
    "kargo çok geç geldi kutu ezikti",
    "malzeme kaliteli kargo da hızlıydı",
    "ürün bozuk geldi kargo geç kaldı",
]


def _labels():
    labels = {aspect: ["yok"] * len(TEXTS) for aspect in ASPECTS}
    labels["kalite"] = ["olumlu", "olumsuz", "yok", "yok", "olumlu", "olumsuz"]
    labels["kargo_paketleme"] = ["yok", "yok", "olumlu", "olumsuz", "olumlu", "olumsuz"]
    return labels


def test_predict_returns_valid_labels_for_every_aspect():
    model = TfidfBaseline().fit(TEXTS, _labels())
    predictions = model.predict(["kargo hızlı ürün kaliteli", "kutu ezik geldi"])

    assert list(predictions) == ASPECTS
    for labels in predictions.values():
        assert len(labels) == 2
        assert all(label in LABELS for label in labels)


def test_single_label_aspect_predicts_that_label():
    model = TfidfBaseline().fit(TEXTS, _labels())
    assert model.predict(["herhangi bir yorum"])["fiyat"] == ["yok"]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        TfidfBaseline().predict(["yorum"])


def test_length_mismatch_raises():
    labels = _labels()
    labels["kalite"] = labels["kalite"][:-1]
    with pytest.raises(ValueError):
        TfidfBaseline().fit(TEXTS, labels)
