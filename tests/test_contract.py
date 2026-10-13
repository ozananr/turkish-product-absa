"""Sözleşme testleri (docs/requirements.md bölüm 6). Gradio'ya bağlı değildir."""

import pytest

from absa.inference import predict
from absa.schema import ASPECTS, LABELS

SAMPLE = "Ürün çok kaliteli ama kargo çok geç geldi."


def test_schema_contents():
    assert ASPECTS == [
        "kalite",
        "fiyat",
        "kargo_paketleme",
        "satici_hizmet",
        "islev_performans",
        "gorunum_tasarim",
        "beden_uyum",
    ]
    assert LABELS == ["yok", "olumlu", "olumsuz", "notr"]


def test_keys_are_ascii_only():
    for key in [*ASPECTS, *LABELS]:
        assert key.isascii(), key


def test_all_aspects_present_in_order():
    assert list(predict(SAMPLE).keys()) == ASPECTS


def test_each_result_has_label_and_confidence_only():
    for aspect, value in predict(SAMPLE).items():
        assert set(value.keys()) == {"label", "confidence"}, aspect


def test_every_label_is_valid():
    for aspect, value in predict(SAMPLE).items():
        assert value["label"] in LABELS, aspect


def test_confidence_is_float_between_0_and_1():
    for aspect, value in predict(SAMPLE).items():
        assert isinstance(value["confidence"], float), aspect
        assert 0.0 <= value["confidence"] <= 1.0, aspect


def test_same_input_returns_same_output():
    assert predict(SAMPLE) == predict(SAMPLE)


def test_turkish_characters_and_emoji_do_not_fail():
    assert list(predict("Çok güzel, ığdır şöyle böyle 😊👍").keys()) == ASPECTS


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_input_raises_value_error(bad):
    with pytest.raises(ValueError):
        predict(bad)


@pytest.mark.parametrize("bad", [None, 123, ["yorum"], b"yorum"])
def test_non_str_input_raises_type_error(bad):
    with pytest.raises(TypeError):
        predict(bad)
