import pytest

from absa.models.evaluate import evaluate, format_report
from absa.schema import ASPECTS, LABELS


def _same_for_all_aspects(labels):
    return {aspect: list(labels) for aspect in ASPECTS}


def test_perfect_predictions_score_one():
    gold = _same_for_all_aspects(["olumlu", "olumsuz", "notr", "yok"])
    results = evaluate(gold, gold)
    assert results["macro_f1"] == pytest.approx(1.0)
    for metrics in results["per_aspect"].values():
        assert metrics["macro_f1"] == pytest.approx(1.0)


def test_known_example():
    # olumlu: precision 2/3, recall 1 -> F1 0.8; olumsuz: F1 0; yok: F1 1
    gold = _same_for_all_aspects(["olumlu", "olumsuz", "olumlu", "yok"])
    pred = _same_for_all_aspects(["olumlu", "olumlu", "olumlu", "yok"])
    results = evaluate(gold, pred)

    kalite = results["per_aspect"]["kalite"]
    assert kalite["per_label"]["olumlu"]["f1"] == pytest.approx(0.8)
    assert kalite["per_label"]["olumsuz"]["f1"] == pytest.approx(0.0)
    assert kalite["macro_f1"] == pytest.approx((0.8 + 0.0 + 1.0) / 3)
    assert results["macro_f1"] == pytest.approx((0.8 + 0.0 + 1.0) / 3)


def test_confusion_matrix_uses_label_order():
    gold = _same_for_all_aspects(["olumlu", "olumsuz"])
    pred = _same_for_all_aspects(["olumlu", "olumlu"])
    matrix = evaluate(gold, pred)["per_aspect"]["kalite"]["confusion_matrix"]

    assert len(matrix) == len(LABELS)
    row, col = LABELS.index("olumsuz"), LABELS.index("olumlu")
    assert matrix[row][col] == 1


def test_length_mismatch_raises():
    gold = _same_for_all_aspects(["olumlu", "yok"])
    pred = _same_for_all_aspects(["olumlu"])
    with pytest.raises(ValueError):
        evaluate(gold, pred)


def test_format_report_lists_every_aspect():
    gold = _same_for_all_aspects(["olumlu", "yok"])
    report = format_report(evaluate(gold, gold))
    for aspect in ASPECTS:
        assert aspect in report
    assert "overall" in report
