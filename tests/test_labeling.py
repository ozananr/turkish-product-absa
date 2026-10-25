import pandas as pd

from absa.labeling.validate import validate_labels
from absa.schema import ASPECTS


def _row(review_id=1, label="yok"):
    return {"review_id": review_id, "text": "x", **{a: label for a in ASPECTS}}


def test_valid_labels_pass():
    assert validate_labels(pd.DataFrame([_row(label="olumlu")])) == []


def test_invalid_label_is_reported():
    assert validate_labels(pd.DataFrame([_row(label="Olumlu")]))


def test_empty_label_is_reported():
    assert validate_labels(pd.DataFrame([_row(label="")]))


def test_duplicate_review_id_is_reported():
    assert validate_labels(pd.DataFrame([_row(1), _row(1)]))


def test_missing_column_is_reported():
    assert validate_labels(pd.DataFrame([{"review_id": 1, "text": "x"}]))
