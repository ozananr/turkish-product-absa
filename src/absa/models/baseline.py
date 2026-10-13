"""TF-IDF + Logistic Regression baseline: one classifier per aspect."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from absa.schema import ASPECTS


class TfidfBaseline:
    """Trains a separate TF-IDF + Logistic Regression pipeline for each aspect.

    If an aspect has only one label in the training data (e.g. always "yok"),
    that label is predicted for every review instead of training a classifier.
    """

    def __init__(self, aspects: Sequence[str] = ASPECTS, random_state: int = 42) -> None:
        self.aspects = list(aspects)
        self.random_state = random_state
        self.models: dict[str, Pipeline | str] = {}

    def _new_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=self.random_state
                    ),
                ),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Mapping[str, Sequence[str]]) -> TfidfBaseline:
        """Train one model per aspect. `labels` maps each aspect to one label per text."""
        texts = list(texts)
        for aspect in self.aspects:
            y = list(labels[aspect])
            if len(y) != len(texts):
                raise ValueError(f"{aspect}: {len(texts)} texts but {len(y)} labels")
            if len(set(y)) == 1:
                self.models[aspect] = y[0]
                continue
            model = self._new_pipeline()
            model.fit(texts, y)
            self.models[aspect] = model
        return self

    def predict(self, texts: Sequence[str]) -> dict[str, list[str]]:
        """Return predicted labels for each aspect, in the same format as `fit` labels."""
        if not self.models:
            raise RuntimeError("Call fit() before predict().")
        texts = list(texts)
        predictions = {}
        for aspect in self.aspects:
            model = self.models[aspect]
            if isinstance(model, str):
                predictions[aspect] = [model] * len(texts)
            else:
                predictions[aspect] = [str(label) for label in model.predict(texts)]
        return predictions
