"""Shared evaluation for all models (baseline, BERTurk, ...).

Every model is evaluated with the same function, so their results can be
compared directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from absa.schema import ASPECTS, LABELS


def evaluate(
    y_true: Mapping[str, Sequence[str]],
    y_pred: Mapping[str, Sequence[str]],
    aspects: Sequence[str] = ASPECTS,
    labels: Sequence[str] = LABELS,
) -> dict:
    """Compute per-aspect precision, recall, F1, macro-F1 and confusion matrices.

    Args:
        y_true: Gold labels, one list per aspect, e.g. {"kalite": ["olumlu", "yok"], ...}.
        y_pred: Predicted labels in the same format.
        aspects: Aspects to evaluate (defaults to all aspects in the schema).
        labels: Label order used for per-label scores and confusion matrices.

    Returns:
        {"macro_f1": float, "per_aspect": {aspect: {...}}}. The overall macro-F1 is the
        mean of the per-aspect macro-F1 scores. Per-aspect macro-F1 only averages over
        labels that appear in the gold or predicted labels of that aspect.
    """
    labels = list(labels)
    per_aspect = {}

    for aspect in aspects:
        true, pred = list(y_true[aspect]), list(y_pred[aspect])
        if len(true) != len(pred):
            raise ValueError(f"{aspect}: {len(true)} gold labels but {len(pred)} predictions")

        precision, recall, f1, support = precision_recall_fscore_support(
            true, pred, labels=labels, average=None, zero_division=0
        )
        per_aspect[aspect] = {
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "per_label": {
                label: {
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "support": int(s),
                }
                for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
            },
            "confusion_matrix": confusion_matrix(true, pred, labels=labels).tolist(),
        }

    return {
        "macro_f1": fmean(m["macro_f1"] for m in per_aspect.values()),
        "per_aspect": per_aspect,
    }


def format_report(results: dict) -> str:
    """Return a short text table of macro-F1 per aspect and overall."""
    lines = [f"{'aspect':<20}{'macro-F1':>10}"]
    for aspect, metrics in results["per_aspect"].items():
        lines.append(f"{aspect:<20}{metrics['macro_f1']:>10.3f}")
    lines.append(f"{'overall':<20}{results['macro_f1']:>10.3f}")
    return "\n".join(lines)
