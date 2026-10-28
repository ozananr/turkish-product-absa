"""Baseline'ı ilk doğrulanmış etiketlerle uçtan uca çalıştırır (issue #33).

Amaç skoru değil, hattın çalıştığını görmek: etiketler -> baseline -> metrikler.
Bu sayılar final raporda kullanılmaz.

Kullanım (bir ya da birden fazla export; checked sütunu varsa yalnızca TRUE satırlar):
    uv run python -m absa.models.first_run --labels \
        data/interim/labeling/export_ozan.csv data/interim/pilot_checked.csv

Çıktılar (--out, varsayılan outputs/baseline_first_run/, git'e girmez):
    report.md     issue'ya yapıştırılacak rapor
    metrics.json  tüm metrikler (aspect başına precision/recall/F1, confusion matrix)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from absa.data.labels import LabelReport, class_distribution, load_labels, rare_classes
from absa.models.baseline import TfidfBaseline
from absa.models.evaluate import evaluate
from absa.schema import ASPECTS, LABELS, NO_MENTION

DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42
RARE_MIN = 5  # bundan az örneği olan sınıflar raporda "nadir" olarak listelenir


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sabit seed ile rastgele train/test bölmesi.

    7 aspect birlikte olduğu için tabakalı bölme yapılmaz; bunun yerine train'de olmayan
    test etiketleri `split_warnings` ile raporlanır.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_warnings(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    warnings = []
    for aspect in ASPECTS:
        train_labels = set(train[aspect])
        if len(train_labels) == 1:
            warnings.append(
                f"`{aspect}`: train'de yalnızca `{next(iter(train_labels))}` var, "
                "model hep bu etiketi tahmin eder"
            )
        unseen = sorted(set(test[aspect]) - train_labels)
        if unseen:
            warnings.append(f"`{aspect}`: test'te var, train'de yok: {unseen}")
    return warnings


def run(df: pd.DataFrame, test_size: float = DEFAULT_TEST_SIZE, seed: int = DEFAULT_SEED) -> dict:
    """Böl, eğit, değerlendir. Karşılaştırma için 'hep yok' tahmini de değerlendirilir."""
    train, test = split(df, test_size, seed)
    model = TfidfBaseline(random_state=seed).fit(
        train["text"], {a: train[a].tolist() for a in ASPECTS}
    )
    y_true = {a: test[a].tolist() for a in ASPECTS}
    y_pred = model.predict(test["text"])
    always_yok = {a: [NO_MENTION] * len(test) for a in ASPECTS}
    return {
        "n_train": len(train),
        "n_test": len(test),
        "seed": seed,
        "test_size": test_size,
        "warnings": split_warnings(train, test),
        "baseline": evaluate(y_true, y_pred),
        "always_yok": evaluate(y_true, always_yok),
    }


def _f1_cell(scores: dict) -> str:
    """Test'te hiç örneği olmayan etiket için F1 anlamsızdır: '-' gösterilir."""
    return "-" if scores["support"] == 0 else f"{scores['f1']:.2f}"


def render_report(report: LabelReport, dist: pd.DataFrame, n_clean: int, results: dict) -> str:
    lines = [
        "> Pipeline check only: these numbers are not results and must not be used in the "
        "final report.",
        "",
        "### Data",
        f"- Files: {report.files}. Checked rows: {report.total_rows} "
        f"(not checked yet, skipped: {report.unchecked_skipped}), usable reviews: {n_clean}, "
        f"dropped: {len(report.dropped_ids)}, auto-fixed labels: {report.auto_fixed}",
    ]
    if report.kappa_agreed or report.conflicting_ids:
        lines.append(
            f"- Kappa set: {report.kappa_agreed} reviews labeled the same by both annotators "
            f"(used once), {len(report.conflicting_ids)} with different labels "
            "(not used until adjudicated)"
        )
    problems = [
        ("Missing columns", report.missing_columns),
        ("Empty text", report.empty_text),
        ("Duplicate review_id", report.duplicate_ids),
        ("Empty labels", report.empty_labels),
        ("Invalid labels (typos)", report.invalid_labels),
    ]
    for name, value in problems:
        if value:
            lines.append(f"- **{name}:** {value}")
    if not report.has_problems:
        lines.append("- No format problems found.")

    lines += [
        "",
        "### Class distribution (all usable rows)",
        "",
        "| Aspect | " + " | ".join(LABELS) + " |",
        "|---|" + "---:|" * len(LABELS),
    ]
    for aspect, row in dist.iterrows():
        lines.append(f"| `{aspect}` | " + " | ".join(str(row[label]) for label in LABELS) + " |")
    rare = [f"`{a}`/{lab} ({n})" for a, lab, n in rare_classes(dist, RARE_MIN) if lab != NO_MENTION]
    if rare:
        lines.append(f"\nRare classes (< {RARE_MIN} examples): " + ", ".join(rare))

    base, yok = results["baseline"], results["always_yok"]
    lines += [
        "",
        f"### Baseline (TF-IDF + Logistic Regression), train {results['n_train']} / "
        f"test {results['n_test']}, seed {results['seed']}",
        "",
        "| Aspect | macro-F1 | always-`yok` macro-F1 | F1 olumlu | F1 olumsuz | F1 notr | "
        "test support (yok/olumlu/olumsuz/notr) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for aspect in ASPECTS:
        m = base["per_aspect"][aspect]
        pl = m["per_label"]
        support = "/".join(str(pl[label]["support"]) for label in LABELS)
        f1 = " | ".join(_f1_cell(pl[label]) for label in ("olumlu", "olumsuz", "notr"))
        lines.append(
            f"| `{aspect}` | {m['macro_f1']:.3f} | {yok['per_aspect'][aspect]['macro_f1']:.3f} | "
            f"{f1} | {support} |"
        )
    lines.append(f"| **overall** | **{base['macro_f1']:.3f}** | {yok['macro_f1']:.3f} | | | | |")
    if results["warnings"]:
        lines += ["", "### Warnings", *[f"- {w}" for w in results["warnings"]]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        default=[Path("data/interim/verified_labels.csv")],
        help="bir ya da birden fazla etiket CSV'si (etiketleyici export'ları, pilot_checked.csv)",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline_first_run"))
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    df, report = load_labels(args.labels)
    if report.missing_columns:
        raise SystemExit(f"Eksik sütunlar: {report.missing_columns}")
    if len(df) < 20:
        raise SystemExit(f"Kullanılabilir yalnızca {len(df)} satır var; en az 20 gerekli.")

    dist = class_distribution(df)
    results = run(df, args.test_size, args.seed)
    markdown = render_report(report, dist, len(df), results)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(markdown, encoding="utf-8")
    (args.out / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(markdown)
    print(f"\nYazıldı: {args.out / 'report.md'}, {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
