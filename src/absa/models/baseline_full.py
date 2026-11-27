"""TF-IDF + Logistic Regression baseline'ı dondurulmuş bölmede eğitir ve değerlendirir (issue #54).

Train parçasında eğitilir, val ve test parçalarında ortak `evaluate()` ile ölçülür. Karşılaştırma
için "hep `yok`" tahmini de değerlendirilir. Test hataları yorum türlerine göre (kısa, karşıtlık,
olumsuzluk, karışık duygu, nötr içeren) özetlenir; hata analizi için test hataları ayrı bir
dosyaya yazılır. Bu sayılar Hafta 6'da BERTurk ile karşılaştırılacak referanstır.

Kullanım:
    uv run python -m absa.models.baseline_full \\
        --labels data/interim/labeling/final_labels.csv \\
        --split data/interim/splits/split_v1.csv

Çıktılar (--out, varsayılan outputs/baseline_full/, git'e girmez):
    report.md        issue'ya yapıştırılacak rapor
    metrics.json     tüm metrikler (val ve test)
    test_errors.csv  en az bir hatalı aspect'i olan test yorumları (hata analizi için)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from absa.data.cleaner import word_count
from absa.models.baseline import TfidfBaseline
from absa.models.evaluate import evaluate
from absa.schema import ASPECTS, LABELS, NO_MENTION

DEFAULT_SEED = 42
SPLITS = ["train", "val", "test"]
SHORT_MAX_WORDS = 5

_CONTRAST = re.compile(r"\b(ama|fakat|ancak|lakin|yalnız|ne var ki)\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(değil|degil|hiç|hic|asla)\b", re.IGNORECASE)

REVIEW_TYPES = {
    "short": f"short (≤ {SHORT_MAX_WORDS} words)",
    "contrast": "contrast (ama / fakat / ancak ...)",
    "negation": "negation word (değil / hiç / asla)",
    "mixed": "mixed sentiment (olumlu and olumsuz in the same review)",
    "notr": "contains a `notr` label",
}


def load_split_data(labels_path: Path, split_path: Path) -> dict[str, pd.DataFrame]:
    """Etiketleri bölme dosyasıyla birleştirir: {"train": df, "val": df, "test": df}.

    Raises:
        ValueError: bölme dosyasında olup etiket dosyasında olmayan yorum varsa.
    """
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(set(split["review_id"]) - set(labels["review_id"]))
    if missing:
        raise ValueError(
            f"Bölme dosyasındaki yorumlar etiket dosyasında yok (ilk 5): {missing[:5]}"
        )
    merged = labels.merge(split[["review_id", "split"]], on="review_id", how="inner")
    return {s: merged[merged["split"] == s].reset_index(drop=True) for s in SPLITS}


def review_types(text: str, gold: dict[str, str]) -> list[str]:
    """Bir yorumun hata analizi türleri (bir yorum birden fazla türde olabilir)."""
    types = []
    if word_count(text) <= SHORT_MAX_WORDS:
        types.append("short")
    if _CONTRAST.search(text):
        types.append("contrast")
    if _NEGATION.search(text):
        types.append("negation")
    values = set(gold.values())
    if {"olumlu", "olumsuz"} <= values:
        types.append("mixed")
    if "notr" in values:
        types.append("notr")
    return types


def error_table(test: pd.DataFrame, pred: dict[str, list[str]]) -> pd.DataFrame:
    """Her test yorumu için hatalı aspect'ler ve yorum türleri."""
    rows = []
    for i, r in test.iterrows():
        gold = {a: r[a] for a in ASPECTS}
        errors = [f"{a}: {gold[a]}→{pred[a][i]}" for a in ASPECTS if gold[a] != pred[a][i]]
        rows.append(
            {
                "review_id": r["review_id"],
                "text": r["text"],
                "types": ", ".join(review_types(r["text"], gold)),
                "n_errors": len(errors),
                "errors": "; ".join(errors),
            }
        )
    return pd.DataFrame(rows)


def run(parts: dict[str, pd.DataFrame], seed: int = DEFAULT_SEED) -> tuple[dict, pd.DataFrame]:
    """Train'de eğit, val ve test'te değerlendir. (metrikler, test hata tablosu)."""
    train = parts["train"]
    model = TfidfBaseline(random_state=seed).fit(
        train["text"], {a: train[a].tolist() for a in ASPECTS}
    )
    results: dict = {
        "model": "TF-IDF (1-2 gram, sublinear tf) + LogisticRegression (balanced), one per aspect",
        "seed": seed,
        "sizes": {s: len(df) for s, df in parts.items()},
    }
    test_pred = None
    for name in ("val", "test"):
        df = parts[name]
        y_true = {a: df[a].tolist() for a in ASPECTS}
        y_pred = model.predict(df["text"])
        results[name] = {
            "baseline": evaluate(y_true, y_pred),
            "always_yok": evaluate(y_true, {a: [NO_MENTION] * len(df) for a in ASPECTS}),
        }
        if name == "test":
            test_pred = y_pred
    return results, error_table(parts["test"], test_pred)


def _f1(scores: dict) -> str:
    """Test'te hiç örneği olmayan etiket için F1 anlamsızdır: '-'."""
    return "-" if scores["support"] == 0 else f"{scores['f1']:.2f}"


def render_report(results: dict, errors: pd.DataFrame) -> str:
    sizes = results["sizes"]
    val, test = results["val"], results["test"]
    lines = [
        "### Setup",
        f"- Model: {results['model']}, `random_state={results['seed']}`",
        f"- Split `split_v1`: train {sizes['train']} / val {sizes['val']} / test {sizes['test']} "
        "(test = adjudicated kappa set, independent human labels)",
        "",
        "### Overall macro-F1",
        "",
        "| | val | test |",
        "|---|---:|---:|",
        f"| **TF-IDF + LR** | **{val['baseline']['macro_f1']:.3f}** | "
        f"**{test['baseline']['macro_f1']:.3f}** |",
        f"| always `yok` | {val['always_yok']['macro_f1']:.3f} | "
        f"{test['always_yok']['macro_f1']:.3f} |",
        "",
        "### Per aspect (test)",
        "",
        "| Aspect | macro-F1 test | macro-F1 val | F1 olumlu | F1 olumsuz | F1 notr | "
        "test support (yok/olumlu/olumsuz/notr) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for a in ASPECTS:
        m = test["baseline"]["per_aspect"][a]
        pl = m["per_label"]
        support = "/".join(str(pl[lab]["support"]) for lab in LABELS)
        f1 = " | ".join(_f1(pl[lab]) for lab in ("olumlu", "olumsuz", "notr"))
        lines.append(
            f"| `{a}` | {m['macro_f1']:.3f} | {val['baseline']['per_aspect'][a]['macro_f1']:.3f} "
            f"| {f1} | {support} |"
        )

    lines += [
        "",
        "<details><summary>Precision / recall / F1 per label (test)</summary>",
        "",
        "| Aspect | label | precision | recall | F1 | support |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for a in ASPECTS:
        for lab in LABELS:
            s = test["baseline"]["per_aspect"][a]["per_label"][lab]
            lines.append(
                f"| `{a}` | {lab} | {s['precision']:.2f} | {s['recall']:.2f} | {_f1(s)} "
                f"| {s['support']} |"
            )
    lines += ["", "</details>", "", "<details><summary>Confusion matrices (test)</summary>", ""]
    for a in ASPECTS:
        cm = test["baseline"]["per_aspect"][a]["confusion_matrix"]
        lines += [
            f"**`{a}`** (rows: gold, columns: predicted)",
            "",
            "| gold \\ pred | " + " | ".join(LABELS) + " |",
            "|---|" + "---:|" * len(LABELS),
            *[
                f"| {lab} | " + " | ".join(str(v) for v in row) + " |"
                for lab, row in zip(LABELS, cm, strict=True)
            ],
            "",
        ]
    lines.append("</details>")

    lines += render_error_summary(errors)
    return "\n".join(lines)


def render_error_summary(errors: pd.DataFrame) -> list[str]:
    n = len(errors)
    wrong = errors["n_errors"] > 0
    lines = [
        "",
        "### Error analysis (test)",
        "",
        f"- Reviews with at least one wrong aspect: {int(wrong.sum())} / {n} "
        f"({wrong.mean():.1%}); wrong cells: {int(errors['n_errors'].sum())} / {n * len(ASPECTS)}",
        "",
        "| Review type | reviews | with an error | error rate |",
        "|---|---:|---:|---:|",
        f"| all | {n} | {int(wrong.sum())} | {wrong.mean():.1%} |",
    ]
    for key, label in REVIEW_TYPES.items():
        mask = errors["types"].str.split(", ").apply(lambda t, k=key: k in t)
        k = int(mask.sum())
        rate = f"{(wrong & mask).sum() / k:.1%}" if k else "-"
        lines.append(f"| {label} | {k} | {int((wrong & mask).sum())} | {rate} |")

    confusions = Counter(
        e.strip() for row in errors["errors"] if row for e in row.split(";") if e.strip()
    )
    lines += [
        "",
        "Most frequent confusions (aspect: gold→predicted):",
        "",
        *[f"- `{c}` × {k}" for c, k in confusions.most_common(10)],
    ]
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True, help="final_labels.csv")
    parser.add_argument("--split", type=Path, required=True, help="split_v1.csv")
    parser.add_argument("--out", type=Path, default=Path("outputs/baseline_full"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    parts = load_split_data(args.labels, args.split)
    results, errors = run(parts, args.seed)
    report = render_report(results, errors)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    errors[errors["n_errors"] > 0].sort_values("n_errors", ascending=False).to_csv(
        args.out / "test_errors.csv", index=False, encoding="utf-8-sig"
    )
    print(report)
    print(f"\nYazıldı: {args.out}/ (report.md, metrics.json, test_errors.csv)")


if __name__ == "__main__":
    main()
