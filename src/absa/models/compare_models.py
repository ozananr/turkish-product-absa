"""BERTurk'ü test kümesinde bir kez ölçer ve temel modelle karşılaştırır (issue #66).

Aynı bölme (`split_v1`), aynı `evaluate()` ve aynı hata analizi kullanılır. Temel model
(TF-IDF + LR, #54) aynı komut içinde yeniden eğitilir; deterministik olduğu için sonuçları #54 ile
aynıdır. BERTurk, #65'te doğrulama kümesinde seçilen modeldir (Drive: absa-data/models/berturk_v1).

Kullanım (CPU yeterli, ~1-2 dk):
    uv run python -m absa.models.compare_models \\
        --labels data/interim/labeling/final_labels.csv \\
        --split data/interim/splits/split_v1.csv \\
        --model models/berturk_v1

Çıktılar (--out, varsayılan outputs/compare/; git'e girmez):
    report.md            karşılaştırma + BERTurk'ün ayrıntılı test raporu (issue'ya yapıştırılır)
    metrics.json         iki modelin val ve test metrikleri
    test_comparison.csv  her test yorumu için iki modelin hataları (örnek seçmek için)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from absa.models.baseline_full import (
    REVIEW_TYPES,
    error_table,
    load_split_data,
    render_error_summary,
    run,
)
from absa.models.evaluate import evaluate
from absa.schema import ASPECTS, LABELS, NO_MENTION


def evaluate_berturk(parts: dict[str, pd.DataFrame], model_dir: Path) -> tuple[dict, pd.DataFrame]:
    """BERTurk'ü val ve test'te değerlendirir: (metrikler, test hata tablosu)."""
    from absa.models.berturk import load_model, predict_labels

    model, tokenizer, meta = load_model(model_dir)
    results: dict = {
        "model": "BERTurk + 7 aspect heads",
        "model_dir": str(model_dir),
        "training": meta.get("training", {}),
    }
    test_pred = None
    for name in ("val", "test"):
        df = parts[name]
        pred = predict_labels(model, tokenizer, df["text"].tolist(), max_length=meta["max_length"])
        results[name] = evaluate({a: df[a].tolist() for a in ASPECTS}, pred)
        if name == "test":
            test_pred = pred
    return results, error_table(parts["test"], test_pred)


def error_kinds(errors: pd.DataFrame) -> dict[str, int]:
    """Hatalı hücreler: bahsedilmeyeni bulma (extra), bahsedileni kaçırma (missed), yanlış duygu."""
    kinds = {"missed": 0, "extra": 0, "sentiment": 0}
    for row in errors["errors"]:
        for e in filter(None, (x.strip() for x in row.split(";"))):
            gold, pred = e.split(": ", 1)[1].split("→")
            if gold == NO_MENTION:
                kinds["extra"] += 1
            elif pred == NO_MENTION:
                kinds["missed"] += 1
            else:
                kinds["sentiment"] += 1
    return kinds


def cell_changes(err_a: pd.DataFrame, err_b: pd.DataFrame) -> dict[str, int]:
    """Hücre bazında: yalnız A'nın hatası (B düzeltti), yalnız B'nin hatası, ikisinin de hatası."""

    def cells(errors):
        return {
            (rid, e.split(":")[0].strip())
            for rid, row in zip(errors["review_id"], errors["errors"], strict=True)
            for e in filter(None, (x.strip() for x in row.split(";")))
        }

    a, b = cells(err_a), cells(err_b)
    return {"fixed": len(a - b), "new": len(b - a), "both": len(a & b)}


def comparison_table(err_base: pd.DataFrame, err_bert: pd.DataFrame) -> pd.DataFrame:
    """Her test yorumu için iki modelin hataları yan yana."""
    merged = err_base[["review_id", "text", "types", "n_errors", "errors"]].merge(
        err_bert[["review_id", "n_errors", "errors"]],
        on="review_id",
        suffixes=("_tfidf", "_berturk"),
    )
    merged["delta"] = merged["n_errors_berturk"] - merged["n_errors_tfidf"]
    return merged.sort_values(["delta", "n_errors_berturk"], ascending=[True, False])


def _rate(errors: pd.DataFrame, key: str | None) -> tuple[int, str]:
    wrong = errors["n_errors"] > 0
    mask = (
        pd.Series(True, index=errors.index)
        if key is None
        else errors["types"].str.split(", ").apply(lambda t: key in t)
    )
    n = int(mask.sum())
    return n, (f"{(wrong & mask).sum() / n:.1%}" if n else "-")


def _f1(scores: dict) -> str:
    return "-" if scores["support"] == 0 else f"{scores['f1']:.2f}"


def render_report(base: dict, bert: dict, err_base: pd.DataFrame, err_bert: pd.DataFrame) -> str:
    b_val, b_test = base["val"]["baseline"], base["test"]["baseline"]
    y_val, y_test = base["val"]["always_yok"], base["test"]["always_yok"]
    sizes = base["sizes"]
    lines = [
        "### Setup",
        f"- Split `split_v1`: train {sizes['train']} / val {sizes['val']} / test {sizes['test']} "
        "(test = adjudicated kappa set, independent human labels); test used once",
        f"- BERTurk: `{bert['model_dir']}` (chosen on val in #65)",
        "",
        "### Overall macro-F1",
        "",
        "| Model | val | test |",
        "|---|---:|---:|",
        f"| always `yok` | {y_val['macro_f1']:.3f} | {y_test['macro_f1']:.3f} |",
        f"| TF-IDF + LR (#54) | {b_val['macro_f1']:.3f} | {b_test['macro_f1']:.3f} |",
        f"| **BERTurk** | **{bert['val']['macro_f1']:.3f}** | **{bert['test']['macro_f1']:.3f}** |",
        "",
        "### Per aspect (test macro-F1 and F1 per label)",
        "",
        "| Aspect | TF-IDF | BERTurk | Δ | olumlu T / B | olumsuz T / B | notr T / B |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for a in ASPECTS:
        t, b = b_test["per_aspect"][a], bert["test"]["per_aspect"][a]
        per = [
            f"{_f1(t['per_label'][lab])} / {_f1(b['per_label'][lab])}"
            for lab in ("olumlu", "olumsuz", "notr")
        ]
        lines.append(
            f"| `{a}` | {t['macro_f1']:.3f} | {b['macro_f1']:.3f} "
            f"| {b['macro_f1'] - t['macro_f1']:+.3f} | " + " | ".join(per) + " |"
        )

    kb, kt = error_kinds(err_base), error_kinds(err_bert)
    ch = cell_changes(err_base, err_bert)
    n_cells = len(err_base) * len(ASPECTS)
    lines += [
        "",
        "### Errors on the test set",
        "",
        "| | TF-IDF | BERTurk |",
        "|---|---:|---:|",
        f"| wrong cells (of {n_cells}) | {sum(kb.values())} | {sum(kt.values())} |",
        f"| missed mention (gold mentioned → `yok`) | {kb['missed']} | {kt['missed']} |",
        f"| extra mention (gold `yok` → mentioned) | {kb['extra']} | {kt['extra']} |",
        f"| wrong sentiment | {kb['sentiment']} | {kt['sentiment']} |",
        "",
        f"Cells: {ch['fixed']} TF-IDF errors fixed by BERTurk, {ch['new']} new BERTurk errors, "
        f"{ch['both']} wrong in both.",
        "",
        "| Review type | reviews | TF-IDF error rate | BERTurk error rate |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [(None, "all"), *REVIEW_TYPES.items()]:
        n, rb = _rate(err_base, key)
        _, rt = _rate(err_bert, key)
        lines.append(f"| {label} | {n} | {rb} | {rt} |")

    lines += [
        "",
        "<details><summary>BERTurk: precision / recall / F1 per label (test)</summary>",
        "",
    ]
    lines += [
        "| Aspect | label | precision | recall | F1 | support |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for a in ASPECTS:
        for lab in LABELS:
            s = bert["test"]["per_aspect"][a]["per_label"][lab]
            lines.append(
                f"| `{a}` | {lab} | {s['precision']:.2f} | {s['recall']:.2f} | {_f1(s)} "
                f"| {s['support']} |"
            )
    lines += [
        "",
        "</details>",
        "",
        "<details><summary>BERTurk: confusion matrices (test)</summary>",
        "",
    ]
    for a in ASPECTS:
        cm = bert["test"]["per_aspect"][a]["confusion_matrix"]
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
    lines += ["</details>", "", "<details><summary>BERTurk: error analysis (test)</summary>"]
    lines += render_error_summary(err_bert)
    lines += ["", "</details>"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True, help="final_labels.csv")
    parser.add_argument("--split", type=Path, required=True, help="split_v1.csv")
    parser.add_argument("--model", type=Path, required=True, help="BERTurk model klasörü")
    parser.add_argument("--out", type=Path, default=Path("outputs/compare"))
    args = parser.parse_args(argv)

    parts = load_split_data(args.labels, args.split)
    base, err_base = run(parts)
    bert, err_bert = evaluate_berturk(parts, args.model)
    report = render_report(base, bert, err_base, err_bert)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "metrics.json").write_text(
        json.dumps({"tfidf": base, "berturk": bert}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    comparison_table(err_base, err_bert).to_csv(
        args.out / "test_comparison.csv", index=False, encoding="utf-8-sig"
    )
    print(report)
    print(f"\nYazıldı: {args.out}/ (report.md, metrics.json, test_comparison.csv)")


if __name__ == "__main__":
    main()
