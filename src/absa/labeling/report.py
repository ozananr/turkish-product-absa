"""#52 tablolarını tek komutla üretir: LLM-insan uyumu ve etiket dağılımı.

Kullanım:
    uv run python -m absa.labeling.report \\
        --prelabels prelabels.csv \\
        --final final_labels.csv \\
        --kappa-prelabels kappa_prelabels.csv \\
        --kappa-final kappa_final.csv \\
        --exports export_salih.csv export_ozan.csv export_yusuf.csv

Dört tablo basılır (issue'ya ve rapora kopyalanabilir markdown):
1. LLM ön etiketleri ve nihai etiketler (yalnızca ön etiketli batch yorumları).
   Kappa setindeki yorumlar `final_labels.csv` içinde olsa bile bu tablodan çıkarılır.
2. LLM ve uzlaştırılmış kappa seti (insan LLM önerisini görmeden etiketledi). 1. tablodaki
   uyum bundan çok daha yüksekse, batch'lerde LLM önerisi etiketleyiciyi etkilemiş demektir.
3. Nihai etiketlerin dağılımı ve nadir sınıflar.
4. (`--exports` verilirse) her etiketleyicinin batch'te değiştirdiği LLM etiketi oranı.

Dosyalar veri içerir, git'e eklenmez (Drive: absa-data/prelabels, absa-data/final).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from absa.data.labels import class_distribution, load_labels, rare_classes
from absa.labeling.adjudicate import prepare_export
from absa.labeling.compare import compare, render_markdown
from absa.schema import ASPECTS

RARE_THRESHOLD = 30


def load_batch_export(path: str | Path) -> tuple[str, pd.DataFrame]:
    """(etiketleyici adı, checked satırlar). Ad `annotator` sütunundan, yoksa dosya adından."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    names = sorted({n.strip() for n in raw.get("annotator", pd.Series(dtype=str)) if n.strip()})
    name = names[0] if len(names) == 1 else Path(path).stem.removeprefix("export_")
    return name, prepare_export(raw)


def split_final(final: pd.DataFrame, kappa_final: pd.DataFrame) -> pd.DataFrame:
    """Nihai etiketlerden kappa setindeki yorumları çıkarır (yalnızca batch yorumları kalır)."""
    return final[~final["review_id"].isin(set(kappa_final["review_id"]))].reset_index(drop=True)


def render_distribution(final: pd.DataFrame, threshold: int = RARE_THRESHOLD) -> str:
    """Etiket dağılımı tablosu ve nadir sınıf listesi (markdown)."""
    dist = class_distribution(final)
    mentioned = len(final) - dist["yok"]
    lines = [
        f"- **Reviews:** {len(final)}",
        "",
        "| Aspect | mentioned | % mentioned | " + " | ".join(dist.columns) + " |",
        "|---|---:|---:|" + "---:|" * len(dist.columns),
    ]
    for aspect in dist.index:
        row = " | ".join(str(int(v)) for v in dist.loc[aspect])
        pct = mentioned[aspect] / len(final)
        lines.append(f"| `{aspect}` | {int(mentioned[aspect])} | {pct:.1%} | {row} |")

    rare = rare_classes(dist, min_count=threshold)
    lines += ["", f"**Rare classes (fewer than {threshold} examples):**", ""]
    if rare:
        lines += [f"- `{a}` / `{label}`: {n}" for a, label, n in rare]
    else:
        lines.append("- none")
    return "\n".join(lines)


def annotator_change_rates(
    prelabels: pd.DataFrame, exports: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Her etiketleyicinin batch'te LLM önerisinden değiştirdiği hücre oranı.

    `exports`: {etiketleyici adı: checked satırlar (review_id + 7 aspect)}. Yalnızca
    prelabels'ta da olan yorumlar karşılaştırılır. Son satır tüm etiketleyicilerin toplamıdır.
    """
    pre = prelabels.drop_duplicates("review_id").set_index("review_id")
    rows = []
    for name, df in exports.items():
        common = df[df["review_id"].isin(pre.index)].set_index("review_id")
        changed = pre.loc[common.index, ASPECTS].to_numpy() != common[ASPECTS].to_numpy()
        rows.append(
            {
                "annotator": name,
                "reviews": len(common),
                "reviews_changed": int(changed.any(axis=1).sum()),
                "cells": int(changed.size),
                "cells_changed": int(changed.sum()),
            }
        )
    table = pd.DataFrame(
        rows, columns=["annotator", "reviews", "reviews_changed", "cells", "cells_changed"]
    )
    total = table[["reviews", "reviews_changed", "cells", "cells_changed"]].sum()
    table.loc[len(table)] = {"annotator": "total", **total.to_dict()}
    table["pct_cells_changed"] = table["cells_changed"] / table["cells"].where(table["cells"] > 0)
    table["pct_cells_changed"] = table["pct_cells_changed"].fillna(0.0)
    return table


def render_change_rates(table: pd.DataFrame) -> str:
    lines = [
        "| Annotator | reviews | reviews with a change | cells changed | % cells changed |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in table.itertuples(index=False):
        name = f"**{r.annotator}**" if r.annotator == "total" else r.annotator
        lines.append(
            f"| {name} | {r.reviews} | {r.reviews_changed} | {r.cells_changed} / {r.cells} "
            f"| {r.pct_cells_changed:.2%} |"
        )
    return "\n".join(lines)


def build_report(
    prelabels: pd.DataFrame,
    final: pd.DataFrame,
    kappa_prelabels: pd.DataFrame,
    kappa_final: pd.DataFrame,
    exports: dict[str, pd.DataFrame] | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """(markdown rapor, batch uyuşmazlıkları, kappa seti uyuşmazlıkları).

    `exports` verilirse 4. bölüm olarak etiketleyici başına değişiklik oranı eklenir.
    """
    batches = split_final(final, kappa_final)
    s_batch, diff_batch = compare(prelabels, batches)
    s_kappa, diff_kappa = compare(kappa_prelabels, kappa_final)

    text = "\n".join(
        [
            "## 1. LLM pre-labels vs final labels (batches, LLM suggestion was visible)",
            "",
            "Label accuracy = share of labels the annotator left unchanged. "
            '"human / LLM" columns = final labels / LLM labels.',
            "",
            render_markdown(s_batch),
            "",
            "## 2. LLM vs adjudicated kappa set (labeled without LLM suggestions)",
            "",
            render_markdown(s_kappa),
            "",
            f"Difference in label accuracy (batches - kappa set): "
            f"{s_batch['label_accuracy'] - s_kappa['label_accuracy']:+.1%}. "
            "A clearly positive value suggests the visible LLM suggestion influenced the "
            "annotators (anchoring).",
            "",
            "## 3. Final label distribution (all checked reviews)",
            "",
            render_distribution(final),
        ]
    )
    if exports:
        text += "\n".join(
            [
                "",
                "",
                "## 4. Changes per annotator (batches, LLM suggestion was visible)",
                "",
                "Share of LLM pre-labels each annotator changed in their checked batch rows. "
                "For comparison, see the label accuracy of the kappa set in table 2, which was "
                "labeled without suggestions.",
                "",
                render_change_rates(annotator_change_rates(prelabels, exports)),
            ]
        )
    return text, diff_batch, diff_kappa


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prelabels", type=Path, required=True, help="prelabels.csv (2500 yorum)")
    parser.add_argument("--final", type=Path, required=True, help="final_labels.csv")
    parser.add_argument("--kappa-prelabels", type=Path, required=True)
    parser.add_argument("--kappa-final", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/interim/report_52"))
    parser.add_argument(
        "--exports",
        type=Path,
        nargs="*",
        default=[],
        help="batch export'ları (export_<isim>.csv); verilirse etiketleyici başına değişiklik",
    )
    args = parser.parse_args(argv)

    final, report = load_labels(args.final)
    kappa_final, kappa_report = load_labels(args.kappa_final)
    for name, rep in (("final", report), ("kappa_final", kappa_report)):
        if rep.has_problems:
            raise SystemExit(f"{name}: etiket dosyasında sorun var: {rep}")

    prelabels = pd.read_csv(args.prelabels, dtype=str, keep_default_na=False)
    kappa_prelabels = pd.read_csv(args.kappa_prelabels, dtype=str, keep_default_na=False)
    exports = dict(load_batch_export(p) for p in args.exports)
    text, diff_batch, diff_kappa = build_report(
        prelabels, final, kappa_prelabels, kappa_final, exports
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    diff_batch.to_csv(args.out / "diff_batches.csv", index=False)
    diff_kappa.to_csv(args.out / "diff_kappa.csv", index=False)
    print(text)
    print(f"\nDosyalar: {args.out}/ (report.md, diff_batches.csv, diff_kappa.csv)")


if __name__ == "__main__":
    main()
