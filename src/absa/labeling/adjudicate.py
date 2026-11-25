"""Kappa setinde iki etiketleyicinin etiketlerini uzlaştırır (issue #51).

İki adım:

1. `conflicts`: iki export'u karşılaştırır. Aynı etiketi verdikleri hücreler otomatik
   uzlaşılmış sayılır; farklı verdikleri hücreler `conflicts.csv` dosyasına yazılır
   (her satır bir yorum x aspect çifti, `final` sütunu boş). Bu dosya Google Sheets'e
   yüklenir ve iki etiketleyici birlikte `final` sütununu doldurur.
2. `finalize`: doldurulmuş `conflicts.csv` ile uzlaşılan etiketleri birleştirir ve
   `kappa_final.csv` yazar. Yorum başına tek satır, `annotator=final`, `checked=TRUE`
   (labeling_guide.md bölüm 5 formatı; `absa.data.labels.load_labels` ile okunabilir).

Kullanım:
    uv run python -m absa.labeling.adjudicate conflicts \\
        --a export_kappa_ozan.csv --b export_kappa_salih.csv --names ozan salih \\
        --out conflicts.csv
    # conflicts.csv'yi Sheets'te doldurun, CSV olarak indirin (conflicts_filled.csv)
    uv run python -m absa.labeling.adjudicate finalize \\
        --a export_kappa_ozan.csv --b export_kappa_salih.csv \\
        --decisions conflicts_filled.csv --out kappa_final.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from absa.schema import ASPECTS, LABELS

FINAL_COLUMNS = ["review_id", "text", *ASPECTS, "annotator", "checked"]
_SAFE_FIXES = {"nötr": "notr", "nötür": "notr"}


def _normalize(value: str) -> str:
    v = value.strip().lower()
    return _SAFE_FIXES.get(v, v)


def load_export(path: str | Path) -> pd.DataFrame:
    """Etiketleyici export'unu okur: yalnızca `checked == TRUE` satırlar, etiketler normalleşir."""
    return prepare_export(pd.read_csv(path, dtype=str, keep_default_na=False))


def prepare_export(df: pd.DataFrame) -> pd.DataFrame:
    """Export tablosunu doğrular ve normalleştirir.

    Raises:
        ValueError: sütun eksikse, review_id tekrar ediyorsa ya da geçersiz etiket varsa.
    """
    required = ["review_id", "text", *ASPECTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütun: {missing}")
    if "checked" in df.columns:
        df = df[df["checked"].str.strip().str.upper() == "TRUE"]
    df = df[required].copy()
    df["review_id"] = df["review_id"].str.strip()
    for aspect in ASPECTS:
        df[aspect] = df[aspect].map(_normalize)
        bad = sorted(set(df[aspect]) - set(LABELS))
        if bad:
            raise ValueError(f"{aspect}: geçersiz etiket {bad}")
    dup = sorted(set(df.loc[df["review_id"].duplicated(), "review_id"]))
    if dup:
        raise ValueError(f"Tekrar eden review_id (ilk 5): {dup[:5]}")
    return df.reset_index(drop=True)


def _align(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """İki export'u review_id ile birleştirir. Kümeler farklıysa hata verir."""
    only_a = sorted(set(a["review_id"]) - set(b["review_id"]))
    only_b = sorted(set(b["review_id"]) - set(a["review_id"]))
    if only_a or only_b:
        raise ValueError(
            f"İki dosyada aynı yorumlar olmalı: yalnızca A'da {len(only_a)}, "
            f"yalnızca B'de {len(only_b)} yorum var (ör. {(only_a + only_b)[:3]}). "
            "Her iki dosyada da tüm satırlar `checked` olmalı."
        )
    return a.merge(b, on="review_id", suffixes=("_a", "_b"))


def build_conflicts(
    a: pd.DataFrame, b: pd.DataFrame, name_a: str = "a", name_b: str = "b"
) -> tuple[pd.DataFrame, dict]:
    """(anlaşmazlık tablosu, özet). Tablo: review_id, aspect, <name_a>, <name_b>, final, text."""
    merged = _align(a, b)
    rows = []
    for _, row in merged.iterrows():
        for aspect in ASPECTS:
            if row[f"{aspect}_a"] != row[f"{aspect}_b"]:
                rows.append(
                    {
                        "review_id": row["review_id"],
                        "aspect": aspect,
                        name_a: row[f"{aspect}_a"],
                        name_b: row[f"{aspect}_b"],
                        "final": "",
                        "text": row["text_a"],
                    }
                )
    columns = ["review_id", "aspect", name_a, name_b, "final", "text"]
    conflicts = pd.DataFrame(rows, columns=columns)
    reviews_with_conflict = conflicts["review_id"].nunique()
    summary = {
        "reviews": len(merged),
        "conflict_cells": len(conflicts),
        "reviews_with_conflict": int(reviews_with_conflict),
        "reviews_fully_agreed": len(merged) - int(reviews_with_conflict),
        "by_aspect": conflicts["aspect"].value_counts().reindex(ASPECTS, fill_value=0).to_dict(),
    }
    return conflicts, summary


def finalize(a: pd.DataFrame, b: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Uzlaşılan etiketler + doldurulmuş kararlardan nihai tabloyu üretir.

    Raises:
        ValueError: bir anlaşmazlık için karar yoksa/geçersizse, ya da kararlar bilinmeyen
            bir hücreye aitse.
    """
    merged = _align(a, b)
    for col in ("review_id", "aspect", "final"):
        if col not in decisions.columns:
            raise ValueError(f"Karar dosyasında '{col}' sütunu yok")

    decided = decisions[["review_id", "aspect", "final"]].copy()
    decided["review_id"] = decided["review_id"].str.strip()
    decided["aspect"] = decided["aspect"].str.strip()
    decided["final"] = decided["final"].map(_normalize)
    if decided.duplicated(subset=["review_id", "aspect"]).any():
        raise ValueError("Karar dosyasında aynı yorum ve aspect birden fazla kez var")
    lookup = {(r, asp): fin for r, asp, fin in decided.itertuples(index=False)}

    expected: set[tuple[str, str]] = set()
    out_rows = []
    problems: list[str] = []
    for _, row in merged.iterrows():
        rid = row["review_id"]
        out = {"review_id": rid, "text": row["text_a"]}
        for aspect in ASPECTS:
            la, lb = row[f"{aspect}_a"], row[f"{aspect}_b"]
            if la == lb:
                out[aspect] = la
                continue
            expected.add((rid, aspect))
            final = lookup.get((rid, aspect), "")
            if final not in LABELS:
                problems.append(f"{rid} / {aspect}: karar yok veya geçersiz ({final!r})")
            out[aspect] = final
        out_rows.append(out)

    unknown = sorted(set(lookup) - expected)
    if unknown:
        problems.append(f"Anlaşmazlık olmayan hücreler için karar var (ilk 3): {unknown[:3]}")
    if problems:
        shown = "\n  ".join(problems[:10])
        raise ValueError(f"{len(problems)} sorun:\n  {shown}")

    final_df = pd.DataFrame(out_rows)
    final_df["annotator"] = "final"
    final_df["checked"] = "TRUE"
    return final_df[FINAL_COLUMNS]


def render_summary(s: dict, name_a: str, name_b: str) -> str:
    by_aspect = ", ".join(f"{a}: {n}" for a, n in s["by_aspect"].items())
    return "\n".join(
        [
            f"- **Reviews compared ({name_a} / {name_b}):** {s['reviews']}",
            f"- **Fully agreed (all 7 aspects):** {s['reviews_fully_agreed']}",
            f"- **Reviews with at least one disagreement:** {s['reviews_with_conflict']}",
            f"- **Disagreeing cells to decide:** {s['conflict_cells']}",
            f"- **By aspect:** {by_aspect}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_conf = sub.add_parser("conflicts", help="anlaşmazlık dosyasını üret")
    p_fin = sub.add_parser("finalize", help="kararlardan kappa_final.csv üret")
    for p in (p_conf, p_fin):
        p.add_argument("--a", type=Path, required=True, help="1. etiketleyici export'u")
        p.add_argument("--b", type=Path, required=True, help="2. etiketleyici export'u")
        p.add_argument("--out", type=Path, required=True)
    p_conf.add_argument("--names", nargs=2, default=["a", "b"], metavar=("NAME_A", "NAME_B"))
    p_fin.add_argument("--decisions", type=Path, required=True, help="doldurulmuş conflicts CSV")

    args = parser.parse_args(argv)
    a, b = load_export(args.a), load_export(args.b)

    if args.command == "conflicts":
        name_a, name_b = args.names
        conflicts, summary = build_conflicts(a, b, name_a, name_b)
        conflicts.to_csv(args.out, index=False)
        print(render_summary(summary, name_a, name_b))
        print(f"\nAnlaşmazlıklar: {args.out} ({len(conflicts)} satır)")
    else:
        decisions = pd.read_csv(args.decisions, dtype=str, keep_default_na=False)
        final_df = finalize(a, b, decisions)
        final_df.to_csv(args.out, index=False)
        print(f"Nihai etiketler: {args.out} ({len(final_df)} yorum)")


if __name__ == "__main__":
    main()
