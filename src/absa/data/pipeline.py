"""Birleştirme + temizleme + maskeleme hattı.

Kullanım:
    uv run python -m absa.data.pipeline --trsa TRSAv1.csv --trendyol trendyol.csv

Ham dosyalar data/raw/ altına kopyalanır, çıktılar data/interim/ altına yazılır.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from absa.data.cleaner import CleaningReport, clean_with_report, word_count
from absa.data.loader import combine, load_trendyol, load_trsa
from absa.data.masker import TAGS

TARGET = 10_000


def compute_metrics(combined: pd.DataFrame, cleaned: pd.DataFrame, report: CleaningReport) -> dict:
    """Son kalite metrikleri."""
    dup_dropped = sum(s["dropped"] for s in report.steps if s["step"].startswith("Tekrar"))
    words = cleaned["text"].map(word_count)
    mask_counts = {t: int(cleaned["text"].str.count(rf"\[{t}\]").sum()) for t in TAGS}
    return {
        "initial": report.initial,
        "final": len(cleaned),
        "total_dropped": report.initial - len(cleaned),
        "retention_rate": len(cleaned) / report.initial if report.initial else 0.0,
        "duplication_rate": dup_dropped / report.initial if report.initial else 0.0,
        "target": TARGET,
        "target_reached": len(cleaned) >= TARGET,
        "steps": report.steps,
        "initial_by_source": combined["source"].value_counts().to_dict(),
        "final_by_source": cleaned["source"].value_counts().to_dict(),
        "label_distribution": cleaned["label"].value_counts().to_dict(),
        "label_distribution_pct": (
            cleaned["label"].value_counts(normalize=True).round(4) * 100
        ).to_dict(),
        "rating_distribution": cleaned["rating"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict(),
        "words_mean": round(float(words.mean()), 2),
        "words_median": float(words.median()),
        "mask_counts": mask_counts,
    }


def render_markdown(m: dict) -> str:
    """Metrikleri docs/data_sources.md için markdown'a çevirir."""
    lines = [
        "| Adım | Önce | Elenen | Sonra |",
        "|---|---:|---:|---:|",
        f"| Birleştirilmiş ham veri | {m['initial']:,} | - | {m['initial']:,} |",
    ]
    lines += [
        f"| {s['step']} | {s['before']:,} | {s['dropped']:,} | {s['after']:,} |" for s in m["steps"]
    ]
    lines += [
        "",
        f"- **Son temiz yorum sayısı:** {m['final']:,} (hedef {m['target']:,}: "
        f"{'ulaşıldı' if m['target_reached'] else 'ulaşılamadı'})",
        f"- **Toplam elenen:** {m['total_dropped']:,} "
        f"(tutma oranı %{m['retention_rate'] * 100:.1f})",
        f"- **Tekrar oranı (ham veriye göre):** %{m['duplication_rate'] * 100:.1f}",
        f"- **Kaynak dağılımı (son):** {m['final_by_source']}",
        "- **Etiket dağılımı (son):** "
        + ", ".join(
            f"{k}: {v:,} (%{m['label_distribution_pct'][k]:.1f})"
            for k, v in m["label_distribution"].items()
        ),
        f"- **Trendyol puan dağılımı (son):** {m['rating_distribution']}",
        f"- **Kelime sayısı:** ortalama {m['words_mean']}, medyan {m['words_median']}",
        f"- **Maskelenen alan sayısı:** {m['mask_counts']}",
    ]
    return "\n".join(lines)


def run(trsa: Path, trendyol: Path, raw_dir: Path, interim_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    raw_trsa = raw_dir / "TRSAv1.csv"
    raw_trendyol = raw_dir / "trendyol.csv"
    for src, dst in ((trsa, raw_trsa), (trendyol, raw_trendyol)):
        if Path(src).resolve() != dst.resolve():
            shutil.copy2(src, dst)

    combined = combine([load_trsa(raw_trsa), load_trendyol(raw_trendyol)])
    combined.to_csv(interim_dir / "reviews_combined.csv", index=False)

    cleaned, report = clean_with_report(combined)
    cleaned.to_csv(interim_dir / "reviews_clean.csv", index=False)

    metrics = compute_metrics(combined, cleaned, report)
    (interim_dir / "cleaning_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=int), encoding="utf-8"
    )
    (interim_dir / "cleaning_report.md").write_text(render_markdown(metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trsa", type=Path, required=True)
    parser.add_argument("--trendyol", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    args = parser.parse_args()
    metrics = run(args.trsa, args.trendyol, args.raw_dir, args.interim_dir)
    print(render_markdown(metrics))


if __name__ == "__main__":
    main()
