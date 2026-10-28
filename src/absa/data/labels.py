"""Doğrulanmış (insan tarafından kontrol edilmiş) etiketleri okur ve kontrol eder.

Beklenen format: `review_id`, `text` ve 7 aspect sütunu. Etiketleyici export'ları
(docs/annotation/labeling_guide.md) ayrıca `annotator` ve `checked` sütunlarını içerir:
- `checked` sütunu varsa yalnızca `TRUE` olan satırlar kullanılır; diğerleri henüz
  kontrol edilmemiş LLM önerileridir.
- Birden fazla dosya birlikte okunabilir (kişi başına export'lar, pilot_checked.csv).
- Kappa setindeki bir yorum iki etiketleyicide bulunur: etiketler aynıysa bir kez
  kullanılır, farklıysa (henüz uzlaştırılmamış) kullanılmaz ve raporlanır.
Diğer fazladan sütunlar (ör. notlar) yok sayılır.

Kontroller: eksik sütun, boş metin, tekrar eden review_id, boş etiket, geçersiz etiket
(yazım hatası). Yalnızca güvenli düzeltmeler otomatik yapılır (büyük/küçük harf, boşluk,
"nötr" -> "notr"); geri kalan sorunlu satırlar ayıklanır ve raporlanır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from absa.schema import ASPECTS, LABELS

REQUIRED_COLUMNS = ["review_id", "text", *ASPECTS]
_SAFE_FIXES = {"nötr": "notr", "nötür": "notr"}


@dataclass
class LabelReport:
    total_rows: int = 0
    missing_columns: list[str] = field(default_factory=list)
    empty_text: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    empty_labels: dict[str, list[str]] = field(default_factory=dict)
    invalid_labels: dict[str, dict[str, int]] = field(default_factory=dict)
    auto_fixed: int = 0
    dropped_ids: list[str] = field(default_factory=list)
    files: int = 1
    unchecked_skipped: int = 0
    kappa_agreed: int = 0
    conflicting_ids: list[str] = field(default_factory=list)

    @property
    def has_problems(self) -> bool:
        return bool(
            self.missing_columns
            or self.empty_text
            or self.duplicate_ids
            or self.empty_labels
            or self.invalid_labels
        )


def _normalize(value: str) -> str:
    v = value.strip().lower()
    return _SAFE_FIXES.get(v, v)


def check_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelReport]:
    """Etiketleri normalleştirir ve kontrol eder. (temiz tablo, rapor) döner."""
    report = LabelReport(total_rows=len(df))
    report.missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if report.missing_columns:
        return df.iloc[0:0][[c for c in REQUIRED_COLUMNS if c in df.columns]], report

    has_annotator = "annotator" in df.columns
    cols = REQUIRED_COLUMNS + (["annotator"] if has_annotator else [])
    df = df[cols].fillna("").astype(str).copy()
    df["review_id"] = df["review_id"].str.strip()
    df["text"] = df["text"].str.strip()
    for aspect in ASPECTS:
        raw = df[aspect]
        fixed = raw.map(_normalize)
        report.auto_fixed += int((fixed != raw).sum())
        df[aspect] = fixed
    if has_annotator:
        df = _resolve_cross_annotator(df, report)
    df = df[REQUIRED_COLUMNS].reset_index(drop=True)
    bad = pd.Series(False, index=df.index)

    empty_text = df["text"] == ""
    report.empty_text = df.loc[empty_text, "review_id"].tolist()
    bad |= empty_text

    duplicated = df["review_id"].duplicated(keep="first")
    report.duplicate_ids = sorted(set(df.loc[duplicated, "review_id"]))
    bad |= duplicated

    for aspect in ASPECTS:
        fixed = df[aspect]
        empty = fixed == ""
        if empty.any():
            report.empty_labels[aspect] = df.loc[empty, "review_id"].tolist()
        invalid = ~fixed.isin(LABELS) & ~empty
        if invalid.any():
            report.invalid_labels[aspect] = fixed[invalid].value_counts().to_dict()
        bad |= empty | invalid

    report.dropped_ids = df.loc[bad, "review_id"].tolist()
    return df[~bad].reset_index(drop=True), report


def _resolve_cross_annotator(df: pd.DataFrame, report: LabelReport) -> pd.DataFrame:
    """Aynı yorumu farklı etiketleyiciler etiketlediyse (kappa seti): aynıysa bir kez tut,
    farklıysa hiçbirini kullanma. Aynı etiketleyicideki tekrarlar dokunulmadan bırakılır
    (check_labels onları tekrar eden id olarak raporlar)."""
    per_review = df.drop_duplicates(subset=["review_id", "annotator"])
    multi = per_review[per_review["review_id"].duplicated(keep=False)]
    if multi.empty:
        return df
    n_label_sets = multi.groupby("review_id")[ASPECTS].nunique().max(axis=1)
    agreed = set(n_label_sets[n_label_sets == 1].index)
    conflicting = set(n_label_sets[n_label_sets > 1].index)
    report.kappa_agreed = len(agreed)
    report.conflicting_ids = sorted(conflicting)
    df = df[~df["review_id"].isin(conflicting)]
    # Uzlaşılmış kappa yorumlarından yalnızca biri kalsın.
    keep = ~(df["review_id"].isin(agreed) & df.duplicated(subset=["review_id"]))
    return df[keep]


def _read(path: str | Path) -> tuple[pd.DataFrame, int]:
    """Bir CSV'yi okur; `checked` sütunu varsa yalnızca TRUE satırları döner."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "checked" not in df.columns:
        return df, 0
    checked = df["checked"].str.strip().str.upper() == "TRUE"
    return df[checked], int((~checked).sum())


def load_labels(paths: str | Path | list[str | Path]) -> tuple[pd.DataFrame, LabelReport]:
    """Bir ya da birden fazla CSV'yi okur (UTF-8, BOM olsa da olur), birleştirir ve kontrol eder."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames, skipped = [], 0
    for path in paths:
        df, n = _read(path)
        frames.append(df)
        skipped += n
    clean, report = check_labels(pd.concat(frames, ignore_index=True))
    report.files = len(paths)
    report.unchecked_skipped = skipped
    return clean, report


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Aspect x etiket sayım tablosu (satırlar ASPECTS, sütunlar LABELS sırasında)."""
    counts = {a: df[a].value_counts().reindex(LABELS, fill_value=0) for a in ASPECTS}
    return pd.DataFrame(counts).T.astype(int)


def rare_classes(dist: pd.DataFrame, min_count: int = 5) -> list[tuple[str, str, int]]:
    """min_count'tan az örneği olan (aspect, etiket, sayı) üçlüleri. 0 olanlar da dahil."""
    return [
        (aspect, label, int(dist.loc[aspect, label]))
        for aspect in dist.index
        for label in dist.columns
        if dist.loc[aspect, label] < min_count
    ]
