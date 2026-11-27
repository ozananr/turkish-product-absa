"""Etiketli veriyi train / val / test olarak böler (issue #53).

Test kümesi: uzlaştırılmış kappa seti (200 yorum). Bu yorumlar LLM önerisi görülmeden iki kişi
tarafından bağımsız etiketlendi; #52, batch etiketlerinin pratikte LLM etiketleriyle aynı
olduğunu gösterdi. Bu yüzden modeller batch etiketleriyle eğitilir, insan etiketleriyle ölçülür.

Kalan yorumlar train / val olarak bölünür (varsayılan %85 / %15, seed 42). Neredeyse aynı
yorumlar (`cleaner.dedup_key`: Türkçe küçük harf, noktalama ve boşluk yok sayılır) aynı parçaya
düşer. Bir test yorumunun neredeyse aynısı train/val havuzunda varsa sızıntı olmaması için
havuzdan çıkarılır ve raporda listelenir.

Kullanım:
    uv run python -m absa.data.split \\
        --labels data/interim/labeling/final_labels.csv \\
        --test-ids data/interim/labeling/kappa_final.csv \\
        --out data/interim/splits/split_v1.csv

Çıktılar (git'e girmez, Drive: absa-data/splits/):
    split_v1.csv         review_id, split (train / val / test)
    split_v1_report.md   parça boyutları ve etiket dağılımı (issue'ya ve docs'a kopyalanır)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from absa.data.cleaner import dedup_key
from absa.schema import ASPECTS, LABELS, NO_MENTION

SPLITS = ["train", "val", "test"]
DEFAULT_VAL_SIZE = 0.15
DEFAULT_SEED = 42


def assign_splits(
    labels: pd.DataFrame,
    test_ids: set[str],
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, list[str]]:
    """(review_id -> split tablosu, sızıntı yüzünden çıkarılan review_id'ler).

    Raises:
        ValueError: test_ids içinde etiket dosyasında olmayan yorum varsa ya da
            train/val için yorum kalmazsa.
    """
    df = labels[["review_id", "text"]].drop_duplicates("review_id").reset_index(drop=True)
    missing = sorted(set(test_ids) - set(df["review_id"]))
    if missing:
        raise ValueError(f"Test yorumları etiket dosyasında yok (ilk 5): {missing[:5]}")

    df["group"] = df["text"].map(dedup_key)
    is_test = df["review_id"].isin(test_ids)
    leak = ~is_test & df["group"].isin(set(df.loc[is_test, "group"]))
    dropped = sorted(df.loc[leak, "review_id"])

    pool = df[~is_test & ~leak].reset_index(drop=True)
    if pool.empty:
        raise ValueError("Train/val için yorum kalmadı")
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss.split(pool, groups=pool["group"]))

    parts = [
        pd.DataFrame({"review_id": pool.loc[train_idx, "review_id"], "split": "train"}),
        pd.DataFrame({"review_id": pool.loc[val_idx, "review_id"], "split": "val"}),
        pd.DataFrame({"review_id": df.loc[is_test, "review_id"], "split": "test"}),
    ]
    out = pd.concat(parts, ignore_index=True).sort_values("review_id", ignore_index=True)
    return out, dropped


def split_counts(labels: pd.DataFrame, splits: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Her parça için aspect x etiket sayıları."""
    merged = labels.merge(splits, on="review_id")
    return {
        name: pd.DataFrame(
            {a: merged.loc[merged["split"] == name, a].value_counts() for a in ASPECTS}
        )
        .reindex(LABELS)
        .fillna(0)
        .astype(int)
        .T
        for name in SPLITS
    }


def missing_classes(counts: dict[str, pd.DataFrame]) -> list[str]:
    """Train'de olup val ya da test'te hiç örneği olmayan (yok dışı) aspect/etiket çiftleri."""
    warnings = []
    for aspect in ASPECTS:
        for label in LABELS:
            if label == NO_MENTION or counts["train"].loc[aspect, label] == 0:
                continue
            absent = [s for s in ("val", "test") if counts[s].loc[aspect, label] == 0]
            if absent:
                warnings.append(f"`{aspect}` / `{label}`: {' ve '.join(absent)} içinde örnek yok")
    return warnings


def render_report(splits: pd.DataFrame, dropped: list[str], counts: dict) -> str:
    sizes = splits["split"].value_counts().reindex(SPLITS, fill_value=0)
    total = int(sizes.sum())
    lines = [
        "### Split sizes",
        "",
        "| Split | reviews | share | labels |",
        "|---|---:|---:|---|",
    ]
    source = {
        "train": "batches (LLM pre-labels, checked)",
        "val": "batches (LLM pre-labels, checked)",
        "test": "adjudicated kappa set (independent human labels)",
    }
    for name in SPLITS:
        lines.append(f"| {name} | {sizes[name]} | {sizes[name] / total:.1%} | {source[name]} |")
    lines.append(f"| **total** | **{total}** | | |")
    lines += [
        "",
        f"Near-duplicates of test reviews removed from train/val: {len(dropped)}"
        + (f" ({', '.join(dropped[:10])})" if dropped else ""),
        "",
        "### Label distribution per split (olumlu / olumsuz / notr; the rest is `yok`)",
        "",
        "| Aspect | train | val | test |",
        "|---|---|---|---|",
    ]
    for aspect in ASPECTS:
        cells = [
            "/".join(str(counts[s].loc[aspect, lab]) for lab in ("olumlu", "olumsuz", "notr"))
            for s in SPLITS
        ]
        lines.append(f"| `{aspect}` | " + " | ".join(cells) + " |")
    warnings = missing_classes(counts)
    lines += ["", "### Classes missing from val or test", ""]
    lines += [f"- {w}" for w in warnings] if warnings else ["- none"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True, help="final_labels.csv")
    parser.add_argument(
        "--test-ids", type=Path, required=True, help="test kümesi (review_id sütunu olan CSV)"
    )
    parser.add_argument("--out", type=Path, default=Path("data/interim/splits/split_v1.csv"))
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    test_ids = set(
        pd.read_csv(args.test_ids, dtype=str, keep_default_na=False, encoding="utf-8-sig")[
            "review_id"
        ].str.strip()
    )
    splits, dropped = assign_splits(labels, test_ids, args.val_size, args.seed)
    report = render_report(splits, dropped, split_counts(labels, splits))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(args.out, index=False)
    report_path = args.out.with_name(args.out.stem + "_report.md")
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nYazıldı: {args.out}, {report_path}")


if __name__ == "__main__":
    main()
