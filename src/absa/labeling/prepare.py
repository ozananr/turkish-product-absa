"""Etiketleme dosyalarını hazırlar: kappa seti ve kişi başına batch'ler (issue #32).

Kullanım:
    uv run python -m absa.labeling.prepare \
        --annotators ozan salih yusuf --kappa-annotators ozan salih

Girdiler (data/interim/, Drive'dan indirilir):
    prelabels.csv       review_id, text ve 7 aspect için LLM ön etiketleri (#31)
    pilot_checked.csv   pilotta ön etiket görülerek kontrol edilen 50 yorum; kappa setine girmez

Çıktılar (data/interim/labeling/, git'e girmez, Google Sheets'e yüklenir):
    kappa_<isim>.csv      kappa setini etiketleyecek iki kişiye aynı 200 yorum, ön etiketsiz
    batch_<isim>.csv      kalan yorumlar, her kişiye çakışmasız bir parça, ön etiketli;
                          kappa yapan iki kişinin batch'i 200 yorum küçüktür, herkesin
                          toplam işi eşit olur; satırlar karışık sırada (yukarıdan
                          aşağı etiketlenir, yarıda kalsa da etiketli kısım temsilidir)
    kappa_prelabels.csv   kappa yorumlarının LLM etiketleri; etiketleyicilere verilmez,
                          sonra LLM ile insanları karşılaştırmak için saklanır
    assignment.md         hangi dosya kimin, kaç yorum (dokümana ve issue'ya kopyalanır)

Aynı girdiler ve aynı seed her zaman aynı dosyaları üretir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from absa.labeling.validate import is_valid_row
from absa.schema import ASPECTS

DEFAULT_KAPPA_SIZE = 200
DEFAULT_SEED = 42
# Etiketleyicinin göreceği sütunlar. Puan, kaynak etiketi, ürün adı gibi bilgiler
# bilerek dışarıda: etiket yalnızca metne göre verilir (boundary_rules.md 2.5).
SHEET_COLUMNS = ["review_id", "text", *ASPECTS, "annotator", "checked", "note"]


def load_prelabels(path: Path) -> pd.DataFrame:
    """prelabels.csv'yi okur, yalnızca gereken sütunları alır ve kontrol eder."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ["review_id", "text", *ASPECTS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: eksik sütun {missing}")
    df = df[["review_id", "text", *ASPECTS]]
    if df["review_id"].duplicated().any():
        raise ValueError(f"{path}: tekrar eden review_id var")
    invalid = [rid for rid, row in zip(df["review_id"], df[ASPECTS].to_dict("records"), strict=True)
               if not is_valid_row(row)]  # fmt: skip
    if invalid:
        raise ValueError(f"{path}: geçersiz etiketli satırlar {invalid[:5]}")
    # Sıralama, dosyadaki satır sırası değişse de aynı sonucu verir.
    return df.sort_values("review_id").reset_index(drop=True)


def load_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return set(pd.read_csv(path, dtype=str, usecols=["review_id"])["review_id"])


def select_kappa(df: pd.DataFrame, size: int, seed: int, exclude: set[str]) -> pd.DataFrame:
    """Kappa setini seçer. exclude içindeki yorumlar (pilot) seçilmez."""
    candidates = df[~df["review_id"].isin(exclude)]
    if len(candidates) < size:
        raise ValueError(f"Kappa seti için {size} yorum gerekli, {len(candidates)} aday var")
    return candidates.sample(n=size, random_state=seed).sort_values("review_id")


def batch_sizes(n_rest: int, names: list[str], kappa_names: list[str], kappa_size: int) -> dict:
    """Herkesin toplam işi (batch + kappa) eşit olacak şekilde batch boyutları.

    Kappa setini etiketleyen iki kişi o 200 yorumu fazladan etiketlediği için
    batch'leri o kadar küçük olur.
    """
    total = n_rest + kappa_size * len(kappa_names)
    base, extra = divmod(total, len(names))
    sizes = {}
    for i, name in enumerate(names):
        sizes[name] = base + (1 if i < extra else 0) - (kappa_size if name in kappa_names else 0)
    if min(sizes.values()) < 0:
        raise ValueError("Kappa seti, kalan yorumlara göre çok büyük")
    return sizes


def split_batches(df: pd.DataFrame, sizes: dict[str, int], seed: int) -> dict[str, pd.DataFrame]:
    """Yorumları karıştırıp verilen boyutlarda çakışmasız parçalara böler.

    Batch'ler karışık sırada kalır, review_id'ye göre sıralanmaz: ID'ler kaynağa göre
    başladığı için (trendyol_..., trsa_...) sıralı bir batch'in ilk yarısı tek kaynaktan
    olurdu. Karışık sırada, yarıda kalan bir batch'in etiketli kısmı da temsili olur.
    """
    shuffled = df.sample(frac=1, random_state=seed)
    bounds = np.cumsum([0, *sizes.values()])
    return {
        name: shuffled.iloc[start:end]
        for name, start, end in zip(sizes, bounds[:-1], bounds[1:], strict=True)
    }


def to_sheet(df: pd.DataFrame, annotator: str, with_prelabels: bool) -> pd.DataFrame:
    """Google Sheets'e yüklenecek tablo. Ön etiketsizse 7 aspect sütunu boş kalır."""
    sheet = df[["review_id", "text", *ASPECTS]].copy()
    if not with_prelabels:
        sheet[ASPECTS] = ""
    sheet["annotator"] = annotator
    sheet["checked"] = "FALSE"
    sheet["note"] = ""
    return sheet[SHEET_COLUMNS].reset_index(drop=True)


def prepare(
    prelabels: pd.DataFrame,
    annotators: list[str],
    kappa_annotators: list[str],
    exclude_from_kappa: set[str] = frozenset(),
    kappa_size: int = DEFAULT_KAPPA_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Dosya adı -> tablo. Dosyaları yazmaz; main() yazar, testler doğrudan kontrol eder."""
    if len(set(annotators)) != len(annotators):
        raise ValueError("Etiketleyici isimleri tekrar ediyor")
    if len(kappa_annotators) != 2 or not set(kappa_annotators) <= set(annotators):
        raise ValueError("--kappa-annotators, --annotators içinden tam olarak 2 kişi olmalı")

    kappa = select_kappa(prelabels, kappa_size, seed, exclude_from_kappa)
    rest = prelabels[~prelabels["review_id"].isin(kappa["review_id"])]

    files = {f"kappa_{n}.csv": to_sheet(kappa, n, with_prelabels=False) for n in kappa_annotators}
    sizes = batch_sizes(len(rest), annotators, kappa_annotators, kappa_size)
    for name, batch in split_batches(rest, sizes, seed).items():
        files[f"batch_{name}.csv"] = to_sheet(batch, name, with_prelabels=True)
    files["kappa_prelabels.csv"] = kappa.reset_index(drop=True)
    return files


def assignment_markdown(files: dict[str, pd.DataFrame], seed: int) -> str:
    lines = [
        f"Seed: {seed}. Source: `prelabels.csv` (LLM pre-labels, prompt v2).",
        "",
        "| File | Annotator | Reviews | Pre-labels |",
        "|---|---|---:|---|",
    ]
    for name, df in files.items():
        if name == "kappa_prelabels.csv":
            continue
        prelabeled = "yes" if name.startswith("batch_") else "no (kappa set)"
        lines.append(f"| `{name}` | {df['annotator'].iloc[0]} | {len(df)} | {prelabeled} |")
    kappa = files["kappa_prelabels.csv"]
    lines.append(
        f"\n`kappa_prelabels.csv` ({len(kappa)} reviews): LLM labels of the kappa set. "
        "Not given to annotators; kept to compare the LLM with humans later."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prelabels", type=Path, default=Path("data/interim/prelabels.csv"))
    parser.add_argument(
        "--exclude-from-kappa", type=Path, default=Path("data/interim/pilot_checked.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("data/interim/labeling"))
    parser.add_argument("--annotators", nargs="+", required=True, help="ör. ozan salih yusuf")
    parser.add_argument("--kappa-annotators", nargs=2, required=True, help="ör. ozan salih")
    parser.add_argument("--kappa-size", type=int, default=DEFAULT_KAPPA_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    exclude = load_ids(args.exclude_from_kappa)
    files = prepare(
        load_prelabels(args.prelabels),
        args.annotators,
        args.kappa_annotators,
        exclude,
        args.kappa_size,
        args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in files.items():
        df.to_csv(args.out / name, index=False)
    summary = assignment_markdown(files, args.seed)
    (args.out / "assignment.md").write_text(summary, encoding="utf-8")
    print(f"Kappa setinden hariç tutulan yorum: {len(exclude)}")
    print(summary)
    print(f"\nYazıldı: {args.out}/")


if __name__ == "__main__":
    main()
