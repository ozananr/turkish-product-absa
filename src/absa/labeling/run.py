"""Etiketleme örneğini LLM ile ön etiketler.

Kullanım:
    # 1) 50 yorumluk deneme (rastgele, sabit seed)
    uv run python -m absa.labeling.run --limit 50 --out data/interim/prelabels_pilot.csv
    # 2) Tam çalıştırma
    uv run python -m absa.labeling.run --out data/interim/prelabels.csv

Ayarlar .env'den okunur (LLM_API_KEY, LLM_MODEL, LLM_BASE_URL). Her sonuç geldiği anda
önbelleğe (--cache) yazılır; komut yarıda kesilirse aynı komutu tekrar çalıştırmak yeterlidir.

Çıktılar (hepsi data/ altında, git'e girmez):
    <out>.csv              review_id, text ve 7 aspect etiketi (yalnızca geçerli cevaplar)
    <out>_failed.csv       geçersiz cevap / API hatası olan yorumlar
    <out>_stats.json / .md istatistikler ve maliyet
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from absa.labeling.cache import JsonlCache
from absa.labeling.client import ChatClient, LLMConfig, OpenAIChatClient
from absa.labeling.labeler import (
    MAX_ATTEMPTS,
    STATUS_ERROR,
    STATUS_INVALID,
    STATUS_OK,
    label_review,
)
from absa.labeling.prompt import PROMPT_VERSION, available_versions
from absa.labeling.stats import compute_stats, render_markdown
from absa.labeling.validate import is_valid_row
from absa.schema import ASPECTS

OUTPUT_COLUMNS = ["review_id", "text", *ASPECTS]


def select_reviews(sample: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """--limit verilirse sabit seed ile rastgele alt örnek (pilot) seçer."""
    sample = sample.sort_values("review_id").reset_index(drop=True)
    if limit and limit < len(sample):
        sample = sample.sample(n=limit, random_state=seed).sort_values("review_id")
    return sample.reset_index(drop=True)


def run(
    reviews: pd.DataFrame,
    client: ChatClient,
    cache: JsonlCache,
    workers: int = 4,
    retry_invalid: bool = False,
    max_attempts: int = MAX_ATTEMPTS,
    progress: bool = True,
) -> list[dict]:
    """Önbellekte olmayanları etiketler, örnekteki tüm yorumların kayıtlarını döner."""
    cached = cache.load()
    redo = {STATUS_ERROR} | ({STATUS_INVALID} if retry_invalid else set())
    todo = reviews[
        [rid not in cached or cached[rid]["status"] in redo for rid in reviews["review_id"]]
    ]
    print(f"{len(reviews)} yorum: {len(reviews) - len(todo)} önbellekte, {len(todo)} etiketlenecek")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(label_review, client, rid, text, cache.prompt_version, max_attempts)
            for rid, text in zip(todo["review_id"], todo["text"], strict=True)
        ]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result().to_record()
            cache.append(record)
            cached[record["review_id"]] = record
            if progress and (i % 25 == 0 or i == len(futures)):
                print(f"  {i}/{len(futures)}", file=sys.stderr)

    return [cached[rid] for rid in reviews["review_id"]]


def build_outputs(reviews: pd.DataFrame, records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(geçerli ön etiketler, başarısız yorumlar) tablolarını üretir."""
    texts = dict(zip(reviews["review_id"], reviews["text"], strict=True))
    ok_rows, failed_rows = [], []
    for r in records:
        if r["status"] == STATUS_OK and r["labels"] and is_valid_row(r["labels"]):
            ok_rows.append(
                {"review_id": r["review_id"], "text": texts[r["review_id"]]} | r["labels"]
            )
        else:
            failed_rows.append(
                {
                    "review_id": r["review_id"],
                    "text": texts[r["review_id"]],
                    "status": r["status"],
                    "error": r["error"],
                    "raw": r["raw"],
                }
            )
    failed_cols = ["review_id", "text", "status", "error", "raw"]
    return (
        pd.DataFrame(ok_rows, columns=OUTPUT_COLUMNS),
        pd.DataFrame(failed_rows, columns=failed_cols),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample", type=Path, default=Path("data/interim/labeling_sample.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/interim/prelabels.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/interim/prelabels_cache.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="pilot için yorum sayısı")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="varsayılan: .env içindeki LLM_MODEL")
    parser.add_argument("--base-url", default=None, help="varsayılan: .env içindeki LLM_BASE_URL")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, choices=available_versions())
    parser.add_argument("--workers", type=int, default=4, help="paralel istek sayısı")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument(
        "--retry-invalid", action="store_true", help="invalid kayıtları tekrar dene"
    )
    parser.add_argument("--price-in", type=float, default=0.0, help="USD / 1M girdi token")
    parser.add_argument("--price-out", type=float, default=0.0, help="USD / 1M çıktı token")
    parser.add_argument("--price-cached", type=float, default=None, help="USD / 1M cached token")
    args = parser.parse_args(argv)

    sample = pd.read_csv(args.sample, usecols=["review_id", "text"], dtype=str)
    reviews = select_reviews(sample, args.limit, args.seed)
    client = OpenAIChatClient(LLMConfig.from_env(model=args.model, base_url=args.base_url))
    cache = JsonlCache(args.cache, client.model, args.prompt_version)

    records = run(
        reviews,
        client,
        cache,
        workers=args.workers,
        retry_invalid=args.retry_invalid,
        max_attempts=args.max_attempts,
    )
    prelabels, failed = build_outputs(reviews, records)

    stem = args.out.with_suffix("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prelabels.to_csv(args.out, index=False)
    failed.to_csv(f"{stem}_failed.csv", index=False)

    prices = {"price_in": args.price_in, "price_out": args.price_out}
    if args.price_cached is not None:
        prices["price_cached"] = args.price_cached
    stats = compute_stats(records, prices)
    Path(f"{stem}_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(f"{stem}_stats.md").write_text(render_markdown(stats), encoding="utf-8")
    print(render_markdown(stats))
    print(
        f"\nYazıldı: {args.out} ({len(prelabels)} satır), {stem}_failed.csv ({len(failed)} satır)"
    )


if __name__ == "__main__":
    main()
