"""Measure model loading and prediction time on CPU (evidence for NFR-1 and NFR-2).

Usage (needs the model in models/berturk_v1/, see README "Run the app"):
    uv run python scripts/measure_latency.py --texts data/interim/kappa_final.csv

Run it on an otherwise idle machine. The first start after copying the model can be much slower
because the 440 MB weight file is not yet in the operating system's disk cache.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True, help="CSV with a `text` column")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    os.environ.pop("ABSA_BACKEND", None)
    t0 = time.perf_counter()
    import pandas as pd
    import torch

    from absa import inference

    t_import = time.perf_counter() - t0
    t1 = time.perf_counter()
    backend = inference.load()
    t_load = time.perf_counter() - t1
    if backend != "berturk":
        raise SystemExit(f"Expected the BERTurk backend, got {backend!r}")

    texts = pd.read_csv(args.texts, dtype=str)["text"].dropna().tolist()[: args.limit]
    inference.predict(texts[0])  # warm-up
    times = []
    for text in texts:
        start = time.perf_counter()
        inference.predict(text)
        times.append(time.perf_counter() - start)
    times.sort()
    p95 = times[int(0.95 * (len(times) - 1))]

    long_text = "Ürün çok kaliteli ama kargo geç geldi. " * 300
    start = time.perf_counter()
    inference.predict(long_text)
    t_long = time.perf_counter() - start

    print(f"Python {sys.version.split()[0]}, torch {torch.__version__}, "
          f"{torch.get_num_threads()} threads, {platform.platform()}")  # fmt: skip
    print(f"NFR-2  import: {t_import:.1f} s, load(): {t_load:.1f} s, "
          f"total {t_import + t_load:.1f} s (limit 60 s)")  # fmt: skip
    print(f"NFR-1  predict() on {len(times)} reviews: median {statistics.median(times):.3f} s, "
          f"p95 {p95:.3f} s, max {max(times):.3f} s (limit 2 s)")  # fmt: skip
    print(f"NFR-1  predict() on a {len(long_text):,}-character review: {t_long:.3f} s")


if __name__ == "__main__":
    main()
