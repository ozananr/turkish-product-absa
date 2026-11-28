"""İşlevsel olmayan gereksinimlerin ölçümü (issue #68): NFR-1 tahmin süresi, NFR-2 yükleme süresi.

Komut yeni bir süreçte çalışır: torch/transformers içe aktarma + model yükleme süresi "soğuk
başlangıç" yükleme süresidir (NFR-2, ≤ 60 sn). Sonra birkaç ısınma tahmini yapılır ve N yorumla
tek tek tahmin süresi ölçülür (NFR-1, CPU'da ≤ 2 sn): medyan, %95'lik değer ve en yüksek değer.

Kullanım:
    uv run python -m absa.nfr                      # yerleşik örnek yorumlar, 100 tahmin
    uv run python -m absa.nfr --texts data/interim/labeling/final_labels.csv --n 100

Çıktılar (--out, varsayılan outputs/nfr/; git'e girmez): report.md, results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

NFR1_LIMIT_S = 2.0
NFR2_LIMIT_S = 60.0

# Yerleşik örnekler: kısa, orta, uzun (128 token'ı aşan) yorumlar; kişisel veri yok.
SAMPLE_TEXTS = [
    "Ürün çok kaliteli ama kargo çok geç geldi.",
    "Fiyatına göre gayet iyi, görüntüsü de resimdeki gibi.",
    "İki günde bozuldu, satıcı da iade etmedi.",
    "Bir beden büyük geldi ama kumaşı güzel.",
    "Berbat.",
    "Hızlı kargo, özenli paketleme, teşekkürler 😊",
    "Rengi fotoğraftakinden daha koyu ama kalıbı tam oldu, kumaşı da yumuşak.",
    "Kokusu güzel fakat kalıcılığı çok kısa, fiyatı da biraz pahalı.",
    "Ürün hiç beklediğim gibi değil, iade etmek istiyorum ama satıcı cevap vermiyor.",
    (
        "Uzun süredir kullanıyorum ve genel olarak memnunum. İlk siparişimde kutu biraz ezik "
        "gelmişti ama ürün sağlamdı. Bu sefer kargo çok hızlıydı, paketleme de özenliydi. "
        "Malzemesi kaliteli, dikişleri düzgün, yıkamada rengi solmadı. Bedeni tam oldu, "
        "normalde M giyiyorum ve M aldım. Fiyatı diğer mağazalara göre biraz yüksek olsa da "
        "kalitesine değdiğini düşünüyorum. Satıcı sorularıma da hızlı cevap verdi. Tek "
        "eksiği renk seçeneklerinin az olması, daha fazla renk olsa iki tane daha alırdım. "
        "Herkese tavsiye ederim, özellikle günlük kullanım için çok rahat bir ürün."
    ),
]
# Uzun örneği ikiye katla: 128 token'ı kesin aşsın (FR-7 kesme yolu da ölçülsün).
SAMPLE_TEXTS.append(SAMPLE_TEXTS[-1] + " " + SAMPLE_TEXTS[-1])


def summarize(times_s: list[float]) -> dict[str, float]:
    ordered = sorted(times_s)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "n": len(ordered),
        "median_s": statistics.median(ordered),
        "p95_s": p95,
        "max_s": ordered[-1],
        "mean_s": statistics.fmean(ordered),
    }


def verdict(load_s: float, stats: dict[str, float]) -> dict[str, bool]:
    """NFR-1 için %95'lik değer, NFR-2 için yükleme süresi kullanılır."""
    return {"NFR-1": stats["p95_s"] <= NFR1_LIMIT_S, "NFR-2": load_s <= NFR2_LIMIT_S}


def load_texts(path: Path | None, n: int) -> list[str]:
    if path is None:
        texts = SAMPLE_TEXTS
    else:
        import pandas as pd

        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        texts = df["text"].sample(min(n, len(df)), random_state=42).tolist()
    return [texts[i % len(texts)] for i in range(n)]


def environment() -> dict[str, str]:
    import torch

    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "torch_threads": str(torch.get_num_threads()),
        "device": "CPU",
    }


def render_report(res: dict) -> str:
    s, v, env = res["predict"], res["verdict"], res["environment"]

    def mark(ok: bool) -> str:
        return "pass" if ok else "**fail**"

    return "\n".join(
        [
            f"- Environment: {env['os']}, Python {env['python']}, {env['cpu']}, torch "
            f"{env['torch']} ({env['torch_threads']} threads), {env['device']}",
            f"- Backend: `{res['backend']}`; {s['n']} single predictions after "
            f"{res['warmup']} warm-up calls; {res['truncated']} of the inputs were truncated "
            "at 128 tokens",
            "",
            "| Requirement | Limit | Measured | Result |",
            "|---|---|---|---|",
            f"| NFR-2 model load (cold start, incl. imports) | ≤ {NFR2_LIMIT_S:.0f} s "
            f"| {res['load_s']:.1f} s | {mark(v['NFR-2'])} |",
            f"| NFR-1 single prediction, 95th percentile | ≤ {NFR1_LIMIT_S:.0f} s "
            f"| {s['p95_s'] * 1000:.0f} ms | {mark(v['NFR-1'])} |",
            f"| NFR-1 single prediction, median / max | – "
            f"| {s['median_s'] * 1000:.0f} ms / {s['max_s'] * 1000:.0f} ms | – |",
        ]
    )


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--texts", type=Path, help="text sütunu olan CSV (yoksa yerleşik örnekler)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("outputs/nfr"))
    args = parser.parse_args(argv)

    start = time.perf_counter()
    from absa import inference

    backend = inference.load()
    load_s = time.perf_counter() - start

    texts = load_texts(args.texts, args.n)
    for text in texts[: args.warmup]:
        inference.predict(text)
    times = []
    for text in texts:
        t0 = time.perf_counter()
        inference.predict(text)
        times.append(time.perf_counter() - t0)

    stats = summarize(times)
    res = {
        "backend": backend,
        "load_s": load_s,
        "warmup": args.warmup,
        "predict": stats,
        "truncated": sum(inference.is_truncated(t) for t in texts),
        "verdict": verdict(load_s, stats),
        "environment": environment(),
    }
    report = render_report(res)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(report)
    print(f"\nYazıldı: {args.out}/ (report.md, results.json)")
    return res


if __name__ == "__main__":
    main()
