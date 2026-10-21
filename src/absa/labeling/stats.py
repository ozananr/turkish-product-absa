"""Ön etiketleme çalıştırmasının istatistikleri (issue raporu için)."""

from __future__ import annotations

import pandas as pd

from absa.labeling.labeler import STATUS_ERROR, STATUS_INVALID, STATUS_OK
from absa.schema import ASPECTS, LABELS, NO_MENTION


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    price_in: float = 0.0,
    price_out: float = 0.0,
    price_cached: float | None = None,
) -> float:
    """Fiyatlar 1M token başına USD. price_cached verilmezse önbellekli token tam fiyattır."""
    cached_price = price_in if price_cached is None else price_cached
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * price_in + cached_tokens * cached_price + output_tokens * price_out) / 1e6


def compute_stats(records: list[dict], prices: dict[str, float | None] | None = None) -> dict:
    prices = prices or {}
    df = pd.DataFrame(records)
    ok = df[df["status"] == STATUS_OK]
    labels = pd.DataFrame(list(ok["labels"]), columns=ASPECTS) if len(ok) else None

    per_aspect = {}
    for aspect in ASPECTS:
        counts = labels[aspect].value_counts() if labels is not None else pd.Series(dtype=int)
        per_aspect[aspect] = {label: int(counts.get(label, 0)) for label in LABELS}

    tokens = {k: int(df[k].sum()) for k in ("input_tokens", "output_tokens", "cached_tokens")}
    has_prices = bool(prices.get("price_in") or prices.get("price_out"))
    return {
        "model": df["model"].iloc[0] if len(df) else None,
        "prompt_version": df["prompt_version"].iloc[0] if len(df) else None,
        "reviews": len(df),
        "ok": int((df["status"] == STATUS_OK).sum()),
        "invalid": int((df["status"] == STATUS_INVALID).sum()),
        "error": int((df["status"] == STATUS_ERROR).sum()),
        "retried": int((df["attempts"] > 1).sum()),
        "label_distribution": per_aspect,
        "mention_rate_pct": {
            a: round(100 * (1 - per_aspect[a][NO_MENTION] / max(len(ok), 1)), 1) for a in ASPECTS
        },
        "all_yok": int((labels == NO_MENTION).all(axis=1).sum()) if labels is not None else 0,
        "tokens": tokens,
        "cost_usd": round(estimate_cost(**tokens, **prices), 4) if has_prices else None,
        "prices_per_1m": prices if has_prices else None,
    }


def render_markdown(s: dict) -> str:
    lines = [
        f"- **Model / prompt:** `{s['model']}` / `{s['prompt_version']}`",
        f"- **Reviews:** {s['reviews']:,} — ok: {s['ok']:,}, invalid: {s['invalid']:,}, "
        f"API error: {s['error']:,} (retried after an invalid answer: {s['retried']:,})",
        f"- **Reviews with all 7 aspects `yok`:** {s['all_yok']:,}",
        f"- **Tokens:** input {s['tokens']['input_tokens']:,} "
        f"(cached {s['tokens']['cached_tokens']:,}), output {s['tokens']['output_tokens']:,}",
        "- **Total cost:** "
        + (f"${s['cost_usd']:.4f}" if s["cost_usd"] is not None else "not computed (no prices)"),
        "",
        "| Aspect | " + " | ".join(LABELS) + " | mentioned % |",
        "|---|" + "---:|" * (len(LABELS) + 1),
    ]
    for aspect, counts in s["label_distribution"].items():
        cells = " | ".join(f"{counts[label]:,}" for label in LABELS)
        lines.append(f"| `{aspect}` | {cells} | {s['mention_rate_pct'][aspect]} |")
    return "\n".join(lines)
