"""LLM ön etiketlerini elle düzeltilmiş etiketlerle karşılaştırır.

Kullanım (50 yorumluk kontrol):
    1. prelabels_pilot.csv dosyasını kopyalayın (ör. pilot_checked.csv),
       yanlış etiketleri düzeltin.
    2. uv run python -m absa.labeling.compare \
           --pred data/interim/prelabels_pilot.csv --gold data/interim/pilot_checked.csv

Aynı araç kappa setinde LLM ile insan etiketlerini karşılaştırmak için de kullanılabilir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from absa.schema import ASPECTS, NO_MENTION


def compare(pred: pd.DataFrame, gold: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """(özet, uyuşmazlık tablosu). Yalnızca iki tabloda da olan review_id'ler karşılaştırılır."""
    merged = pred.merge(gold, on="review_id", suffixes=("_pred", "_gold"))
    if merged.empty:
        raise ValueError("Ortak review_id yok.")

    per_aspect = {}
    rows = []
    for a in ASPECTS:
        p, g = merged[f"{a}_pred"], merged[f"{a}_gold"]
        both_mentioned = (p != NO_MENTION) & (g != NO_MENTION)
        per_aspect[a] = {
            "accuracy": round(float((p == g).mean()), 3),
            "kappa": round(float(cohen_kappa_score(p, g)), 3)
            if p.nunique() + g.nunique() > 2
            else None,
            "gold_mentioned": int((g != NO_MENTION).sum()),
            "llm_mentioned": int((p != NO_MENTION).sum()),
            "missed": int(((p == NO_MENTION) & (g != NO_MENTION)).sum()),
            "extra": int(((p != NO_MENTION) & (g == NO_MENTION)).sum()),
            "wrong_sentiment": int((both_mentioned & (p != g)).sum()),
        }
        for _, row in merged[p != g].iterrows():
            rows.append(
                {
                    "review_id": row["review_id"],
                    "aspect": a,
                    "llm": row[f"{a}_pred"],
                    "gold": row[f"{a}_gold"],
                    "text": row.get("text_pred", row.get("text", "")),
                }
            )

    pred_m = merged[[f"{a}_pred" for a in ASPECTS]].to_numpy()
    gold_m = merged[[f"{a}_gold" for a in ASPECTS]].to_numpy()
    summary = {
        "reviews": len(merged),
        "review_exact_match": round(float((pred_m == gold_m).all(axis=1).mean()), 3),
        "label_accuracy": round(float((pred_m == gold_m).mean()), 3),
        "per_aspect": per_aspect,
    }
    return summary, pd.DataFrame(rows, columns=["review_id", "aspect", "llm", "gold", "text"])


def render_markdown(s: dict) -> str:
    lines = [
        f"- **Reviews compared:** {s['reviews']}",
        f"- **All 7 labels correct:** {s['review_exact_match']:.1%} of reviews",
        f"- **Label accuracy (reviews x aspects):** {s['label_accuracy']:.1%}",
        "",
        "| Aspect | accuracy | kappa | mentioned (human / LLM) | missed | extra | wrong sent. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, m in s["per_aspect"].items():
        kappa = "-" if m["kappa"] is None else f"{m['kappa']:.2f}"
        lines.append(
            f"| `{a}` | {m['accuracy']:.0%} | {kappa} | {m['gold_mentioned']} / "
            f"{m['llm_mentioned']} | {m['missed']} | {m['extra']} | {m['wrong_sentiment']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred", type=Path, required=True, help="LLM ön etiketleri")
    parser.add_argument("--gold", type=Path, required=True, help="elle düzeltilmiş etiketler")
    parser.add_argument("--diff-out", type=Path, default=None, help="uyuşmazlıklar CSV")
    args = parser.parse_args(argv)

    summary, diff = compare(pd.read_csv(args.pred, dtype=str), pd.read_csv(args.gold, dtype=str))
    print(render_markdown(summary))
    diff_out = args.diff_out or args.pred.with_name(args.pred.stem + "_diff.csv")
    diff.to_csv(diff_out, index=False)
    print(f"\nUyuşmazlıklar: {diff_out} ({len(diff)} satır)")


if __name__ == "__main__":
    main()
