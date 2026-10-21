import json

import pandas as pd

from absa.labeling.cache import JsonlCache
from absa.labeling.client import Completion
from absa.labeling.run import OUTPUT_COLUMNS, build_outputs, run, select_reviews
from absa.labeling.stats import compute_stats, estimate_cost, render_markdown
from absa.labeling.validate import is_valid_row
from absa.schema import ASPECTS, LABELS

VALID = json.dumps({a: "yok" for a in ASPECTS} | {"kargo_paketleme": "olumsuz"})


class ScriptedClient:
    """review metnine göre cevap verir: 'BOZUK' geçersiz, 'HATA' API hatası."""

    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        text = messages[1]["content"]
        if "HATA" in text:
            raise ConnectionError("ağ yok")
        answer = "geçersiz" if "BOZUK" in text else VALID
        return Completion(text=answer, input_tokens=1000, output_tokens=50, cached_tokens=800)


def _reviews(n=6):
    texts = [f"Kargo geç geldi {i}" for i in range(n)]
    texts[1] = "BOZUK cevap"
    texts[2] = "HATA verir"
    return pd.DataFrame({"review_id": [f"r{i}" for i in range(n)], "text": texts})


def test_pilot_secimi_sabit_seed_ile_ayni():
    df = pd.DataFrame({"review_id": [f"r{i:03d}" for i in range(200)], "text": "x"})
    a = select_reviews(df, 50, seed=42)
    b = select_reviews(df.sample(frac=1, random_state=3), 50, seed=42)
    assert len(a) == 50 and a["review_id"].tolist() == b["review_id"].tolist()
    assert len(select_reviews(df, None, 42)) == 200


def test_run_onbellekten_devam_eder(tmp_path):
    cache = JsonlCache(tmp_path / "c.jsonl", "fake-model", "v1")
    client = ScriptedClient()
    records = run(_reviews(), client, cache, workers=2, max_attempts=2, progress=False)
    assert [r["status"] for r in records] == ["ok", "invalid", "error", "ok", "ok", "ok"]
    assert client.calls == 4 + 2 + 1  # 4 geçerli, 1x2 geçersiz deneme, 1 hata

    # İkinci çalıştırma: yalnızca API hatası olan yorum tekrar denenir.
    client2 = ScriptedClient()
    run(_reviews(), client2, cache, workers=2, max_attempts=2, progress=False)
    assert client2.calls == 1

    # --retry-invalid ile geçersiz olan da tekrar denenir.
    client3 = ScriptedClient()
    run(_reviews(), client3, cache, retry_invalid=True, max_attempts=2, progress=False)
    assert client3.calls == 1 + 2


def test_ciktilar_yalnizca_gecerli_satirlari_icerir(tmp_path):
    reviews = _reviews()
    cache = JsonlCache(tmp_path / "c.jsonl", "fake-model", "v1")
    records = run(reviews, ScriptedClient(), cache, max_attempts=2, progress=False)
    prelabels, failed = build_outputs(reviews, records)
    assert prelabels.columns.tolist() == OUTPUT_COLUMNS
    assert len(prelabels) == 4 and set(failed["status"]) == {"invalid", "error"}
    for _, row in prelabels.iterrows():
        assert is_valid_row({a: row[a] for a in ASPECTS})


def test_istatistik_ve_maliyet(tmp_path):
    cache = JsonlCache(tmp_path / "c.jsonl", "fake-model", "v1")
    records = run(_reviews(), ScriptedClient(), cache, max_attempts=2, progress=False)
    stats = compute_stats(records, {"price_in": 1.0, "price_out": 4.0, "price_cached": 0.1})
    assert (stats["reviews"], stats["ok"], stats["invalid"], stats["error"]) == (6, 4, 1, 1)
    assert stats["label_distribution"]["kargo_paketleme"]["olumsuz"] == 4
    assert all(sum(c.values()) == 4 for c in stats["label_distribution"].values())
    assert set(stats["label_distribution"]["fiyat"]) == set(LABELS)
    # 6 başarılı çağrı (4 ok + 2 invalid deneme): 6000 girdi (4800 cached), 300 çıktı
    expected = estimate_cost(6000, 300, 4800, 1.0, 4.0, 0.1)
    assert stats["cost_usd"] == round(expected, 4)
    assert "kargo_paketleme" in render_markdown(stats)


def test_maliyet_formulu():
    assert estimate_cost(1_000_000, 1_000_000, 0, 0.5, 2.0) == 2.5
    assert estimate_cost(1_000_000, 0, 1_000_000, 0.5, 2.0, 0.05) == 0.05
