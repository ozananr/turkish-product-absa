import json

from absa.labeling.cache import JsonlCache
from absa.labeling.client import Completion
from absa.labeling.labeler import STATUS_ERROR, STATUS_INVALID, STATUS_OK, label_review
from absa.schema import ASPECTS

VALID = json.dumps({a: "yok" for a in ASPECTS} | {"fiyat": "olumlu"})


class FakeClient:
    """Sırayla verilen cevapları döndürür; Exception ise fırlatır."""

    model = "fake-model"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Completion(text=answer, input_tokens=100, output_tokens=20, cached_tokens=50)


def test_ilk_denemede_gecerli():
    client = FakeClient([VALID])
    r = label_review(client, "r1", "Fiyatı çok uygun")
    assert r.status == STATUS_OK and r.attempts == 1
    assert r.labels["fiyat"] == "olumlu"
    assert (r.input_tokens, r.output_tokens, r.cached_tokens) == (100, 20, 50)


def test_gecersiz_cevapta_duzeltme_ister_ve_tekrar_dener():
    client = FakeClient(["bozuk", VALID])
    r = label_review(client, "r1", "Fiyatı çok uygun")
    assert r.status == STATUS_OK and r.attempts == 2
    assert r.input_tokens == 200  # iki denemenin token'ları toplanır
    second = client.calls[1]
    assert second[-2] == {"role": "assistant", "content": "bozuk"}
    assert "invalid" in second[-1]["content"]


def test_hep_gecersizse_invalid_olarak_isaretlenir():
    client = FakeClient(["{}", "{}", "{}"])
    r = label_review(client, "r1", "metin", max_attempts=3)
    assert r.status == STATUS_INVALID and r.attempts == 3
    assert r.labels is None and "eksik aspect" in r.error


def test_api_hatasi_error_olarak_isaretlenir():
    r = label_review(FakeClient([TimeoutError("zaman aşımı")]), "r1", "metin")
    assert r.status == STATUS_ERROR and "TimeoutError" in r.error


def test_onbellek_model_ve_surume_gore_ayrilir(tmp_path):
    path = tmp_path / "cache.jsonl"
    a = JsonlCache(path, "m1", "v1")
    a.append({"review_id": "r1", "model": "m1", "prompt_version": "v1", "status": "ok"})
    a.append({"review_id": "r2", "model": "m2", "prompt_version": "v1", "status": "ok"})
    a.append({"review_id": "r1", "model": "m1", "prompt_version": "v1", "status": "error"})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"review_id": "r3", "mod')  # yarım kalmış satır
    records = a.load()
    assert set(records) == {"r1"}
    assert records["r1"]["status"] == "error"  # son kayıt geçerlidir
    assert JsonlCache(path, "m2", "v1").load().keys() == {"r2"}
