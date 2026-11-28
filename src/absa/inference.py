"""predict(): eğitilmiş BERTurk modeliyle tahmin (issue #67). Sözleşme: docs/requirements.md.

Model, ilk kullanımda (ya da `load()` ile uygulama açılırken) BİR KEZ yüklenir ve bellekte kalır
(NFR-2). Yorum, eğitim verisine uygulanan temizleme ve kişisel veri maskelemesinden geçer,
128 token'da kesilir; her aspect için en olası etiket ve onun olasılığı (güven) döner.

Model klasörü: `models/berturk_v1/` (ya da ABSA_MODEL_DIR ortam değişkeni). Klasör Drive'dan
indirilir (absa-data/models/berturk_v1). ABSA_BACKEND=mock ile model olmadan sahte tahmin yapılır
(CI ve model indirilmeden arayüz geliştirme için).
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
from dataclasses import dataclass
from pathlib import Path

from .data.cleaner import normalize_text
from .data.masker import mask_personal_data
from .schema import ASPECTS, LABELS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "berturk_v1"
MODEL_MARKER = "absa_config.json"
MAX_LENGTH = 128


class ModelNotFoundError(FileNotFoundError):
    """Model klasörü yok ya da eksik."""


@dataclass
class _Backend:
    name: str  # "berturk" ya da "mock"
    model: object = None
    tokenizer: object = None
    max_length: int = MAX_LENGTH


_lock = threading.Lock()
_backend: _Backend | None = None


def model_dir() -> Path:
    return Path(os.environ.get("ABSA_MODEL_DIR", DEFAULT_MODEL_DIR))


def model_available(path: Path | None = None) -> bool:
    return ((path or model_dir()) / MODEL_MARKER).exists()


def _load_backend() -> _Backend:
    if os.environ.get("ABSA_BACKEND", "").lower() == "mock":
        return _Backend("mock")
    path = model_dir()
    if not model_available(path):
        raise ModelNotFoundError(
            f"Model bulunamadı: {path}\n"
            "Drive'daki absa-data/models/berturk_v1 klasörünü indirip bu yola koyun "
            "(README: 'Run the app'). Model olmadan denemek için: ABSA_BACKEND=mock"
        )
    from .models.berturk import load_model

    model, tokenizer, meta = load_model(path, device="cpu")
    return _Backend("berturk", model, tokenizer, meta.get("max_length", MAX_LENGTH))


def load() -> str:
    """Modeli (bir kez) yükler; kullanılan arka ucu döndürür: 'berturk' ya da 'mock'."""
    global _backend
    with _lock:
        if _backend is None:
            _backend = _load_backend()
        return _backend.name


def reset() -> None:
    """Yüklü modeli bırakır (testler ve ortam değişkeni değişince)."""
    global _backend
    with _lock:
        _backend = None


def preprocess(text: str) -> str:
    """Eğitim verisine uygulanan temizleme ve maskeleme."""
    return mask_personal_data(normalize_text(text))


def token_count(text: str) -> int:
    """Yorumun model için token sayısı ([CLS]/[SEP] dahil). Mock'ta kelime sayısı + 2."""
    load()
    clean = preprocess(text)
    if _backend.name == "mock":
        return len(clean.split()) + 2
    return len(_backend.tokenizer(clean, truncation=False)["input_ids"])


def is_truncated(text: str) -> bool:
    """Yorum model sınırından uzun olduğu için kesildi mi? (FR-7)"""
    return token_count(text) > _backend.max_length


def _validate(text: object) -> None:
    if not isinstance(text, str):
        raise TypeError("Girdi str olmalıdır.")
    if not text.strip():
        raise ValueError("Yorum metni boş olamaz.")


def _predict_mock(text: str) -> dict[str, dict[str, str | float]]:
    """Sahte tahmin: aynı girdi her zaman aynı çıktıyı verir."""
    normalized = " ".join(text.split()).casefold()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return {
        aspect: {
            "label": rng.choices(LABELS, weights=[4, 3, 2, 1], k=1)[0],
            "confidence": round(rng.uniform(0.50, 0.99), 2),
        }
        for aspect in ASPECTS
    }


def _predict_berturk(text: str) -> dict[str, dict[str, str | float]]:
    import torch

    enc = _backend.tokenizer(
        preprocess(text), truncation=True, max_length=_backend.max_length, return_tensors="pt"
    )
    with torch.inference_mode():
        logits = _backend.model(enc["input_ids"], enc["attention_mask"])[0]
        probs = torch.softmax(logits.float(), dim=-1)
    conf, idx = probs.max(dim=-1)
    return {
        aspect: {"label": LABELS[int(idx[i])], "confidence": round(float(conf[i]), 4)}
        for i, aspect in enumerate(ASPECTS)
    }


def predict(text: str) -> dict[str, dict[str, str | float]]:
    """Yorum için her aspect'in etiketini ve güven skorunu döndürür.

    Returns:
        {"kalite": {"label": "olumlu", "confidence": 0.91}, ...}
        Çıktı her zaman 7 aspect'i ASPECTS sırasında içerir.

    Raises:
        TypeError: Girdi str değilse.
        ValueError: Girdi boş ya da yalnızca boşluktan oluşuyorsa.
        ModelNotFoundError: Model klasörü yoksa (ABSA_BACKEND=mock değilse).
    """
    _validate(text)
    if load() == "mock":
        return _predict_mock(text)
    return _predict_berturk(text)
