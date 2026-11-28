"""Model dosyası yoksa (CI) testler sahte predict() ile, varsa gerçek BERTurk ile çalışır."""

import os

from absa import inference

if not inference.model_available():
    os.environ.setdefault("ABSA_BACKEND", "mock")
