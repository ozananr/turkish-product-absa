"""Satır satır JSONL önbellek: her yorumun sonucu geldiği anda diske yazılır.

Çalışma yarıda kesilirse (çökme, Ctrl+C, kota) aynı komut tekrar çalıştırıldığında
önbellekteki yorumlar atlanır. Kayıtlar model ve prompt sürümüyle birlikte tutulur;
farklı model ya da prompt sürümünün sonuçları birbirine karışmaz.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path


class JsonlCache:
    def __init__(self, path: Path, model: str, prompt_version: str):
        self.path = Path(path)
        self.model = model
        self.prompt_version = prompt_version
        self._lock = threading.Lock()

    def _matches(self, record: dict) -> bool:
        return record.get("model") == self.model and (
            record.get("prompt_version") == self.prompt_version
        )

    def load(self) -> dict[str, dict]:
        """review_id -> son kayıt. Bozuk satırlar (yarım yazılmış son satır) atlanır."""
        records: dict[str, dict] = {}
        if not self.path.exists():
            return records
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self._matches(record):
                    records[record["review_id"]] = record
        return records

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
