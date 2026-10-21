"""Ön etiketleme prompt'u.

Sistem prompt'u `prompts/<sürüm>.md` dosyasındadır ve docs/annotation/aspects.md ile
docs/annotation/boundary_rules.md'den türetilmiştir. Prompt değişince yeni bir sürüm
dosyası eklenir (v1 -> v2), böylece önbellek ve sonuçlar hangi prompt'la üretildiğini bilir.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_VERSION = "v1"


def available_versions() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("v*.md"))


@cache
def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    path = PROMPTS_DIR / f"{version}.md"
    if not path.is_file():
        raise ValueError(f"Prompt sürümü bulunamadı: {version} (mevcut: {available_versions()})")
    return path.read_text(encoding="utf-8").strip()


def build_messages(text: str, version: str = PROMPT_VERSION) -> list[dict[str, str]]:
    """Chat API için mesaj listesi. Yorum metni ayraçlar arasında verilir."""
    user = f'Review: "{text.strip()}"\n\nReturn only the JSON object.'
    return [
        {"role": "system", "content": load_system_prompt(version)},
        {"role": "user", "content": user},
    ]
