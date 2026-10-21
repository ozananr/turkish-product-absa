"""Tek bir yorumu LLM ile etiketler: geçersiz cevapta düzeltme isteyerek yeniden dener."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from absa.labeling.client import ChatClient
from absa.labeling.prompt import PROMPT_VERSION, build_messages
from absa.labeling.validate import InvalidResponse, parse_labels

STATUS_OK = "ok"
STATUS_INVALID = "invalid"  # tüm denemelerde şemaya uymayan cevap
STATUS_ERROR = "error"  # API hatası (ağ, kota, içerik filtresi...); sonraki çalıştırmada denenir

MAX_ATTEMPTS = 3


@dataclass
class LabelResult:
    review_id: str
    status: str
    model: str
    prompt_version: str
    labels: dict[str, str] | None = None
    attempts: int = 0
    error: str | None = None
    raw: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def to_record(self) -> dict:
        return asdict(self)


def _correction(error: str) -> str:
    return (
        f"Your answer was invalid: {error}. Return only one JSON object with exactly the "
        "7 aspect keys and only the labels yok, olumlu, olumsuz, notr."
    )


def label_review(
    client: ChatClient,
    review_id: str,
    text: str,
    prompt_version: str = PROMPT_VERSION,
    max_attempts: int = MAX_ATTEMPTS,
) -> LabelResult:
    result = LabelResult(
        review_id=review_id,
        status=STATUS_INVALID,
        model=client.model,
        prompt_version=prompt_version,
    )
    messages = build_messages(text, prompt_version)
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        try:
            completion = client.complete(messages)
        except Exception as exc:  # SDK kendi yeniden denemelerini bitirdikten sonra kalan hata
            result.status = STATUS_ERROR
            result.error = f"{type(exc).__name__}: {exc}"[:500]
            return result

        result.input_tokens += completion.input_tokens
        result.output_tokens += completion.output_tokens
        result.cached_tokens += completion.cached_tokens
        result.raw = completion.text
        try:
            result.labels = parse_labels(completion.text)
        except InvalidResponse as exc:
            result.error = str(exc)
            messages = [
                *messages,
                {"role": "assistant", "content": completion.text or ""},
                {"role": "user", "content": _correction(str(exc))},
            ]
            continue
        result.status = STATUS_OK
        result.error = None
        return result
    return result
