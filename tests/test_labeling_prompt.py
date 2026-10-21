import re

import pytest

from absa.labeling.prompt import (
    PROMPT_VERSION,
    available_versions,
    build_messages,
    load_system_prompt,
)
from absa.labeling.validate import parse_labels
from absa.schema import ASPECTS, LABELS

JSON_BLOCK = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def test_varsayilan_surum_mevcut():
    assert PROMPT_VERSION in available_versions()


@pytest.mark.parametrize("version", available_versions())
def test_prompt_tum_aspect_ve_etiketleri_icerir(version):
    prompt = load_system_prompt(version)
    for key in [*ASPECTS, *LABELS]:
        assert f"`{key}`" in prompt, key


@pytest.mark.parametrize("version", available_versions())
def test_prompttaki_ornekler_semaya_uyar(version):
    blocks = JSON_BLOCK.findall(load_system_prompt(version))
    assert len(blocks) >= 4
    for block in blocks:
        parse_labels(block)


def test_mesajlar_yorumu_icerir():
    messages = build_messages("  Kargo hızlıydı 😊 ")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert 'Review: "Kargo hızlıydı 😊"' in messages[1]["content"]


def test_bilinmeyen_surum_hata_verir():
    with pytest.raises(ValueError):
        load_system_prompt("v999")
