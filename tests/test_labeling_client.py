import pytest

from absa.labeling.client import LLMConfig


@pytest.fixture
def env(monkeypatch):
    for key in ("LLM_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_TEMPERATURE",
                "LLM_JSON_MODE", "LLM_REASONING_EFFORT"):  # fmt: skip
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # gerçek .env okunmasın
    return monkeypatch


def test_ayarlar_ortamdan_okunur(env):
    env.setenv("LLM_API_KEY", "k")
    env.setenv("LLM_MODEL", "gemini-x")
    env.setenv("LLM_BASE_URL", "https://example.com/v1/")
    env.setenv("LLM_REASONING_EFFORT", "none")
    cfg = LLMConfig.from_env()
    assert (cfg.model, cfg.base_url, cfg.temperature, cfg.json_mode) == (
        "gemini-x",
        "https://example.com/v1/",
        0.0,
        True,
    )
    assert cfg.reasoning_effort == "none"


def test_openai_anahtarina_geri_duser_ve_cli_modeli_onceliklidir(env):
    env.setenv("OPENAI_API_KEY", "k")
    env.setenv("LLM_MODEL", "env-model")
    env.setenv("LLM_TEMPERATURE", "none")
    cfg = LLMConfig.from_env(model="cli-model")
    assert cfg.api_key == "k" and cfg.model == "cli-model" and cfg.temperature is None


def test_anahtar_ya_da_model_yoksa_hata(env):
    with pytest.raises(RuntimeError):
        LLMConfig.from_env(model="m")
    env.setenv("LLM_API_KEY", "k")
    with pytest.raises(RuntimeError):
        LLMConfig.from_env()
