import os

import pytest

from ai_meme_bot.config import AppConfig, ConfigError


def _clear_ai_env(monkeypatch):
    for name in (
        "TRADING_MODE",
        "AI_PROVIDER",
        "AI_BASE_URL",
        "AI_API_KEY",
        "AI_MODEL",
        "DB_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_custom_provider_config_and_safe_identity(monkeypatch, tmp_path):
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "custom")
    monkeypatch.setenv("AI_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("AI_API_KEY", "do-not-print")
    monkeypatch.setenv("AI_MODEL", "model-a")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "trades.db"))

    config = AppConfig.from_env()

    assert config.trading_mode == "PAPER"
    assert config.ai_identity == "custom:model-a"
    assert "do-not-print" not in config.ai_identity


@pytest.mark.parametrize("missing", ["AI_BASE_URL", "AI_API_KEY", "AI_MODEL"])
def test_custom_provider_requires_all_ai_values(monkeypatch, missing):
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "custom")
    values = {
        "AI_BASE_URL": "https://router.example/v1",
        "AI_API_KEY": "token",
        "AI_MODEL": "model-a",
    }
    for name, value in values.items():
        if name != missing:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv(missing, "")

    with pytest.raises(ConfigError, match=missing):
        AppConfig.from_env()
