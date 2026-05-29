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
        "ENTRY_SCORE_THRESHOLD",
        "SCOUT_ENABLED",
        "MIN_TRADE_AMOUNT_SOL",
        "MAX_TRADE_AMOUNT_SOL",
        "POSITION_REVIEW_SECONDS",
        "BLOCKED_ENTRY_UTC_HOURS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_custom_provider_config_and_safe_identity(monkeypatch, tmp_path):
    _clear_ai_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "custom")
    monkeypatch.setenv("AI_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("AI_API_KEY", "do-not-print")
    monkeypatch.setenv("AI_MODEL", "model-a")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "trades.db"))

    config = AppConfig.from_env(env_file=tmp_path / "missing.env")

    assert config.trading_mode == "PAPER"
    assert config.entry_score_threshold == 25
    assert config.scout_enabled is False
    assert config.position_review_seconds == 15.0
    assert config.min_trade_amount_sol == 0.1
    assert config.max_trade_amount_sol == 0.3
    assert config.blocked_entry_utc_hours == "20"
    assert config.realtime_price_feed_enabled is False
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
