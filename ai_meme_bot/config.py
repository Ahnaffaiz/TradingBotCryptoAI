"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ai_meme_bot.models import StrategySettings


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError("{0} must be a number.".format(name)) from exc


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigError("{0} must be an integer.".format(name)) from exc


def _bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _id_set_env(name: str) -> frozenset[int]:
    values = set()
    for item in os.getenv(name, "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError as exc:
            raise ConfigError("{0} must contain comma-separated integers.".format(name)) from exc
    return frozenset(values)


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime settings for the bot."""

    trading_mode: str
    base_trade_amount: float
    db_path: Path
    ai_provider: str
    ai_base_url: str
    ai_api_key: str
    ai_model: str
    telegram_bot_token: Optional[str]
    telegram_admin_user_ids: frozenset[int]
    hermes_operator_enabled: bool
    dexscreener_profile_url: str
    dexscreener_token_url: str
    solana_rpc_url: Optional[str]
    x_bearer_token: Optional[str]
    x_recent_search_url: str
    x_search_minutes: int
    geckoterminal_trending_url: str
    tracker_poll_seconds: float
    position_review_seconds: float
    min_liquidity_usd: float
    min_pair_age_seconds: int
    entry_score_threshold: int
    reflection_time: str
    reflection_timezone: str

    @classmethod
    def from_env(
        cls,
        env_file: Optional[Path] = None,
        validate_ai: bool = True,
    ) -> "AppConfig":
        """Load `.env` values and validate the provider-neutral v1 contract."""

        load_dotenv(env_file)
        trading_mode = os.getenv("TRADING_MODE", "PAPER").strip().upper()
        if trading_mode not in {"PAPER", "REAL"}:
            raise ConfigError("TRADING_MODE must be PAPER or REAL.")

        base_trade_amount = _float_env("BASE_TRADE_AMOUNT", 0.1)
        if base_trade_amount <= 0:
            raise ConfigError("BASE_TRADE_AMOUNT must be greater than zero.")

        provider = os.getenv("AI_PROVIDER", "custom").strip().lower()
        base_url = os.getenv("AI_BASE_URL", "").strip()
        api_key = os.getenv("AI_API_KEY", "").strip()
        model = os.getenv("AI_MODEL", "").strip()
        if validate_ai and provider == "custom":
            missing = [
                name
                for name, value in (
                    ("AI_BASE_URL", base_url),
                    ("AI_API_KEY", api_key),
                    ("AI_MODEL", model),
                )
                if not value
            ]
            if missing:
                raise ConfigError(
                    "Custom AI provider requires {0}.".format(", ".join(missing))
                )
        if validate_ai and provider != "custom" and not model:
            raise ConfigError("AI_MODEL is required for AI_PROVIDER={0}.".format(provider))

        raw_db_path = Path(os.getenv("DB_PATH", "ai_meme_bot/database.db"))
        db_path = raw_db_path if raw_db_path.is_absolute() else Path.cwd() / raw_db_path
        rpc_url = os.getenv("HELIUS_RPC_URL", "").strip() or os.getenv(
            "SOLANA_RPC_URL", ""
        ).strip()

        entry_threshold = _int_env("ENTRY_SCORE_THRESHOLD", 80)
        if entry_threshold < 0 or entry_threshold > 100:
            raise ConfigError("ENTRY_SCORE_THRESHOLD must be between 0 and 100.")

        return cls(
            trading_mode=trading_mode,
            base_trade_amount=base_trade_amount,
            db_path=db_path,
            ai_provider=provider,
            ai_base_url=base_url,
            ai_api_key=api_key,
            ai_model=model,
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None,
            telegram_admin_user_ids=_id_set_env("TELEGRAM_ADMIN_USER_IDS"),
            hermes_operator_enabled=_bool_env("HERMES_OPERATOR_ENABLED"),
            dexscreener_profile_url=os.getenv(
                "DEXSCREENER_PROFILE_URL",
                "https://api.dexscreener.com/token-profiles/latest/v1",
            ).strip(),
            dexscreener_token_url=os.getenv(
                "DEXSCREENER_TOKEN_URL",
                "https://api.dexscreener.com/tokens/v1",
            ).rstrip("/"),
            solana_rpc_url=rpc_url or None,
            x_bearer_token=os.getenv("X_BEARER_TOKEN", "").strip() or None,
            x_recent_search_url=os.getenv(
                "X_RECENT_SEARCH_URL", "https://api.x.com/2/tweets/search/recent"
            ).strip(),
            x_search_minutes=_int_env("X_SEARCH_MINUTES", 30),
            geckoterminal_trending_url=os.getenv(
                "GECKOTERMINAL_TRENDING_URL",
                "https://api.geckoterminal.com/api/v2/networks/solana/trending_pools",
            ).strip(),
            tracker_poll_seconds=_float_env("TRACKER_POLL_SECONDS", 30.0),
            position_review_seconds=_float_env("POSITION_REVIEW_SECONDS", 45.0),
            min_liquidity_usd=_float_env("MIN_LIQUIDITY_USD", 10000.0),
            min_pair_age_seconds=_int_env("MIN_PAIR_AGE_SECONDS", 60),
            entry_score_threshold=entry_threshold,
            reflection_time=os.getenv("REFLECTION_TIME", "00:00").strip(),
            reflection_timezone=os.getenv("REFLECTION_TIMEZONE", "Asia/Jakarta").strip(),
        )

    @property
    def ai_identity(self) -> str:
        """Return a status-safe provider/model name."""

        return "{0}:{1}".format(self.ai_provider, self.ai_model or "unset")

    @property
    def strategy_defaults(self) -> StrategySettings:
        """Return startup strategy settings before any reflection tuning."""

        return StrategySettings(
            entry_score_threshold=self.entry_score_threshold,
            tracker_poll_seconds=self.tracker_poll_seconds,
            base_trade_amount=self.base_trade_amount,
            position_review_seconds=self.position_review_seconds,
            reflection_time=self.reflection_time,
        )
