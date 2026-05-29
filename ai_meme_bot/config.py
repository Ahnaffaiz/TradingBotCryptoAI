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


def _hour_list_env(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    value = default if raw_value is None or raw_value.strip() == "" else raw_value
    hours = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            hour = int(item)
        except ValueError as exc:
            raise ConfigError(
                "{0} must contain comma-separated UTC hours from 0 to 23.".format(name)
            ) from exc
        if hour < 0 or hour > 23:
            raise ConfigError("{0} hours must be between 0 and 23.".format(name))
        hours.append(str(hour))
    return ",".join(hours)


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
    birdeye_api_key: Optional[str]
    birdeye_ws_url: str
    realtime_price_feed_enabled: bool
    tracker_poll_seconds: float
    position_review_seconds: float
    min_liquidity_usd: float
    min_pair_age_seconds: int
    entry_score_threshold: int
    launch_enabled: bool
    scout_enabled: bool
    launch_score_threshold: int
    scout_score_threshold: int
    take_profit_pct: float
    stop_loss_pct: float
    trailing_stop_pct: float
    max_hold_seconds: float
    scout_min_liquidity_usd: float
    scout_min_volume_5m_usd: float
    reflection_time: str
    reflection_timezone: str
    min_trade_amount_sol: float = 0.1
    max_trade_amount_sol: float = 0.3
    blocked_entry_utc_hours: str = "20"
    min_buy_sell_ratio: float = 1.15
    min_volume_liquidity_ratio_5m: float = 0.03
    max_top_holder_share_pct: float = 35.0
    max_momentum_5m_pct: float = 80.0
    momentum_exhaustion_min_buy_sell_ratio: float = 2.0
    max_buy_more_count: int = 2
    buy_more_cooldown_seconds: float = 120.0

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
        min_trade_amount_sol = _float_env("MIN_TRADE_AMOUNT_SOL", 0.1)
        max_trade_amount_sol = _float_env("MAX_TRADE_AMOUNT_SOL", 0.3)
        if min_trade_amount_sol < 0.01 or min_trade_amount_sol > 2.0:
            raise ConfigError("MIN_TRADE_AMOUNT_SOL must be between 0.01 and 2.0.")
        if max_trade_amount_sol < 0.01 or max_trade_amount_sol > 2.0:
            raise ConfigError("MAX_TRADE_AMOUNT_SOL must be between 0.01 and 2.0.")
        if min_trade_amount_sol > max_trade_amount_sol:
            raise ConfigError(
                "MIN_TRADE_AMOUNT_SOL must be less than or equal to MAX_TRADE_AMOUNT_SOL."
            )

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

        entry_threshold = _int_env("ENTRY_SCORE_THRESHOLD", 25)
        if entry_threshold < 0 or entry_threshold > 100:
            raise ConfigError("ENTRY_SCORE_THRESHOLD must be between 0 and 100.")
        launch_threshold = _int_env("LAUNCH_SCORE_THRESHOLD", entry_threshold)
        scout_threshold = _int_env("SCOUT_SCORE_THRESHOLD", 70)
        if not 0 <= launch_threshold <= 100:
            raise ConfigError("LAUNCH_SCORE_THRESHOLD must be between 0 and 100.")
        if not 0 <= scout_threshold <= 100:
            raise ConfigError("SCOUT_SCORE_THRESHOLD must be between 0 and 100.")

        take_profit_pct = _float_env("TAKE_PROFIT_PCT", 18.0)
        stop_loss_pct = _float_env("STOP_LOSS_PCT", 8.0)
        trailing_stop_pct = _float_env("TRAILING_STOP_PCT", 7.0)
        max_hold_seconds = _float_env("MAX_HOLD_SECONDS", 3600.0)
        if take_profit_pct <= 0:
            raise ConfigError("TAKE_PROFIT_PCT must be greater than zero.")
        if stop_loss_pct <= 0:
            raise ConfigError("STOP_LOSS_PCT must be greater than zero.")
        if trailing_stop_pct < 0:
            raise ConfigError("TRAILING_STOP_PCT must be zero or greater.")
        if max_hold_seconds <= 0:
            raise ConfigError("MAX_HOLD_SECONDS must be greater than zero.")
        min_buy_sell_ratio = _float_env("MIN_BUY_SELL_RATIO", 1.15)
        min_volume_liquidity_ratio_5m = _float_env(
            "MIN_VOLUME_LIQUIDITY_RATIO_5M", 0.03
        )
        max_top_holder_share_pct = _float_env("MAX_TOP_HOLDER_SHARE_PCT", 35.0)
        max_momentum_5m_pct = _float_env("MAX_MOMENTUM_5M_PCT", 80.0)
        momentum_exhaustion_min_buy_sell_ratio = _float_env(
            "MOMENTUM_EXHAUSTION_MIN_BUY_SELL_RATIO", 2.0
        )
        max_buy_more_count = _int_env("MAX_BUY_MORE_COUNT", 2)
        buy_more_cooldown_seconds = _float_env("BUY_MORE_COOLDOWN_SECONDS", 120.0)
        if min_buy_sell_ratio < 0:
            raise ConfigError("MIN_BUY_SELL_RATIO must be zero or greater.")
        if min_volume_liquidity_ratio_5m < 0:
            raise ConfigError("MIN_VOLUME_LIQUIDITY_RATIO_5M must be zero or greater.")
        if not 0 <= max_top_holder_share_pct <= 100:
            raise ConfigError("MAX_TOP_HOLDER_SHARE_PCT must be between 0 and 100.")
        if max_momentum_5m_pct < 0:
            raise ConfigError("MAX_MOMENTUM_5M_PCT must be zero or greater.")
        if momentum_exhaustion_min_buy_sell_ratio < 0:
            raise ConfigError(
                "MOMENTUM_EXHAUSTION_MIN_BUY_SELL_RATIO must be zero or greater."
            )
        if max_buy_more_count < 0:
            raise ConfigError("MAX_BUY_MORE_COUNT must be zero or greater.")
        if buy_more_cooldown_seconds < 0:
            raise ConfigError("BUY_MORE_COOLDOWN_SECONDS must be zero or greater.")

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
            birdeye_api_key=os.getenv("BIRDEYE_API_KEY", "").strip() or None,
            birdeye_ws_url=os.getenv(
                "BIRDEYE_WS_URL",
                "wss://public-api.birdeye.so/socket/solana",
            ).strip(),
            realtime_price_feed_enabled=_bool_env("REALTIME_PRICE_FEED_ENABLED", False),
            tracker_poll_seconds=_float_env("TRACKER_POLL_SECONDS", 30.0),
            position_review_seconds=_float_env("POSITION_REVIEW_SECONDS", 15.0),
            min_liquidity_usd=_float_env("MIN_LIQUIDITY_USD", 10000.0),
            min_pair_age_seconds=_int_env("MIN_PAIR_AGE_SECONDS", 60),
            entry_score_threshold=entry_threshold,
            launch_enabled=_bool_env("LAUNCH_ENABLED", True),
            scout_enabled=_bool_env("SCOUT_ENABLED", False),
            launch_score_threshold=launch_threshold,
            scout_score_threshold=scout_threshold,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_hold_seconds=max_hold_seconds,
            scout_min_liquidity_usd=_float_env("SCOUT_MIN_LIQUIDITY_USD", 15000.0),
            scout_min_volume_5m_usd=_float_env("SCOUT_MIN_VOLUME_5M_USD", 500.0),
            reflection_time=os.getenv("REFLECTION_TIME", "00:00").strip(),
            reflection_timezone=os.getenv("REFLECTION_TIMEZONE", "Asia/Jakarta").strip(),
            min_trade_amount_sol=min_trade_amount_sol,
            max_trade_amount_sol=max_trade_amount_sol,
            blocked_entry_utc_hours=_hour_list_env("BLOCKED_ENTRY_UTC_HOURS", "20"),
            min_buy_sell_ratio=min_buy_sell_ratio,
            min_volume_liquidity_ratio_5m=min_volume_liquidity_ratio_5m,
            max_top_holder_share_pct=max_top_holder_share_pct,
            max_momentum_5m_pct=max_momentum_5m_pct,
            momentum_exhaustion_min_buy_sell_ratio=(
                momentum_exhaustion_min_buy_sell_ratio
            ),
            max_buy_more_count=max_buy_more_count,
            buy_more_cooldown_seconds=buy_more_cooldown_seconds,
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
            launch_enabled=self.launch_enabled,
            scout_enabled=self.scout_enabled,
            launch_score_threshold=self.launch_score_threshold,
            scout_score_threshold=self.scout_score_threshold,
            take_profit_pct=self.take_profit_pct,
            stop_loss_pct=self.stop_loss_pct,
            trailing_stop_pct=self.trailing_stop_pct,
            max_hold_seconds=self.max_hold_seconds,
            scout_min_liquidity_usd=self.scout_min_liquidity_usd,
            scout_min_volume_5m_usd=self.scout_min_volume_5m_usd,
            dynamic_setup_enabled=True,
            min_trade_amount_sol=self.min_trade_amount_sol,
            max_trade_amount_sol=self.max_trade_amount_sol,
            blocked_entry_utc_hours=self.blocked_entry_utc_hours,
            min_buy_sell_ratio=self.min_buy_sell_ratio,
            min_volume_liquidity_ratio_5m=self.min_volume_liquidity_ratio_5m,
            max_top_holder_share_pct=self.max_top_holder_share_pct,
            max_momentum_5m_pct=self.max_momentum_5m_pct,
            momentum_exhaustion_min_buy_sell_ratio=(
                self.momentum_exhaustion_min_buy_sell_ratio
            ),
            max_buy_more_count=self.max_buy_more_count,
            buy_more_cooldown_seconds=self.buy_more_cooldown_seconds,
        )
