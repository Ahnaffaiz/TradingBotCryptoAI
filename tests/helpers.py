from pathlib import Path

from ai_meme_bot.config import AppConfig
from ai_meme_bot.models import TokenSnapshot


def make_config(db_path: Path, **overrides) -> AppConfig:
    values = {
        "trading_mode": "PAPER",
        "base_trade_amount": 1.0,
        "db_path": db_path,
        "ai_provider": "custom",
        "ai_base_url": "https://ai.example/v1",
        "ai_api_key": "secret-key",
        "ai_model": "paper-model",
        "telegram_bot_token": None,
        "telegram_admin_user_ids": frozenset(),
        "hermes_operator_enabled": False,
        "dexscreener_profile_url": "https://dex.example/profiles",
        "dexscreener_token_url": "https://dex.example/tokens",
        "solana_rpc_url": None,
        "x_bearer_token": None,
        "x_recent_search_url": "https://x.example/recent",
        "x_search_minutes": 30,
        "geckoterminal_trending_url": "https://gecko.example/trending",
        "birdeye_api_key": None,
        "birdeye_ws_url": "wss://birdeye.example/socket/solana",
        "realtime_price_feed_enabled": False,
        "tracker_poll_seconds": 0.01,
        "position_review_seconds": 0.01,
        "min_liquidity_usd": 10000.0,
        "min_pair_age_seconds": 60,
        "entry_score_threshold": 25,
        "launch_enabled": True,
        "scout_enabled": False,
        "launch_score_threshold": 25,
        "scout_score_threshold": 70,
        "take_profit_pct": 18.0,
        "stop_loss_pct": 8.0,
        "trailing_stop_pct": 7.0,
        "max_hold_seconds": 3600.0,
        "scout_min_liquidity_usd": 15000.0,
        "scout_min_volume_5m_usd": 500.0,
        "reflection_time": "00:00",
        "reflection_timezone": "UTC",
        "min_trade_amount_sol": 0.01,
        "max_trade_amount_sol": 2.0,
        "blocked_entry_utc_hours": "",
        "min_buy_sell_ratio": 0.0,
        "min_volume_liquidity_ratio_5m": 0.0,
        "max_top_holder_share_pct": 100.0,
        "max_momentum_5m_pct": 1000.0,
        "momentum_exhaustion_min_buy_sell_ratio": 0.0,
        "max_buy_more_count": 2,
        "buy_more_cooldown_seconds": 0.0,
    }
    values.update(overrides)
    return AppConfig(**values)


def make_snapshot(price: float = 2.0, token: str = "mint-1") -> TokenSnapshot:
    return TokenSnapshot(
        token_address=token,
        pair_address="pair-1",
        price_usd=price,
        liquidity_usd=20000.0,
        volume_5m_usd=900.0,
        pair_age_seconds=120.0,
        token_name="Mint One",
        token_symbol="MINT",
        token_logo_url="https://cdn.example/mint.png",
        raw_context={"priceUsd": str(price)},
    )
