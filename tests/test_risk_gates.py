from datetime import datetime, timezone

from dataclasses import replace

from ai_meme_bot.main import _entry_risk_rejection
from tests.helpers import make_config, make_snapshot


def test_entry_risk_gate_blocks_configured_utc_hour(tmp_path):
    settings = make_config(tmp_path / "trades.db").strategy_defaults
    settings = replace(settings, blocked_entry_utc_hours="20")

    reason = _entry_risk_rejection(
        make_snapshot(),
        settings,
        now=datetime(2026, 5, 29, 20, 0, tzinfo=timezone.utc),
    )

    assert "UTC hour 20" in reason


def test_entry_risk_gate_blocks_sell_pressure_and_holder_concentration(tmp_path):
    settings = make_config(tmp_path / "trades.db").strategy_defaults
    settings = replace(
        settings,
        min_buy_sell_ratio=1.15,
        max_top_holder_share_pct=35.0,
    )
    sell_pressure = make_snapshot()
    sell_pressure.buys_5m = 10
    sell_pressure.sells_5m = 20
    concentrated = make_snapshot()
    concentrated.top_holder_share_pct = 45.0

    assert "buy/sell" in _entry_risk_rejection(sell_pressure, settings)
    assert "top holders" in _entry_risk_rejection(concentrated, settings)


def test_entry_risk_gate_blocks_exhausted_momentum(tmp_path):
    settings = make_config(tmp_path / "trades.db").strategy_defaults
    settings = replace(
        settings,
        max_momentum_5m_pct=80.0,
        momentum_exhaustion_min_buy_sell_ratio=2.0,
    )
    snapshot = make_snapshot()
    snapshot.price_change_5m_pct = 120.0
    snapshot.buys_5m = 10
    snapshot.sells_5m = 8

    assert "momentum" in _entry_risk_rejection(snapshot, settings)
