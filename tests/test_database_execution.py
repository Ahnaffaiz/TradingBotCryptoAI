from dataclasses import replace

import pytest

from ai_meme_bot.core.database import (
    Database,
    DuplicateOpenTradeError,
    InsufficientBalanceError,
)
from ai_meme_bot.core.execution import TradeExecutor
from ai_meme_bot.agent.tools import TradingTools
from ai_meme_bot.main import _hard_exit_reason
from ai_meme_bot.models import StrategySettings, TokenEvaluation, TradePlan
from tests.helpers import make_config, make_snapshot


@pytest.mark.asyncio
async def test_paper_trade_lifecycle_updates_balance_and_pnl(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    executor = TradeExecutor(config, database)

    opened = await executor.execute_trade("mint-1", "BUY", 1.0, make_snapshot(price=2.0))
    trade = await database.get_trade(opened.trade_id)

    assert opened.success is True
    assert await database.get_balance() == pytest.approx(0.0)
    assert trade.token_quantity == pytest.approx(0.5)

    closed = await executor.close_trade(trade, make_snapshot(price=4.0), "take gains")

    assert closed.success is True
    assert closed.pnl == pytest.approx(1.0)
    assert await database.get_balance() == pytest.approx(2.0)
    assert (await database.get_trade(trade.id)).status == "CLOSED"


@pytest.mark.asyncio
async def test_database_rejects_duplicate_and_insufficient_opens(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    snapshot = make_snapshot()
    await database.insert_trade("mint-1", 2.0, 0.5, 0.25, snapshot.stored_payload())

    with pytest.raises(DuplicateOpenTradeError):
        await database.insert_trade("mint-1", 2.0, 0.5, 0.25, snapshot.stored_payload())

    with pytest.raises(InsufficientBalanceError):
        await database.insert_trade("mint-2", 2.0, 2.0, 1.0, snapshot.stored_payload())


@pytest.mark.asyncio
async def test_ai_trade_plan_persists_and_controls_hard_exit(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    trade_id = await database.insert_trade(
        "mint-1",
        1.0,
        0.5,
        0.5,
        make_snapshot(price=1.0).stored_payload(),
        {
            "entry_amount_sol": 0.5,
            "stop_loss_pct": 12,
            "take_profit_targets_pct": [30, 60],
            "trailing_stop_pct": 5,
            "max_hold_seconds": 3600,
        },
    )
    trade = await database.get_trade(trade_id)
    settings = StrategySettings(
        25,
        30.0,
        0.1,
        45.0,
        "00:00",
        take_profit_pct=10.0,
        stop_loss_pct=5.0,
    )

    assert '"take_profit_targets_pct": [30, 60]' in trade.trade_plan_json
    assert _hard_exit_reason(trade, make_snapshot(price=1.2), settings, 1.2) is None
    assert "TP1 30" in _hard_exit_reason(
        trade, make_snapshot(price=1.35), settings, 1.35
    )
    assert "stop loss" in _hard_exit_reason(
        trade, make_snapshot(price=0.87), settings, 1.0
    )
    assert "trailing stop" in _hard_exit_reason(
        trade, make_snapshot(price=1.2), settings, 1.3
    )


@pytest.mark.asyncio
async def test_real_mode_fails_closed(tmp_path):
    config = make_config(tmp_path / "trades.db", trading_mode="REAL")
    database = Database(config.db_path)
    await database.init_db()

    result = await TradeExecutor(config, database).execute_trade(
        "mint-1", "BUY", 1.0, make_snapshot()
    )

    assert result.success is False
    assert "disabled in v1" in result.message
    assert await database.get_open_trades() == []


@pytest.mark.asyncio
async def test_strategy_settings_persist_over_startup_defaults(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    tuned = StrategySettings(88, 12.0, 0.15, 20.0, "02:45")

    await database.set_strategy_settings(tuned)

    assert await database.get_strategy_settings(config.strategy_defaults) == tuned


@pytest.mark.asyncio
async def test_dynamic_setup_setting_persists(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    defaults = await database.get_strategy_settings(config.strategy_defaults)

    await database.set_strategy_settings(
        replace(defaults, dynamic_setup_enabled=False)
    )

    loaded = await database.get_strategy_settings(config.strategy_defaults)
    assert defaults.dynamic_setup_enabled is True
    assert loaded.dynamic_setup_enabled is False


@pytest.mark.asyncio
async def test_dynamic_buy_size_is_clamped_to_strategy_range(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    settings = await database.get_strategy_settings(config.strategy_defaults)
    await database.set_strategy_settings(
        replace(settings, min_trade_amount_sol=0.05, max_trade_amount_sol=0.2)
    )
    tools = TradingTools(
        config,
        database,
        tracker=object(),
        executor=TradeExecutor(config, database),
    )
    evaluation = TokenEvaluation(
        score=90,
        decision="buy",
        rationale="strong",
        trade_plan=TradePlan(
            entry_amount_sol=0.5,
            stop_loss_pct=8.0,
            take_profit_targets_pct=[18.0],
            trailing_stop_pct=7.0,
            max_hold_seconds=3600.0,
        ),
    )

    result = await tools.trigger_buy("mint-1", make_snapshot(), evaluation)
    trade = await database.get_trade(result.trade_id)

    assert result.success is True
    assert result.entry_amount_sol == pytest.approx(0.2)
    assert trade.entry_amount_sol == pytest.approx(0.2)


def test_hard_exit_rules_trigger_take_profit_and_stop_loss():
    settings = StrategySettings(
        25,
        30.0,
        0.1,
        45.0,
        "00:00",
        take_profit_pct=15.0,
        stop_loss_pct=8.0,
    )
    trade = type(
        "Trade",
        (),
        {"buy_price": 1.0, "opened_at": "2026-05-22T00:00:00+00:00"},
    )()

    assert "take profit" in _hard_exit_reason(
        trade, make_snapshot(price=1.2), settings, 1.2
    )
    assert "stop loss" in _hard_exit_reason(
        trade, make_snapshot(price=0.9), settings, 1.0
    )


@pytest.mark.asyncio
async def test_learning_history_labels_skip_outcomes_and_reflection_evidence(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    initial = make_snapshot(price=1.0, token="missed-mint")
    analysis_id = await database.record_analysis(
        initial, TokenEvaluation(score=35, decision="skip", rationale="unknown holders")
    )
    await database.add_outcome_snapshot(
        analysis_id,
        900,
        initial.stored_payload(),
        make_snapshot(price=1.7, token="missed-mint"),
        "skip",
    )
    await database.add_activity(
        "filter_reject",
        "liquidity below floor",
        "filtered-mint",
        {"liquidity_usd": 1200},
    )
    await database.add_activity("error", "rpc timeout", payload={"stage": "holders"})

    evidence = await database.get_reflection_evidence()

    assert evidence.recent_analyses[0]["token_address"] == "missed-mint"
    assert evidence.missed_winners[0]["token_address"] == "missed-mint"
    assert evidence.failed_filters[0]["token_address"] == "filtered-mint"
    assert evidence.recurring_errors[0]["detail"] == "rpc timeout"
