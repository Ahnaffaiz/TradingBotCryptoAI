import pytest

from ai_meme_bot.core.database import (
    Database,
    DuplicateOpenTradeError,
    InsufficientBalanceError,
)
from ai_meme_bot.core.execution import TradeExecutor
from ai_meme_bot.models import StrategySettings, TokenEvaluation
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
    assert await database.get_balance() == pytest.approx(99.0)
    assert trade.token_quantity == pytest.approx(0.5)

    closed = await executor.close_trade(trade, make_snapshot(price=4.0), "take gains")

    assert closed.success is True
    assert closed.pnl == pytest.approx(1.0)
    assert await database.get_balance() == pytest.approx(101.0)
    assert (await database.get_trade(trade.id)).status == "CLOSED"


@pytest.mark.asyncio
async def test_database_rejects_duplicate_and_insufficient_opens(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    snapshot = make_snapshot()
    await database.insert_trade("mint-1", 2.0, 1.0, 0.5, snapshot.stored_payload())

    with pytest.raises(DuplicateOpenTradeError):
        await database.insert_trade("mint-1", 2.0, 1.0, 0.5, snapshot.stored_payload())

    with pytest.raises(InsufficientBalanceError):
        await database.insert_trade("mint-2", 2.0, 200.0, 100.0, snapshot.stored_payload())


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
