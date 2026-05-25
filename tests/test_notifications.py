import asyncio
from dataclasses import replace

import pytest

from ai_meme_bot.core.database import Database
from ai_meme_bot.core.execution import TradeExecutor
from ai_meme_bot.main import run_discovery_loop, run_position_review_loop
from ai_meme_bot.models import ExitDecision, TokenEvaluation, TradePlan, TradeResult
from tests.helpers import make_config, make_snapshot


class OneSnapshotTracker:
    async def discover(self):
        yield make_snapshot()


class OneScoutSnapshotTracker:
    async def discover(self):
        snapshot = make_snapshot()
        snapshot.strategy = "scout"
        yield snapshot


class TakeProfitTracker:
    async def snapshot_for_token(self, token_address, apply_filters=True):
        return make_snapshot(price=1.2, token=token_address)


def dynamic_plan() -> TradePlan:
    return TradePlan(
        entry_amount_sol=0.1,
        stop_loss_pct=8.0,
        take_profit_targets_pct=[18.0],
        trailing_stop_pct=7.0,
        max_hold_seconds=3600.0,
        rationale="bounded setup",
    )


class BuyAI:
    async def evaluate_entry(self, _snapshot, _rules):
        return TokenEvaluation(
            score=95,
            decision="buy",
            rationale="paper entry",
            trade_plan=dynamic_plan(),
        )


class ThresholdBuyAI:
    async def evaluate_entry(self, _snapshot, _rules):
        return TokenEvaluation(
            score=25,
            decision="buy",
            rationale="paper entry",
            trade_plan=dynamic_plan(),
        )


class LegacyBuyAI:
    async def evaluate_entry(self, _snapshot, _rules):
        return TokenEvaluation(score=95, decision="buy", rationale="no setup")


class ThresholdSkipAI:
    async def evaluate_entry(self, _snapshot, _rules):
        return TokenEvaluation(score=35, decision="skip", rationale="score gate")


class UnavailableAI:
    async def evaluate_entry(self, _snapshot, _rules):
        return TokenEvaluation()


class HoldAI:
    async def evaluate_exit(self, _trade, _snapshot, _rules):
        return ExitDecision("hold", "wait")


class BuyTools:
    async def trigger_buy(self, _token_address, _snapshot):
        return TradeResult(True, "Opened paper trade.")


class CaptureNotifier:
    def __init__(self):
        self.events = []

    async def entry_analysis(self, snapshot, evaluation):
        self.events.append(("analysis", snapshot.token_address, evaluation.score))

    async def buy_result(self, snapshot, evaluation, result):
        self.events.append(("buy", snapshot.token_address, evaluation.score, result.trade_id))

    async def exit_analysis(self, *_args):
        return None

    async def sell_result(self, *_args):
        return None

    async def reflection(self, *_args):
        return None

    async def error(self, stage, detail):
        self.events.append(("error", stage, detail))


@pytest.mark.asyncio
async def test_discovery_reports_analysis_before_buy_result(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    await database.set_auto_trading(True)
    notifier = CaptureNotifier()

    await run_discovery_loop(
        database,
        OneSnapshotTracker(),
        BuyAI(),
        BuyTools(),
        config,
        notifier,
    )

    assert notifier.events == [
        ("analysis", "mint-1", 95),
        ("buy", "mint-1", 95, None),
    ]


@pytest.mark.asyncio
async def test_discovery_buys_when_score_meets_threshold(tmp_path):
    config = make_config(tmp_path / "trades.db", entry_score_threshold=25)
    database = Database(config.db_path)
    await database.init_db()
    await database.set_auto_trading(True)
    notifier = CaptureNotifier()

    await run_discovery_loop(
        database,
        OneSnapshotTracker(),
        ThresholdBuyAI(),
        BuyTools(),
        config,
        notifier,
    )

    assert notifier.events == [
        ("analysis", "mint-1", 25),
        ("buy", "mint-1", 25, None),
    ]


@pytest.mark.asyncio
async def test_dynamic_discovery_requires_trade_plan(tmp_path):
    config = make_config(tmp_path / "trades.db", entry_score_threshold=20)
    database = Database(config.db_path)
    await database.init_db()
    await database.set_auto_trading(True)
    notifier = CaptureNotifier()

    await run_discovery_loop(
        database,
        OneSnapshotTracker(),
        LegacyBuyAI(),
        BuyTools(),
        config,
        notifier,
    )

    assert notifier.events == [("analysis", "mint-1", 95)]


@pytest.mark.asyncio
async def test_dynamic_discovery_requires_buy_decision(tmp_path):
    config = make_config(
        tmp_path / "trades.db", entry_score_threshold=20, launch_score_threshold=20
    )
    database = Database(config.db_path)
    await database.init_db()
    await database.set_auto_trading(True)
    notifier = CaptureNotifier()

    await run_discovery_loop(
        database,
        OneSnapshotTracker(),
        ThresholdSkipAI(),
        BuyTools(),
        config,
        notifier,
    )

    assert notifier.events == [("analysis", "mint-1", 35)]


@pytest.mark.asyncio
async def test_static_discovery_score_threshold_overrides_skip_decision(tmp_path):
    config = make_config(
        tmp_path / "trades.db", entry_score_threshold=20, launch_score_threshold=20
    )
    database = Database(config.db_path)
    await database.init_db()
    settings = await database.get_strategy_settings(config.strategy_defaults)
    await database.set_strategy_settings(
        replace(settings, dynamic_setup_enabled=False)
    )
    await database.set_auto_trading(True)
    notifier = CaptureNotifier()

    await run_discovery_loop(
        database,
        OneSnapshotTracker(),
        ThresholdSkipAI(),
        BuyTools(),
        config,
        notifier,
    )

    assert notifier.events == [
        ("analysis", "mint-1", 35),
        ("buy", "mint-1", 35, None),
    ]


@pytest.mark.asyncio
async def test_discovery_does_not_buy_unavailable_zero_score(tmp_path):
    config = make_config(tmp_path / "trades.db", entry_score_threshold=0)
    database = Database(config.db_path)
    await database.init_db()
    await database.set_auto_trading(True)
    notifier = CaptureNotifier()

    await run_discovery_loop(
        database,
        OneSnapshotTracker(),
        UnavailableAI(),
        BuyTools(),
        config,
        notifier,
    )

    assert notifier.events == [("analysis", "mint-1", 0)]


@pytest.mark.asyncio
async def test_scout_mode_uses_separate_higher_threshold(tmp_path):
    config = make_config(tmp_path / "trades.db", scout_score_threshold=70)
    database = Database(config.db_path)
    await database.init_db()
    await database.set_auto_trading(True)
    notifier = CaptureNotifier()

    await run_discovery_loop(
        database,
        OneScoutSnapshotTracker(),
        ThresholdSkipAI(),
        BuyTools(),
        config,
        notifier,
    )

    assert notifier.events == [("analysis", "mint-1", 35)]


@pytest.mark.asyncio
async def test_position_review_hard_exit_uses_exit_decision(tmp_path):
    config = make_config(
        tmp_path / "trades.db",
        take_profit_pct=10.0,
        position_review_seconds=999.0,
    )
    database = Database(config.db_path)
    await database.init_db()
    await database.insert_trade(
        "mint-1", 1.0, 0.5, 0.5, make_snapshot(price=1.0).stored_payload()
    )
    notifier = CaptureNotifier()

    task = run_position_review_loop(
        database,
        TakeProfitTracker(),
        HoldAI(),
        TradeExecutor(config, database),
        config,
        notifier,
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(task, timeout=0.05)

    closed = await database.get_closed_trades()
    assert len(closed) == 1
    assert closed[0].exit_reason.startswith("take profit")
