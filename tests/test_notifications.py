import pytest

from ai_meme_bot.core.database import Database
from ai_meme_bot.main import run_discovery_loop
from ai_meme_bot.models import TokenEvaluation, TradeResult
from tests.helpers import make_config, make_snapshot


class OneSnapshotTracker:
    async def discover(self):
        yield make_snapshot()


class BuyAI:
    async def evaluate_entry(self, _snapshot, _rules):
        return TokenEvaluation(score=95, decision="buy", rationale="paper entry")


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
