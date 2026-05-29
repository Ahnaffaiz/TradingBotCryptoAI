import json

import asyncio
import pytest

from ai_meme_bot.agent.hermes_bot import NullPaperNotifier
from ai_meme_bot.core.database import Database
from ai_meme_bot.core.execution import TradeExecutor
from ai_meme_bot.core.price_feed import BirdeyePriceFeed
from ai_meme_bot.core.price_feed import PriceTick
from ai_meme_bot.main import run_realtime_exit_loop
from tests.helpers import make_config, make_snapshot


def test_birdeye_feed_builds_simple_pair_subscription():
    message = BirdeyePriceFeed.subscription_message(["pair-1"])

    assert message == {
        "type": "SUBSCRIBE_PRICE",
        "data": {
            "queryType": "simple",
            "chartType": "1m",
            "address": "pair-1",
            "currency": "pair",
        },
    }


def test_birdeye_feed_builds_complex_pair_subscription():
    message = BirdeyePriceFeed.subscription_message(["pair-1", "pair-2"])

    assert message["type"] == "SUBSCRIBE_PRICE"
    assert message["data"]["queryType"] == "complex"
    assert "address = pair-1" in message["data"]["query"]
    assert "currency = pair" in message["data"]["query"]


def test_birdeye_feed_parses_price_data_and_ignores_other_messages():
    payload = {
        "type": "PRICE_DATA",
        "data": {"address": "pair-1", "c": 0.123, "symbol": "MINT-SOL"},
    }

    tick = BirdeyePriceFeed.parse_message(json.dumps(payload))

    assert tick.address == "pair-1"
    assert tick.price_usd == 0.123
    assert BirdeyePriceFeed.parse_message('{"type": "WELCOME"}') is None


class OneTickFeed:
    def __init__(self):
        self.addresses = []

    async def stream_prices(self, addresses):
        self.addresses = list(addresses)
        yield PriceTick("pair-1", 0.9, {"type": "PRICE_DATA"})
        await asyncio.sleep(999)


@pytest.mark.asyncio
async def test_realtime_exit_loop_closes_on_streamed_stop_loss(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    executor = TradeExecutor(config, database)
    opened = await executor.execute_trade(
        "mint-1", "BUY", 0.5, make_snapshot(price=1.0)
    )
    feed = OneTickFeed()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            run_realtime_exit_loop(
                database, executor, config, NullPaperNotifier(), price_feed=feed
            ),
            timeout=0.05,
        )

    closed = await database.get_trade(opened.trade_id)
    assert feed.addresses == ["pair-1"]
    assert closed.status == "CLOSED"
    assert closed.exit_reason.startswith("stop loss")
