import pytest

from ai_meme_bot.agent.hermes_bot import TelegramPaperNotifier, TelegramTradingBot
from ai_meme_bot.core.database import Database
from ai_meme_bot.models import (
    ExitDecision,
    StrategySettings,
    TokenEvaluation,
    TradeRecord,
    TradeResult,
)
from tests.helpers import make_config, make_snapshot


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))


class FakeUpdate:
    def __init__(self, user_id=5):
        self.effective_message = FakeMessage()
        self.effective_chat = type("FakeChat", (), {"id": 94721})()
        self.effective_user = type(
            "FakeUser", (), {"id": user_id, "username": "admin-user"}
        )()


class FakeContext:
    def __init__(self, args):
        self.args = args


class FakeOperator:
    def __init__(self):
        self.prompts = []

    async def operator_chat(self, prompt, user_id, user_name=""):
        self.prompts.append((prompt, user_id, user_name))
        return "Changed config and ran tests."


class FakeTelegramBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))


@pytest.mark.asyncio
async def test_status_redacts_api_key_and_auto_handlers_toggle_state(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    bot = TelegramTradingBot(config, database)
    update = FakeUpdate()

    await bot.auto_on(update, None)
    status = await bot.render_status()
    await bot.auto_off(update, None)

    assert "AI:</b> custom:paper-model" in status
    assert "secret-key" not in status
    assert "Auto entries:</b> 🟢 ON" in status
    assert "Reports:</b> 🔔 ON in this chat" in status
    assert "Thresholds:</b> launch ≥ 25 | scout ≥ 70" in status
    assert (
        "Trade size:</b> static 1.000000 SOL | dynamic 0.010000..2.000000 SOL"
        in status
    )
    assert await database.get_auto_trading() is False
    assert await database.get_notification_chat_id() == 94721
    assert "Auto entries enabled" in update.effective_message.replies[0][0]
    assert "Auto entries paused" in update.effective_message.replies[1][0]
    assert update.effective_message.replies[0][1]["parse_mode"] == "HTML"


@pytest.mark.asyncio
async def test_threshold_command_updates_live_strategy_settings(tmp_path):
    config = make_config(tmp_path / "trades.db", entry_score_threshold=80)
    database = Database(config.db_path)
    await database.init_db()
    bot = TelegramTradingBot(config, database)
    update = FakeUpdate()

    await bot.threshold(update, FakeContext([]))
    await bot.threshold(update, FakeContext(["25/100"]))
    await bot.threshold(update, FakeContext(["101"]))

    assert await database.get_strategy_settings(config.strategy_defaults) == (
        StrategySettings(25, 0.01, 1.0, 0.01, "00:00")
    )
    assert "Use <code>/threshold 25</code>" in update.effective_message.replies[0][0]
    assert "Launch threshold updated" in update.effective_message.replies[1][0]
    assert "Invalid threshold" in update.effective_message.replies[2][0]


@pytest.mark.asyncio
async def test_dynamic_setup_commands_toggle_strategy_settings(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    bot = TelegramTradingBot(config, database)
    update = FakeUpdate()

    await bot.dynamic_setup(update, None)
    await bot.dynamic_setup_off(update, None)
    disabled = await database.get_strategy_settings(config.strategy_defaults)
    await bot.dynamic_setup_on(update, None)
    enabled = await database.get_strategy_settings(config.strategy_defaults)

    assert "Dynamic setup:</b> ON" in update.effective_message.replies[0][0]
    assert "AI size:</b> 0.01..2 SOL" in update.effective_message.replies[0][0]
    assert "Dynamic setup updated" in update.effective_message.replies[1][0]
    assert disabled.dynamic_setup_enabled is False
    assert enabled.dynamic_setup_enabled is True


@pytest.mark.asyncio
async def test_trade_size_bound_commands_update_strategy_settings(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    bot = TelegramTradingBot(config, database)
    update = FakeUpdate()

    await bot.min_trade_size(update, FakeContext(["0.05"]))
    await bot.max_trade_size(update, FakeContext(["0.25"]))
    settings = await database.get_strategy_settings(config.strategy_defaults)
    await bot.min_trade_size(update, FakeContext(["0.3"]))

    assert settings.min_trade_amount_sol == 0.05
    assert settings.max_trade_amount_sol == 0.25
    assert "Minimum trade size updated" in update.effective_message.replies[0][0]
    assert "Maximum trade size updated" in update.effective_message.replies[1][0]
    assert "Invalid range" in update.effective_message.replies[2][0]


@pytest.mark.asyncio
async def test_telegram_notifier_sends_analysis_and_trade_reports(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    await database.set_notification_chat_id(1234)
    telegram_bot = FakeTelegramBot()
    notifier = TelegramPaperNotifier(database, telegram_bot)
    snapshot = make_snapshot()
    evaluation = TokenEvaluation(score=92, decision="buy", rationale="liquid enough")

    await notifier.entry_analysis(snapshot, evaluation)
    await notifier.buy_result(
        snapshot,
        evaluation,
        TradeResult(True, "Opened paper trade.", trade_id=9, entry_amount_sol=0.25),
    )

    assert [chat_id for chat_id, _text, _kwargs in telegram_bot.messages] == [1234, 1234]
    assert "Paper Entry Analysis" in telegram_bot.messages[0][1]
    assert "liquid enough" in telegram_bot.messages[0][1]
    assert "Paper Buy Opened" in telegram_bot.messages[1][1]
    assert "Entry size:</b> 0.250000 SOL" in telegram_bot.messages[1][1]
    assert telegram_bot.messages[0][2]["parse_mode"] == "HTML"


def test_sell_result_displays_entry_size():
    trade = TradeRecord(
        id=4,
        token_address="mint-1",
        buy_price=1.0,
        sell_price=None,
        pnl=None,
        status="OPEN",
        timestamp="2026-05-22T00:00:00+00:00",
        entry_amount_sol=0.25,
        token_quantity=0.25,
        entry_snapshot_json="{}",
        trade_plan_json="{}",
        exit_snapshot_json=None,
        exit_reason=None,
        opened_at="2026-05-22T00:00:00+00:00",
        closed_at=None,
    )

    from ai_meme_bot.agent.hermes_bot import format_sell_result

    rendered = format_sell_result(
        trade,
        make_snapshot(price=1.2),
        ExitDecision("close", "take profit"),
        TradeResult(True, "Closed paper trade.", trade_id=4, pnl=0.05),
    )

    assert "Entry size:</b> 0.250000 SOL" in rendered


@pytest.mark.asyncio
async def test_notification_switch_mutes_and_resumes_reports(tmp_path):
    config = make_config(tmp_path / "trades.db")
    database = Database(config.db_path)
    await database.init_db()
    await database.set_notification_chat_id(1234)
    telegram_bot = FakeTelegramBot()
    notifier = TelegramPaperNotifier(database, telegram_bot)
    controller = TelegramTradingBot(config, database)
    update = FakeUpdate()
    snapshot = make_snapshot()
    evaluation = TokenEvaluation(score=25, decision="skip", rationale="too risky")

    await controller.notify_off(update, None)
    await notifier.entry_analysis(snapshot, evaluation)
    await controller.notify_on(update, None)
    await notifier.entry_analysis(snapshot, evaluation)

    assert await database.get_notifications_enabled() is True
    assert len(telegram_bot.messages) == 1
    assert "Reports muted" in update.effective_message.replies[0][0]
    assert "Reports enabled" in update.effective_message.replies[1][0]


@pytest.mark.asyncio
async def test_hermes_operator_is_admin_gated(tmp_path):
    config = make_config(
        tmp_path / "trades.db",
        hermes_operator_enabled=True,
        telegram_admin_user_ids=frozenset({42}),
    )
    database = Database(config.db_path)
    await database.init_db()
    operator = FakeOperator()
    controller = TelegramTradingBot(config, database, operator_backend=operator)
    denied = FakeUpdate(user_id=7)
    allowed = FakeUpdate(user_id=42)

    await controller.hermes(denied, FakeContext(["edit", "README"]))
    await controller.hermes(allowed, FakeContext(["edit", "README"]))

    assert "Admin only" in denied.effective_message.replies[0][0]
    assert operator.prompts == [("edit README", "42", "admin-user")]
    assert "Hermes operator working" in allowed.effective_message.replies[0][0]
    assert "Changed config" in allowed.effective_message.replies[1][0]
