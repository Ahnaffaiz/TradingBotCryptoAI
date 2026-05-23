"""Telegram controls for the repo-owned Hermes paper trader."""

from __future__ import annotations

from html import escape
import logging
from typing import Any, List, Optional, Protocol

from ai_meme_bot.config import AppConfig
from ai_meme_bot.agent.ai_service import HermesChatBackend
from ai_meme_bot.core.database import Database
from ai_meme_bot.models import (
    ExitDecision,
    ReflectionRules,
    TokenEvaluation,
    TokenSnapshot,
    TradeRecord,
    TradeResult,
)


LOGGER = logging.getLogger(__name__)
PARSE_MODE = "HTML"

MENU_STATUS = "📊 Status"
MENU_AUTO_ON = "🟢 Auto On"
MENU_AUTO_OFF = "🔴 Auto Off"
MENU_NOTIFY_ON = "🔔 Notify On"
MENU_NOTIFY_OFF = "🔕 Notify Off"


class PaperNotifier(Protocol):
    """Notification surface used by the paper-trading loops."""

    async def entry_analysis(
        self, snapshot: TokenSnapshot, evaluation: TokenEvaluation
    ) -> None:
        """Report a paper entry review."""

    async def buy_result(
        self, snapshot: TokenSnapshot, evaluation: TokenEvaluation, result: TradeResult
    ) -> None:
        """Report a paper buy outcome."""

    async def exit_analysis(
        self, trade: TradeRecord, snapshot: TokenSnapshot, decision: ExitDecision
    ) -> None:
        """Report an open paper position review."""

    async def sell_result(
        self,
        trade: TradeRecord,
        snapshot: TokenSnapshot,
        decision: ExitDecision,
        result: TradeResult,
    ) -> None:
        """Report a paper sell outcome."""

    async def reflection(self, rules: ReflectionRules) -> None:
        """Report learned daily rules."""

    async def error(self, stage: str, detail: str) -> None:
        """Report a non-fatal paper pipeline error."""


class NullPaperNotifier:
    """Do nothing when Telegram is not configured or not started."""

    async def entry_analysis(
        self, snapshot: TokenSnapshot, evaluation: TokenEvaluation
    ) -> None:
        return None

    async def buy_result(
        self, snapshot: TokenSnapshot, evaluation: TokenEvaluation, result: TradeResult
    ) -> None:
        return None

    async def exit_analysis(
        self, trade: TradeRecord, snapshot: TokenSnapshot, decision: ExitDecision
    ) -> None:
        return None

    async def sell_result(
        self,
        trade: TradeRecord,
        snapshot: TokenSnapshot,
        decision: ExitDecision,
        result: TradeResult,
    ) -> None:
        return None

    async def reflection(self, rules: ReflectionRules) -> None:
        return None

    async def error(self, stage: str, detail: str) -> None:
        return None


class TelegramPaperNotifier:
    """Send paper-trading reports to the last registered Telegram chat."""

    def __init__(self, database: Database, bot: Any) -> None:
        self.database = database
        self.bot = bot

    async def entry_analysis(
        self, snapshot: TokenSnapshot, evaluation: TokenEvaluation
    ) -> None:
        await self._send(format_entry_analysis(snapshot, evaluation))

    async def buy_result(
        self, snapshot: TokenSnapshot, evaluation: TokenEvaluation, result: TradeResult
    ) -> None:
        await self._send(format_buy_result(snapshot, evaluation, result))

    async def exit_analysis(
        self, trade: TradeRecord, snapshot: TokenSnapshot, decision: ExitDecision
    ) -> None:
        await self._send(format_exit_analysis(trade, snapshot, decision))

    async def sell_result(
        self,
        trade: TradeRecord,
        snapshot: TokenSnapshot,
        decision: ExitDecision,
        result: TradeResult,
    ) -> None:
        await self._send(format_sell_result(trade, snapshot, decision, result))

    async def reflection(self, rules: ReflectionRules) -> None:
        if rules.rules:
            await self._send(format_reflection(rules))

    async def error(self, stage: str, detail: str) -> None:
        await self._send(format_error(stage, detail))

    async def _send(self, text: str) -> None:
        chat_id = await self.database.get_notification_chat_id()
        if chat_id is None or not await self.database.get_notifications_enabled():
            return
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text[:4000],
                parse_mode=PARSE_MODE,
            )
        except Exception as exc:
            LOGGER.warning("Telegram event notification failed: %s", exc)


class TelegramTradingBot:
    """Handlers and status rendering for `python-telegram-bot`."""

    def __init__(
        self,
        config: AppConfig,
        database: Database,
        operator_backend: Optional[HermesChatBackend] = None,
    ) -> None:
        self.config = config
        self.database = database
        self.operator_backend = operator_backend or HermesChatBackend(config)

    def build_application(self) -> Any:
        """Build the Telegram application when a token is configured."""

        if not self.config.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required to start Telegram polling.")
        from telegram.ext import (
            ApplicationBuilder,
            CommandHandler,
            MessageHandler,
            filters,
        )

        application = (
            ApplicationBuilder()
            .token(self.config.telegram_bot_token)
            .post_init(self._post_init)
            .build()
        )
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("whoami", self.whoami))
        application.add_handler(CommandHandler("menu", self.menu))
        application.add_handler(CommandHandler("status", self.status))
        application.add_handler(CommandHandler("auto_on", self.auto_on))
        application.add_handler(CommandHandler("auto_off", self.auto_off))
        application.add_handler(CommandHandler("notify_on", self.notify_on))
        application.add_handler(CommandHandler("notify_off", self.notify_off))
        application.add_handler(CommandHandler("hermes", self.hermes))
        application.add_handler(
            MessageHandler(filters.Regex("^{0}$".format(MENU_STATUS)), self.status)
        )
        application.add_handler(
            MessageHandler(filters.Regex("^{0}$".format(MENU_AUTO_ON)), self.auto_on)
        )
        application.add_handler(
            MessageHandler(filters.Regex("^{0}$".format(MENU_AUTO_OFF)), self.auto_off)
        )
        application.add_handler(
            MessageHandler(filters.Regex("^{0}$".format(MENU_NOTIFY_ON)), self.notify_on)
        )
        application.add_handler(
            MessageHandler(
                filters.Regex("^{0}$".format(MENU_NOTIFY_OFF)), self.notify_off
            )
        )
        return application

    async def start(self, update: Any, _context: Any) -> None:
        """Explain the paper-first bot control surface."""

        await self._register_notification_chat(update)
        await _reply(
            update,
            welcome_text(),
            reply_markup=menu_markup(),
        )

    async def menu(self, update: Any, _context: Any) -> None:
        """Show the persistent Telegram controls."""

        await self._register_notification_chat(update)
        await _reply(update, menu_text(), reply_markup=menu_markup())

    async def whoami(self, update: Any, _context: Any) -> None:
        """Show Telegram IDs needed by admin-only operator config."""

        user = getattr(update, "effective_user", None)
        chat = getattr(update, "effective_chat", None)
        await _reply(
            update,
            "🪪 <b>Telegram IDs</b>\n"
            "👤 <b>User:</b> <code>{0}</code>\n"
            "💬 <b>Chat:</b> <code>{1}</code>".format(
                _html(getattr(user, "id", "unknown")),
                _html(getattr(chat, "id", "unknown")),
            ),
        )

    async def status(self, update: Any, _context: Any) -> None:
        """Send a concise runtime status report."""

        await self._register_notification_chat(update)
        await _reply(update, await self.render_status(), reply_markup=menu_markup())

    async def auto_on(self, update: Any, _context: Any) -> None:
        """Allow the discovery loop to open new AI-approved paper entries."""

        await self._register_notification_chat(update)
        await self.database.set_auto_trading(True)
        await _reply(
            update,
            "🟢 <b>Auto entries enabled</b>\n"
            "Eligible paper candidates will be analyzed and may open simulated buys.",
            reply_markup=menu_markup(),
        )

    async def auto_off(self, update: Any, _context: Any) -> None:
        """Stop new entries while leaving open-position reviews active."""

        await self._register_notification_chat(update)
        await self.database.set_auto_trading(False)
        await _reply(
            update,
            "🔴 <b>Auto entries paused</b>\n"
            "Open paper trades still receive exit analysis.",
            reply_markup=menu_markup(),
        )

    async def notify_on(self, update: Any, _context: Any) -> None:
        """Enable Telegram analysis and paper-trade reports."""

        await self._register_notification_chat(update)
        await self.database.set_notifications_enabled(True)
        await _reply(
            update,
            "🔔 <b>Reports enabled</b>\n"
            "This chat will receive analysis, paper trade, reflection, and error reports.",
            reply_markup=menu_markup(),
        )

    async def notify_off(self, update: Any, _context: Any) -> None:
        """Mute Telegram analysis and paper-trade reports."""

        await self._register_notification_chat(update)
        await self.database.set_notifications_enabled(False)
        await _reply(
            update,
            "🔕 <b>Reports muted</b>\n"
            "Use /notify_on or the menu to resume paper-trading reports.",
            reply_markup=menu_markup(),
        )

    async def hermes(self, update: Any, context: Any) -> None:
        """Run an admin-only Hermes workspace task from Telegram."""

        if not self.config.hermes_operator_enabled:
            await _reply(
                update,
                "🔒 <b>Hermes operator disabled</b>\n"
                "Set <code>HERMES_OPERATOR_ENABLED=1</code> and configure "
                "<code>TELEGRAM_ADMIN_USER_IDS</code> before using it.",
            )
            return
        user = getattr(update, "effective_user", None)
        user_id = getattr(user, "id", None)
        if user_id not in self.config.telegram_admin_user_ids:
            await _reply(update, "⛔ <b>Admin only</b>")
            return
        prompt = " ".join(getattr(context, "args", []) or []).strip()
        if not prompt:
            await _reply(
                update,
                "🛠 <b>Hermes operator</b>\n"
                "Use <code>/hermes task</code>. This can edit project files and "
                "run local tools for an admin request.",
            )
            return
        await self._register_notification_chat(update)
        await _reply(update, "🛠 <b>Hermes operator working</b>")
        try:
            response = await self.operator_backend.operator_chat(
                prompt,
                user_id=str(user_id),
                user_name=str(getattr(user, "username", "") or ""),
            )
            await self.database.add_activity(
                "hermes_operator",
                "admin workspace task",
                payload={"telegram_user_id": user_id, "prompt": prompt[:500]},
            )
            await _reply(update, "🧠 <b>Hermes</b>\n{0}".format(_html(response[:3800])))
        except Exception as exc:
            LOGGER.exception("Hermes Telegram operator failed")
            await self.database.add_activity(
                "error",
                str(exc),
                payload={"stage": "hermes operator", "telegram_user_id": user_id},
            )
            await _reply(update, format_error("hermes operator", str(exc)))

    async def render_status(self) -> str:
        """Render status without leaking provider credentials."""

        balance = await self.database.get_balance()
        auto_enabled = await self.database.get_auto_trading()
        notification_chat_id = await self.database.get_notification_chat_id()
        notifications_enabled = await self.database.get_notifications_enabled()
        open_trades = await self.database.get_open_trades()
        closed_trades = await self.database.get_closed_trades(limit=3)
        settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
        lines = [
            "📊 <b>Paper Bot Status</b>",
            "🧪 <b>Mode:</b> {0}".format(_html(self.config.trading_mode)),
            "🧠 <b>AI:</b> {0}".format(_html(self.config.ai_identity)),
            "🤖 <b>Auto entries:</b> {0}".format(
                "🟢 ON" if auto_enabled else "🔴 OFF"
            ),
            "🔔 <b>Reports:</b> {0}".format(
                _report_state(notification_chat_id, notifications_enabled)
            ),
            "💰 <b>Paper balance:</b> {0:.6f} SOL".format(balance),
            "🎚 <b>Entry threshold:</b> score &gt; {0}".format(
                settings.entry_score_threshold
            ),
            "📦 <b>Trade size:</b> {0:.6f} SOL".format(settings.base_trade_amount),
            "⏱ <b>Cadence:</b> discover {0:g}s | exits {1:g}s".format(
                settings.tracker_poll_seconds, settings.position_review_seconds
            ),
            "🌙 <b>Reflection:</b> {0} {1}".format(
                _html(settings.reflection_time), _html(self.config.reflection_timezone)
            ),
            "📂 <b>Open positions:</b> {0}".format(len(open_trades)),
        ]
        lines.extend(_format_open_trades(open_trades))
        if closed_trades:
            lines.append("")
            lines.append("✅ <b>Recent closed trades</b>")
            lines.extend(_format_closed_trades(closed_trades))
        return "\n".join(lines)

    async def _post_init(self, application: Any) -> None:
        """Publish Telegram's slash-command menu after startup."""

        try:
            from telegram import BotCommand

            await application.bot.set_my_commands(
                [
                    BotCommand("menu", "show paper bot controls"),
                    BotCommand("whoami", "show Telegram user id"),
                    BotCommand("status", "show paper trading status"),
                    BotCommand("auto_on", "enable paper auto entries"),
                    BotCommand("auto_off", "pause paper auto entries"),
                    BotCommand("notify_on", "enable analysis reports"),
                    BotCommand("notify_off", "mute analysis reports"),
                    BotCommand("hermes", "admin workspace operator"),
                ]
            )
        except Exception as exc:
            LOGGER.warning("Telegram command menu registration failed: %s", exc)

    async def _register_notification_chat(self, update: Any) -> None:
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
        if chat_id is not None:
            await self.database.set_notification_chat_id(int(chat_id))


async def _reply(update: Any, text: str, reply_markup: Any = None) -> None:
    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if message is not None:
        await message.reply_text(
            text,
            parse_mode=PARSE_MODE,
            reply_markup=reply_markup,
        )


def _format_open_trades(trades: List[TradeRecord]) -> List[str]:
    return [
        "• #{0} <code>{1}</code> entry ${2:.10g}".format(
            trade.id, _html(_short_token(trade.token_address)), trade.buy_price
        )
        for trade in trades[:5]
    ]


def _format_closed_trades(trades: List[TradeRecord]) -> List[str]:
    return [
        "• #{0} <code>{1}</code> PnL {2:+.6f} SOL".format(
            trade.id, _html(_short_token(trade.token_address)), trade.pnl or 0.0
        )
        for trade in trades
    ]


def _short_token(token_address: str) -> str:
    if len(token_address) <= 14:
        return token_address
    return "{0}...{1}".format(token_address[:7], token_address[-5:])


def format_entry_analysis(snapshot: TokenSnapshot, evaluation: TokenEvaluation) -> str:
    """Format one candidate review before any paper buy."""

    return "\n".join(
        [
            "{0} <b>Paper Entry Analysis</b>".format(_entry_icon(evaluation)),
            "🪙 <b>Token:</b> <code>{0}</code>".format(_html(snapshot.token_address)),
            "🔗 <b>Pair:</b> <code>{0}</code>".format(_html(snapshot.pair_address)),
            "🧠 <b>Decision:</b> {0} | <b>Score:</b> {1}/100".format(
                _decision_label(evaluation.decision), evaluation.score
            ),
            "💵 <b>Price:</b> ${0:.10g}".format(snapshot.price_usd),
            "💧 <b>Liquidity:</b> ${0:,.2f}".format(snapshot.liquidity_usd),
            "📈 <b>5m volume:</b> ${0:,.2f}".format(snapshot.volume_5m_usd),
            "🌊 <b>Trend:</b> 5m {0} | 1h {1}".format(
                _metric(snapshot.price_change_5m_pct, "%"),
                _metric(snapshot.price_change_1h_pct, "%"),
            ),
            "🔁 <b>5m txns:</b> buys {0} | sells {1}".format(
                _count(snapshot.buys_5m), _count(snapshot.sells_5m)
            ),
            "👥 <b>Top holders:</b> {0}".format(
                _metric(snapshot.top_holder_share_pct, "%")
            ),
            "𝕏 <b>Recent mentions:</b> {0} from {1} authors | {2}".format(
                _count(snapshot.x_recent_mentions),
                _count(snapshot.x_recent_author_count),
                _html(snapshot.x_sentiment_hint or "unknown"),
            ),
            "📊 <b>GeckoTerminal trend:</b> {0}".format(
                (
                    "trending rank #{0}".format(snapshot.geckoterminal_trending_rank)
                    if snapshot.geckoterminal_trending_rank is not None
                    else "not on current list"
                    if snapshot.geckoterminal_trending is False
                    else "unknown"
                )
            ),
            "📝 <b>Reason:</b> {0}".format(
                _html(evaluation.rationale or "No rationale returned.")
            ),
        ]
    )


def format_buy_result(
    snapshot: TokenSnapshot, evaluation: TokenEvaluation, result: TradeResult
) -> str:
    """Format a paper buy success or rejection."""

    return "\n".join(
        [
            "{0} <b>Paper Buy {1}</b>".format(
                "🟢" if result.success else "🟠",
                "Opened" if result.success else "Rejected",
            ),
            "🪙 <b>Token:</b> <code>{0}</code>".format(_html(snapshot.token_address)),
            "🧾 <b>Trade:</b> {0}".format(
                "#{0}".format(result.trade_id) if result.trade_id else "n/a"
            ),
            "🧠 <b>AI score:</b> {0}/100".format(evaluation.score),
            "ℹ️ <b>Detail:</b> {0}".format(_html(result.message)),
        ]
    )


def format_exit_analysis(
    trade: TradeRecord, snapshot: TokenSnapshot, decision: ExitDecision
) -> str:
    """Format one AI review before any paper sell."""

    return "\n".join(
        [
            "{0} <b>Paper Exit Analysis</b>".format(_exit_icon(decision)),
            "🧾 <b>Trade:</b> #{0}".format(trade.id),
            "🪙 <b>Token:</b> <code>{0}</code>".format(_html(trade.token_address)),
            "🧠 <b>Decision:</b> {0}".format(_decision_label(decision.decision)),
            "🏁 <b>Entry:</b> ${0:.10g}".format(trade.buy_price),
            "💵 <b>Current:</b> ${0:.10g}".format(snapshot.price_usd),
            "📝 <b>Reason:</b> {0}".format(
                _html(decision.rationale or "No rationale returned.")
            ),
        ]
    )


def format_sell_result(
    trade: TradeRecord,
    snapshot: TokenSnapshot,
    decision: ExitDecision,
    result: TradeResult,
) -> str:
    """Format a paper close success or failure."""

    return "\n".join(
        [
            "{0} <b>Paper Sell {1}</b>".format(
                "💸" if result.success else "🟠",
                "Closed" if result.success else "Rejected",
            ),
            "🧾 <b>Trade:</b> #{0}".format(trade.id),
            "🪙 <b>Token:</b> <code>{0}</code>".format(_html(snapshot.token_address)),
            "📝 <b>Exit reason:</b> {0}".format(
                _html(decision.rationale or "AI close")
            ),
            "📊 <b>PnL:</b> {0}".format(
                "{0:+.6f} SOL".format(result.pnl)
                if result.pnl is not None
                else "unavailable"
            ),
            "ℹ️ <b>Detail:</b> {0}".format(_html(result.message)),
        ]
    )


def format_reflection(rules: ReflectionRules) -> str:
    """Format nightly learned rules."""

    rendered_rules = "\n".join("• {0}".format(_html(rule)) for rule in rules.rules)
    return "🧠 <b>Daily Paper Reflection</b>\n{0}".format(rendered_rules)


def format_error(stage: str, detail: str) -> str:
    """Format a runtime pipeline failure for Telegram."""

    return "🚨 <b>Paper Bot Error</b>\n⚙️ <b>Stage:</b> {0}\n🧾 <b>Detail:</b> {1}".format(
        _html(stage), _html(detail)
    )


def welcome_text() -> str:
    """Render the start screen."""

    return (
        "🧪 <b>Paper Trader Ready</b>\n"
        "🧠 AI analysis reports and paper trade outcomes can land in this chat.\n"
        "⚡ Use the menu below to control auto entries and notifications.\n"
        "🛑 REAL trading remains disabled in v1."
    )


def menu_text() -> str:
    """Render a compact menu response."""

    return (
        "🎛 <b>Paper Bot Menu</b>\n"
        "📊 Check status, 🟢 enable entries, 🔴 pause entries,\n"
        "🔔 receive reports, or 🔕 mute reports."
    )


def menu_markup() -> Any:
    """Build the persistent Telegram menu keyboard."""

    from telegram import ReplyKeyboardMarkup

    return ReplyKeyboardMarkup(
        [
            [MENU_STATUS],
            [MENU_AUTO_ON, MENU_AUTO_OFF],
            [MENU_NOTIFY_ON, MENU_NOTIFY_OFF],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Choose a paper bot action",
    )


def _entry_icon(evaluation: TokenEvaluation) -> str:
    return "🚀" if evaluation.wants_buy else "🔎"


def _exit_icon(decision: ExitDecision) -> str:
    return "💸" if decision.wants_close else "🛡️"


def _decision_label(decision: str) -> str:
    labels = {
        "buy": "🟢 BUY",
        "skip": "🟠 SKIP",
        "close": "💸 CLOSE",
        "hold": "🛡️ HOLD",
    }
    return labels.get(decision.lower(), _html(decision.upper()))


def _report_state(chat_id: Optional[int], notifications_enabled: bool) -> str:
    if chat_id is None:
        return "⚪ send /start"
    return "🔔 ON in this chat" if notifications_enabled else "🔕 MUTED"


def _html(value: Any) -> str:
    return escape(str(value), quote=False)


def _metric(value: Optional[float], suffix: str) -> str:
    if value is None:
        return "unknown"
    return "{0:.4g}{1}".format(value, suffix)


def _count(value: Optional[int]) -> str:
    return "unknown" if value is None else str(value)
