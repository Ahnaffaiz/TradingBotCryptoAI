"""Telegram controls for the repo-owned Hermes paper trader."""

from __future__ import annotations

from dataclasses import replace
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
MENU_THRESHOLD = "🎚 Threshold"
MENU_SETTINGS = "⚙️ Settings"


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
        application.add_handler(CommandHandler("settings", self.status))
        application.add_handler(CommandHandler("auto_on", self.auto_on))
        application.add_handler(CommandHandler("auto_off", self.auto_off))
        application.add_handler(CommandHandler("threshold", self.threshold))
        application.add_handler(CommandHandler("set_threshold", self.threshold))
        application.add_handler(CommandHandler("launch_on", self.launch_on))
        application.add_handler(CommandHandler("launch_off", self.launch_off))
        application.add_handler(CommandHandler("scout_on", self.scout_on))
        application.add_handler(CommandHandler("scout_off", self.scout_off))
        application.add_handler(CommandHandler("launch_threshold", self.launch_threshold))
        application.add_handler(CommandHandler("scout_threshold", self.scout_threshold))
        application.add_handler(CommandHandler("take_profit", self.take_profit))
        application.add_handler(CommandHandler("stop_loss", self.stop_loss))
        application.add_handler(CommandHandler("trailing_stop", self.trailing_stop))
        application.add_handler(CommandHandler("max_hold", self.max_hold))
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
            MessageHandler(filters.Regex("^{0}$".format(MENU_THRESHOLD)), self.threshold)
        )
        application.add_handler(
            MessageHandler(filters.Regex("^{0}$".format(MENU_SETTINGS)), self.status)
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

    async def threshold(self, update: Any, context: Any) -> None:
        """Show or update the live launch score threshold."""

        await self._register_notification_chat(update)
        settings = await self.database.get_strategy_settings(
            self.config.strategy_defaults
        )
        args = getattr(context, "args", []) if context is not None else []
        raw_value = " ".join(args or []).strip()
        if not raw_value:
            await _reply(
                update,
                "🎚 <b>Launch threshold:</b> score ≥ {0}\n"
                "Use <code>/threshold 25</code> to change it live.".format(
                    settings.launch_score_threshold
                ),
                reply_markup=menu_markup(),
            )
            return
        threshold = _parse_score_threshold(raw_value)
        if threshold is None:
            await _reply(
                update,
                "⚠️ <b>Invalid threshold</b>\n"
                "Send a whole number from <code>0</code> to <code>100</code>, "
                "for example <code>/threshold 25</code>.",
                reply_markup=menu_markup(),
            )
            return
        updated = replace(
            settings,
            entry_score_threshold=threshold,
            launch_score_threshold=threshold,
        )
        await self.database.set_strategy_settings(updated)
        await self.database.add_activity(
            "strategy_threshold",
            "launch score threshold set to {0}".format(threshold),
            payload={"launch_score_threshold": threshold},
        )
        await _reply(
            update,
            "✅ <b>Launch threshold updated</b>\n"
            "Launch mode may now buy when the AI score is positive and ≥ {0}/100.".format(
                threshold
            ),
            reply_markup=menu_markup(),
        )

    async def launch_threshold(self, update: Any, context: Any) -> None:
        """Set launch-mode score threshold."""

        await self._update_int_setting(
            update,
            context,
            "launch_score_threshold",
            "Launch threshold",
            0,
            100,
        )

    async def launch_on(self, update: Any, _context: Any) -> None:
        """Enable launch discovery."""

        await self._toggle_strategy(update, "launch_enabled", "Launch mode", True)

    async def launch_off(self, update: Any, _context: Any) -> None:
        """Disable launch discovery."""

        await self._toggle_strategy(update, "launch_enabled", "Launch mode", False)

    async def scout_threshold(self, update: Any, context: Any) -> None:
        """Set scout-mode score threshold."""

        await self._update_int_setting(
            update,
            context,
            "scout_score_threshold",
            "Scout threshold",
            0,
            100,
        )

    async def scout_on(self, update: Any, _context: Any) -> None:
        """Enable scout discovery."""

        await self._toggle_strategy(update, "scout_enabled", "Scout mode", True)

    async def scout_off(self, update: Any, _context: Any) -> None:
        """Disable scout discovery."""

        await self._toggle_strategy(update, "scout_enabled", "Scout mode", False)

    async def take_profit(self, update: Any, context: Any) -> None:
        """Set hard take-profit percentage."""

        await self._update_float_setting(
            update, context, "take_profit_pct", "Take profit", 0.1, 500.0, "%"
        )

    async def stop_loss(self, update: Any, context: Any) -> None:
        """Set hard stop-loss percentage."""

        await self._update_float_setting(
            update, context, "stop_loss_pct", "Stop loss", 0.1, 100.0, "%"
        )

    async def trailing_stop(self, update: Any, context: Any) -> None:
        """Set trailing-stop percentage; zero disables it."""

        await self._update_float_setting(
            update, context, "trailing_stop_pct", "Trailing stop", 0.0, 100.0, "%"
        )

    async def max_hold(self, update: Any, context: Any) -> None:
        """Set maximum paper trade hold time."""

        await self._register_notification_chat(update)
        settings = await self.database.get_strategy_settings(
            self.config.strategy_defaults
        )
        raw_value = _context_text(context)
        if not raw_value:
            await _reply(
                update,
                "⏳ <b>Max hold:</b> {0}\nUse <code>/max_hold 60m</code>, "
                "<code>/max_hold 2h</code>, or <code>/max_hold 1d</code>.".format(
                    _duration_label(settings.max_hold_seconds)
                ),
                reply_markup=menu_markup(),
            )
            return
        seconds = _parse_duration_seconds(raw_value)
        if seconds is None or seconds <= 0:
            await _reply(
                update,
                "⚠️ <b>Invalid max hold</b>\nUse values like <code>30m</code>, "
                "<code>1h</code>, or <code>1d</code>.",
                reply_markup=menu_markup(),
            )
            return
        await self._store_settings(
            update,
            replace(settings, max_hold_seconds=seconds),
            "Max hold",
            _duration_label(seconds),
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
            "🎯 <b>Thresholds:</b> launch ≥ {0} | scout ≥ {1}".format(
                settings.launch_score_threshold, settings.scout_score_threshold
            ),
            "🧭 <b>Strategies:</b> launch {0} | scout {1}".format(
                "ON" if settings.launch_enabled else "OFF",
                "ON" if settings.scout_enabled else "OFF",
            ),
            "📦 <b>Trade size:</b> {0:.6f} SOL".format(settings.base_trade_amount),
            "🛡 <b>Hard exits:</b> TP {0:g}% | SL {1:g}% | trail {2:g}% | max {3}".format(
                settings.take_profit_pct,
                settings.stop_loss_pct,
                settings.trailing_stop_pct,
                _duration_label(settings.max_hold_seconds),
            ),
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
                    BotCommand("settings", "show strategy settings"),
                    BotCommand("auto_on", "enable paper auto entries"),
                    BotCommand("auto_off", "pause paper auto entries"),
                    BotCommand("launch_on", "enable launch scanner"),
                    BotCommand("launch_off", "disable launch scanner"),
                    BotCommand("scout_on", "enable scout scanner"),
                    BotCommand("scout_off", "disable scout scanner"),
                    BotCommand("threshold", "set launch buy score threshold"),
                    BotCommand("scout_threshold", "set scout buy score threshold"),
                    BotCommand("take_profit", "set hard take profit percent"),
                    BotCommand("stop_loss", "set hard stop loss percent"),
                    BotCommand("trailing_stop", "set trailing stop percent"),
                    BotCommand("max_hold", "set maximum hold time"),
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

    async def _toggle_strategy(
        self, update: Any, field_name: str, label: str, enabled: bool
    ) -> None:
        await self._register_notification_chat(update)
        settings = await self.database.get_strategy_settings(
            self.config.strategy_defaults
        )
        await self._store_settings(
            update,
            replace(settings, **{field_name: enabled}),
            label,
            "ON" if enabled else "OFF",
        )

    async def _update_int_setting(
        self,
        update: Any,
        context: Any,
        field_name: str,
        label: str,
        minimum: int,
        maximum: int,
    ) -> None:
        await self._register_notification_chat(update)
        settings = await self.database.get_strategy_settings(
            self.config.strategy_defaults
        )
        raw_value = _context_text(context)
        current_value = getattr(settings, field_name)
        if not raw_value:
            await _reply(
                update,
                "🎚 <b>{0}:</b> {1}\nUse <code>/{2} {1}</code> to change it.".format(
                    _html(label), current_value, field_name.replace("_score", "")
                ),
                reply_markup=menu_markup(),
            )
            return
        value = _parse_score_threshold(raw_value)
        if value is None or value < minimum or value > maximum:
            await _reply(
                update,
                "⚠️ <b>Invalid {0}</b>\nSend a whole number from <code>{1}</code> "
                "to <code>{2}</code>.".format(_html(label.lower()), minimum, maximum),
                reply_markup=menu_markup(),
            )
            return
        await self._store_settings(
            update, replace(settings, **{field_name: value}), label, str(value)
        )

    async def _update_float_setting(
        self,
        update: Any,
        context: Any,
        field_name: str,
        label: str,
        minimum: float,
        maximum: float,
        suffix: str,
    ) -> None:
        await self._register_notification_chat(update)
        settings = await self.database.get_strategy_settings(
            self.config.strategy_defaults
        )
        raw_value = _context_text(context)
        current_value = getattr(settings, field_name)
        if not raw_value:
            await _reply(
                update,
                "🎚 <b>{0}:</b> {1:g}{2}".format(
                    _html(label), current_value, _html(suffix)
                ),
                reply_markup=menu_markup(),
            )
            return
        value = _parse_float(raw_value)
        if value is None or value < minimum or value > maximum:
            await _reply(
                update,
                "⚠️ <b>Invalid {0}</b>\nSend a number from <code>{1:g}</code> "
                "to <code>{2:g}</code>.".format(_html(label.lower()), minimum, maximum),
                reply_markup=menu_markup(),
            )
            return
        await self._store_settings(
            update,
            replace(settings, **{field_name: value}),
            label,
            "{0:g}{1}".format(value, suffix),
        )

    async def _store_settings(
        self, update: Any, settings: Any, label: str, rendered_value: str
    ) -> None:
        await self.database.set_strategy_settings(settings)
        await self.database.add_activity(
            "strategy_settings",
            "{0} set to {1}".format(label, rendered_value),
            payload=settings.prompt_payload(),
        )
        await _reply(
            update,
            "✅ <b>{0} updated</b>\nNow set to <code>{1}</code>.".format(
                _html(label), _html(rendered_value)
            ),
            reply_markup=menu_markup(),
        )


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
            "🎯 <b>Strategy:</b> {0}".format(_html(snapshot.strategy.upper())),
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
            "🎯 <b>Strategy:</b> {0}".format(_html(snapshot.strategy.upper())),
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
        "🎚 inspect threshold, ⚙️ settings, 🔔 reports, or 🔕 mute."
    )


def menu_markup() -> Any:
    """Build the persistent Telegram menu keyboard."""

    from telegram import ReplyKeyboardMarkup

    return ReplyKeyboardMarkup(
        [
            [MENU_STATUS, MENU_THRESHOLD, MENU_SETTINGS],
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


def _parse_score_threshold(raw_value: str) -> Optional[int]:
    value = raw_value.strip()
    if value.endswith("%"):
        value = value[:-1].strip()
    if "/" in value:
        value = value.split("/", 1)[0].strip()
    try:
        threshold = int(value)
    except ValueError:
        return None
    if threshold < 0 or threshold > 100:
        return None
    return threshold


def _context_text(context: Any) -> str:
    return " ".join(getattr(context, "args", []) or []).strip()


def _parse_float(raw_value: str) -> Optional[float]:
    value = raw_value.strip()
    if value.endswith("%"):
        value = value[:-1].strip()
    try:
        return float(value)
    except ValueError:
        return None


def _parse_duration_seconds(raw_value: str) -> Optional[float]:
    value = raw_value.strip().lower()
    multiplier = 1.0
    if value.endswith("m"):
        multiplier = 60.0
        value = value[:-1]
    elif value.endswith("h"):
        multiplier = 3600.0
        value = value[:-1]
    elif value.endswith("d"):
        multiplier = 86400.0
        value = value[:-1]
    elif value.endswith("s"):
        value = value[:-1]
    try:
        return float(value.strip()) * multiplier
    except ValueError:
        return None


def _duration_label(seconds: float) -> str:
    if seconds >= 86400 and seconds % 86400 == 0:
        return "{0:g}d".format(seconds / 86400)
    if seconds >= 3600 and seconds % 3600 == 0:
        return "{0:g}h".format(seconds / 3600)
    if seconds >= 60 and seconds % 60 == 0:
        return "{0:g}m".format(seconds / 60)
    return "{0:g}s".format(seconds)


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
