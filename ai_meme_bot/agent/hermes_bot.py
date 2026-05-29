"""Telegram controls for the repo-owned Hermes paper trader."""

from __future__ import annotations

import json
from dataclasses import replace
from html import escape
import logging
from pathlib import Path
import re
from typing import Any, List, Optional, Protocol

from ai_meme_bot.config import AppConfig
from ai_meme_bot.agent.ai_service import HermesChatBackend
from ai_meme_bot.core.database import Database
from ai_meme_bot.core.execution import TradeExecutor
from ai_meme_bot.core.pnl_image import render_pnl_chart
from ai_meme_bot.models import (
    ExitDecision,
    ReflectionRules,
    TokenEvaluation,
    TokenSnapshot,
    TradePlan,
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
MENU_HISTORY = "📜 History"


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

    async def buy_more_result(
        self,
        trade: TradeRecord,
        snapshot: TokenSnapshot,
        decision: ExitDecision,
        result: TradeResult,
    ) -> None:
        """Report a paper add-on buy outcome."""

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

    async def buy_more_result(
        self,
        trade: TradeRecord,
        snapshot: TokenSnapshot,
        decision: ExitDecision,
        result: TradeResult,
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
        await self._send(
            format_entry_analysis(snapshot, evaluation),
            photo_url=snapshot.token_logo_url,
        )

    async def buy_result(
        self, snapshot: TokenSnapshot, evaluation: TokenEvaluation, result: TradeResult
    ) -> None:
        await self._send(format_buy_result(snapshot, evaluation, result))

    async def exit_analysis(
        self, trade: TradeRecord, snapshot: TokenSnapshot, decision: ExitDecision
    ) -> None:
        await self._send(
            format_exit_analysis(trade, snapshot, decision),
            photo_url=snapshot.token_logo_url,
        )

    async def buy_more_result(
        self,
        trade: TradeRecord,
        snapshot: TokenSnapshot,
        decision: ExitDecision,
        result: TradeResult,
    ) -> None:
        await self._send(format_buy_more_result(trade, snapshot, decision, result))

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

    async def _send(self, text: str, photo_url: Optional[str] = None) -> None:
        chat_id = await self.database.get_notification_chat_id()
        if chat_id is None or not await self.database.get_notifications_enabled():
            return
        if photo_url:
            try:
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_url,
                    caption=text[:1000],
                    parse_mode=PARSE_MODE,
                )
                if len(text) > 1000:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=text[:4000],
                        parse_mode=PARSE_MODE,
                    )
                return
            except Exception as exc:
                LOGGER.warning("Telegram photo notification failed: %s", exc)
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
        tracker: Any = None,
        executor: Optional[TradeExecutor] = None,
    ) -> None:
        self.config = config
        self.database = database
        self.operator_backend = operator_backend or HermesChatBackend(config)
        self.tracker = tracker
        self.executor = executor

    def build_application(self) -> Any:
        """Build the Telegram application when a token is configured."""

        if not self.config.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required to start Telegram polling.")
        from telegram.ext import (
            ApplicationBuilder,
            CallbackQueryHandler,
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
        application.add_handler(CommandHandler("history", self.history))
        application.add_handler(CommandHandler("position", self.position))
        application.add_handler(CommandHandler("pos", self.position))
        application.add_handler(CommandHandler("close", self.close_position))
        application.add_handler(CommandHandler("tp", self.take_profit_position))
        application.add_handler(CommandHandler("sl", self.cut_loss_position))
        application.add_handler(CommandHandler("cut_loss", self.cut_loss_position))
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
        application.add_handler(CommandHandler("min_trade_size", self.min_trade_size))
        application.add_handler(CommandHandler("max_trade_size", self.max_trade_size))
        application.add_handler(CommandHandler("min_size", self.min_trade_size))
        application.add_handler(CommandHandler("max_size", self.max_trade_size))
        application.add_handler(CommandHandler("size_range", self.size_range))
        application.add_handler(CommandHandler("take_profit", self.take_profit))
        application.add_handler(CommandHandler("stop_loss", self.stop_loss))
        application.add_handler(CommandHandler("trailing_stop", self.trailing_stop))
        application.add_handler(CommandHandler("max_hold", self.max_hold))
        application.add_handler(CommandHandler("dynamic_setup", self.dynamic_setup))
        application.add_handler(CommandHandler("dynamic_setup_on", self.dynamic_setup_on))
        application.add_handler(CommandHandler("dynamic_setup_off", self.dynamic_setup_off))
        application.add_handler(CommandHandler("notify_on", self.notify_on))
        application.add_handler(CommandHandler("notify_off", self.notify_off))
        application.add_handler(CommandHandler("hermes", self.hermes))
        application.add_handler(CallbackQueryHandler(self.button, pattern=r"^bot:"))
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
            MessageHandler(filters.Regex("^{0}$".format(MENU_HISTORY)), self.history)
        )
        application.add_handler(
            MessageHandler(filters.Regex("^{0}$".format(MENU_NOTIFY_ON)), self.notify_on)
        )
        application.add_handler(
            MessageHandler(
                filters.Regex("^{0}$".format(MENU_NOTIFY_OFF)), self.notify_off
            )
        )
        application.add_handler(
            MessageHandler(filters.Regex(r"^#?\d+$"), self.position)
        )
        application.add_handler(
            MessageHandler(
                filters.Regex(r"^(✅\s*)?(Take Profit|TP)\s+#?\d+$"),
                self.take_profit_position,
            )
        )
        application.add_handler(
            MessageHandler(
                filters.Regex(r"^(🛑\s*)?(Cut Loss|SL)\s+#?\d+$"),
                self.cut_loss_position,
            )
        )
        return application

    async def button(self, update: Any, _context: Any) -> None:
        """Handle inline menu button taps."""

        query = getattr(update, "callback_query", None)
        if query is None:
            return
        try:
            await query.answer()
        except Exception:
            pass
        data = str(getattr(query, "data", "") or "")
        await self._register_notification_chat(update)

        if data == "bot:menu":
            await _respond(update, menu_text(), reply_markup=menu_markup())
            return
        if data == "bot:status":
            await self.status(update, None)
            return
        if data == "bot:history":
            trades = await self.database.get_trade_history(limit=50)
            await _respond(
                update, format_trade_history(trades), reply_markup=history_menu_markup()
            )
            return
        if data == "bot:history:all":
            trades = await self.database.get_trade_history(limit=None)
            await _respond(
                update, format_trade_history(trades), reply_markup=history_menu_markup()
            )
            return
        if data == "bot:positions":
            await _respond(
                update,
                await self.render_positions_menu(),
                reply_markup=position_menu_markup(await self.database.get_open_trades()),
            )
            return
        if data.startswith("bot:position:"):
            trade_id = _parse_callback_int(data, "bot:position:")
            if trade_id is not None:
                text = await self.render_position_detail(trade_id)
                trade = await self.database.get_trade(trade_id)
                await _respond(
                    update,
                    text,
                    reply_markup=position_action_markup(trade) if trade else menu_markup(),
                )
            return
        if data in {"bot:tp:", "bot:sl:", "bot:close:"}:
            return
        if data.startswith("bot:tp:"):
            await self._manual_close_by_callback(update, data, "bot:tp:", "manual take profit")
            return
        if data.startswith("bot:sl:"):
            await self._manual_close_by_callback(update, data, "bot:sl:", "manual cut loss")
            return
        if data.startswith("bot:close:"):
            await self._manual_close_by_callback(update, data, "bot:close:", "manual close")
            return
        if data == "bot:trading":
            await _respond(
                update,
                await self.render_trading_menu(),
                reply_markup=trading_menu_markup(
                    await self.database.get_strategy_settings(self.config.strategy_defaults)
                ),
            )
            return
        if data == "bot:settings":
            await _respond(
                update,
                await self.render_settings_menu(),
                reply_markup=settings_menu_markup(
                    await self.database.get_strategy_settings(self.config.strategy_defaults)
                ),
            )
            return
        if data == "bot:size":
            settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
            await _respond(
                update,
                _size_menu_text(settings),
                reply_markup=size_menu_markup(settings),
            )
            return
        if data == "bot:exits":
            settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
            await _respond(
                update,
                _exit_menu_text(settings),
                reply_markup=size_menu_markup(settings),
            )
            return
        if data == "bot:risk":
            settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
            await _respond(
                update,
                _risk_menu_text(settings),
                reply_markup=risk_menu_markup(settings),
            )
            return
        if data == "bot:reports":
            await _respond(
                update,
                await self.render_reports_menu(),
                reply_markup=reports_menu_markup(
                    await self.database.get_notifications_enabled()
                ),
            )
            return

        handled = await self._handle_setting_button(update, data)
        if handled:
            return
        await _respond(update, menu_text(), reply_markup=menu_markup())

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
        text = await self.render_status()
        open_trades = await self.database.get_open_trades()
        image_path = await self.render_pnl_image(open_trades)
        reply_markup = position_menu_markup(open_trades)
        if image_path is not None:
            await _reply_photo(update, image_path, text, reply_markup=reply_markup)
        else:
            await _reply(update, text, reply_markup=reply_markup)

    async def history(self, update: Any, context: Any) -> None:
        """Show trading history, newest first."""

        await self._register_notification_chat(update)
        raw_value = _context_text(context)
        limit = (
            None
            if raw_value.strip().lower() == "all"
            else _parse_history_limit(raw_value)
        )
        if limit is None and raw_value.strip().lower() != "all":
            await _reply(
                update,
                "⚠️ <b>Invalid history limit</b>\nUse the History buttons to choose latest or all.",
                reply_markup=size_menu_markup(settings),
            )
            return
        trades = await self.database.get_trade_history(limit=limit)
        await _reply(update, format_trade_history(trades), reply_markup=menu_markup())

    async def position(self, update: Any, context: Any) -> None:
        """Show one position by id."""

        await self._register_notification_chat(update)
        trade_id = _trade_id_from_update(update, context)
        if trade_id is None:
            await _reply(
                update,
                "🧾 <b>Position detail</b>\nTap <b>Positions</b> and choose an open trade.",
                reply_markup=position_menu_markup(await self.database.get_open_trades()),
            )
            return
        text = await self.render_position_detail(trade_id)
        trade = await self.database.get_trade(trade_id)
        await _reply(
            update,
            text,
            reply_markup=position_action_markup(trade) if trade else menu_markup(),
        )

    async def close_position(self, update: Any, context: Any) -> None:
        """Manually close one open paper position."""

        await self._manual_close(update, context, "manual close")

    async def take_profit_position(self, update: Any, context: Any) -> None:
        """Manually take profit on one open paper position."""

        await self._manual_close(update, context, "manual take profit")

    async def cut_loss_position(self, update: Any, context: Any) -> None:
        """Manually cut loss on one open paper position."""

        await self._manual_close(update, context, "manual cut loss")

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
                "Use the Settings buttons to change it live.".format(
                    settings.launch_score_threshold
                ),
                reply_markup=exit_menu_markup(settings),
            )
            return
        threshold = _parse_score_threshold(raw_value)
        if threshold is None:
            await _reply(
                update,
                "⚠️ <b>Invalid threshold</b>\n"
                "Send a whole number from <code>0</code> to <code>100</code>, "
                "for example <code>/threshold 25</code>.",
                reply_markup=size_menu_markup(settings),
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
            reply_markup=trading_menu_markup(settings),
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

    async def min_trade_size(self, update: Any, context: Any) -> None:
        """Set minimum dynamic trade size."""

        await self._update_trade_size_bound(
            update,
            context,
            "min_trade_amount_sol",
            "Minimum trade size",
            is_minimum=True,
        )

    async def max_trade_size(self, update: Any, context: Any) -> None:
        """Set maximum dynamic trade size."""

        await self._update_trade_size_bound(
            update,
            context,
            "max_trade_amount_sol",
            "Maximum trade size",
            is_minimum=False,
        )

    async def size_range(self, update: Any, context: Any) -> None:
        """Set minimum and maximum dynamic trade size together."""

        await self._register_notification_chat(update)
        settings = await self.database.get_strategy_settings(
            self.config.strategy_defaults
        )
        args = getattr(context, "args", []) if context is not None else []
        if len(args) != 2:
            await _reply(
                update,
                "📦 <b>Dynamic size range:</b> {0:g}..{1:g} SOL\n"
                "Tap a Size preset below or use <code>/size_range 0.1 0.3</code> for a custom range.".format(
                    settings.min_trade_amount_sol,
                    settings.max_trade_amount_sol,
                ),
                reply_markup=exit_menu_markup(settings),
            )
            return
        minimum = _parse_float(args[0])
        maximum = _parse_float(args[1])
        if (
            minimum is None
            or maximum is None
            or minimum < 0.01
            or maximum > 2.0
            or minimum > maximum
        ):
            await _reply(
                update,
                "⚠️ <b>Invalid size range</b>\nUse two numbers from "
                "<code>0.01</code> to <code>2</code> SOL, with min <= max.",
                reply_markup=exit_menu_markup(settings),
            )
            return
        await self._store_settings(
            update,
            replace(
                settings,
                min_trade_amount_sol=minimum,
                max_trade_amount_sol=maximum,
            ),
            "Dynamic size range",
            "{0:g}..{1:g} SOL".format(minimum, maximum),
        )

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
                "⏳ <b>Max hold:</b> {0}\nUse the Exit buttons for presets, or send a custom duration if needed.".format(
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

    async def dynamic_setup(self, update: Any, _context: Any) -> None:
        """Show dynamic per-trade setup mode and guardrails."""

        await self._register_notification_chat(update)
        settings = await self.database.get_strategy_settings(
            self.config.strategy_defaults
        )
        min_entry_size, max_entry_size = settings.trade_size_bounds()
        await _reply(
            update,
            "🧠 <b>Dynamic setup:</b> {0}\n"
            "📦 <b>AI size:</b> {1:g}..{2:g} SOL\n"
            "🛡 <b>AI exits:</b> SL 1..50% | TP 3..500% | trail 0..50% | max 60s..1d\n"
            "Use the Trading and Size buttons to change these controls.".format(
                "ON" if settings.dynamic_setup_enabled else "OFF",
                min_entry_size,
                max_entry_size,
            ),
            reply_markup=trading_menu_markup(settings),
        )

    async def dynamic_setup_on(self, update: Any, _context: Any) -> None:
        """Require AI buy decisions and per-trade setup for new entries."""

        await self._toggle_strategy(
            update, "dynamic_setup_enabled", "Dynamic setup", True
        )

    async def dynamic_setup_off(self, update: Any, _context: Any) -> None:
        """Use static size and exit settings for new entries."""

        await self._toggle_strategy(
            update, "dynamic_setup_enabled", "Dynamic setup", False
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
            "Use the Reports buttons to resume paper-trading reports.",
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
            "🎯 <b>Launch threshold:</b> ≥ {0}".format(
                settings.launch_score_threshold
            ),
            "🧭 <b>Launch scanner:</b> {0}".format(
                "ON" if settings.launch_enabled else "OFF"
            ),
            "📦 <b>Trade size:</b> static {0:.6f} SOL | dynamic {1:.6f}..{2:.6f} SOL".format(
                settings.base_trade_amount,
                settings.min_trade_amount_sol,
                settings.max_trade_amount_sol,
            ),
            "🧠 <b>Setup:</b> {0}".format(
                "dynamic AI per trade"
                if settings.dynamic_setup_enabled
                else "static settings"
            ),
            "🛡 <b>Hard exits:</b> TP {0:g}% | SL {1:g}% | trail {2:g}% | max {3}".format(
                settings.take_profit_pct,
                settings.stop_loss_pct,
                settings.trailing_stop_pct,
                _duration_label(settings.max_hold_seconds),
            ),
            "⏱ <b>Cadence:</b> discover {0:g}s | exits {1:g}s".format(
                settings.tracker_poll_seconds, settings.position_review_seconds
            ),
            "🚧 <b>Risk gates:</b> block UTC {0} | buy/sell ≥ {1:g} | vol/liq ≥ {2:g}".format(
                _html(settings.blocked_entry_utc_hours or "none"),
                settings.min_buy_sell_ratio,
                settings.min_volume_liquidity_ratio_5m,
            ),
            "🌙 <b>Reflection:</b> {0} {1}".format(
                _html(settings.reflection_time), _html(self.config.reflection_timezone)
            ),
            "📂 <b>Open positions:</b> {0}".format(len(open_trades)),
        ]
        if settings.scout_enabled:
            lines.insert(
                8,
                "🔎 <b>Scout scanner:</b> ON | threshold ≥ {0}".format(
                    settings.scout_score_threshold
                ),
            )
        lines.extend(_format_open_trades(open_trades))
        if closed_trades:
            lines.append("")
            lines.append("✅ <b>Recent closed trades</b>")
            lines.extend(_format_closed_trades(closed_trades))
        return "\n".join(lines)

    async def render_trading_menu(self) -> str:
        """Render trading controls for button menus."""

        auto_enabled = await self.database.get_auto_trading()
        settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
        return "\n".join(
            [
                "🤖 <b>Trading Controls</b>",
                "Auto entries: <b>{0}</b>".format("ON" if auto_enabled else "OFF"),
                "Launch scanner: <b>{0}</b>".format(
                    "ON" if settings.launch_enabled else "OFF"
                ),
                "Scout scanner: <b>{0}</b>".format(
                    "ON" if settings.scout_enabled else "OFF"
                ),
                "Dynamic setup: <b>{0}</b>".format(
                    "ON" if settings.dynamic_setup_enabled else "OFF"
                ),
            ]
        )

    async def render_settings_menu(self) -> str:
        """Render high-level settings for button menus."""

        settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
        return "\n".join(
            [
                "⚙️ <b>Strategy Settings</b>",
                "Launch threshold: <b>{0}/100</b>".format(
                    settings.launch_score_threshold
                ),
                "Dynamic size: <b>{0:g}..{1:g} SOL</b>".format(
                    settings.min_trade_amount_sol,
                    settings.max_trade_amount_sol,
                ),
                "Hard exits: <b>TP {0:g}% | SL {1:g}% | trail {2:g}% | max {3}</b>".format(
                    settings.take_profit_pct,
                    settings.stop_loss_pct,
                    settings.trailing_stop_pct,
                    _duration_label(settings.max_hold_seconds),
                ),
                "Risk gates: <b>UTC {0} | buy/sell ≥ {1:g} | vol/liq ≥ {2:g}</b>".format(
                    _html(settings.blocked_entry_utc_hours or "none"),
                    settings.min_buy_sell_ratio,
                    settings.min_volume_liquidity_ratio_5m,
                ),
            ]
        )

    async def render_reports_menu(self) -> str:
        """Render notification controls for button menus."""

        chat_id = await self.database.get_notification_chat_id()
        enabled = await self.database.get_notifications_enabled()
        return "\n".join(
            [
                "🔔 <b>Reports</b>",
                "State: <b>{0}</b>".format(_report_state(chat_id, enabled)),
                "This chat receives entry analysis, buys, position reviews, sells, reflection, and errors.",
            ]
        )

    async def render_positions_menu(self) -> str:
        """Render open-position chooser text."""

        open_trades = await self.database.get_open_trades()
        if not open_trades:
            return "🧾 <b>Positions</b>\nNo open paper positions."
        lines = ["🧾 <b>Positions</b>", "Tap a position to view or manage it."]
        lines.extend(_format_open_trades(open_trades))
        return "\n".join(lines)

    async def render_pnl_image(
        self, open_trades: Optional[List[TradeRecord]] = None
    ) -> Optional[Path]:
        """Generate a status PnL chart image."""

        try:
            closed = list(reversed(await self.database.get_closed_trades(limit=60)))
            open_rows = (
                open_trades
                if open_trades is not None
                else await self.database.get_open_trades()
            )
            open_pnls = []
            for trade in open_rows:
                snapshot = await self._current_snapshot(trade)
                if snapshot is not None:
                    open_pnls.append(_unrealized_pnl(trade, snapshot.price_usd))
            output_path = (
                self.config.db_path.parent
                / "pnl_charts"
                / "paper_status_pnl.png"
            )
            return render_pnl_chart(
                output_path,
                [trade.pnl or 0.0 for trade in closed],
                open_pnls,
            )
        except Exception as exc:
            LOGGER.warning("PnL image generation failed: %s", exc)
            return None

    async def render_position_detail(self, trade_id: int) -> str:
        """Render one position detail with current market condition when available."""

        trade = await self.database.get_trade(trade_id)
        if trade is None:
            return "⚠️ <b>Position not found</b>\nNo trade with id <code>#{0}</code>.".format(
                trade_id
            )
        snapshot = await self._current_snapshot(trade)
        plan = _trade_plan_from_json(trade)
        current_price = snapshot.price_usd if snapshot is not None else trade.sell_price
        pnl = _unrealized_pnl(trade, current_price) if current_price else trade.pnl
        pnl_pct = _pct_change(trade.buy_price, current_price) if current_price else None
        lines = [
            "🧾 <b>Position #{0}</b> {1}".format(trade.id, _html(trade.status)),
            "🪙 <b>Token:</b> {0}".format(
                _format_trade_token_identity(trade, snapshot)
            ),
            "🏁 <b>Entry:</b> ${0:.10g} | {1:.6f} SOL".format(
                trade.buy_price, trade.entry_amount_sol
            ),
            "📦 <b>Quantity:</b> {0:.10g}".format(trade.token_quantity),
            "🕒 <b>Opened:</b> {0}".format(_html(trade.opened_at)),
        ]
        if current_price:
            lines.extend(
                [
                    "💵 <b>Current:</b> ${0:.10g}".format(current_price),
                    "📊 <b>PnL:</b> {0} ({1})".format(
                        _pnl_label(pnl), _metric(pnl_pct, "%")
                    ),
                ]
            )
        if snapshot is not None:
            lines.extend(
                [
                    "💧 <b>Liquidity:</b> ${0:,.2f}".format(snapshot.liquidity_usd),
                    "📈 <b>5m volume:</b> ${0:,.2f}".format(snapshot.volume_5m_usd),
                    "🌊 <b>Trend:</b> 5m {0} | 1h {1}".format(
                        _metric(snapshot.price_change_5m_pct, "%"),
                        _metric(snapshot.price_change_1h_pct, "%"),
                    ),
                    "🔁 <b>5m txns:</b> buys {0} | sells {1}".format(
                        _count(snapshot.buys_5m), _count(snapshot.sells_5m)
                    ),
                ]
            )
        lines.extend(
            [
                "🛡 <b>AI setup:</b> SL {0:g}% | TP {1} | trail {2:g}% | max {3}".format(
                    plan.stop_loss_pct,
                    _format_targets(plan.take_profit_targets_pct),
                    plan.trailing_stop_pct,
                    _duration_label(plan.max_hold_seconds),
                ),
                "🧠 <b>Setup mode:</b> {0}".format(_trade_setup_label(trade)),
                "📝 <b>Plan:</b> {0}".format(
                    _html(plan.rationale or "No AI setup rationale stored.")
                ),
            ]
        )
        if trade.status == "OPEN":
            lines.append("")
            lines.append(
                "Use <code>/tp {0}</code>, <code>/sl {0}</code>, or <code>/close {0}</code>.".format(
                    trade.id
                )
            )
        elif trade.exit_reason:
            lines.append("🏁 <b>Exit:</b> {0}".format(_html(trade.exit_reason)))
        return "\n".join(lines)

    async def _manual_close(self, update: Any, context: Any, reason: str) -> None:
        await self._register_notification_chat(update)
        trade_id = _trade_id_from_update(update, context)
        if trade_id is None:
            await _reply(
                update,
                "⚠️ <b>Position id required</b>\nUse <code>/tp 1</code>, "
                "<code>/sl 1</code>, or <code>/close 1</code>.",
                reply_markup=position_menu_markup(
                    await self.database.get_open_trades()
                ),
            )
            return
        trade = await self.database.get_trade(trade_id)
        if trade is None or trade.status != "OPEN":
            await _reply(
                update,
                "⚠️ <b>Open position not found</b>\nTrade <code>#{0}</code> is not open.".format(
                    trade_id
                ),
                reply_markup=position_menu_markup(
                    await self.database.get_open_trades()
                ),
            )
            return
        if self.executor is None or self.tracker is None:
            await _reply(
                update,
                "⚠️ <b>Manual close unavailable</b>\nThe Telegram bot needs tracker "
                "and executor services to close at current price.",
                reply_markup=position_action_markup(trade),
            )
            return
        snapshot = await self._current_snapshot(trade)
        if snapshot is None:
            await _reply(
                update,
                "⚠️ <b>Current price unavailable</b>\nCould not refresh this token.",
                reply_markup=position_action_markup(trade),
            )
            return
        result = await self.executor.close_trade(trade, snapshot, reason)
        await self.database.add_activity(
            "manual_sell" if result.success else "manual_sell_rejected",
            result.message,
            trade.token_address,
            {"trade_id": trade.id, "pnl": result.pnl, "reason": reason},
        )
        decision = ExitDecision("close", reason)
        await _reply(
            update,
            format_sell_result(trade, snapshot, decision, result),
            reply_markup=position_menu_markup(await self.database.get_open_trades()),
        )

    async def _manual_close_by_callback(
        self, update: Any, data: str, prefix: str, reason: str
    ) -> None:
        trade_id = _parse_callback_int(data, prefix)
        if trade_id is None:
            await _respond(
                update,
                "⚠️ <b>Position id missing</b>",
                reply_markup=position_menu_markup(await self.database.get_open_trades()),
            )
            return
        trade = await self.database.get_trade(trade_id)
        if trade is None or trade.status != "OPEN":
            await _respond(
                update,
                "⚠️ <b>Open position not found</b>\nTrade <code>#{0}</code> is not open.".format(
                    trade_id
                ),
                reply_markup=position_menu_markup(await self.database.get_open_trades()),
            )
            return
        if self.executor is None or self.tracker is None:
            await _respond(
                update,
                "⚠️ <b>Manual close unavailable</b>\nThe Telegram bot needs tracker and executor services to close at current price.",
                reply_markup=position_action_markup(trade),
            )
            return
        snapshot = await self._current_snapshot(trade)
        if snapshot is None:
            await _respond(
                update,
                "⚠️ <b>Current price unavailable</b>\nCould not refresh this token.",
                reply_markup=position_action_markup(trade),
            )
            return
        result = await self.executor.close_trade(trade, snapshot, reason)
        await self.database.add_activity(
            "manual_sell" if result.success else "manual_sell_rejected",
            result.message,
            trade.token_address,
            {"trade_id": trade.id, "pnl": result.pnl, "reason": reason},
        )
        decision = ExitDecision("sell_now", reason)
        await _respond(
            update,
            format_sell_result(trade, snapshot, decision, result),
            reply_markup=position_menu_markup(await self.database.get_open_trades()),
        )

    async def _handle_setting_button(self, update: Any, data: str) -> bool:
        settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
        if data == "bot:auto:on":
            await self.database.set_auto_trading(True)
            await _respond(
                update,
                "🟢 <b>Auto entries enabled</b>",
                reply_markup=trading_menu_markup(settings),
            )
            return True
        if data == "bot:auto:off":
            await self.database.set_auto_trading(False)
            await _respond(
                update,
                "🔴 <b>Auto entries paused</b>",
                reply_markup=trading_menu_markup(settings),
            )
            return True
        toggle_map = {
            "bot:launch:on": ("launch_enabled", True, "Launch mode"),
            "bot:launch:off": ("launch_enabled", False, "Launch mode"),
            "bot:scout:on": ("scout_enabled", True, "Scout mode"),
            "bot:scout:off": ("scout_enabled", False, "Scout mode"),
            "bot:dynamic:on": ("dynamic_setup_enabled", True, "Dynamic setup"),
            "bot:dynamic:off": ("dynamic_setup_enabled", False, "Dynamic setup"),
        }
        if data in toggle_map:
            field_name, enabled, label = toggle_map[data]
            updated = replace(settings, **{field_name: enabled})
            await self.database.set_strategy_settings(updated)
            await self.database.add_activity(
                "strategy_settings",
                "{0} set to {1}".format(label, "ON" if enabled else "OFF"),
                payload=updated.prompt_payload(),
            )
            await _respond(
                update,
                "✅ <b>{0} updated</b>\nNow set to <code>{1}</code>.".format(
                    _html(label), "ON" if enabled else "OFF"
                ),
                reply_markup=trading_menu_markup(updated),
            )
            return True
        if data.startswith("bot:threshold:"):
            value = _parse_callback_int(data, "bot:threshold:")
            if value is not None:
                updated = replace(
                    settings,
                    entry_score_threshold=value,
                    launch_score_threshold=value,
                )
                await self.database.set_strategy_settings(updated)
                await self.database.add_activity(
                    "strategy_threshold",
                    "launch score threshold set to {0}".format(value),
                    payload={"launch_score_threshold": value},
                )
                await _respond(
                    update,
                    "✅ <b>Launch threshold updated</b>\nNow set to <code>{0}/100</code>.".format(
                        value
                    ),
                    reply_markup=settings_menu_markup(updated),
                )
            return True
        if data.startswith("bot:size:"):
            raw = data[len("bot:size:") :]
            try:
                min_text, max_text = raw.split(":", 1)
                minimum = float(min_text)
                maximum = float(max_text)
            except ValueError:
                return True
            updated = replace(
                settings,
                min_trade_amount_sol=minimum,
                max_trade_amount_sol=maximum,
            )
            await self.database.set_strategy_settings(updated)
            await self.database.add_activity(
                "strategy_settings",
                "Dynamic size range set to {0:g}..{1:g} SOL".format(minimum, maximum),
                payload=updated.prompt_payload(),
            )
            await _respond(
                update,
                "✅ <b>Dynamic size range updated</b>\nNow set to <code>{0:g}..{1:g} SOL</code>.".format(
                    minimum, maximum
                ),
                reply_markup=size_menu_markup(updated),
            )
            return True
        if data.startswith("bot:exit:"):
            updated, label, value = _updated_exit_setting(settings, data)
            if updated is not None:
                await self.database.set_strategy_settings(updated)
                await self.database.add_activity(
                    "strategy_settings",
                    "{0} set to {1}".format(label, value),
                    payload=updated.prompt_payload(),
                )
                await _respond(
                    update,
                    "✅ <b>{0} updated</b>\nNow set to <code>{1}</code>.".format(
                        _html(label), _html(value)
                    ),
                    reply_markup=exit_menu_markup(updated),
                )
            return True
        if data.startswith("bot:risk:"):
            updated, label, value = _updated_risk_setting(settings, data)
            if updated is not None:
                await self.database.set_strategy_settings(updated)
                await self.database.add_activity(
                    "strategy_settings",
                    "{0} set to {1}".format(label, value),
                    payload=updated.prompt_payload(),
                )
                await _respond(
                    update,
                    "✅ <b>{0} updated</b>\nNow set to <code>{1}</code>.".format(
                        _html(label), _html(value)
                    ),
                    reply_markup=risk_menu_markup(updated),
                )
            return True
        if data == "bot:notify:on":
            await self.database.set_notifications_enabled(True)
            await _respond(
                update,
                "🔔 <b>Reports enabled</b>",
                reply_markup=reports_menu_markup(True),
            )
            return True
        if data == "bot:notify:off":
            await self.database.set_notifications_enabled(False)
            await _respond(
                update,
                "🔕 <b>Reports muted</b>",
                reply_markup=reports_menu_markup(False),
            )
            return True
        return False

    async def _current_snapshot(self, trade: TradeRecord) -> Optional[TokenSnapshot]:
        if self.tracker is None:
            return None
        try:
            return await self.tracker.snapshot_for_token(
                trade.token_address, apply_filters=False
            )
        except Exception as exc:
            LOGGER.warning("Position snapshot failed for trade=%s: %s", trade.id, exc)
            return None

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
                    BotCommand("history", "show trading history"),
                    BotCommand("position", "show position detail by id"),
                    BotCommand("tp", "manually take profit by position id"),
                    BotCommand("sl", "manually cut loss by position id"),
                    BotCommand("close", "manually close position by id"),
                    BotCommand("auto_on", "enable paper auto entries"),
                    BotCommand("auto_off", "pause paper auto entries"),
                    BotCommand("launch_on", "enable launch scanner"),
                    BotCommand("launch_off", "disable launch scanner"),
                    BotCommand("scout_on", "enable scout scanner"),
                    BotCommand("scout_off", "disable scout scanner"),
                    BotCommand("threshold", "set launch buy score threshold"),
                    BotCommand("scout_threshold", "set scout buy score threshold"),
                    BotCommand("min_trade_size", "set minimum dynamic trade size"),
                    BotCommand("max_trade_size", "set maximum dynamic trade size"),
                    BotCommand("size_range", "set dynamic trade size range"),
                    BotCommand("take_profit", "set hard take profit percent"),
                    BotCommand("stop_loss", "set hard stop loss percent"),
                    BotCommand("trailing_stop", "set trailing stop percent"),
                    BotCommand("max_hold", "set maximum hold time"),
                    BotCommand("dynamic_setup", "show dynamic setup mode"),
                    BotCommand("dynamic_setup_on", "enable AI trade setup"),
                    BotCommand("dynamic_setup_off", "use static trade setup"),
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
                "🎚 <b>{0}:</b> {1}\nUse the Settings buttons to change presets.".format(
                    _html(label), current_value
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

    async def _update_trade_size_bound(
        self,
        update: Any,
        context: Any,
        field_name: str,
        label: str,
        is_minimum: bool,
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
                "📦 <b>{0}:</b> {1:g} SOL\n"
                "Dynamic AI size range is <code>{2:g}..{3:g} SOL</code>.".format(
                    _html(label),
                    current_value,
                    settings.min_trade_amount_sol,
                    settings.max_trade_amount_sol,
                ),
                reply_markup=menu_markup(),
            )
            return
        value = _parse_float(raw_value)
        if value is None or value < 0.01 or value > 2.0:
            await _reply(
                update,
                "⚠️ <b>Invalid {0}</b>\nSend a number from <code>0.01</code> "
                "to <code>2</code> SOL.".format(_html(label.lower())),
                reply_markup=menu_markup(),
            )
            return
        if is_minimum and value > settings.max_trade_amount_sol:
            await _reply(
                update,
                "⚠️ <b>Invalid range</b>\nMinimum trade size cannot be above "
                "the current maximum of <code>{0:g} SOL</code>.".format(
                    settings.max_trade_amount_sol
                ),
                reply_markup=menu_markup(),
            )
            return
        if not is_minimum and value < settings.min_trade_amount_sol:
            await _reply(
                update,
                "⚠️ <b>Invalid range</b>\nMaximum trade size cannot be below "
                "the current minimum of <code>{0:g} SOL</code>.".format(
                    settings.min_trade_amount_sol
                ),
                reply_markup=menu_markup(),
            )
            return
        await self._store_settings(
            update,
            replace(settings, **{field_name: value}),
            label,
            "{0:g} SOL".format(value),
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


async def _respond(update: Any, text: str, reply_markup: Any = None) -> None:
    query = getattr(update, "callback_query", None)
    if query is not None:
        try:
            await query.edit_message_text(
                text=text,
                parse_mode=PARSE_MODE,
                reply_markup=reply_markup,
            )
            return
        except Exception:
            pass
    await _reply(update, text, reply_markup=reply_markup)


async def _reply_photo(
    update: Any, image_path: Path, caption: str, reply_markup: Any = None
) -> None:
    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if message is None:
        return
    short_caption = caption if len(caption) <= 1000 else "📊 <b>Paper PnL</b>"
    try:
        with image_path.open("rb") as image_file:
            await message.reply_photo(
                photo=image_file,
                caption=short_caption,
                parse_mode=PARSE_MODE,
                reply_markup=reply_markup if short_caption == caption else None,
            )
        if short_caption != caption:
            await _reply(update, caption, reply_markup=reply_markup)
    except AttributeError:
        await _reply(update, caption, reply_markup=reply_markup)


def _format_open_trades(trades: List[TradeRecord]) -> List[str]:
    return [
        "• #{0} <code>{1}</code> entry ${2:.10g} size {3:.6f} SOL".format(
            trade.id,
            _html(_short_token(trade.token_address)),
            trade.buy_price,
            trade.entry_amount_sol,
        )
        for trade in trades[:5]
    ]


def _format_closed_trades(trades: List[TradeRecord]) -> List[str]:
    lines = []
    for trade in trades:
        pnl_pct = _pct_change(trade.buy_price, trade.sell_price)
        lines.append(
            "• #{0} {1} size {2:.6f} SOL sell {3} PnL {4} ({5}) exit {6}".format(
                trade.id,
                _format_trade_token_identity(trade),
                trade.entry_amount_sol,
                "${0:.10g}".format(trade.sell_price)
                if trade.sell_price is not None
                else "n/a",
                _pnl_label(trade.pnl),
                _metric(pnl_pct, "%"),
                _html(_short_reason(trade.exit_reason)),
            )
        )
    return lines


def _short_token(token_address: str) -> str:
    if len(token_address) <= 14:
        return token_address
    return "{0}...{1}".format(token_address[:7], token_address[-5:])


def _short_reason(reason: Optional[str]) -> str:
    text = str(reason or "n/a").strip()
    if len(text) <= 48:
        return text
    return "{0}...".format(text[:45])


def _format_snapshot_token_identity(snapshot: TokenSnapshot) -> str:
    return _format_token_identity(
        snapshot.token_address,
        snapshot.token_name,
        snapshot.token_symbol,
    )


def _format_trade_token_identity(
    trade: TradeRecord, snapshot: Optional[TokenSnapshot] = None
) -> str:
    if snapshot is not None:
        return _format_snapshot_token_identity(snapshot)
    metadata = _trade_token_metadata(trade)
    return _format_token_identity(
        trade.token_address,
        metadata.get("token_name"),
        metadata.get("token_symbol"),
    )


def _format_token_identity(
    token_address: str, token_name: Any = None, token_symbol: Any = None
) -> str:
    name = str(token_name or "").strip()
    symbol = str(token_symbol or "").strip()
    if name and symbol:
        label = "{0} ({1})".format(name, symbol)
    elif name:
        label = name
    elif symbol:
        label = symbol
    else:
        label = "Unknown token"
    return "{0} | CA <code>{1}</code>".format(_html(label), _html(token_address))


def _trade_token_metadata(trade: TradeRecord) -> dict[str, Any]:
    for raw_json in (trade.exit_snapshot_json, trade.entry_snapshot_json):
        try:
            payload = json.loads(raw_json or "{}")
        except (TypeError, ValueError):
            payload = {}
        if isinstance(payload, dict):
            name = payload.get("token_name")
            symbol = payload.get("token_symbol")
            if name or symbol:
                return {"token_name": name, "token_symbol": symbol}
    return {}


def format_trade_history(trades: List[TradeRecord]) -> str:
    if not trades:
        return "📜 <b>Trading History</b>\nNo paper trades yet."
    lines = ["📜 <b>Trading History</b>"]
    for trade in trades:
        pnl = _pnl_label(trade.pnl)
        pnl_pct = _pct_change(trade.buy_price, trade.sell_price)
        lines.append(
            "• #{0} {1} {2} entry ${3:.10g} size {4:.6f} SOL PnL {5} ({6})".format(
                trade.id,
                _html(trade.status),
                _format_trade_token_identity(trade),
                trade.buy_price,
                trade.entry_amount_sol,
                pnl,
                _metric(pnl_pct, "%"),
            )
        )
    return "\n".join(lines[:80])


def format_entry_analysis(snapshot: TokenSnapshot, evaluation: TokenEvaluation) -> str:
    """Format one candidate review before any paper buy."""

    lines = [
            "{0} <b>Paper Entry Analysis</b>".format(_entry_icon(evaluation)),
            "🎯 <b>Strategy:</b> {0}".format(_html(snapshot.strategy.upper())),
            "🪙 <b>Token:</b> {0}".format(_format_snapshot_token_identity(snapshot)),
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
    if evaluation.trade_plan is not None:
        lines.extend(
            [
                "🧠 <b>Setup mode:</b> dynamic AI",
                "📦 <b>AI size:</b> {0:.6f} SOL".format(
                    evaluation.trade_plan.entry_amount_sol
                ),
                "🛡 <b>AI setup:</b> SL {0:g}% | TP {1} | trail {2:g}% | max {3}".format(
                    evaluation.trade_plan.stop_loss_pct,
                    _format_targets(evaluation.trade_plan.take_profit_targets_pct),
                    evaluation.trade_plan.trailing_stop_pct,
                    _duration_label(evaluation.trade_plan.max_hold_seconds),
                ),
            ]
        )
    return "\n".join(lines)


def format_buy_result(
    snapshot: TokenSnapshot, evaluation: TokenEvaluation, result: TradeResult
) -> str:
    """Format a paper buy success or rejection."""

    planned_size = (
        evaluation.trade_plan.entry_amount_sol
        if evaluation.trade_plan is not None
        else None
    )
    actual_size = (
        result.entry_amount_sol
        if result.entry_amount_sol is not None
        else planned_size
    )
    return "\n".join(
        [
            "{0} <b>Paper Buy {1}</b>".format(
                "🟢" if result.success else "🟠",
                "Opened" if result.success else "Rejected",
            ),
            "🪙 <b>Token:</b> {0}".format(_format_snapshot_token_identity(snapshot)),
            "🧾 <b>Trade:</b> {0}".format(
                "#{0}".format(result.trade_id) if result.trade_id else "n/a"
            ),
            "🧠 <b>AI score:</b> {0}/100".format(evaluation.score),
            "🎯 <b>Strategy:</b> {0}".format(_html(snapshot.strategy.upper())),
            "📦 <b>Entry size:</b> {0}".format(
                "{0:.6f} SOL".format(actual_size)
                if actual_size is not None
                else "default"
            ),
            "🧮 <b>AI requested:</b> {0}".format(
                "{0:.6f} SOL".format(planned_size)
                if planned_size is not None and planned_size != actual_size
                else "same as entry"
                if planned_size is not None
                else "static settings"
            ),
            "🧠 <b>Setup mode:</b> {0}".format(
                "dynamic AI" if evaluation.trade_plan else "static settings"
            ),
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
            "🪙 <b>Token:</b> {0}".format(
                _format_trade_token_identity(trade, snapshot)
            ),
            "🧠 <b>Decision:</b> {0}".format(_decision_label(decision.decision)),
            "🏁 <b>Entry:</b> ${0:.10g}".format(trade.buy_price),
            "💵 <b>Current:</b> ${0:.10g}".format(snapshot.price_usd),
            "📦 <b>Entry size:</b> {0:.6f} SOL".format(trade.entry_amount_sol),
            "📊 <b>Unrealized PnL:</b> {0} ({1})".format(
                _pnl_label(_unrealized_pnl(trade, snapshot.price_usd)),
                _metric(_pct_change(trade.buy_price, snapshot.price_usd), "%"),
            ),
            "📝 <b>Reason:</b> {0}".format(
                _html(decision.rationale or "No rationale returned.")
            ),
        ]
    )


def format_buy_more_result(
    trade: TradeRecord,
    snapshot: TokenSnapshot,
    decision: ExitDecision,
    result: TradeResult,
) -> str:
    """Format a paper add-on buy success or rejection."""

    return "\n".join(
        [
            "{0} <b>Paper Buy More {1}</b>".format(
                "➕" if result.success else "🟠",
                "Added" if result.success else "Rejected",
            ),
            "🧾 <b>Trade:</b> #{0}".format(trade.id),
            "🪙 <b>Token:</b> {0}".format(_format_snapshot_token_identity(snapshot)),
            "📦 <b>Add size:</b> {0}".format(
                "{0:.6f} SOL".format(result.entry_amount_sol)
                if result.entry_amount_sol is not None
                else "n/a"
            ),
            "📦 <b>Previous size:</b> {0:.6f} SOL".format(trade.entry_amount_sol),
            "💵 <b>Add price:</b> ${0:.10g}".format(snapshot.price_usd),
            "📊 <b>Unrealized PnL before add:</b> {0} ({1})".format(
                _pnl_label(_unrealized_pnl(trade, snapshot.price_usd)),
                _metric(_pct_change(trade.buy_price, snapshot.price_usd), "%"),
            ),
            "📝 <b>AI reason:</b> {0}".format(
                _html(decision.rationale or "AI buy-more")
            ),
            "ℹ️ <b>Detail:</b> {0}".format(_html(result.message)),
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
            "🪙 <b>Token:</b> {0}".format(_format_snapshot_token_identity(snapshot)),
            "📦 <b>Entry size:</b> {0:.6f} SOL".format(trade.entry_amount_sol),
            "💵 <b>Sell price:</b> ${0:.10g}".format(snapshot.price_usd),
            "📝 <b>Exit reason:</b> {0}".format(
                _html(decision.rationale or "AI close")
            ),
            "📊 <b>PnL:</b> {0} ({1})".format(
                "{0:+.6f} SOL".format(result.pnl)
                if result.pnl is not None
                else "unavailable",
                _metric(_pct_change(trade.buy_price, snapshot.price_usd), "%"),
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
        "⚡ Use the buttons below to control entries, positions, settings, and reports.\n"
        "🛑 REAL trading remains disabled in v1."
    )


def menu_text() -> str:
    """Render a compact menu response."""

    return (
        "🎛 <b>Paper Bot Menu</b>\n"
        "Tap a button to manage status, positions, trading, settings, history, or reports."
    )


def menu_markup() -> Any:
    """Build the main inline Telegram menu."""

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Status", callback_data="bot:status"),
                InlineKeyboardButton("🧾 Positions", callback_data="bot:positions"),
            ],
            [
                InlineKeyboardButton("🤖 Trading", callback_data="bot:trading"),
                InlineKeyboardButton("⚙️ Settings", callback_data="bot:settings"),
            ],
            [
                InlineKeyboardButton("📜 History", callback_data="bot:history"),
                InlineKeyboardButton("🔔 Reports", callback_data="bot:reports"),
            ],
        ]
    )


def position_menu_markup(open_trades: List[TradeRecord]) -> Any:
    """Build an inline menu that exposes open position ids as direct buttons."""

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    position_rows = [
        [
            InlineKeyboardButton(
                "#{0}".format(trade.id),
                callback_data="bot:position:{0}".format(trade.id),
            )
            for trade in open_trades[index : index + 3]
        ]
        for index in range(0, len(open_trades), 3)
    ]
    rows = [
        [
            InlineKeyboardButton("📊 Status", callback_data="bot:status"),
            InlineKeyboardButton("📜 History", callback_data="bot:history"),
        ],
        *position_rows,
        [
            InlineKeyboardButton("🤖 Trading", callback_data="bot:trading"),
            InlineKeyboardButton("⚙️ Settings", callback_data="bot:settings"),
        ],
        [InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu")],
    ]
    return InlineKeyboardMarkup(rows)


def position_action_markup(trade: Optional[TradeRecord]) -> Any:
    """Build manual action buttons for one selected position."""

    if trade is None or trade.status != "OPEN":
        return menu_markup()
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ Take Profit", callback_data="bot:tp:{0}".format(trade.id)
                ),
                InlineKeyboardButton(
                    "🛑 Cut Loss", callback_data="bot:sl:{0}".format(trade.id)
                ),
            ],
            [
                InlineKeyboardButton(
                    "💸 Close", callback_data="bot:close:{0}".format(trade.id)
                ),
                InlineKeyboardButton("🧾 Positions", callback_data="bot:positions"),
            ],
            [InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu")],
        ]
    )


def history_menu_markup() -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Latest 50", callback_data="bot:history"),
                InlineKeyboardButton("All", callback_data="bot:history:all"),
            ],
            [InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu")],
        ]
    )


def trading_menu_markup(settings: Any) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🟢 Auto On", callback_data="bot:auto:on"),
                InlineKeyboardButton("🔴 Auto Off", callback_data="bot:auto:off"),
            ],
            [
                InlineKeyboardButton(
                    "Launch {0}".format("OFF" if settings.launch_enabled else "ON"),
                    callback_data=(
                        "bot:launch:off" if settings.launch_enabled else "bot:launch:on"
                    ),
                ),
                InlineKeyboardButton(
                    "Scout {0}".format("OFF" if settings.scout_enabled else "ON"),
                    callback_data=(
                        "bot:scout:off" if settings.scout_enabled else "bot:scout:on"
                    ),
                ),
            ],
            [
                InlineKeyboardButton(
                    "Dynamic {0}".format(
                        "OFF" if settings.dynamic_setup_enabled else "ON"
                    ),
                    callback_data=(
                        "bot:dynamic:off"
                        if settings.dynamic_setup_enabled
                        else "bot:dynamic:on"
                    ),
                ),
                InlineKeyboardButton("⚙️ Settings", callback_data="bot:settings"),
            ],
            [InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu")],
        ]
    )


def settings_menu_markup(settings: Any) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Threshold 25", callback_data="bot:threshold:25"),
                InlineKeyboardButton("Threshold 30", callback_data="bot:threshold:30"),
                InlineKeyboardButton("Threshold 35", callback_data="bot:threshold:35"),
            ],
            [
                InlineKeyboardButton("📦 Size", callback_data="bot:size"),
                InlineKeyboardButton("🛡 Exits", callback_data="bot:exits"),
                InlineKeyboardButton("🚧 Risk", callback_data="bot:risk"),
            ],
            [
                InlineKeyboardButton("🤖 Trading", callback_data="bot:trading"),
                InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu"),
            ],
        ]
    )


def size_menu_markup(settings: Any) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("0.05..0.15", callback_data="bot:size:0.05:0.15"),
                InlineKeyboardButton("0.1..0.3", callback_data="bot:size:0.1:0.3"),
            ],
            [
                InlineKeyboardButton("0.15..0.4", callback_data="bot:size:0.15:0.4"),
                InlineKeyboardButton("0.2..0.5", callback_data="bot:size:0.2:0.5"),
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="bot:settings"),
                InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu"),
            ],
        ]
    )


def exit_menu_markup(settings: Any) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("SL 6%", callback_data="bot:exit:sl:6"),
                InlineKeyboardButton("SL 8%", callback_data="bot:exit:sl:8"),
                InlineKeyboardButton("SL 10%", callback_data="bot:exit:sl:10"),
            ],
            [
                InlineKeyboardButton("TP 12%", callback_data="bot:exit:tp:12"),
                InlineKeyboardButton("TP 18%", callback_data="bot:exit:tp:18"),
                InlineKeyboardButton("TP 25%", callback_data="bot:exit:tp:25"),
            ],
            [
                InlineKeyboardButton("Trail 5%", callback_data="bot:exit:trail:5"),
                InlineKeyboardButton("Trail 7%", callback_data="bot:exit:trail:7"),
                InlineKeyboardButton("Max 10m", callback_data="bot:exit:max:600"),
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="bot:settings"),
                InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu"),
            ],
        ]
    )


def risk_menu_markup(settings: Any) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Block UTC 20", callback_data="bot:risk:block:20"),
                InlineKeyboardButton("No Hour Block", callback_data="bot:risk:block:none"),
            ],
            [
                InlineKeyboardButton("Buy/Sell 1.15", callback_data="bot:risk:ratio:1.15"),
                InlineKeyboardButton("Buy/Sell 1.5", callback_data="bot:risk:ratio:1.5"),
            ],
            [
                InlineKeyboardButton("Top Holders 35%", callback_data="bot:risk:holders:35"),
                InlineKeyboardButton("Top Holders 45%", callback_data="bot:risk:holders:45"),
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="bot:settings"),
                InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu"),
            ],
        ]
    )


def reports_menu_markup(enabled: bool) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🔔 Reports On", callback_data="bot:notify:on"),
                InlineKeyboardButton("🔕 Reports Off", callback_data="bot:notify:off"),
            ],
            [InlineKeyboardButton("⬅️ Main Menu", callback_data="bot:menu")],
        ]
    )


def _entry_icon(evaluation: TokenEvaluation) -> str:
    return "🚀" if evaluation.wants_buy else "🔎"


def _exit_icon(decision: ExitDecision) -> str:
    if decision.wants_buy_more:
        return "➕"
    return "💸" if decision.wants_close else "🛡️"


def _decision_label(decision: str) -> str:
    labels = {
        "buy": "🟢 BUY",
        "skip": "🟠 SKIP",
        "close": "💸 CLOSE",
        "sell_now": "💸 SELL NOW",
        "buy_more": "➕ BUY MORE",
        "hold": "🛡️ HOLD",
    }
    return labels.get(decision.lower(), _html(decision.upper()))


def _size_menu_text(settings: Any) -> str:
    return (
        "📦 <b>Dynamic Size</b>\n"
        "Current range: <b>{0:g}..{1:g} SOL</b>\n"
        "Tap a preset below.".format(
            settings.min_trade_amount_sol,
            settings.max_trade_amount_sol,
        )
    )


def _exit_menu_text(settings: Any) -> str:
    return (
        "🛡 <b>Hard Exit Settings</b>\n"
        "TP <b>{0:g}%</b> | SL <b>{1:g}%</b> | trail <b>{2:g}%</b> | max <b>{3}</b>\n"
        "Tap a preset below.".format(
            settings.take_profit_pct,
            settings.stop_loss_pct,
            settings.trailing_stop_pct,
            _duration_label(settings.max_hold_seconds),
        )
    )


def _risk_menu_text(settings: Any) -> str:
    return (
        "🚧 <b>Entry Risk Gates</b>\n"
        "Blocked UTC: <b>{0}</b>\n"
        "Buy/sell ≥ <b>{1:g}</b> | vol/liq ≥ <b>{2:g}</b> | top holders ≤ <b>{3:g}%</b>".format(
            _html(settings.blocked_entry_utc_hours or "none"),
            settings.min_buy_sell_ratio,
            settings.min_volume_liquidity_ratio_5m,
            settings.max_top_holder_share_pct,
        )
    )


def _parse_callback_int(data: str, prefix: str) -> Optional[int]:
    if not data.startswith(prefix):
        return None
    try:
        return int(data[len(prefix) :])
    except ValueError:
        return None


def _updated_exit_setting(settings: Any, data: str) -> tuple[Any, str, str]:
    parts = data.split(":")
    if len(parts) != 4:
        return None, "", ""
    kind = parts[2]
    try:
        value = float(parts[3])
    except ValueError:
        return None, "", ""
    if kind == "sl":
        return replace(settings, stop_loss_pct=value), "Stop loss", "{0:g}%".format(value)
    if kind == "tp":
        return replace(settings, take_profit_pct=value), "Take profit", "{0:g}%".format(value)
    if kind == "trail":
        return (
            replace(settings, trailing_stop_pct=value),
            "Trailing stop",
            "{0:g}%".format(value),
        )
    if kind == "max":
        return (
            replace(settings, max_hold_seconds=value),
            "Max hold",
            _duration_label(value),
        )
    return None, "", ""


def _updated_risk_setting(settings: Any, data: str) -> tuple[Any, str, str]:
    parts = data.split(":")
    if len(parts) != 4:
        return None, "", ""
    kind = parts[2]
    raw_value = parts[3]
    if kind == "block":
        value = "" if raw_value == "none" else raw_value
        return (
            replace(settings, blocked_entry_utc_hours=value),
            "Blocked UTC hours",
            value or "none",
        )
    try:
        value = float(raw_value)
    except ValueError:
        return None, "", ""
    if kind == "ratio":
        return (
            replace(settings, min_buy_sell_ratio=value),
            "Minimum buy/sell ratio",
            "{0:g}".format(value),
        )
    if kind == "holders":
        return (
            replace(settings, max_top_holder_share_pct=value),
            "Max top-holder share",
            "{0:g}%".format(value),
        )
    return None, "", ""


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


def _parse_history_limit(raw_value: str) -> Optional[int]:
    value = raw_value.strip().lower()
    if not value:
        return 50
    try:
        limit = int(value)
    except ValueError:
        return None
    return max(1, min(500, limit))


def _trade_id_from_update(update: Any, context: Any) -> Optional[int]:
    raw_value = _context_text(context)
    if not raw_value:
        message = getattr(update, "effective_message", None) or getattr(update, "message", None)
        raw_value = str(getattr(message, "text", "") or "")
    match = re.search(r"#?(\d+)", raw_value)
    return int(match.group(1)) if match else None


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


def _trade_plan_from_json(trade: TradeRecord) -> TradePlan:
    fallback = TradePlan(
        entry_amount_sol=trade.entry_amount_sol,
        stop_loss_pct=8.0,
        take_profit_targets_pct=[],
        trailing_stop_pct=0.0,
        max_hold_seconds=3600.0,
    )
    try:
        payload = json.loads(trade.trade_plan_json or "{}")
    except (TypeError, ValueError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    try:
        targets = payload.get("take_profit_targets_pct") or []
        if not isinstance(targets, list):
            targets = []
        return TradePlan(
            entry_amount_sol=float(
                payload.get("entry_amount_sol", trade.entry_amount_sol)
            ),
            stop_loss_pct=float(payload.get("stop_loss_pct", fallback.stop_loss_pct)),
            take_profit_targets_pct=[float(target) for target in targets],
            trailing_stop_pct=float(
                payload.get("trailing_stop_pct", fallback.trailing_stop_pct)
            ),
            max_hold_seconds=float(
                payload.get("max_hold_seconds", fallback.max_hold_seconds)
            ),
            rationale=str(payload.get("rationale", "")),
        )
    except (TypeError, ValueError):
        return fallback


def _trade_setup_label(trade: TradeRecord) -> str:
    try:
        payload = json.loads(trade.trade_plan_json or "{}")
    except (TypeError, ValueError):
        payload = {}
    if isinstance(payload, dict) and bool(payload):
        return "dynamic AI"
    return "static settings"


def _format_targets(targets: List[float]) -> str:
    if not targets:
        return "not set"
    return " | ".join(
        "TP{0} {1:g}%".format(index + 1, target)
        for index, target in enumerate(targets)
    )


def _unrealized_pnl(trade: TradeRecord, current_price: Optional[float]) -> Optional[float]:
    if current_price is None:
        return None
    return trade.token_quantity * current_price - trade.entry_amount_sol


def _pct_change(start: float, end: Optional[float]) -> Optional[float]:
    if end is None or start <= 0:
        return None
    return ((end - start) / start) * 100


def _pnl_label(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    return "{0:+.6f} SOL".format(value)


def _html(value: Any) -> str:
    return escape(str(value), quote=False)


def _metric(value: Optional[float], suffix: str) -> str:
    if value is None:
        return "unknown"
    return "{0:.4g}{1}".format(value, suffix)


def _count(value: Optional[int]) -> str:
    return "unknown" if value is None else str(value)
