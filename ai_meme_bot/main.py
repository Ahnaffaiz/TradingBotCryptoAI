"""Async application orchestrator."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from datetime import datetime, time as clock_time, timedelta, timezone
from typing import Any, List
from zoneinfo import ZoneInfo

from ai_meme_bot.agent.ai_service import HermesChatBackend, TradingAIService
from ai_meme_bot.agent.hermes_bot import (
    NullPaperNotifier,
    PaperNotifier,
    TelegramPaperNotifier,
    TelegramTradingBot,
)
from ai_meme_bot.agent.reflection import generate_daily_rules
from ai_meme_bot.agent.tools import TradingTools
from ai_meme_bot.config import AppConfig
from ai_meme_bot.core.database import Database
from ai_meme_bot.core.execution import TradeExecutor
from ai_meme_bot.core.price_feed import BirdeyePriceFeed, PriceTick
from ai_meme_bot.core.tracker import TokenTracker
from ai_meme_bot.models import ExitDecision, TokenSnapshot, TradePlan, TradeResult


LOGGER = logging.getLogger(__name__)
OUTCOME_HORIZONS_SECONDS = (300, 900, 3600)


async def run_discovery_loop(
    database: Database,
    tracker: TokenTracker,
    ai_service: TradingAIService,
    tools: TradingTools,
    config: AppConfig,
    notifier: PaperNotifier,
) -> None:
    """Evaluate discovered snapshots and open approved paper positions."""

    async for snapshot in tracker.discover():
        if not await database.get_auto_trading():
            continue
        try:
            rules = await database.get_latest_rules()
            settings = await database.get_strategy_settings(config.strategy_defaults)
            evaluation = await _evaluate_entry(ai_service, snapshot, rules, settings)
            analysis_id = await database.record_analysis(snapshot, evaluation)
            await database.add_activity(
                "analysis",
                "{0} {1} score {2}".format(
                    snapshot.strategy, evaluation.decision, evaluation.score
                ),
                snapshot.token_address,
                {
                    "analysis_id": analysis_id,
                    "pair_address": snapshot.pair_address,
                    "rationale": evaluation.rationale,
                    "strategy": snapshot.strategy,
                },
            )
            await notifier.entry_analysis(snapshot, evaluation)
            threshold = _entry_threshold(settings, snapshot.strategy)
            threshold_approved = evaluation.score > 0 and evaluation.score >= threshold
            dynamic_approved = (
                not settings.dynamic_setup_enabled
                or (evaluation.wants_buy and evaluation.trade_plan is not None)
            )
            risk_rejection = (
                _entry_risk_rejection(snapshot, settings)
                if threshold_approved and dynamic_approved
                else None
            )
            if risk_rejection is not None:
                await database.add_activity(
                    "entry_risk_reject",
                    risk_rejection,
                    snapshot.token_address,
                    {
                        "analysis_id": analysis_id,
                        "pair_address": snapshot.pair_address,
                        "strategy": snapshot.strategy,
                    },
                )
                LOGGER.info(
                    "Entry risk gate rejected token=%s reason=%s",
                    snapshot.token_address,
                    risk_rejection,
                )
                continue
            if threshold_approved and dynamic_approved:
                result = await _trigger_buy(tools, snapshot, evaluation)
                if result.success:
                    await database.mark_analysis_bought(analysis_id, result.trade_id)
                await database.add_activity(
                    "paper_buy" if result.success else "paper_buy_rejected",
                    result.message,
                    snapshot.token_address,
                    {
                        "analysis_id": analysis_id,
                        "trade_id": result.trade_id,
                        "entry_amount_sol": result.entry_amount_sol,
                        "strategy": snapshot.strategy,
                        "threshold": threshold,
                    },
                )
                await notifier.buy_result(snapshot, evaluation, result)
                LOGGER.info(
                    "Entry decision token=%s score=%s success=%s detail=%s",
                    snapshot.token_address,
                    evaluation.score,
                    result.success,
                    result.message,
                )
        except Exception as exc:
            LOGGER.exception("Entry pipeline failed for token=%s", snapshot.token_address)
            await database.add_activity("error", str(exc), snapshot.token_address, {"stage": "entry analysis"})
            await notifier.error(
                "entry analysis",
                "{0}: {1}".format(snapshot.token_address, exc),
            )


async def run_position_review_loop(
    database: Database,
    tracker: TokenTracker,
    ai_service: TradingAIService,
    executor: TradeExecutor,
    config: AppConfig,
    notifier: PaperNotifier,
) -> None:
    """Refresh open trades and close only on validated AI exit decisions."""

    high_watermarks: dict[int, float] = {}
    while True:
        try:
            rules = await database.get_latest_rules()
            trades = await database.get_open_trades()
        except Exception as exc:
            LOGGER.exception("Open position lookup failed")
            await database.add_activity("error", str(exc), payload={"stage": "position lookup"})
            await notifier.error("position lookup", str(exc))
            settings = await database.get_strategy_settings(config.strategy_defaults)
            await asyncio.sleep(settings.position_review_seconds)
            continue

        for trade in trades:
            try:
                snapshot = await tracker.snapshot_for_token(
                    trade.token_address, apply_filters=False
                )
                if snapshot is None:
                    continue
                settings = await database.get_strategy_settings(config.strategy_defaults)
                high_watermarks[trade.id] = max(
                    high_watermarks.get(trade.id, trade.buy_price),
                    snapshot.price_usd,
                )
                hard_exit_reason = _hard_exit_reason(
                    trade, snapshot, settings, high_watermarks[trade.id]
                )
                if hard_exit_reason is not None:
                    decision = ExitDecision("sell_now", hard_exit_reason)
                    result = await executor.close_trade(trade, snapshot, hard_exit_reason)
                    high_watermarks.pop(trade.id, None)
                    await database.add_activity(
                        "paper_sell" if result.success else "paper_sell_rejected",
                        result.message,
                        trade.token_address,
                        {
                            "trade_id": trade.id,
                            "pnl": result.pnl,
                            "exit_type": "hard_rule",
                            "reason": hard_exit_reason,
                        },
                    )
                    await notifier.sell_result(trade, snapshot, decision, result)
                    continue
                decision = await _evaluate_position(
                    ai_service, trade, snapshot, rules, settings
                )
                await database.add_activity(
                    "position_analysis",
                    decision.decision,
                    trade.token_address,
                    {
                        "trade_id": trade.id,
                        "rationale": decision.rationale,
                        "price_usd": snapshot.price_usd,
                        "add_amount_sol": decision.add_amount_sol,
                    },
                )
                await notifier.exit_analysis(trade, snapshot, decision)
                if decision.wants_close:
                    result = await executor.close_trade(trade, snapshot, decision.rationale)
                    high_watermarks.pop(trade.id, None)
                    await database.add_activity(
                        "paper_sell" if result.success else "paper_sell_rejected",
                        result.message,
                        trade.token_address,
                        {"trade_id": trade.id, "pnl": result.pnl},
                    )
                    await notifier.sell_result(trade, snapshot, decision, result)
                    LOGGER.info(
                        "Exit decision trade=%s success=%s detail=%s",
                        trade.id,
                        result.success,
                        result.message,
                    )
                elif decision.wants_buy_more:
                    result = await _trigger_buy_more(
                        database, executor, trade, snapshot, decision, settings
                    )
                    await database.add_activity(
                        "paper_buy_more" if result.success else "paper_buy_more_rejected",
                        result.message,
                        trade.token_address,
                        {
                            "trade_id": trade.id,
                            "entry_amount_sol": result.entry_amount_sol,
                            "reason": decision.rationale,
                        },
                    )
                    await notifier.buy_more_result(trade, snapshot, decision, result)
            except Exception as exc:
                LOGGER.exception("Exit pipeline failed for trade=%s", trade.id)
                await database.add_activity(
                    "error",
                    str(exc),
                    trade.token_address,
                    {"stage": "exit analysis", "trade_id": trade.id},
                )
                await notifier.error(
                    "exit analysis",
                    "trade #{0} {1}: {2}".format(trade.id, trade.token_address, exc),
                )
        settings = await database.get_strategy_settings(config.strategy_defaults)
        await asyncio.sleep(settings.position_review_seconds)


async def run_realtime_exit_loop(
    database: Database,
    executor: TradeExecutor,
    config: AppConfig,
    notifier: PaperNotifier,
    price_feed: Any | None = None,
) -> None:
    """Close open trades quickly from Birdeye WebSocket price ticks."""

    if price_feed is None:
        if not config.realtime_price_feed_enabled or not config.birdeye_api_key:
            return
        price_feed = BirdeyePriceFeed(config.birdeye_api_key, config.birdeye_ws_url)
    high_watermarks: dict[int, float] = {}
    while True:
        settings = await database.get_strategy_settings(config.strategy_defaults)
        trades = await database.get_open_trades()
        address_to_trade = {
            pair_address: trade
            for trade in trades
            if (pair_address := _pair_address_for_trade(trade))
        }
        if not address_to_trade:
            await asyncio.sleep(max(5.0, min(30.0, settings.position_review_seconds)))
            continue
        try:
            async with asyncio.timeout(60):
                async for tick in price_feed.stream_prices(address_to_trade.keys()):
                    trade = address_to_trade.get(tick.address)
                    if trade is None:
                        continue
                    current_trade = await database.get_trade(trade.id)
                    if current_trade is None or current_trade.status != "OPEN":
                        continue
                    snapshot = _snapshot_from_tick(current_trade, tick)
                    high_watermarks[current_trade.id] = max(
                        high_watermarks.get(current_trade.id, current_trade.buy_price),
                        snapshot.price_usd,
                    )
                    settings = await database.get_strategy_settings(
                        config.strategy_defaults
                    )
                    hard_exit_reason = _hard_exit_reason(
                        current_trade,
                        snapshot,
                        settings,
                        high_watermarks[current_trade.id],
                    )
                    if hard_exit_reason is None:
                        continue
                    decision = ExitDecision("sell_now", hard_exit_reason)
                    result = await executor.close_trade(
                        current_trade, snapshot, hard_exit_reason
                    )
                    high_watermarks.pop(current_trade.id, None)
                    await database.add_activity(
                        "paper_sell" if result.success else "paper_sell_rejected",
                        result.message,
                        current_trade.token_address,
                        {
                            "trade_id": current_trade.id,
                            "pnl": result.pnl,
                            "exit_type": "realtime_hard_rule",
                            "reason": hard_exit_reason,
                            "price_feed": "birdeye",
                        },
                    )
                    await notifier.sell_result(current_trade, snapshot, decision, result)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Realtime exit monitor failed")
            await database.add_activity(
                "error", str(exc), payload={"stage": "realtime exit"}
            )
            await notifier.error("realtime exit", str(exc))
            await asyncio.sleep(max(5.0, settings.position_review_seconds))


def _entry_threshold(settings: Any, strategy: str) -> int:
    if strategy == "scout":
        return settings.scout_score_threshold
    if strategy == "launch":
        return settings.launch_score_threshold
    return settings.entry_score_threshold


async def _evaluate_entry(
    ai_service: Any, snapshot: Any, rules: str, settings: Any
) -> Any:
    signature = inspect.signature(ai_service.evaluate_entry)
    if len(signature.parameters) >= 3:
        return await ai_service.evaluate_entry(snapshot, rules, settings)
    return await ai_service.evaluate_entry(snapshot, rules)


async def _trigger_buy(tools: Any, snapshot: Any, evaluation: Any) -> Any:
    signature = inspect.signature(tools.trigger_buy)
    if len(signature.parameters) >= 3:
        return await tools.trigger_buy(snapshot.token_address, snapshot, evaluation)
    return await tools.trigger_buy(snapshot.token_address, snapshot)


async def _evaluate_position(
    ai_service: Any, trade: Any, snapshot: Any, rules: str, settings: Any
) -> Any:
    signature = inspect.signature(ai_service.evaluate_exit)
    if len(signature.parameters) >= 4:
        return await ai_service.evaluate_exit(trade, snapshot, rules, settings)
    return await ai_service.evaluate_exit(trade, snapshot, rules)


async def _trigger_buy_more(
    database: Database,
    executor: TradeExecutor,
    trade: Any,
    snapshot: TokenSnapshot,
    decision: ExitDecision,
    settings: Any,
) -> TradeResult:
    reason = await _buy_more_rejection(database, trade, settings)
    requested_amount = decision.add_amount_sol
    min_size, max_size = settings.trade_size_bounds()
    amount = max(min_size, min(float(requested_amount or min_size), max_size))
    remaining = max(0.0, float(settings.max_trade_amount_sol) - trade.entry_amount_sol)
    balance = await database.get_balance()
    amount = min(amount, remaining, balance)
    if reason is None and amount < min_size:
        reason = (
            "buy-more amount below minimum after balance and max-position guardrails"
        )
    if reason is not None:
        return TradeResult(
            False,
            "Buy-more rejected: {0}.".format(reason),
            trade_id=trade.id,
            entry_amount_sol=amount,
        )
    return await executor.add_to_trade(
        trade,
        round(amount, 6),
        snapshot,
        decision.rationale or "AI buy-more",
    )


async def _buy_more_rejection(
    database: Database, trade: Any, settings: Any
) -> str | None:
    additions = await database.get_trade_additions(trade.id)
    if len(additions) >= int(settings.max_buy_more_count):
        return "max buy-more count reached"
    min_size, _max_size = settings.trade_size_bounds()
    remaining = float(settings.max_trade_amount_sol) - float(trade.entry_amount_sol)
    if remaining < min_size:
        return "position is already at the dynamic max size"
    balance = await database.get_balance()
    if balance < min_size:
        return "paper balance is below the minimum dynamic trade size"
    if additions and float(settings.buy_more_cooldown_seconds) > 0:
        created_at = _parse_iso_datetime(str(additions[0].get("created_at", "")))
        if created_at is not None:
            age = (datetime.now(timezone.utc) - created_at).total_seconds()
            if age < float(settings.buy_more_cooldown_seconds):
                return "buy-more cooldown is still active"
    return None


def _entry_risk_rejection(
    snapshot: TokenSnapshot, settings: Any, now: datetime | None = None
) -> str | None:
    current = now or datetime.now(timezone.utc)
    if current.astimezone(timezone.utc).hour in settings.blocked_hours():
        return "UTC hour {0} is blocked for new entries".format(current.hour)
    ratio = _buy_sell_ratio(snapshot)
    if ratio is not None and ratio < float(settings.min_buy_sell_ratio):
        return "5m buy/sell ratio {0:.2f} below {1:g}".format(
            ratio, settings.min_buy_sell_ratio
        )
    if snapshot.liquidity_usd > 0:
        volume_liquidity = snapshot.volume_5m_usd / snapshot.liquidity_usd
        if volume_liquidity < float(settings.min_volume_liquidity_ratio_5m):
            return "5m volume/liquidity {0:.4f} below {1:g}".format(
                volume_liquidity, settings.min_volume_liquidity_ratio_5m
            )
    if (
        snapshot.top_holder_share_pct is not None
        and snapshot.top_holder_share_pct > float(settings.max_top_holder_share_pct)
    ):
        return "top holders {0:g}% above {1:g}%".format(
            snapshot.top_holder_share_pct, settings.max_top_holder_share_pct
        )
    if (
        snapshot.price_change_5m_pct is not None
        and snapshot.price_change_5m_pct > float(settings.max_momentum_5m_pct)
        and (
            ratio is None
            or ratio < float(settings.momentum_exhaustion_min_buy_sell_ratio)
        )
    ):
        return "5m momentum {0:g}% is exhausted without dominant buy pressure".format(
            snapshot.price_change_5m_pct
        )
    return None


def _buy_sell_ratio(snapshot: TokenSnapshot) -> float | None:
    if snapshot.buys_5m is None or snapshot.sells_5m is None:
        return None
    if snapshot.sells_5m <= 0:
        return float(snapshot.buys_5m) if snapshot.buys_5m > 0 else None
    return float(snapshot.buys_5m) / float(snapshot.sells_5m)


def _pair_address_for_trade(trade: Any) -> str | None:
    try:
        payload = json.loads(trade.entry_snapshot_json or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        return None
    pair_address = payload.get("pair_address")
    return str(pair_address).strip() if pair_address else None


def _snapshot_from_tick(trade: Any, tick: PriceTick) -> TokenSnapshot:
    try:
        payload = json.loads(trade.entry_snapshot_json or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload = dict(payload)
    payload["price_usd"] = tick.price_usd
    payload["pair_address"] = payload.get("pair_address") or tick.address
    payload["token_address"] = payload.get("token_address") or trade.token_address
    payload["liquidity_usd"] = float(payload.get("liquidity_usd") or 0.0)
    payload["volume_5m_usd"] = float(payload.get("volume_5m_usd") or 0.0)
    payload["pair_age_seconds"] = float(payload.get("pair_age_seconds") or 0.0)
    payload["raw_context"] = dict(payload.get("raw_context") or {})
    payload["raw_context"]["birdeye_tick"] = tick.raw_payload
    valid_keys = set(TokenSnapshot.__dataclass_fields__)
    return TokenSnapshot(
        **{key: value for key, value in payload.items() if key in valid_keys}
    )


def _hard_exit_reason(
    trade: Any, snapshot: Any, settings: Any, high_price: float
) -> str | None:
    plan = _trade_plan_for_hard_exit(trade, settings)
    change_pct = _pct_change(trade.buy_price, snapshot.price_usd)
    if change_pct is None:
        return None
    take_profit_pct = (
        plan.take_profit_targets_pct[0]
        if plan.take_profit_targets_pct
        else settings.take_profit_pct
    )
    if change_pct >= take_profit_pct:
        return "take profit {0:.2f}% >= TP1 {1:g}%".format(
            change_pct, take_profit_pct
        )
    if change_pct <= -plan.stop_loss_pct:
        return "stop loss {0:.2f}% <= -{1:g}%".format(
            change_pct, plan.stop_loss_pct
        )
    if plan.trailing_stop_pct > 0 and high_price > trade.buy_price:
        trail_pct = _pct_change(high_price, snapshot.price_usd)
        if trail_pct is not None and trail_pct <= -plan.trailing_stop_pct:
            return "trailing stop {0:.2f}% from high <= -{1:g}%".format(
                trail_pct, plan.trailing_stop_pct
            )
    opened_at = _parse_iso_datetime(trade.opened_at)
    if opened_at is not None:
        age_seconds = (datetime.now(timezone.utc) - opened_at).total_seconds()
        if age_seconds >= plan.max_hold_seconds:
            return "max hold {0:g}s reached".format(plan.max_hold_seconds)
    return None


def _trade_plan_for_hard_exit(trade: Any, settings: Any) -> TradePlan:
    fallback = TradePlan(
        entry_amount_sol=float(
            getattr(trade, "entry_amount_sol", settings.base_trade_amount)
        ),
        stop_loss_pct=float(settings.stop_loss_pct),
        take_profit_targets_pct=[float(settings.take_profit_pct)],
        trailing_stop_pct=float(settings.trailing_stop_pct),
        max_hold_seconds=float(settings.max_hold_seconds),
    )
    raw_json = getattr(trade, "trade_plan_json", "") or ""
    try:
        payload = json.loads(raw_json)
    except (TypeError, ValueError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    try:
        targets = (
            payload.get("take_profit_targets_pct")
            or fallback.take_profit_targets_pct
        )
        if not isinstance(targets, list):
            targets = fallback.take_profit_targets_pct
        normalized_targets = []
        for target in targets:
            parsed = float(target)
            if parsed > 0:
                normalized_targets.append(parsed)
        return TradePlan(
            entry_amount_sol=float(
                payload.get("entry_amount_sol", fallback.entry_amount_sol)
            ),
            stop_loss_pct=float(payload.get("stop_loss_pct", fallback.stop_loss_pct)),
            take_profit_targets_pct=(
                normalized_targets or fallback.take_profit_targets_pct
            ),
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


def _pct_change(start: float, end: float) -> float | None:
    if start <= 0:
        return None
    return ((end - start) / start) * 100


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def run_reflection_loop(
    database: Database,
    ai_service: TradingAIService,
    config: AppConfig,
    notifier: PaperNotifier,
) -> None:
    """Run daily rule generation at the configured wall-clock time."""

    while True:
        settings = await database.get_strategy_settings(config.strategy_defaults)
        delay = _seconds_until_reflection(settings.reflection_time, config.reflection_timezone)
        await asyncio.sleep(delay)
        try:
            rules = await generate_daily_rules(
                database, ai_service, config.strategy_defaults
            )
            await database.add_activity(
                "reflection",
                "stored {0} rules".format(len(rules.rules)),
                payload={"rules": rules.rules},
            )
            await notifier.reflection(rules)
            LOGGER.info("Daily reflection stored %s rules.", len(rules.rules))
        except Exception as exc:
            LOGGER.exception("Daily reflection failed")
            await database.add_activity("error", str(exc), payload={"stage": "daily reflection"})
            await notifier.error("daily reflection", str(exc))
        await asyncio.sleep(1)


async def run_outcome_loop(
    database: Database,
    tracker: TokenTracker,
    config: AppConfig,
    notifier: PaperNotifier,
) -> None:
    """Capture follow-up snapshots for analyses at 5m, 15m, and 1h."""

    while True:
        for analysis in await database.get_due_analysis_outcomes(OUTCOME_HORIZONS_SECONDS):
            try:
                snapshot = await tracker.snapshot_for_token(
                    analysis["token_address"], apply_filters=False
                )
                if snapshot is None:
                    continue
                initial_snapshot = json.loads(analysis["snapshot_json"])
                await database.add_outcome_snapshot(
                    analysis["id"],
                    int(analysis["horizon_seconds"]),
                    initial_snapshot,
                    snapshot,
                    str(analysis["ai_decision"]),
                )
                await database.add_activity(
                    "outcome_snapshot",
                    "{0}s follow-up captured".format(analysis["horizon_seconds"]),
                    snapshot.token_address,
                    {
                        "analysis_id": analysis["id"],
                        "horizon_seconds": analysis["horizon_seconds"],
                    },
                )
            except Exception as exc:
                LOGGER.exception("Outcome snapshot failed for analysis=%s", analysis["id"])
                await database.add_activity(
                    "error",
                    str(exc),
                    analysis["token_address"],
                    {"stage": "outcome snapshot", "analysis_id": analysis["id"]},
                )
                await notifier.error(
                    "outcome snapshot",
                    "analysis #{0} {1}: {2}".format(
                        analysis["id"], analysis["token_address"], exc
                    ),
                )
        settings = await database.get_strategy_settings(config.strategy_defaults)
        await asyncio.sleep(max(30.0, settings.position_review_seconds))


def _seconds_until_reflection(reflection_time: str, timezone_name: str) -> float:
    """Return seconds until the next configured HH:MM schedule."""

    hour_text, minute_text = reflection_time.split(":", 1)
    target_time = clock_time(hour=int(hour_text), minute=int(minute_text))
    zone = ZoneInfo(timezone_name)
    now = datetime.now(zone)
    next_run = datetime.combine(now.date(), target_time, tzinfo=zone)
    if next_run <= now:
        next_run += timedelta(days=1)
    return max(1.0, (next_run - now).total_seconds())


async def _start_telegram(application: Any) -> None:
    await application.initialize()
    if application.post_init is not None:
        await application.post_init(application)
    await application.start()
    if application.updater is None:
        raise RuntimeError("Telegram updater is unavailable.")
    await application.updater.start_polling()


async def _stop_telegram(application: Any) -> None:
    if application.updater is not None:
        await application.updater.stop()
    await application.stop()
    await application.shutdown()


async def run(config: AppConfig) -> None:
    """Create services, launch background tasks, and keep the app alive."""

    database = Database(config.db_path)
    await database.init_db()
    ai_service = TradingAIService(HermesChatBackend(config))
    executor = TradeExecutor(config, database)

    async def record_filtered_token(filtered: Any) -> None:
        await database.add_activity(
            "filter_reject",
            filtered.reason,
            filtered.token_address,
            {
                "pair_address": filtered.pair_address,
                "metrics": filtered.payload,
            },
        )

    async def poll_seconds() -> float:
        return (
            await database.get_strategy_settings(config.strategy_defaults)
        ).tracker_poll_seconds

    async def strategy_settings() -> Any:
        return await database.get_strategy_settings(config.strategy_defaults)

    async with TokenTracker(
        config,
        filtered_callback=record_filtered_token,
        poll_seconds_callback=poll_seconds,
        strategy_settings_callback=strategy_settings,
    ) as tracker:
        tools = TradingTools(config, database, tracker, executor)
        telegram = TelegramTradingBot(
            config, database, tracker=tracker, executor=executor
        )
        application = (
            telegram.build_application() if config.telegram_bot_token else None
        )
        if application is not None:
            await _start_telegram(application)
            notifier: PaperNotifier = TelegramPaperNotifier(database, application.bot)
        else:
            LOGGER.warning("TELEGRAM_BOT_TOKEN is unset; background loops only.")
            notifier = NullPaperNotifier()

        tasks: List[asyncio.Task[Any]] = [
            asyncio.create_task(
                run_discovery_loop(
                    database, tracker, ai_service, tools, config, notifier
                ),
                name="discovery",
            ),
            asyncio.create_task(
                run_position_review_loop(
                    database, tracker, ai_service, executor, config, notifier
                ),
                name="position-review",
            ),
            asyncio.create_task(
                run_reflection_loop(database, ai_service, config, notifier),
                name="reflection",
            ),
            asyncio.create_task(
                run_outcome_loop(database, tracker, config, notifier),
                name="outcomes",
            ),
        ]
        if config.realtime_price_feed_enabled and config.birdeye_api_key:
            tasks.append(
                asyncio.create_task(
                    run_realtime_exit_loop(database, executor, config, notifier),
                    name="realtime-exits",
                )
            )
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if application is not None:
                await _stop_telegram(application)


def main() -> None:
    """CLI entry point for `python -m ai_meme_bot.main`."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run(AppConfig.from_env()))


if __name__ == "__main__":
    main()
