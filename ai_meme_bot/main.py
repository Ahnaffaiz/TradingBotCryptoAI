"""Async application orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, time as clock_time, timedelta
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
from ai_meme_bot.core.tracker import TokenTracker


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
            evaluation = await ai_service.evaluate_entry(snapshot, rules)
            analysis_id = await database.record_analysis(snapshot, evaluation)
            await database.add_activity(
                "analysis",
                "{0} score {1}".format(evaluation.decision, evaluation.score),
                snapshot.token_address,
                {
                    "analysis_id": analysis_id,
                    "pair_address": snapshot.pair_address,
                    "rationale": evaluation.rationale,
                },
            )
            await notifier.entry_analysis(snapshot, evaluation)
            settings = await database.get_strategy_settings(config.strategy_defaults)
            if (
                evaluation.score > 0
                and evaluation.score >= settings.entry_score_threshold
            ):
                result = await tools.trigger_buy(snapshot.token_address, snapshot)
                if result.success:
                    await database.mark_analysis_bought(analysis_id, result.trade_id)
                await database.add_activity(
                    "paper_buy" if result.success else "paper_buy_rejected",
                    result.message,
                    snapshot.token_address,
                    {"analysis_id": analysis_id, "trade_id": result.trade_id},
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
                decision = await ai_service.evaluate_exit(trade, snapshot, rules)
                await database.add_activity(
                    "exit_analysis",
                    decision.decision,
                    trade.token_address,
                    {
                        "trade_id": trade.id,
                        "rationale": decision.rationale,
                        "price_usd": snapshot.price_usd,
                    },
                )
                await notifier.exit_analysis(trade, snapshot, decision)
                if decision.wants_close:
                    result = await executor.close_trade(trade, snapshot, decision.rationale)
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

    async with TokenTracker(
        config,
        filtered_callback=record_filtered_token,
        poll_seconds_callback=poll_seconds,
    ) as tracker:
        tools = TradingTools(config, database, tracker, executor)
        telegram = TelegramTradingBot(config, database)
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
