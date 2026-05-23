"""Daily closed-trade reflection workflow."""

from __future__ import annotations

from ai_meme_bot.agent.ai_service import TradingAIService
from ai_meme_bot.core.database import Database
from ai_meme_bot.models import ReflectionRules, StrategySettings


async def generate_daily_rules(
    database: Database, ai_service: TradingAIService, config_settings: StrategySettings
) -> ReflectionRules:
    """Generate prompt rules and persist bounded adaptive strategy settings."""

    evidence = await database.get_reflection_evidence()
    current_settings = await database.get_strategy_settings(config_settings)
    reflection = await ai_service.generate_reflection(evidence, current_settings)
    rules = reflection.rules
    if rules.rules:
        await database.add_rules(rules.text)
    if reflection.settings is not None:
        await database.set_strategy_settings(reflection.settings)
        await database.add_activity(
            "strategy_tuning",
            reflection.settings_rationale or "AI reflection tuned paper settings.",
            payload=reflection.settings.prompt_payload(),
        )
    return rules
