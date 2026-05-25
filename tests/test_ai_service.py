import pytest

from ai_meme_bot.agent.ai_service import TradingAIService
from ai_meme_bot.models import ReflectionEvidence, StrategySettings, TradeRecord
from tests.helpers import make_snapshot


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, _system_prompt, _user_prompt):
        return self.responses.pop(0)


def make_trade(status="OPEN"):
    return TradeRecord(
        id=7,
        token_address="mint-1",
        buy_price=2.0,
        sell_price=None,
        pnl=None,
        status=status,
        timestamp="2026-05-22T00:00:00+00:00",
        entry_amount_sol=1.0,
        token_quantity=0.5,
        entry_snapshot_json="{}",
        trade_plan_json="{}",
        exit_snapshot_json=None,
        exit_reason=None,
        opened_at="2026-05-22T00:00:00+00:00",
        closed_at=None,
    )


@pytest.mark.asyncio
async def test_entry_accepts_valid_json_and_rejects_bad_json():
    service = TradingAIService(
        FakeBackend(
            [
                '{"score": 91, "decision": "buy", "rationale": "liquid"}',
                "this is not a decision",
            ]
        )
    )

    approved = await service.evaluate_entry(make_snapshot(), "")
    rejected = await service.evaluate_entry(make_snapshot(), "")

    assert approved.score == 91
    assert approved.wants_buy is True
    assert rejected.wants_buy is False
    assert rejected.score == 0


@pytest.mark.asyncio
async def test_entry_accepts_bounded_ai_trade_plan():
    service = TradingAIService(
        FakeBackend(
            [
                (
                    '{"score": 88, "decision": "buy", "rationale": "strong setup", '
                    '"trade_plan": {"position_size_sol": 5, "stop_loss_pct": 0.5, '
                    '"take_profit_targets_pct": [12, 24, 600], '
                    '"trailing_stop_pct": 7, "max_hold_seconds": 120, '
                    '"rationale": "high liquidity"}}'
                )
            ]
        )
    )

    evaluation = await service.evaluate_entry(
        make_snapshot(),
        "",
        StrategySettings(
            25,
            30.0,
            0.5,
            45.0,
            "00:00",
            max_trade_amount_sol=1.5,
        ),
    )

    assert evaluation.trade_plan.entry_amount_sol == 1.5
    assert evaluation.trade_plan.stop_loss_pct == 1.0
    assert evaluation.trade_plan.take_profit_targets_pct == [12.0, 24.0]
    assert evaluation.trade_plan.trailing_stop_pct == 7.0


@pytest.mark.asyncio
async def test_entry_infers_missing_dynamic_trade_plan():
    service = TradingAIService(
        FakeBackend(['{"score": 80, "decision": "buy", "rationale": "momentum"}'])
    )

    evaluation = await service.evaluate_entry(
        make_snapshot(),
        "",
        StrategySettings(
            25,
            30.0,
            0.5,
            45.0,
            "00:00",
            max_trade_amount_sol=1.5,
        ),
    )

    assert evaluation.wants_buy is True
    assert evaluation.trade_plan is not None
    assert 0.01 <= evaluation.trade_plan.entry_amount_sol <= 1.5
    assert evaluation.trade_plan.stop_loss_pct > 0
    assert evaluation.trade_plan.take_profit_targets_pct


@pytest.mark.asyncio
async def test_exit_and_reflection_require_structured_decisions():
    service = TradingAIService(
        FakeBackend(
            [
                '{"decision": "close", "rationale": "liquidity faded"}',
                '{"rules": ["Require liquidity", "Avoid concentration", "Wait for volume"]}',
            ]
        )
    )

    exit_decision = await service.evaluate_exit(make_trade(), make_snapshot(), "")
    rules = await service.generate_reflection_rules(
        ReflectionEvidence(
            profitable_trades=[{"token_address": "mint-1", "pnl": 0.4}],
            missed_winners=[{"token_address": "mint-skip", "price_change_pct": 120}],
        )
    )

    assert exit_decision.wants_close is True
    assert rules.rules == ["Require liquidity", "Avoid concentration", "Wait for volume"]


@pytest.mark.asyncio
async def test_reflection_accepts_only_bounded_strategy_settings():
    evidence = ReflectionEvidence(recent_analyses=[{"token_address": "mint-1"}])
    service = TradingAIService(
        FakeBackend(
            [
                (
                    '{"rules": ["Rule a", "Rule b", "Rule c"], "settings": {'
                    '"entry_score_threshold": 84, "tracker_poll_seconds": 20, '
                    '"base_trade_amount": 0.2, "position_review_seconds": 25, '
                    '"reflection_time": "01:30"}, "settings_rationale": "more samples"}'
                ),
                (
                    '{"rules": ["Rule a", "Rule b", "Rule c"], "settings": {'
                    '"entry_score_threshold": 10, "tracker_poll_seconds": 1, '
                    '"base_trade_amount": 99, "position_review_seconds": 1, '
                    '"reflection_time": "bad"}}'
                ),
            ]
        )
    )
    defaults = StrategySettings(25, 30.0, 0.1, 45.0, "00:00")

    tuned = await service.generate_reflection(evidence, defaults)
    rejected = await service.generate_reflection(evidence, defaults)

    assert tuned.settings == StrategySettings(
        84, 20.0, 0.2, 25.0, "01:30", launch_score_threshold=84
    )
    assert rejected.settings is None


@pytest.mark.asyncio
async def test_reflection_preserves_telegram_dynamic_setup_toggle():
    evidence = ReflectionEvidence(recent_analyses=[{"token_address": "mint-1"}])
    service = TradingAIService(
        FakeBackend(
            [
                (
                    '{"rules": ["Rule a", "Rule b", "Rule c"], "settings": {'
                    '"entry_score_threshold": 84, "tracker_poll_seconds": 20, '
                    '"base_trade_amount": 0.2, "position_review_seconds": 25, '
                    '"reflection_time": "01:30"}}'
                )
            ]
        )
    )
    current = StrategySettings(
        25, 30.0, 0.1, 45.0, "00:00", dynamic_setup_enabled=False
    )

    tuned = await service.generate_reflection(evidence, current)

    assert tuned.settings.dynamic_setup_enabled is False
