"""Typed application tools exposed to the AI-facing layer."""

from __future__ import annotations

from typing import Dict, Optional

from ai_meme_bot.config import AppConfig
from ai_meme_bot.core.database import Database
from ai_meme_bot.core.execution import TradeExecutor
from ai_meme_bot.core.tracker import TokenTracker
from ai_meme_bot.models import TokenEvaluation, TokenSnapshot, TradePlan, TradeResult


class TradingTools:
    """Safe wrappers around state, analysis, and paper buy operations."""

    def __init__(
        self,
        config: AppConfig,
        database: Database,
        tracker: TokenTracker,
        executor: TradeExecutor,
    ) -> None:
        self.config = config
        self.database = database
        self.tracker = tracker
        self.executor = executor

    async def get_current_balance(self) -> float:
        """Return the current paper wallet balance in relative SOL units."""

        return await self.database.get_balance()

    async def analyze_token(self, token_address: str) -> Optional[Dict[str, object]]:
        """Return a filtered PumpSwap snapshot for one Solana token address."""

        snapshot = await self.tracker.snapshot_for_token(token_address)
        return snapshot.prompt_payload() if snapshot else None

    async def trigger_buy(
        self,
        token_address: str,
        snapshot: Optional[TokenSnapshot] = None,
        evaluation: Optional[TokenEvaluation] = None,
    ) -> TradeResult:
        """Open a paper buy after the orchestrator has approved the token."""

        chosen_snapshot = snapshot or await self.tracker.snapshot_for_token(token_address)
        if chosen_snapshot is None:
            return TradeResult(False, "Token has no eligible PumpSwap snapshot.")
        settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
        plan = evaluation.trade_plan if evaluation else None
        entry_amount = plan.entry_amount_sol if plan else settings.base_trade_amount
        balance = await self.database.get_balance()
        if balance <= 0:
            return TradeResult(False, "Dummy balance is insufficient.")
        entry_amount = min(entry_amount, balance)
        if plan is not None and entry_amount != plan.entry_amount_sol:
            plan = TradePlan(
                entry_amount_sol=entry_amount,
                stop_loss_pct=plan.stop_loss_pct,
                take_profit_targets_pct=plan.take_profit_targets_pct[:],
                trailing_stop_pct=plan.trailing_stop_pct,
                max_hold_seconds=plan.max_hold_seconds,
                rationale="{0} Size capped by available paper balance.".format(
                    plan.rationale
                ).strip(),
            )
        return await self.executor.execute_trade(
            token_address,
            "BUY",
            entry_amount,
            chosen_snapshot,
            plan,
        )
