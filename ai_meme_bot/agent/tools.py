"""Typed application tools exposed to the AI-facing layer."""

from __future__ import annotations

from typing import Dict, Optional

from ai_meme_bot.config import AppConfig
from ai_meme_bot.core.database import Database
from ai_meme_bot.core.execution import TradeExecutor
from ai_meme_bot.core.tracker import TokenTracker
from ai_meme_bot.models import TokenSnapshot, TradeResult


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
        self, token_address: str, snapshot: Optional[TokenSnapshot] = None
    ) -> TradeResult:
        """Open a paper buy after the orchestrator has approved the token."""

        chosen_snapshot = snapshot or await self.tracker.snapshot_for_token(token_address)
        if chosen_snapshot is None:
            return TradeResult(False, "Token has no eligible PumpSwap snapshot.")
        settings = await self.database.get_strategy_settings(self.config.strategy_defaults)
        return await self.executor.execute_trade(
            token_address,
            "BUY",
            settings.base_trade_amount,
            chosen_snapshot,
        )
