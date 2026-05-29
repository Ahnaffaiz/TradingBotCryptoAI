"""Mode-routed paper trade execution."""

from __future__ import annotations

import logging
from typing import Optional

from ai_meme_bot.config import AppConfig
from ai_meme_bot.core.database import (
    Database,
    DatabaseError,
    DuplicateOpenTradeError,
    InsufficientBalanceError,
)
from ai_meme_bot.models import TokenSnapshot, TradePlan, TradeRecord, TradeResult


LOGGER = logging.getLogger(__name__)


class RealTradingUnavailable(RuntimeError):
    """V1 guardrail for the intentionally closed broadcast path."""


class TradeExecutor:
    """Execute paper trades and retain a hard mode boundary for REAL."""

    def __init__(self, config: AppConfig, database: Database) -> None:
        self.config = config
        self.database = database

    async def execute_trade(
        self,
        token_address: str,
        action: str,
        amount_sol: Optional[float],
        snapshot: TokenSnapshot,
        trade_plan: Optional[TradePlan] = None,
    ) -> TradeResult:
        """Dispatch a trade action based on `TRADING_MODE`."""

        action = action.upper()
        try:
            if self.config.trading_mode == "PAPER":
                if action == "BUY":
                    return await self._paper_buy(
                        token_address,
                        amount_sol or self.config.base_trade_amount,
                        snapshot,
                        trade_plan,
                    )
                if action in {"SELL", "CLOSE"}:
                    return await self._paper_close_by_token(token_address, snapshot, "AI close")
                return TradeResult(False, "Unsupported paper trade action: {0}".format(action))
            if self.config.trading_mode == "REAL":
                raise RealTradingUnavailable(
                    "REAL mode is disabled in v1; PumpSwap/Jito broadcast is not implemented."
                )
            return TradeResult(False, "Unsupported trading mode: {0}".format(self.config.trading_mode))
        except RealTradingUnavailable as exc:
            LOGGER.warning("%s", exc)
            return TradeResult(False, str(exc))
        except (DatabaseError, ValueError, ZeroDivisionError) as exc:
            LOGGER.warning("Trade action rejected: %s", exc)
            return TradeResult(False, str(exc))
        except Exception as exc:  # Keeps tracker loops alive on unexpected timeouts/state errors.
            LOGGER.exception("Trade action failed unexpectedly")
            return TradeResult(False, "Trade action failed safely: {0}".format(exc))

    async def close_trade(
        self, trade: TradeRecord, snapshot: TokenSnapshot, reason: str
    ) -> TradeResult:
        """Close a known open paper trade after a validated AI exit decision."""

        if self.config.trading_mode != "PAPER":
            return await self.execute_trade(
                trade.token_address, "CLOSE", trade.entry_amount_sol, snapshot
            )
        try:
            pnl = await self.database.close_trade(
                trade.id,
                snapshot.price_usd,
                snapshot.stored_payload(),
                reason,
            )
        except DatabaseError as exc:
            return TradeResult(False, str(exc), trade_id=trade.id)
        return TradeResult(
            True,
            "Closed paper trade.",
            trade_id=trade.id,
            pnl=pnl,
            entry_amount_sol=trade.entry_amount_sol,
        )

    async def add_to_trade(
        self, trade: TradeRecord, amount_sol: float, snapshot: TokenSnapshot, reason: str
    ) -> TradeResult:
        """Blend an additional paper buy into an existing open position."""

        if self.config.trading_mode != "PAPER":
            return TradeResult(
                False,
                "REAL mode add-on buys are disabled in v1.",
                trade_id=trade.id,
                entry_amount_sol=amount_sol,
            )
        if snapshot.price_usd <= 0:
            return TradeResult(
                False,
                "Paper add-on needs a positive snapshot price.",
                trade_id=trade.id,
                entry_amount_sol=amount_sol,
            )
        token_quantity = amount_sol / snapshot.price_usd
        try:
            await self.database.add_to_trade(
                trade.id,
                snapshot.price_usd,
                amount_sol,
                token_quantity,
                snapshot.stored_payload(),
                reason,
            )
        except DatabaseError as exc:
            return TradeResult(False, str(exc), trade_id=trade.id, entry_amount_sol=amount_sol)
        return TradeResult(
            True,
            "Added to paper trade.",
            trade_id=trade.id,
            entry_amount_sol=amount_sol,
        )

    async def _paper_buy(
        self,
        token_address: str,
        amount_sol: float,
        snapshot: TokenSnapshot,
        trade_plan: Optional[TradePlan] = None,
    ) -> TradeResult:
        if snapshot.price_usd <= 0:
            raise ValueError("Paper buy needs a positive snapshot price.")
        # For v1 the paper wallet tracks relative capital units named SOL. Prices remain
        # USD quotes, so quantities preserve percentage PnL without a SOL/USD oracle.
        token_quantity = amount_sol / snapshot.price_usd
        try:
            trade_id = await self.database.insert_trade(
                token_address=token_address,
                buy_price=snapshot.price_usd,
                entry_amount_sol=amount_sol,
                token_quantity=token_quantity,
                entry_snapshot=snapshot.stored_payload(),
                trade_plan=trade_plan.stored_payload() if trade_plan else None,
            )
        except (InsufficientBalanceError, DuplicateOpenTradeError):
            raise
        return TradeResult(
            True,
            "Opened paper trade.",
            trade_id=trade_id,
            entry_amount_sol=amount_sol,
        )

    async def _paper_close_by_token(
        self, token_address: str, snapshot: TokenSnapshot, reason: str
    ) -> TradeResult:
        open_trade = next(
            (
                trade
                for trade in await self.database.get_open_trades()
                if trade.token_address == token_address
            ),
            None,
        )
        if open_trade is None:
            return TradeResult(False, "No open paper trade for token.")
        return await self.close_trade(open_trade, snapshot, reason)


async def execute_trade(
    executor: TradeExecutor,
    token_address: str,
    action: str,
    amount_sol: Optional[float],
    snapshot: TokenSnapshot,
    trade_plan: Optional[TradePlan] = None,
) -> TradeResult:
    """PRD-style function wrapper around the mode-aware executor."""

    return await executor.execute_trade(
        token_address, action, amount_sol, snapshot, trade_plan
    )
