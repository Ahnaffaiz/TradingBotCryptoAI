"""Async SQLite state management for the paper trader."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import aiosqlite

from ai_meme_bot.models import (
    ReflectionEvidence,
    StrategySettings,
    TokenEvaluation,
    TokenSnapshot,
    TradeRecord,
    isoformat_utc,
    utc_now,
)


DEFAULT_BALANCE = 1.0


class DatabaseError(RuntimeError):
    """Raised when a state transition cannot be persisted."""


class InsufficientBalanceError(DatabaseError):
    """Raised when a paper buy exceeds the dummy wallet."""


class DuplicateOpenTradeError(DatabaseError):
    """Raised when the same token already has an open paper trade."""


class Database:
    """Small SQLite repository with explicit paper-trade transitions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self.db_path)) as connection:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            yield connection

    async def init_db(self) -> None:
        """Create tables and seed the paper wallet/runtime switches."""

        async with self._connect() as connection:
            await connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS wallet (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    balance REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    buy_price REAL NOT NULL,
                    sell_price REAL,
                    pnl REAL,
                    status TEXT NOT NULL CHECK(status IN ('OPEN', 'CLOSED')),
                    timestamp TEXT NOT NULL,
                    entry_amount_sol REAL NOT NULL,
                    token_quantity REAL NOT NULL,
                    entry_snapshot_json TEXT NOT NULL,
                    trade_plan_json TEXT NOT NULL DEFAULT '{}',
                    exit_snapshot_json TEXT,
                    exit_reason TEXT,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS one_open_trade_per_token
                ON trade_history(token_address)
                WHERE status = 'OPEN';

                CREATE TABLE IF NOT EXISTS ai_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rules_text TEXT NOT NULL,
                    date TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS token_analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    pair_address TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    ai_score INTEGER NOT NULL,
                    ai_decision TEXT NOT NULL,
                    ai_rationale TEXT NOT NULL,
                    bought INTEGER NOT NULL DEFAULT 0,
                    trade_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(trade_id) REFERENCES trade_history(id)
                );

                CREATE INDEX IF NOT EXISTS analysis_created_at_idx
                ON token_analysis_history(created_at);

                CREATE TABLE IF NOT EXISTS token_outcome_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL,
                    token_address TEXT NOT NULL,
                    horizon_seconds INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    price_change_pct REAL,
                    liquidity_change_pct REAL,
                    outcome_label TEXT NOT NULL,
                    FOREIGN KEY(analysis_id) REFERENCES token_analysis_history(id),
                    UNIQUE(analysis_id, horizon_seconds)
                );

                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    token_address TEXT,
                    detail TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS activity_created_at_idx
                ON activity_log(created_at);

                CREATE TABLE IF NOT EXISTS runtime_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            await _ensure_column(
                connection,
                "trade_history",
                "trade_plan_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            await connection.execute(
                "INSERT OR IGNORE INTO wallet(id, balance) VALUES(1, ?)",
                (DEFAULT_BALANCE,),
            )
            await connection.execute(
                """
                INSERT OR IGNORE INTO runtime_settings(key, value)
                VALUES('auto_trading_enabled', '0')
                """
            )
            await connection.execute(
                """
                INSERT OR IGNORE INTO runtime_settings(key, value)
                VALUES('notifications_enabled', '1')
                """
            )
            await connection.commit()

    async def get_balance(self) -> float:
        """Return the current dummy wallet balance."""

        async with self._connect() as connection:
            cursor = await connection.execute("SELECT balance FROM wallet WHERE id = 1")
            row = await cursor.fetchone()
        return float(row["balance"]) if row else DEFAULT_BALANCE

    async def update_dummy_balance(self, delta: float) -> float:
        """Apply a balance delta and return the new dummy balance."""

        async with self._connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute("SELECT balance FROM wallet WHERE id = 1")
            row = await cursor.fetchone()
            balance = float(row["balance"]) if row else DEFAULT_BALANCE
            new_balance = balance + delta
            if new_balance < -1e-9:
                await connection.rollback()
                raise InsufficientBalanceError("Dummy balance is insufficient.")
            await connection.execute(
                "UPDATE wallet SET balance = ? WHERE id = 1", (new_balance,)
            )
            await connection.commit()
        return new_balance

    async def insert_trade(
        self,
        token_address: str,
        buy_price: float,
        entry_amount_sol: float,
        token_quantity: float,
        entry_snapshot: Dict[str, Any],
        trade_plan: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Debit the wallet and create one open paper trade atomically."""

        if buy_price <= 0 or entry_amount_sol <= 0 or token_quantity <= 0:
            raise DatabaseError("Trade price, amount, and quantity must be positive.")
        opened_at = isoformat_utc()
        try:
            async with self._connect() as connection:
                await connection.execute("BEGIN IMMEDIATE")
                cursor = await connection.execute(
                    "SELECT balance FROM wallet WHERE id = 1"
                )
                row = await cursor.fetchone()
                balance = float(row["balance"]) if row else DEFAULT_BALANCE
                if balance < entry_amount_sol:
                    await connection.rollback()
                    raise InsufficientBalanceError("Dummy balance is insufficient.")
                await connection.execute(
                    "UPDATE wallet SET balance = ? WHERE id = 1",
                    (balance - entry_amount_sol,),
                )
                cursor = await connection.execute(
                    """
                    INSERT INTO trade_history(
                        token_address, buy_price, status, timestamp,
                        entry_amount_sol, token_quantity, entry_snapshot_json,
                        trade_plan_json, opened_at
                    ) VALUES(?, ?, 'OPEN', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        token_address,
                        buy_price,
                        opened_at,
                        entry_amount_sol,
                        token_quantity,
                        json.dumps(entry_snapshot, sort_keys=True),
                        json.dumps(trade_plan or {}, sort_keys=True),
                        opened_at,
                    ),
                )
                await connection.commit()
                return int(cursor.lastrowid)
        except aiosqlite.IntegrityError as exc:
            raise DuplicateOpenTradeError(
                "Token already has an open paper trade."
            ) from exc

    async def close_trade(
        self,
        trade_id: int,
        sell_price: float,
        exit_snapshot: Dict[str, Any],
        exit_reason: str,
    ) -> float:
        """Close an open paper trade, credit proceeds, and return PnL."""

        if sell_price <= 0:
            raise DatabaseError("Sell price must be positive.")
        closed_at = isoformat_utc()
        async with self._connect() as connection:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                "SELECT * FROM trade_history WHERE id = ? AND status = 'OPEN'",
                (trade_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                await connection.rollback()
                raise DatabaseError("Open trade {0} was not found.".format(trade_id))
            proceeds = float(row["token_quantity"]) * sell_price
            pnl = proceeds - float(row["entry_amount_sol"])
            cursor = await connection.execute("SELECT balance FROM wallet WHERE id = 1")
            wallet = await cursor.fetchone()
            balance = float(wallet["balance"]) if wallet else DEFAULT_BALANCE
            await connection.execute(
                """
                UPDATE trade_history
                SET sell_price = ?, pnl = ?, status = 'CLOSED',
                    exit_snapshot_json = ?, exit_reason = ?, closed_at = ?
                WHERE id = ?
                """,
                (
                    sell_price,
                    pnl,
                    json.dumps(exit_snapshot, sort_keys=True),
                    exit_reason,
                    closed_at,
                    trade_id,
                ),
            )
            await connection.execute(
                "UPDATE wallet SET balance = ? WHERE id = 1",
                (balance + proceeds,),
            )
            await connection.commit()
        return pnl

    async def update_trade_status(
        self,
        trade_id: int,
        status: str,
        sell_price: Optional[float] = None,
        pnl: Optional[float] = None,
    ) -> None:
        """Compatibility update helper for non-paper fixtures and migrations."""

        if status not in {"OPEN", "CLOSED"}:
            raise DatabaseError("Trade status must be OPEN or CLOSED.")
        async with self._connect() as connection:
            await connection.execute(
                """
                UPDATE trade_history
                SET status = ?, sell_price = COALESCE(?, sell_price),
                    pnl = COALESCE(?, pnl),
                    closed_at = CASE WHEN ? = 'CLOSED' THEN COALESCE(closed_at, ?)
                                     ELSE closed_at END
                WHERE id = ?
                """,
                (status, sell_price, pnl, status, isoformat_utc(), trade_id),
            )
            await connection.commit()

    async def get_trade(self, trade_id: int) -> Optional[TradeRecord]:
        """Fetch one trade by id."""

        async with self._connect() as connection:
            cursor = await connection.execute(
                "SELECT * FROM trade_history WHERE id = ?", (trade_id,)
            )
            row = await cursor.fetchone()
        return self._trade_from_row(row) if row else None

    async def get_open_trades(self) -> List[TradeRecord]:
        """Fetch current open trades."""

        return await self._fetch_trades(
            "SELECT * FROM trade_history WHERE status = 'OPEN' ORDER BY opened_at"
        )

    async def get_closed_trades(self, limit: Optional[int] = None) -> List[TradeRecord]:
        """Fetch recent closed trades for reflection."""

        sql = "SELECT * FROM trade_history WHERE status = 'CLOSED' ORDER BY closed_at DESC"
        params: Iterable[Any] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return await self._fetch_trades(sql, params)

    async def get_recent_trades(self, limit: int = 10) -> List[TradeRecord]:
        """Fetch recent trades regardless of status."""

        return await self._fetch_trades(
            "SELECT * FROM trade_history ORDER BY opened_at DESC LIMIT ?", (limit,)
        )

    async def get_trade_history(
        self, limit: Optional[int] = None
    ) -> List[TradeRecord]:
        """Fetch trade history regardless of status, newest first."""

        sql = "SELECT * FROM trade_history ORDER BY opened_at DESC"
        params: Iterable[Any] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return await self._fetch_trades(sql, params)

    async def _fetch_trades(
        self, sql: str, params: Iterable[Any] = ()
    ) -> List[TradeRecord]:
        async with self._connect() as connection:
            cursor = await connection.execute(sql, tuple(params))
            rows = await cursor.fetchall()
        return [self._trade_from_row(row) for row in rows]

    @staticmethod
    def _trade_from_row(row: aiosqlite.Row) -> TradeRecord:
        return TradeRecord(**dict(row))

    async def set_auto_trading(self, enabled: bool) -> None:
        """Persist whether background discovery may open new positions."""

        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO runtime_settings(key, value) VALUES('auto_trading_enabled', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("1" if enabled else "0",),
            )
            await connection.commit()

    async def get_auto_trading(self) -> bool:
        """Return the persisted auto-trading switch."""

        async with self._connect() as connection:
            cursor = await connection.execute(
                "SELECT value FROM runtime_settings WHERE key = 'auto_trading_enabled'"
            )
            row = await cursor.fetchone()
        return bool(row and row["value"] == "1")

    async def set_notification_chat_id(self, chat_id: int) -> None:
        """Remember the Telegram chat that should receive paper-trade reports."""

        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO runtime_settings(key, value) VALUES('notification_chat_id', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(chat_id),),
            )
            await connection.commit()

    async def get_notification_chat_id(self) -> Optional[int]:
        """Return the last Telegram chat that interacted with the bot."""

        async with self._connect() as connection:
            cursor = await connection.execute(
                "SELECT value FROM runtime_settings WHERE key = 'notification_chat_id'"
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    async def set_notifications_enabled(self, enabled: bool) -> None:
        """Persist whether Telegram reports should be delivered."""

        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT INTO runtime_settings(key, value) VALUES('notifications_enabled', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("1" if enabled else "0",),
            )
            await connection.commit()

    async def get_notifications_enabled(self) -> bool:
        """Return the Telegram paper-report switch."""

        async with self._connect() as connection:
            cursor = await connection.execute(
                "SELECT value FROM runtime_settings WHERE key = 'notifications_enabled'"
            )
            row = await cursor.fetchone()
        return row is None or row["value"] == "1"

    async def set_strategy_settings(self, settings: StrategySettings) -> None:
        """Persist AI-tuned paper strategy settings."""

        async with self._connect() as connection:
            for key, value in settings.prompt_payload().items():
                await connection.execute(
                    """
                    INSERT INTO runtime_settings(key, value) VALUES(?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    ("strategy_{0}".format(key), str(value)),
                )
            await connection.commit()

    async def get_strategy_settings(self, defaults: StrategySettings) -> StrategySettings:
        """Load persisted strategy values and fall back to startup config."""

        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                SELECT key, value FROM runtime_settings
                WHERE key LIKE 'strategy_%'
                """
            )
            stored = {
                str(row["key"])[len("strategy_") :]: row["value"]
                for row in await cursor.fetchall()
            }
        try:
            return StrategySettings(
                entry_score_threshold=int(
                    stored.get("entry_score_threshold", defaults.entry_score_threshold)
                ),
                tracker_poll_seconds=float(
                    stored.get("tracker_poll_seconds", defaults.tracker_poll_seconds)
                ),
                base_trade_amount=float(
                    stored.get("base_trade_amount", defaults.base_trade_amount)
                ),
                position_review_seconds=float(
                    stored.get(
                        "position_review_seconds", defaults.position_review_seconds
                    )
                ),
                reflection_time=str(
                    stored.get("reflection_time", defaults.reflection_time)
                ),
                launch_enabled=_stored_bool(
                    stored.get("launch_enabled"), defaults.launch_enabled
                ),
                scout_enabled=_stored_bool(
                    stored.get("scout_enabled"), defaults.scout_enabled
                ),
                launch_score_threshold=int(
                    stored.get(
                        "launch_score_threshold", defaults.launch_score_threshold
                    )
                ),
                scout_score_threshold=int(
                    stored.get("scout_score_threshold", defaults.scout_score_threshold)
                ),
                take_profit_pct=float(
                    stored.get("take_profit_pct", defaults.take_profit_pct)
                ),
                stop_loss_pct=float(
                    stored.get("stop_loss_pct", defaults.stop_loss_pct)
                ),
                trailing_stop_pct=float(
                    stored.get("trailing_stop_pct", defaults.trailing_stop_pct)
                ),
                max_hold_seconds=float(
                    stored.get("max_hold_seconds", defaults.max_hold_seconds)
                ),
                scout_min_liquidity_usd=float(
                    stored.get(
                        "scout_min_liquidity_usd", defaults.scout_min_liquidity_usd
                    )
                ),
                scout_min_volume_5m_usd=float(
                    stored.get(
                        "scout_min_volume_5m_usd",
                        defaults.scout_min_volume_5m_usd,
                    )
                ),
                dynamic_setup_enabled=_stored_bool(
                    stored.get("dynamic_setup_enabled"),
                    defaults.dynamic_setup_enabled,
                ),
            )
        except (TypeError, ValueError):
            return defaults

    async def add_rules(self, rules_text: str, date: Optional[str] = None) -> int:
        """Persist reflection rules and return their id."""

        stored_date = date or isoformat_utc()
        async with self._connect() as connection:
            cursor = await connection.execute(
                "INSERT INTO ai_rules(rules_text, date) VALUES(?, ?)",
                (rules_text, stored_date),
            )
            await connection.commit()
        return int(cursor.lastrowid)

    async def get_latest_rules(self) -> str:
        """Fetch the newest reflection text for prompt injection."""

        async with self._connect() as connection:
            cursor = await connection.execute(
                "SELECT rules_text FROM ai_rules ORDER BY date DESC, id DESC LIMIT 1"
            )
            row = await cursor.fetchone()
        return str(row["rules_text"]) if row else ""

    async def record_analysis(
        self, snapshot: TokenSnapshot, evaluation: TokenEvaluation
    ) -> int:
        """Store one AI entry analysis for later outcome tracking."""

        created_at = isoformat_utc()
        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                INSERT INTO token_analysis_history(
                    token_address, pair_address, snapshot_json, ai_score,
                    ai_decision, ai_rationale, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.token_address,
                    snapshot.pair_address,
                    json.dumps(snapshot.stored_payload(), sort_keys=True),
                    evaluation.score,
                    evaluation.decision,
                    evaluation.rationale,
                    created_at,
                ),
            )
            await connection.commit()
        return int(cursor.lastrowid)

    async def mark_analysis_bought(self, analysis_id: int, trade_id: Optional[int]) -> None:
        """Link a successful paper buy to its entry analysis."""

        async with self._connect() as connection:
            await connection.execute(
                """
                UPDATE token_analysis_history
                SET bought = 1, trade_id = COALESCE(?, trade_id)
                WHERE id = ?
                """,
                (trade_id, analysis_id),
            )
            await connection.commit()

    async def add_outcome_snapshot(
        self,
        analysis_id: int,
        horizon_seconds: int,
        initial_snapshot: Dict[str, Any],
        current_snapshot: TokenSnapshot,
        analysis_decision: str,
    ) -> None:
        """Persist a later market snapshot and label skip outcomes."""

        price_change = _percent_change(
            initial_snapshot.get("price_usd"), current_snapshot.price_usd
        )
        liquidity_change = _percent_change(
            initial_snapshot.get("liquidity_usd"), current_snapshot.liquidity_usd
        )
        outcome_label = _outcome_label(analysis_decision, price_change, liquidity_change)
        async with self._connect() as connection:
            await connection.execute(
                """
                INSERT OR IGNORE INTO token_outcome_snapshots(
                    analysis_id, token_address, horizon_seconds, captured_at,
                    snapshot_json, price_change_pct, liquidity_change_pct, outcome_label
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    current_snapshot.token_address,
                    horizon_seconds,
                    isoformat_utc(),
                    json.dumps(current_snapshot.stored_payload(), sort_keys=True),
                    price_change,
                    liquidity_change,
                    outcome_label,
                ),
            )
            await connection.commit()

    async def get_due_analysis_outcomes(
        self, horizons: Iterable[int], limit: int = 60
    ) -> List[Dict[str, Any]]:
        """Return analyses that need a later market snapshot."""

        due_rows: List[Dict[str, Any]] = []
        async with self._connect() as connection:
            for horizon in horizons:
                threshold = isoformat_utc(utc_now() - timedelta(seconds=horizon))
                cursor = await connection.execute(
                    """
                    SELECT a.id, a.token_address, a.snapshot_json, a.ai_decision, ?
                           AS horizon_seconds
                    FROM token_analysis_history AS a
                    LEFT JOIN token_outcome_snapshots AS o
                      ON o.analysis_id = a.id AND o.horizon_seconds = ?
                    WHERE a.created_at <= ? AND o.id IS NULL
                    ORDER BY a.created_at
                    LIMIT ?
                    """,
                    (horizon, horizon, threshold, limit),
                )
                due_rows.extend(dict(row) for row in await cursor.fetchall())
        return due_rows

    async def add_activity(
        self,
        event_type: str,
        detail: str,
        token_address: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Append a structured activity event for reflection and diagnostics."""

        async with self._connect() as connection:
            cursor = await connection.execute(
                """
                INSERT INTO activity_log(
                    event_type, token_address, detail, payload_json, created_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    token_address,
                    detail,
                    json.dumps(payload or {}, sort_keys=True),
                    isoformat_utc(),
                ),
            )
            await connection.commit()
        return int(cursor.lastrowid)

    async def get_reflection_evidence(self, limit: int = 40) -> ReflectionEvidence:
        """Build bounded learning evidence from trades, analyses, outcomes, and logs."""

        closed_trades = await self.get_closed_trades(limit=limit)
        profitable = [_trade_evidence(trade) for trade in closed_trades if (trade.pnl or 0) > 0]
        losing = [_trade_evidence(trade) for trade in closed_trades if (trade.pnl or 0) <= 0]
        async with self._connect() as connection:
            analyses = await _fetch_dicts(
                connection,
                """
                SELECT id, token_address, ai_score, ai_decision, ai_rationale,
                       bought, created_at
                FROM token_analysis_history
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            correct_skips = await _fetch_dicts(
                connection,
                """
                SELECT o.analysis_id, o.token_address, o.horizon_seconds,
                       o.price_change_pct, o.liquidity_change_pct, o.captured_at,
                       a.ai_score, a.ai_rationale
                FROM token_outcome_snapshots AS o
                JOIN token_analysis_history AS a ON a.id = o.analysis_id
                WHERE o.outcome_label = 'correct_skip'
                ORDER BY o.captured_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            missed_winners = await _fetch_dicts(
                connection,
                """
                SELECT o.analysis_id, o.token_address, o.horizon_seconds,
                       o.price_change_pct, o.liquidity_change_pct, o.captured_at,
                       a.ai_score, a.ai_rationale
                FROM token_outcome_snapshots AS o
                JOIN token_analysis_history AS a ON a.id = o.analysis_id
                WHERE o.outcome_label = 'missed_winner'
                ORDER BY o.captured_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            failed_filters = await _activity_evidence(
                connection, "filter_reject", limit
            )
            errors = await _activity_evidence(connection, "error", limit)
        return ReflectionEvidence(
            profitable_trades=profitable,
            losing_trades=losing,
            recent_analyses=analyses,
            correct_skips=correct_skips,
            missed_winners=missed_winners,
            failed_filters=failed_filters,
            recurring_errors=errors,
        )


async def _fetch_dicts(
    connection: aiosqlite.Connection, sql: str, params: Iterable[Any]
) -> List[Dict[str, Any]]:
    cursor = await connection.execute(sql, tuple(params))
    return [dict(row) for row in await cursor.fetchall()]


async def _ensure_column(
    connection: aiosqlite.Connection, table: str, column: str, definition: str
) -> None:
    cursor = await connection.execute("PRAGMA table_info({0})".format(table))
    columns = {str(row["name"]) for row in await cursor.fetchall()}
    if column not in columns:
        await connection.execute(
            "ALTER TABLE {0} ADD COLUMN {1} {2}".format(table, column, definition)
        )


async def _activity_evidence(
    connection: aiosqlite.Connection, event_type: str, limit: int
) -> List[Dict[str, Any]]:
    return await _fetch_dicts(
        connection,
        """
        SELECT token_address, detail, payload_json, created_at
        FROM activity_log
        WHERE event_type = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (event_type, limit),
    )


def _trade_evidence(trade: TradeRecord) -> Dict[str, Any]:
    return {
        "trade_id": trade.id,
        "token_address": trade.token_address,
        "buy_price": trade.buy_price,
        "sell_price": trade.sell_price,
        "pnl": trade.pnl,
        "exit_reason": trade.exit_reason,
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
        "entry_snapshot_json": trade.entry_snapshot_json,
        "trade_plan_json": trade.trade_plan_json,
        "exit_snapshot_json": trade.exit_snapshot_json,
    }


def _percent_change(start: Any, end: Any) -> Optional[float]:
    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError):
        return None
    if start_value <= 0:
        return None
    return round(((end_value - start_value) / start_value) * 100, 4)


def _stored_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _outcome_label(
    analysis_decision: str,
    price_change_pct: Optional[float],
    liquidity_change_pct: Optional[float],
) -> str:
    if analysis_decision.lower() != "skip":
        return "observed_buy"
    if price_change_pct is not None and price_change_pct >= 50:
        return "missed_winner"
    if (
        price_change_pct is not None
        and price_change_pct <= -30
        or liquidity_change_pct is not None
        and liquidity_change_pct <= -50
    ):
        return "correct_skip"
    return "skip_observed"


async def init_db(db_path: Path) -> Database:
    """PRD-style helper that creates and returns a database repository."""

    database = Database(db_path)
    await database.init_db()
    return database


async def update_dummy_balance(db_path: Path, delta: float) -> float:
    """PRD-style wrapper for paper balance updates."""

    return await Database(db_path).update_dummy_balance(delta)


async def insert_trade(
    db_path: Path,
    token_address: str,
    buy_price: float,
    entry_amount_sol: float,
    token_quantity: float,
    entry_snapshot: Dict[str, Any],
    trade_plan: Optional[Dict[str, Any]] = None,
) -> int:
    """PRD-style wrapper for opening a paper trade."""

    return await Database(db_path).insert_trade(
        token_address,
        buy_price,
        entry_amount_sol,
        token_quantity,
        entry_snapshot,
        trade_plan,
    )


async def update_trade_status(
    db_path: Path,
    trade_id: int,
    status: str,
    sell_price: Optional[float] = None,
    pnl: Optional[float] = None,
) -> None:
    """PRD-style wrapper for direct trade status updates."""

    await Database(db_path).update_trade_status(trade_id, status, sell_price, pnl)


async def get_recent_trades(db_path: Path, limit: int = 10) -> List[TradeRecord]:
    """PRD-style wrapper for recent trade lookup."""

    return await Database(db_path).get_recent_trades(limit)
