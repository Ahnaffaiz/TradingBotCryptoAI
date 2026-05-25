"""Typed data contracts shared by the bot subsystems."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


def isoformat_utc(value: Optional[datetime] = None) -> str:
    """Serialize a timestamp in UTC for SQLite and prompts."""

    return (value or utc_now()).astimezone(timezone.utc).isoformat()


@dataclass
class TokenSnapshot:
    """Market and holder metrics evaluated by the AI and paper engine."""

    token_address: str
    pair_address: str
    price_usd: float
    liquidity_usd: float
    volume_5m_usd: float
    pair_age_seconds: float
    dex_id: str = "pumpswap"
    pair_created_at_ms: Optional[int] = None
    top_holder_share_pct: Optional[float] = None
    developer_holding_pct: Optional[float] = None
    price_change_5m_pct: Optional[float] = None
    price_change_1h_pct: Optional[float] = None
    buys_5m: Optional[int] = None
    sells_5m: Optional[int] = None
    x_recent_mentions: Optional[int] = None
    x_recent_author_count: Optional[int] = None
    x_sentiment_hint: Optional[str] = None
    geckoterminal_trending: Optional[bool] = None
    geckoterminal_trending_rank: Optional[int] = None
    strategy: str = "launch"
    raw_context: Dict[str, Any] = field(default_factory=dict)

    def prompt_payload(self) -> Dict[str, Any]:
        """Return a bounded prompt payload instead of every raw API field."""

        return {
            "token_address": self.token_address,
            "pair_address": self.pair_address,
            "dex_id": self.dex_id,
            "price_usd": self.price_usd,
            "liquidity_usd": self.liquidity_usd,
            "volume_5m_usd": self.volume_5m_usd,
            "pair_age_seconds": self.pair_age_seconds,
            "top_holder_share_pct": self.top_holder_share_pct,
            "developer_holding_pct": self.developer_holding_pct,
            "price_change_5m_pct": self.price_change_5m_pct,
            "price_change_1h_pct": self.price_change_1h_pct,
            "buys_5m": self.buys_5m,
            "sells_5m": self.sells_5m,
            "x_recent_mentions": self.x_recent_mentions,
            "x_recent_author_count": self.x_recent_author_count,
            "x_sentiment_hint": self.x_sentiment_hint,
            "geckoterminal_trending": self.geckoterminal_trending,
            "geckoterminal_trending_rank": self.geckoterminal_trending_rank,
            "strategy": self.strategy,
        }

    def stored_payload(self) -> Dict[str, Any]:
        """Return all state needed for later reflection and tests."""

        return asdict(self)


@dataclass
class TradePlan:
    """AI-generated paper position sizing and exit setup."""

    entry_amount_sol: float
    stop_loss_pct: float
    take_profit_targets_pct: List[float] = field(default_factory=list)
    trailing_stop_pct: float = 0.0
    max_hold_seconds: float = 3600.0
    rationale: str = ""

    def stored_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenEvaluation:
    """AI entry review result."""

    score: int = 0
    decision: str = "skip"
    rationale: str = "AI evaluation unavailable."
    trade_plan: Optional[TradePlan] = None

    @property
    def wants_buy(self) -> bool:
        return self.decision.lower() == "buy"


@dataclass
class ExitDecision:
    """AI review result for an open paper trade."""

    decision: str = "hold"
    rationale: str = "AI exit evaluation unavailable."

    @property
    def wants_close(self) -> bool:
        return self.decision.lower() == "close"


@dataclass
class ReflectionRules:
    """Strict rules generated from closed trade history."""

    rules: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join("- {0}".format(rule) for rule in self.rules)


@dataclass
class StrategySettings:
    """Paper strategy controls that may be tuned within app guardrails."""

    entry_score_threshold: int
    tracker_poll_seconds: float
    base_trade_amount: float
    position_review_seconds: float
    reflection_time: str
    launch_enabled: bool = True
    scout_enabled: bool = True
    launch_score_threshold: int = 25
    scout_score_threshold: int = 70
    take_profit_pct: float = 18.0
    stop_loss_pct: float = 8.0
    trailing_stop_pct: float = 7.0
    max_hold_seconds: float = 3600.0
    scout_min_liquidity_usd: float = 15000.0
    scout_min_volume_5m_usd: float = 500.0
    dynamic_setup_enabled: bool = True

    def prompt_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyReflection:
    """Nightly rules plus optional bounded adaptive runtime settings."""

    rules: ReflectionRules = field(default_factory=ReflectionRules)
    settings: Optional[StrategySettings] = None
    settings_rationale: str = ""


@dataclass
class ReflectionEvidence:
    """Compact learning evidence provided to nightly AI reflection."""

    profitable_trades: List[Dict[str, Any]] = field(default_factory=list)
    losing_trades: List[Dict[str, Any]] = field(default_factory=list)
    recent_analyses: List[Dict[str, Any]] = field(default_factory=list)
    correct_skips: List[Dict[str, Any]] = field(default_factory=list)
    missed_winners: List[Dict[str, Any]] = field(default_factory=list)
    failed_filters: List[Dict[str, Any]] = field(default_factory=list)
    recurring_errors: List[Dict[str, Any]] = field(default_factory=list)

    def has_learning_data(self) -> bool:
        """Return whether reflection has any observed evidence."""

        return any(
            (
                self.profitable_trades,
                self.losing_trades,
                self.recent_analyses,
                self.correct_skips,
                self.missed_winners,
                self.failed_filters,
                self.recurring_errors,
            )
        )


@dataclass
class FilteredToken:
    """One candidate rejected by the base market filters."""

    token_address: str
    reason: str
    pair_address: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """A paper or future live trade persisted in SQLite."""

    id: int
    token_address: str
    buy_price: float
    sell_price: Optional[float]
    pnl: Optional[float]
    status: str
    timestamp: str
    entry_amount_sol: float
    token_quantity: float
    entry_snapshot_json: str
    trade_plan_json: str
    exit_snapshot_json: Optional[str]
    exit_reason: Optional[str]
    opened_at: str
    closed_at: Optional[str]


@dataclass
class TradeResult:
    """Outcome of a mode-routed trade operation."""

    success: bool
    message: str
    trade_id: Optional[int] = None
    pnl: Optional[float] = None
