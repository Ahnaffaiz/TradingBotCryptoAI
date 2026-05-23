"""Fail-closed structured AI decisions backed by embedded Hermes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Protocol

from ai_meme_bot.config import AppConfig
from ai_meme_bot.models import (
    ExitDecision,
    ReflectionEvidence,
    ReflectionRules,
    StrategyReflection,
    StrategySettings,
    TokenEvaluation,
    TokenSnapshot,
    TradeRecord,
)


LOGGER = logging.getLogger(__name__)


class ChatBackend(Protocol):
    """Small async interface used by the strategy service and tests."""

    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model response text."""


class HermesChatBackend:
    """Instantiate Hermes agents for provider-neutral OpenAI-compatible calls."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Run the synchronous Hermes Python API off the event loop."""

        return await asyncio.to_thread(self._chat_sync, system_prompt, user_prompt)

    async def operator_chat(
        self, prompt: str, user_id: str, user_name: str = ""
    ) -> str:
        """Run an explicitly-enabled workspace operator Hermes turn."""

        return await asyncio.to_thread(
            self._operator_chat_sync, prompt, user_id, user_name
        )

    def _chat_sync(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from run_agent import AIAgent
        except ImportError as exc:
            raise RuntimeError(
                "Hermes is unavailable; install project dependencies before AI calls."
            ) from exc

        kwargs: Dict[str, Any] = {
            "model": self.config.ai_model,
            "api_key": self.config.ai_api_key or None,
            "quiet_mode": True,
            "skip_context_files": True,
            "skip_memory": True,
            "ephemeral_system_prompt": system_prompt,
            "disabled_toolsets": ["browser", "file", "terminal", "web"],
            "platform": "telegram",
        }
        if self.config.ai_base_url:
            kwargs["base_url"] = self.config.ai_base_url
        agent = AIAgent(**kwargs)
        response = agent.chat(user_prompt)
        return str(response)

    def _operator_chat_sync(self, prompt: str, user_id: str, user_name: str) -> str:
        try:
            from run_agent import AIAgent
        except ImportError as exc:
            raise RuntimeError(
                "Hermes is unavailable; install project dependencies before operator calls."
            ) from exc

        kwargs: Dict[str, Any] = {
            "model": self.config.ai_model,
            "api_key": self.config.ai_api_key or None,
            "quiet_mode": True,
            "ephemeral_system_prompt": _operator_system_prompt(),
            "enabled_toolsets": ["file", "terminal"],
            "disabled_toolsets": ["browser", "web", "messaging"],
            "platform": "telegram",
            "user_id": user_id,
            "user_name": user_name,
            "max_iterations": 40,
        }
        if self.config.ai_base_url:
            kwargs["base_url"] = self.config.ai_base_url
        agent = AIAgent(**kwargs)
        return str(agent.chat(prompt))


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object even if a provider wraps it in light prose."""

    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        start = text.find("{") if isinstance(text, str) else -1
        end = text.rfind("}") if isinstance(text, str) else -1
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


class TradingAIService:
    """Prompts the model and validates every trading decision."""

    def __init__(self, backend: ChatBackend) -> None:
        self.backend = backend

    async def evaluate_entry(
        self, snapshot: TokenSnapshot, rules: str
    ) -> TokenEvaluation:
        """Return a validated entry review or a safe skip."""

        prompt = {
            "task": "Evaluate whether a paper trader may buy this PumpSwap candidate.",
            "candidate": snapshot.prompt_payload(),
            "latest_rules": rules or "No learned rules yet.",
            "response_schema": {
                "score": "integer from 0 to 100",
                "decision": "buy or skip",
                "rationale": "short string",
            },
        }
        try:
            payload = _extract_json(
                await self.backend.chat(_system_prompt(), json.dumps(prompt, sort_keys=True))
            )
            if payload is None:
                raise ValueError("entry response is not JSON")
            score = int(payload["score"])
            decision = str(payload["decision"]).lower()
            rationale = str(payload.get("rationale", "")).strip()
            if score < 0 or score > 100 or decision not in {"buy", "skip"}:
                raise ValueError("entry response fields are invalid")
            return TokenEvaluation(score=score, decision=decision, rationale=rationale)
        except Exception as exc:
            LOGGER.warning("Entry AI response rejected: %s", exc)
            return TokenEvaluation()

    async def evaluate_exit(
        self, trade: TradeRecord, snapshot: TokenSnapshot, rules: str
    ) -> ExitDecision:
        """Return a validated hold/close result for an open paper trade."""

        prompt = {
            "task": "Evaluate whether this open paper position should close now.",
            "trade": _trade_prompt_payload(trade),
            "current_snapshot": snapshot.prompt_payload(),
            "latest_rules": rules or "No learned rules yet.",
            "response_schema": {
                "decision": "hold or close",
                "rationale": "short string",
            },
        }
        try:
            payload = _extract_json(
                await self.backend.chat(_system_prompt(), json.dumps(prompt, sort_keys=True))
            )
            if payload is None:
                raise ValueError("exit response is not JSON")
            decision = str(payload["decision"]).lower()
            rationale = str(payload.get("rationale", "")).strip()
            if decision not in {"hold", "close"}:
                raise ValueError("exit decision is invalid")
            return ExitDecision(decision=decision, rationale=rationale)
        except Exception as exc:
            LOGGER.warning("Exit AI response rejected: %s", exc)
            return ExitDecision()

    async def generate_reflection(
        self, evidence: ReflectionEvidence, current_settings: StrategySettings
    ) -> StrategyReflection:
        """Generate strict rules and bounded paper-strategy tuning."""

        prompt = {
            "task": (
                "Review paper-trading evidence. Learn from profitable trades, losing "
                "trades, correct skips, missed winners, filter rejections, recent "
                "analyses, and recurring error patterns."
            ),
            "evidence": {
                "profitable_trades": evidence.profitable_trades,
                "losing_trades": evidence.losing_trades,
                "recent_analyses": evidence.recent_analyses,
                "correct_skips": evidence.correct_skips,
                "missed_winners": evidence.missed_winners,
                "tokens_failed_base_filters": evidence.failed_filters,
                "recurring_errors": evidence.recurring_errors,
            },
            "response_schema": {
                "rules": ["strict rule 1", "strict rule 2", "strict rule 3"],
                "settings": {
                    "entry_score_threshold": "integer 60..95",
                    "tracker_poll_seconds": "number 10..300",
                    "base_trade_amount": "number 0.01..2.0",
                    "position_review_seconds": "number 15..300",
                    "reflection_time": "HH:MM 24-hour wall clock",
                },
                "settings_rationale": "short string",
            },
            "current_settings": current_settings.prompt_payload(),
            "tuning_guardrails": (
                "Keep changes conservative. Tune for paper learning quality and risk "
                "control, not trade frequency alone. Preserve current values when "
                "evidence is weak."
            ),
        }
        if not evidence.has_learning_data():
            return StrategyReflection()
        try:
            payload = _extract_json(
                await self.backend.chat(_system_prompt(), json.dumps(prompt, sort_keys=True))
            )
            rules = payload.get("rules") if payload else None
            if not isinstance(rules, list):
                raise ValueError("reflection rules are missing")
            normalized = [str(rule).strip() for rule in rules if str(rule).strip()]
            if len(normalized) != 3:
                raise ValueError("reflection must return exactly three rules")
            return StrategyReflection(
                rules=ReflectionRules(normalized),
                settings=_validated_settings(payload.get("settings")),
                settings_rationale=str(payload.get("settings_rationale", "")).strip(),
            )
        except Exception as exc:
            LOGGER.warning("Reflection AI response rejected: %s", exc)
            return StrategyReflection()

    async def generate_reflection_rules(
        self, evidence: ReflectionEvidence
    ) -> ReflectionRules:
        """Compatibility helper for tests and rule-only callers."""

        settings = StrategySettings(80, 30.0, 0.1, 45.0, "00:00")
        return (await self.generate_reflection(evidence, settings)).rules


def _system_prompt() -> str:
    return (
        "You are a cautious paper-trading risk evaluator for volatile Solana tokens. "
        "Return only one JSON object matching the requested schema. Treat missing "
        "metrics as unknown and never invent holder or developer data."
    )


def _operator_system_prompt() -> str:
    return (
        "You are the admin-only Hermes operator for this AI meme bot workspace. "
        "You may inspect and edit local project files and use terminal tools for "
        "the admin's explicit Telegram request. Keep changes focused, protect "
        "secrets in .env and private keys, do not enable REAL trading or send "
        "transactions unless the admin explicitly asks and the app supports it, "
        "and report changed files plus verification."
    )


def _trade_prompt_payload(trade: TradeRecord) -> Dict[str, Any]:
    return {
        "id": trade.id,
        "token_address": trade.token_address,
        "buy_price": trade.buy_price,
        "sell_price": trade.sell_price,
        "pnl": trade.pnl,
        "status": trade.status,
        "entry_amount_sol": trade.entry_amount_sol,
        "token_quantity": trade.token_quantity,
        "opened_at": trade.opened_at,
        "closed_at": trade.closed_at,
        "exit_reason": trade.exit_reason,
        "entry_snapshot_json": trade.entry_snapshot_json,
        "exit_snapshot_json": trade.exit_snapshot_json,
    }


def _validated_settings(payload: Any) -> Optional[StrategySettings]:
    """Reject settings unless every paper guardrail is satisfied."""

    if not isinstance(payload, dict):
        return None
    try:
        threshold = int(payload["entry_score_threshold"])
        tracker_poll = float(payload["tracker_poll_seconds"])
        trade_amount = float(payload["base_trade_amount"])
        review_seconds = float(payload["position_review_seconds"])
        reflection_time = str(payload["reflection_time"]).strip()
    except (KeyError, TypeError, ValueError):
        return None
    if not 60 <= threshold <= 95:
        return None
    if not 10 <= tracker_poll <= 300:
        return None
    if not 0.01 <= trade_amount <= 2.0:
        return None
    if not 15 <= review_seconds <= 300:
        return None
    if not _valid_hhmm(reflection_time):
        return None
    return StrategySettings(
        threshold, tracker_poll, trade_amount, review_seconds, reflection_time
    )


def _valid_hhmm(value: str) -> bool:
    try:
        hour_text, minute_text = value.split(":", 1)
        return len(hour_text) == 2 and len(minute_text) == 2 and (
            0 <= int(hour_text) <= 23 and 0 <= int(minute_text) <= 59
        )
    except (TypeError, ValueError):
        return False
