"""Real-time price feed helpers for open-position protection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from typing import Any, AsyncIterator, Iterable, Optional
from urllib.parse import quote

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceTick:
    """One parsed real-time price update."""

    address: str
    price_usd: float
    raw_payload: dict[str, Any]


class BirdeyePriceFeed:
    """Birdeye WebSocket SUBSCRIBE_PRICE client for Solana pair prices."""

    def __init__(
        self,
        api_key: str,
        ws_url: str = "wss://public-api.birdeye.so/socket/solana",
        reconnect_base_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.ws_url = ws_url.rstrip("?")
        self.reconnect_base_seconds = reconnect_base_seconds
        self.reconnect_max_seconds = reconnect_max_seconds

    @property
    def websocket_url(self) -> str:
        """Return the authenticated Birdeye WebSocket URL."""

        separator = "&" if "?" in self.ws_url else "?"
        return "{0}{1}x-api-key={2}".format(
            self.ws_url, separator, quote(self.api_key, safe="")
        )

    @staticmethod
    def subscription_message(addresses: Iterable[str]) -> dict[str, Any]:
        """Build a SUBSCRIBE_PRICE message for one or more pair addresses."""

        clean_addresses = [address for address in addresses if address]
        if len(clean_addresses) == 1:
            return {
                "type": "SUBSCRIBE_PRICE",
                "data": {
                    "queryType": "simple",
                    "chartType": "1m",
                    "address": clean_addresses[0],
                    "currency": "pair",
                },
            }
        query = " OR ".join(
            "(address = {0} AND chartType = 1m AND currency = pair)".format(address)
            for address in clean_addresses[:100]
        )
        return {
            "type": "SUBSCRIBE_PRICE",
            "data": {"queryType": "complex", "query": query},
        }

    @staticmethod
    def parse_message(message: str | bytes) -> Optional[PriceTick]:
        """Parse a Birdeye PRICE_DATA message."""

        if isinstance(message, bytes):
            message = message.decode("utf-8")
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("type") != "PRICE_DATA":
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        address = str(data.get("address") or "").strip()
        try:
            price = float(data.get("c"))
        except (TypeError, ValueError):
            return None
        if not address or price <= 0:
            return None
        return PriceTick(address=address, price_usd=price, raw_payload=payload)

    async def stream_prices(self, addresses: Iterable[str]) -> AsyncIterator[PriceTick]:
        """Yield live pair-price ticks and reconnect on transient failures."""

        clean_addresses = [address for address in addresses if address][:100]
        if not clean_addresses:
            return
        delay = self.reconnect_base_seconds
        while True:
            try:
                async for tick in self._connect_once(clean_addresses):
                    delay = self.reconnect_base_seconds
                    yield tick
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("Birdeye price feed disconnected: %s", exc)
                await asyncio.sleep(delay)
                delay = min(self.reconnect_max_seconds, delay * 2)

    async def _connect_once(self, addresses: list[str]) -> AsyncIterator[PriceTick]:
        import websockets

        async with websockets.connect(
            self.websocket_url,
            subprotocols=["echo-protocol"],
            origin="https://birdeye.so",
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            await websocket.send(json.dumps(self.subscription_message(addresses)))
            async for message in websocket:
                tick = self.parse_message(message)
                if tick is not None:
                    yield tick
