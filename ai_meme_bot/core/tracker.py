"""Dexscreener discovery and Solana holder enrichment."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
)

import aiohttp

from ai_meme_bot.config import AppConfig
from ai_meme_bot.models import FilteredToken, TokenSnapshot, utc_now


class TrackerError(RuntimeError):
    """Raised when a market payload cannot produce a usable snapshot."""


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_top_holder_share(
    largest_accounts: Dict[str, Any], token_supply: Dict[str, Any]
) -> Optional[float]:
    """Return raw top-10 token-account share as a percentage."""

    accounts = largest_accounts.get("result", {}).get("value", [])
    supply_value = token_supply.get("result", {}).get("value", {})
    supply_ui_amount = _as_float(
        supply_value.get("uiAmountString", supply_value.get("uiAmount"))
    )
    if not accounts or supply_ui_amount <= 0:
        return None
    top_ten = accounts[:10]
    total = sum(
        _as_float(account.get("uiAmountString", account.get("uiAmount")))
        for account in top_ten
    )
    return round((total / supply_ui_amount) * 100, 4)


class TokenTracker:
    """Discovers Solana profiles and emits filtered PumpSwap snapshots."""

    def __init__(
        self,
        config: AppConfig,
        session: Optional[aiohttp.ClientSession] = None,
        time_fn=time.time,
        filtered_callback: Optional[Callable[[FilteredToken], Awaitable[None]]] = None,
        poll_seconds_callback: Optional[Callable[[], Awaitable[float]]] = None,
    ) -> None:
        self.config = config
        self._session = session
        self._owned_session: Optional[aiohttp.ClientSession] = None
        self._time_fn = time_fn
        self._filtered_callback = filtered_callback
        self._poll_seconds_callback = poll_seconds_callback
        self._logged_filter_keys = set()

    async def __aenter__(self) -> "TokenTracker":
        if self._session is None:
            self._owned_session = aiohttp.ClientSession()
            self._session = self._owned_session
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._owned_session is not None:
            await self._owned_session.close()
            self._owned_session = None
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise TrackerError("TokenTracker must be entered before network use.")
        return self._session

    async def fetch_profiles(self) -> List[str]:
        """Fetch current Solana token addresses from Dexscreener profiles."""

        try:
            async with self.session.get(self.config.dexscreener_profile_url) as response:
                response.raise_for_status()
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return []
        if not isinstance(payload, list):
            return []
        addresses = []
        seen = set()
        for profile in payload:
            if not isinstance(profile, dict) or profile.get("chainId") != "solana":
                continue
            address = profile.get("tokenAddress")
            if isinstance(address, str) and address and address not in seen:
                seen.add(address)
                addresses.append(address)
        return addresses

    async def fetch_pairs(self, token_address: str) -> List[Dict[str, Any]]:
        """Fetch Dexscreener pair data for one Solana token address."""

        return await self.fetch_pairs_for_tokens([token_address])

    async def fetch_pairs_for_tokens(
        self, token_addresses: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """Fetch Dexscreener pair data for up to 30 Solana token addresses."""

        if not token_addresses:
            return []
        addresses = ",".join(token_addresses[:30])
        url = "{0}/solana/{1}".format(self.config.dexscreener_token_url, addresses)
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return []
        if isinstance(payload, list):
            return [pair for pair in payload if isinstance(pair, dict)]
        if isinstance(payload, dict):
            pairs = payload.get("pairs", [])
            return [pair for pair in pairs if isinstance(pair, dict)]
        return []

    async def snapshot_for_token(
        self, token_address: str, apply_filters: bool = True
    ) -> Optional[TokenSnapshot]:
        """Build the best PumpSwap pair snapshot for a token."""

        pairs = await self.fetch_pairs(token_address)
        pair = self._select_pumpswap_pair(pairs, apply_filters=apply_filters)
        if pair is None:
            return None
        snapshot = self.snapshot_from_pair(token_address, pair)
        snapshot.top_holder_share_pct, x_trend, trending_pools = await asyncio.gather(
            self.fetch_top_holder_share(token_address),
            self.fetch_x_trend(token_address),
            self.fetch_geckoterminal_trending_pools(),
        )
        self._apply_x_trend(snapshot, x_trend)
        self._apply_geckoterminal_trend(snapshot, trending_pools)
        return snapshot

    async def snapshots_for_tokens(
        self, token_addresses: Sequence[str], apply_filters: bool = True
    ) -> List[TokenSnapshot]:
        """Batch-enrich token addresses into eligible PumpSwap snapshots."""

        grouped_pairs = {token_address: [] for token_address in token_addresses}
        for pair in await self.fetch_pairs_for_tokens(token_addresses):
            for token_key in ("baseToken", "quoteToken"):
                address = (pair.get(token_key) or {}).get("address")
                if address in grouped_pairs:
                    grouped_pairs[address].append(pair)

        snapshots = []
        for token_address in token_addresses:
            pair = self._select_pumpswap_pair(
                grouped_pairs[token_address], apply_filters=apply_filters
            )
            if pair is None:
                if apply_filters:
                    await self._record_filtered_token(
                        self._filter_rejection(token_address, grouped_pairs[token_address])
                    )
                continue
            try:
                snapshots.append(self.snapshot_from_pair(token_address, pair))
            except TrackerError:
                continue
        shares = await asyncio.gather(
            *(self.fetch_top_holder_share(snapshot.token_address) for snapshot in snapshots)
        )
        x_trends = await asyncio.gather(
            *(self.fetch_x_trend(snapshot.token_address) for snapshot in snapshots)
        )
        trending_pools = await self.fetch_geckoterminal_trending_pools()
        for snapshot, share, x_trend in zip(snapshots, shares, x_trends):
            snapshot.top_holder_share_pct = share
            self._apply_x_trend(snapshot, x_trend)
            self._apply_geckoterminal_trend(snapshot, trending_pools)
        return snapshots

    async def _record_filtered_token(self, filtered: FilteredToken) -> None:
        """Emit each rejection reason once per tracker run."""

        filter_key = (filtered.token_address, filtered.reason)
        if filter_key in self._logged_filter_keys:
            return
        self._logged_filter_keys.add(filter_key)
        if self._filtered_callback is not None:
            try:
                await self._filtered_callback(filtered)
            except Exception:
                return

    def _filter_rejection(
        self, token_address: str, pairs: Sequence[Dict[str, Any]]
    ) -> FilteredToken:
        pumpswap_pairs = [pair for pair in pairs if pair.get("dexId") == "pumpswap"]
        if not pumpswap_pairs:
            return FilteredToken(token_address, "no PumpSwap pair in Dexscreener batch")

        pair = max(
            pumpswap_pairs,
            key=lambda item: _as_float((item.get("liquidity") or {}).get("usd")),
        )
        reasons = []
        liquidity = _as_float((pair.get("liquidity") or {}).get("usd"))
        age = self._pair_age_seconds(pair.get("pairCreatedAt"))
        if liquidity < self.config.min_liquidity_usd:
            reasons.append("liquidity ${0:,.2f} below ${1:,.2f}".format(
                liquidity, self.config.min_liquidity_usd
            ))
        if age < self.config.min_pair_age_seconds:
            reasons.append(
                "pair age {0:.1f}s below {1}s".format(
                    age, self.config.min_pair_age_seconds
                )
            )
        if _as_float(pair.get("priceUsd")) <= 0:
            reasons.append("missing positive price")
        return FilteredToken(
            token_address=token_address,
            pair_address=str(pair.get("pairAddress", "")) or None,
            reason="; ".join(reasons) or "pair failed base filters",
            payload={
                "dex_id": pair.get("dexId"),
                "liquidity_usd": liquidity,
                "pair_age_seconds": age,
                "price_usd": _as_float(pair.get("priceUsd")),
            },
        )

    def snapshot_from_pair(
        self, token_address: str, pair: Dict[str, Any]
    ) -> TokenSnapshot:
        """Normalize one Dexscreener pair record."""

        pair_created_at = pair.get("pairCreatedAt")
        pair_age = self._pair_age_seconds(pair_created_at)
        price = _as_float(pair.get("priceUsd"))
        liquidity = _as_float((pair.get("liquidity") or {}).get("usd"))
        if price <= 0:
            raise TrackerError("Dexscreener pair is missing a positive price.")
        return TokenSnapshot(
            token_address=token_address,
            pair_address=str(pair.get("pairAddress", "")),
            price_usd=price,
            liquidity_usd=liquidity,
            volume_5m_usd=_as_float((pair.get("volume") or {}).get("m5")),
            pair_age_seconds=pair_age,
            dex_id=str(pair.get("dexId", "")),
            pair_created_at_ms=int(pair_created_at) if pair_created_at else None,
            price_change_5m_pct=_nullable_float((pair.get("priceChange") or {}).get("m5")),
            price_change_1h_pct=_nullable_float((pair.get("priceChange") or {}).get("h1")),
            buys_5m=_nullable_int(((pair.get("txns") or {}).get("m5") or {}).get("buys")),
            sells_5m=_nullable_int(((pair.get("txns") or {}).get("m5") or {}).get("sells")),
            raw_context=pair,
        )

    def _select_pumpswap_pair(
        self, pairs: Sequence[Dict[str, Any]], apply_filters: bool = True
    ) -> Optional[Dict[str, Any]]:
        candidates = []
        for pair in pairs:
            if pair.get("dexId") != "pumpswap":
                continue
            liquidity = _as_float((pair.get("liquidity") or {}).get("usd"))
            age = self._pair_age_seconds(pair.get("pairCreatedAt"))
            if apply_filters and (
                liquidity < self.config.min_liquidity_usd
                or age < self.config.min_pair_age_seconds
            ):
                continue
            if _as_float(pair.get("priceUsd")) <= 0:
                continue
            candidates.append(pair)
        if not candidates:
            return None
        return max(candidates, key=lambda pair: _as_float((pair.get("liquidity") or {}).get("usd")))

    def _pair_age_seconds(self, pair_created_at_ms: Any) -> float:
        created_at_ms = _as_float(pair_created_at_ms)
        if created_at_ms <= 0:
            return 0.0
        return max(0.0, self._time_fn() - (created_at_ms / 1000.0))

    async def fetch_top_holder_share(self, token_address: str) -> Optional[float]:
        """Query raw largest token accounts and token supply over Solana RPC."""

        if not self.config.solana_rpc_url:
            return None
        largest, supply = await asyncio.gather(
            self._rpc("getTokenLargestAccounts", [token_address]),
            self._rpc("getTokenSupply", [token_address]),
        )
        return calculate_top_holder_share(largest or {}, supply or {})

    async def fetch_x_trend(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch recent X mention trend for a mint address when configured."""

        if not self.config.x_bearer_token:
            return None
        params = {
            "query": '"{0}" -is:retweet'.format(token_address),
            "max_results": "10",
            "tweet.fields": "author_id",
            "start_time": (
                utc_now() - timedelta(minutes=max(10, self.config.x_search_minutes))
            ).isoformat().replace("+00:00", "Z"),
        }
        headers = {"Authorization": "Bearer {0}".format(self.config.x_bearer_token)}
        try:
            async with self.session.get(
                self.config.x_recent_search_url, params=params, headers=headers
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        posts = payload.get("data") or []
        if not isinstance(posts, list):
            posts = []
        authors = {
            str(post.get("author_id"))
            for post in posts
            if isinstance(post, dict) and post.get("author_id")
        }
        return {
            "mentions": len(posts),
            "author_count": len(authors),
            "sentiment_hint": _x_sentiment_hint(posts),
        }

    @staticmethod
    def _apply_x_trend(snapshot: TokenSnapshot, trend: Optional[Dict[str, Any]]) -> None:
        if trend is None:
            return
        snapshot.x_recent_mentions = _nullable_int(trend.get("mentions"))
        snapshot.x_recent_author_count = _nullable_int(trend.get("author_count"))
        sentiment = trend.get("sentiment_hint")
        snapshot.x_sentiment_hint = str(sentiment) if sentiment else None

    async def fetch_geckoterminal_trending_pools(
        self,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetch current Solana trending pools keyed by token and pool address."""

        if not self.config.geckoterminal_trending_url:
            return None
        try:
            async with self.session.get(
                self.config.geckoterminal_trending_url,
                params={"include": "base_token", "page": "1"},
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            return None

        pools: Dict[str, Dict[str, Any]] = {}
        for rank, pool in enumerate(payload["data"], start=1):
            if not isinstance(pool, dict):
                continue
            attributes = pool.get("attributes") or {}
            relationships = pool.get("relationships") or {}
            base_data = (relationships.get("base_token") or {}).get("data") or {}
            pool_address = attributes.get("address")
            token_address = _geckoterminal_solana_address(base_data.get("id"))
            trend = {"rank": rank, "pool_address": pool_address}
            if isinstance(token_address, str) and token_address:
                pools["token:{0}".format(token_address)] = trend
            if isinstance(pool_address, str) and pool_address:
                pools["pool:{0}".format(pool_address)] = trend
        return pools

    @staticmethod
    def _apply_geckoterminal_trend(
        snapshot: TokenSnapshot, trending_pools: Optional[Dict[str, Dict[str, Any]]]
    ) -> None:
        if trending_pools is None:
            return
        match = trending_pools.get("token:{0}".format(snapshot.token_address))
        match = match or trending_pools.get("pool:{0}".format(snapshot.pair_address))
        snapshot.geckoterminal_trending = match is not None
        snapshot.geckoterminal_trending_rank = (
            _nullable_int(match.get("rank")) if match else None
        )

    async def _rpc(self, method: str, params: Iterable[Any]) -> Optional[Dict[str, Any]]:
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)}
        try:
            async with self.session.post(self.config.solana_rpc_url, json=body) as response:
                response.raise_for_status()
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    async def discover(self) -> AsyncIterator[TokenSnapshot]:
        """Continuously yield new filtered PumpSwap snapshots."""

        seen = set()
        while True:
            addresses = [
                token_address
                for token_address in await self.fetch_profiles()
                if token_address not in seen
            ]
            for chunk in _chunks(addresses, 30):
                for snapshot in await self.snapshots_for_tokens(chunk):
                    seen.add(snapshot.token_address)
                    yield snapshot
            poll_seconds = self.config.tracker_poll_seconds
            if self._poll_seconds_callback is not None:
                poll_seconds = await self._poll_seconds_callback()
            await asyncio.sleep(poll_seconds)


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _nullable_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _as_float(value)


def _nullable_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _x_sentiment_hint(posts: Sequence[Dict[str, Any]]) -> str:
    """Return a shallow safety hint without treating X posts as trusted facts."""

    text = " ".join(str(post.get("text", "")).lower() for post in posts if isinstance(post, dict))
    risk_hits = sum(word in text for word in ("rug", "scam", "dump", "honeypot"))
    hype_hits = sum(word in text for word in ("moon", "send", "ape", "gem", "pump"))
    if risk_hits > hype_hits:
        return "risk-language"
    if hype_hits > risk_hits:
        return "hype-language"
    return "mixed-or-neutral"


def _geckoterminal_solana_address(resource_id: Any) -> Optional[str]:
    """Read GeckoTerminal JSON:API Solana resource ids like `solana_<address>`."""

    if not isinstance(resource_id, str) or not resource_id.startswith("solana_"):
        return None
    return resource_id[len("solana_") :]
