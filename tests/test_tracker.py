import pytest

from ai_meme_bot.core.tracker import TokenTracker, calculate_top_holder_share
from tests.helpers import make_config


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    async def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = responses

    def get(self, url, **_kwargs):
        return FakeResponse(self.responses.get(url, {}))


class RetryTracker(TokenTracker):
    def __init__(self, config):
        super().__init__(config, session=FakeSession({}))
        self.snapshot_attempts = 0

    async def fetch_profiles(self):
        return ["mint-late"]

    async def snapshots_for_tokens(self, token_addresses, apply_filters=True):
        self.snapshot_attempts += 1
        if self.snapshot_attempts == 1:
            return []
        from tests.helpers import make_snapshot

        return [make_snapshot(token=token_addresses[0])]


class CaptureFilters:
    def __init__(self):
        self.items = []

    async def __call__(self, filtered):
        self.items.append(filtered)


@pytest.mark.asyncio
async def test_profiles_keep_unique_solana_addresses(tmp_path):
    config = make_config(tmp_path / "trades.db")
    session = FakeSession(
        {
            config.dexscreener_profile_url: [
                {"chainId": "solana", "tokenAddress": "mint-a"},
                {"chainId": "ethereum", "tokenAddress": "0x1"},
                {"chainId": "solana", "tokenAddress": "mint-a"},
            ]
        }
    )

    assert await TokenTracker(config, session=session).fetch_profiles() == ["mint-a"]


@pytest.mark.asyncio
async def test_batch_snapshots_group_pair_data_by_token(tmp_path):
    config = make_config(tmp_path / "trades.db")
    session = FakeSession(
        {
            "{0}/solana/mint-a,mint-b".format(config.dexscreener_token_url): [
                {
                    "dexId": "pumpswap",
                    "pairAddress": "pair-a",
                    "baseToken": {"address": "mint-a"},
                    "priceUsd": "0.5",
                    "liquidity": {"usd": 20000},
                    "volume": {"m5": 40},
                    "pairCreatedAt": 100000,
                },
                {
                    "dexId": "pumpswap",
                    "pairAddress": "pair-b",
                    "baseToken": {"address": "mint-b"},
                    "priceUsd": "1.5",
                    "liquidity": {"usd": 22000},
                    "volume": {"m5": 90},
                    "pairCreatedAt": 100000,
                },
            ]
        }
    )
    tracker = TokenTracker(config, session=session, time_fn=lambda: 200.0)

    snapshots = await tracker.snapshots_for_tokens(["mint-a", "mint-b"])

    assert [snapshot.pair_address for snapshot in snapshots] == ["pair-a", "pair-b"]


@pytest.mark.asyncio
async def test_batch_snapshots_records_base_filter_rejections(tmp_path):
    config = make_config(tmp_path / "trades.db")
    filters = CaptureFilters()
    session = FakeSession(
        {
            "{0}/solana/thin-mint".format(config.dexscreener_token_url): [
                {
                    "dexId": "pumpswap",
                    "pairAddress": "thin-pair",
                    "baseToken": {"address": "thin-mint"},
                    "priceUsd": "0.01",
                    "liquidity": {"usd": 900},
                    "pairCreatedAt": 100000,
                }
            ]
        }
    )
    tracker = TokenTracker(
        config, session=session, time_fn=lambda: 200.0, filtered_callback=filters
    )

    snapshots = await tracker.snapshots_for_tokens(["thin-mint"])

    assert snapshots == []
    assert filters.items[0].token_address == "thin-mint"
    assert "liquidity" in filters.items[0].reason


def test_pumpswap_filters_and_normalizes_snapshot(tmp_path):
    config = make_config(tmp_path / "trades.db")
    tracker = TokenTracker(config, session=FakeSession({}), time_fn=lambda: 200.0)
    pairs = [
        {
            "dexId": "pumpfun",
            "pairAddress": "pre-graduation",
            "priceUsd": "1",
            "liquidity": {"usd": 99999},
            "pairCreatedAt": 1,
        },
        {
            "dexId": "pumpswap",
            "pairAddress": "too-new",
            "priceUsd": "1",
            "liquidity": {"usd": 20000},
            "pairCreatedAt": 199000,
        },
        {
            "dexId": "pumpswap",
            "pairAddress": "eligible",
            "priceUsd": "0.25",
                    "liquidity": {"usd": 25000},
                    "volume": {"m5": 700},
                    "priceChange": {"m5": 12.5, "h1": -4.2},
                    "txns": {"m5": {"buys": 18, "sells": 9}},
                    "pairCreatedAt": 100000,
        },
    ]

    pair = tracker._select_pumpswap_pair(pairs)
    snapshot = tracker.snapshot_from_pair("mint-a", pair)

    assert snapshot.pair_address == "eligible"
    assert snapshot.liquidity_usd == 25000
    assert snapshot.volume_5m_usd == 700
    assert snapshot.pair_age_seconds == 100
    assert snapshot.price_change_5m_pct == 12.5
    assert snapshot.price_change_1h_pct == -4.2
    assert snapshot.buys_5m == 18
    assert snapshot.sells_5m == 9


def test_top_holder_share_uses_largest_accounts_and_supply():
    share = calculate_top_holder_share(
        {
            "result": {
                "value": [
                    {"uiAmountString": "30"},
                    {"uiAmountString": "20"},
                    {"uiAmountString": "10"},
                ]
            }
        },
        {"result": {"value": {"uiAmountString": "100"}}},
    )

    assert share == 60.0


@pytest.mark.asyncio
async def test_discovery_retries_profiles_that_are_not_eligible_yet(tmp_path):
    config = make_config(tmp_path / "trades.db", tracker_poll_seconds=0)
    tracker = RetryTracker(config)
    discovery = tracker.discover()

    snapshot = await discovery.__anext__()
    await discovery.aclose()

    assert snapshot.token_address == "mint-late"
    assert tracker.snapshot_attempts == 2


@pytest.mark.asyncio
async def test_x_trend_uses_recent_mentions_when_configured(tmp_path):
    config = make_config(
        tmp_path / "trades.db",
        x_bearer_token="x-token",
        x_recent_search_url="https://x.example/recent",
    )
    tracker = TokenTracker(
        config,
        session=FakeSession(
            {
                "https://x.example/recent": {
                    "data": [
                        {"author_id": "a", "text": "mint looks like a gem"},
                        {"author_id": "b", "text": "ape this mint to moon"},
                    ]
                }
            }
        ),
    )

    trend = await tracker.fetch_x_trend("mint-a")

    assert trend == {
        "mentions": 2,
        "author_count": 2,
        "sentiment_hint": "hype-language",
    }


@pytest.mark.asyncio
async def test_geckoterminal_trend_matches_solana_base_token(tmp_path):
    config = make_config(tmp_path / "trades.db")
    tracker = TokenTracker(
        config,
        session=FakeSession(
            {
                config.geckoterminal_trending_url: {
                    "data": [
                        {
                            "attributes": {"address": "pool-a"},
                            "relationships": {
                                "base_token": {
                                    "data": {"id": "solana_mint-a", "type": "token"}
                                }
                            },
                        }
                    ]
                }
            }
        ),
    )

    pools = await tracker.fetch_geckoterminal_trending_pools()
    snapshot = tracker.snapshot_from_pair(
        "mint-a",
        {
            "dexId": "pumpswap",
            "pairAddress": "pair-a",
            "priceUsd": "0.1",
            "liquidity": {"usd": 20000},
            "pairCreatedAt": 1,
        },
    )
    tracker._apply_geckoterminal_trend(snapshot, pools)

    assert snapshot.geckoterminal_trending is True
    assert snapshot.geckoterminal_trending_rank == 1


@pytest.mark.asyncio
async def test_scout_discovery_uses_trending_pools_and_filters(tmp_path):
    config = make_config(
        tmp_path / "trades.db",
        launch_enabled=False,
        scout_enabled=True,
        tracker_poll_seconds=0,
    )
    session = FakeSession(
        {
            config.geckoterminal_trending_url: {
                "data": [
                    {
                        "attributes": {"address": "pool-a"},
                        "relationships": {
                            "base_token": {
                                "data": {"id": "solana_mint-a", "type": "token"}
                            }
                        },
                    }
                ]
            },
            "{0}/solana/mint-a".format(config.dexscreener_token_url): [
                {
                    "dexId": "pumpswap",
                    "pairAddress": "pair-a",
                    "baseToken": {"address": "mint-a"},
                    "priceUsd": "0.5",
                    "liquidity": {"usd": 20000},
                    "volume": {"m5": 900},
                    "priceChange": {"m5": 4, "h1": -12},
                    "txns": {"m5": {"buys": 21, "sells": 10}},
                    "pairCreatedAt": 100000,
                }
            ],
        }
    )
    tracker = TokenTracker(config, session=session, time_fn=lambda: 200.0)
    discovery = tracker.discover()

    snapshot = await discovery.__anext__()
    await discovery.aclose()

    assert snapshot.token_address == "mint-a"
    assert snapshot.strategy == "scout"
