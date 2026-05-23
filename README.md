# AI Meme Bot

`ai_meme_bot` is a paper-first PumpSwap candidate trader. It discovers Solana token
profiles through Dexscreener, enriches PumpSwap pairs and top-holder concentration,
asks an embedded Hermes agent for structured entry and exit decisions, keeps paper
trades in SQLite, and exposes Telegram controls.

The v1 `REAL` branch is intentionally closed. It keeps the trading mode boundary in
place but never broadcasts a Solana transaction until the PumpSwap and Jito path is
implemented and verified separately.

## Setup

Use Python 3.11 or newer. The current Hermes Agent package requires Python 3.11.
On this macOS workspace, `python3` may still be Apple's older Python, so call the
newer interpreter explicitly:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
cp ai_meme_bot/.env.example .env
```

Or use the bootstrap runner:

```bash
chmod +x run.sh
./run.sh
```

The first `./run.sh` creates `.venv` when needed, installs the editable app, and
creates `.env` if it is missing. After the required `.env` values are filled, the
same command starts the bot.

Set an OpenAI-compatible AI endpoint in `.env`. `AI_BASE_URL` should include the
provider's `/v1` prefix when that endpoint expects it.

```env
AI_PROVIDER=custom
AI_BASE_URL=https://provider.example/v1
AI_API_KEY=replace-me
AI_MODEL=replace-me
TELEGRAM_BOT_TOKEN=replace-me
```

Initialize and run the bot:

```bash
python -m ai_meme_bot.main
```

Telegram commands:

- `/start` describes the paper bot and opens the menu.
- `/menu` reopens the button menu.
- `/whoami` shows the Telegram user ID for admin-only features.
- `/status` shows mode, configured AI model, paper balance, and trades.
- `/auto_on` allows new AI-approved paper entries.
- `/auto_off` stops new entries while open trades remain under review.
- `/notify_on` enables Telegram reports.
- `/notify_off` mutes Telegram reports.
- `/hermes <task>` runs the opt-in admin workspace operator.

The last Telegram chat that sends a command becomes the paper report destination.
The menu provides buttons for status, auto entries, and notification controls.
While reports are enabled, the bot sends entry analysis before paper buys, paper
buy outcomes, exit analysis for open positions, paper sell outcomes, daily
reflection rules, and pipeline errors.

`/hermes` is a powerful embedded Hermes operator for the same Telegram bot. It is
disabled by default because it can inspect and edit project files and use local
terminal tools. Enable it only for admin Telegram user IDs:

```env
HERMES_OPERATOR_ENABLED=1
TELEGRAM_ADMIN_USER_IDS=123456789
```

Run this bot's `/whoami` before filling the ID. Do not run a separate Hermes
Telegram gateway on the same `TELEGRAM_BOT_TOKEN`; this app already polls that
bot token.

## Configuration

See [`ai_meme_bot/.env.example`](ai_meme_bot/.env.example). Important defaults:

- `TRADING_MODE=PAPER`
- `BASE_TRADE_AMOUNT=0.1`
- `MIN_LIQUIDITY_USD=10000`
- `MIN_PAIR_AGE_SECONDS=60`
- `ENTRY_SCORE_THRESHOLD=80`

Paper trading does not require a Pump.fun RPC, wallet key, or Jito setup. Dexscreener
provides discovery and pair pricing. `HELIUS_RPC_URL` or `SOLANA_RPC_URL` is optional
in paper mode and enables Solana top-holder concentration enrichment. Without RPC
data the tracker leaves `top_holder_share_pct` empty and the AI prompt must treat
that field as unknown.

The AI candidate payload also includes Dexscreener market trend fields: 5m and 1h
price change plus 5m buy/sell transaction counts. GeckoTerminal's Solana
trending-pools list adds a second research signal: whether the mint or current
pool is on that list and its rank. Optional X recent-search enrichment can add
mint-address mention count, unique author count, and a shallow risk/hype language
hint:

```env
X_BEARER_TOKEN=your-x-api-bearer-token
X_RECENT_SEARCH_URL=https://api.x.com/2/tweets/search/recent
X_SEARCH_MINUTES=30
GECKOTERMINAL_TRENDING_URL=https://api.geckoterminal.com/api/v2/networks/solana/trending_pools
```

Without `X_BEARER_TOKEN`, X trend fields stay unknown and the bot continues with
Dexscreener, GeckoTerminal, and RPC metrics.

`PRIVATE_KEY_BASE58` and `JITO_BLOCK_ENGINE_URL` are reserved for the future live
trading branch. V1 refuses `REAL` execution even if they are configured.

## Learning History

The bot stores learning evidence in SQLite:

- `token_analysis_history` keeps AI scores, decisions, rationales, and entry metrics.
- `token_outcome_snapshots` samples analyzed tokens after 5m, 15m, and 1h so
  reflection can distinguish correct skips from missed winners.
- `activity_log` records filter rejections, analysis actions, paper buys/sells,
  outcome captures, reflection runs, and runtime errors.

Nightly reflection learns from profitable and losing closed trades, recent analyses,
correct skips, missed winners, tokens rejected by base filters, and recurring error
activity. The AI writes three strict rules back into the active prompt and may tune
paper-mode runtime settings within app limits: entry score threshold, discovery poll
cadence, paper trade size, exit review cadence, and next reflection wall-clock time.
Invalid or out-of-range tuning output is ignored.

## Tests

```bash
pytest
```
# TradingBotCryptoAI
