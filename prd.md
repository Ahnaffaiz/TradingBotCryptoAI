# Product Requirements Document (PRD) & System Architecture
**Project Name:** AI Meme Coin Trader (Pump Swap Sniper)
**Version:** 2.0 (Complete Technical Specification)
**Environment:** Local Development (macOS / Apple Silicon) -> Production (Ubuntu VPS)
**Target:** Codex / AI Coding Assistant
**Objective:** Generate a complete, interconnected Python project based on the specifications below.

---

## 1. Executive Summary
An autonomous AI-agent-based trading bot integrated with Telegram. It operates on the Solana network, specifically focusing on post-bonding curve tokens transitioning to Pump Swap. The system features a toggle-based architecture (Paper/Real Trading) and Continuous Learning capabilities through RAG (Retrieval-Augmented Generation) evaluation based on local transaction history data to consistently improve the win rate.

## 2. Technology Stack & Dependencies

### Core Engine
*   **Language:** Python 3.10+ (`asyncio` is mandatory for high-frequency processing).
*   **Blockchain SDK:** 
    *   `solana-py` (Standard RPC interactions).
    *   `solders` (High-speed transaction construction, keypair management, instruction serialization).
*   **Database:** `sqlite3` / `aiosqlite` (Zero-configuration local database for dummy balances and trade logs).
*   **Networking:** `aiohttp` (Asynchronous HTTP requests) and `websockets` (Real-time data streaming).

### AI & Agent Framework
*   **Agent Framework:** Hermes Agent (LLM orchestration and tool-calling protocol framework).
*   **LLM Provider:** Google Gemini API (Model: `gemini-1.5-flash` for millisecond-latency reasoning and large context window).
*   **Messaging Interface:** `python-telegram-bot` (v20+ utilizing `asyncio`).

## 3. Environment Variables (`.env`)
Codex must create a module to securely load the following variables using `python-dotenv`:
```env
# Application State
TRADING_MODE=PAPER # Options: PAPER or REAL
BASE_TRADE_AMOUNT=0.1 # Amount of SOL per real trade

# Blockchain & MEV
PRIVATE_KEY_BASE58=your_solana_wallet_private_key
HELIUS_RPC_URL=[https://mainnet.core.jito.wtf/your_api_key](https://mainnet.core.jito.wtf/your_api_key)
HELIUS_WSS_URL=wss://mainnet.core.jito.wtf/your_api_key
JITO_BLOCK_ENGINE_URL=[https://mainnet.block-engine.jito.wtf/api/v1/bundles](https://mainnet.block-engine.jito.wtf/api/v1/bundles)

# AI & APIs
GEMINI_API_KEY=your_google_gemini_api_key
TELEGRAM_BOT_TOKEN=your_botfather_token
DEXSCREENER_API_BASE=[https://api.dexscreener.com/latest/dex/tokens/](https://api.dexscreener.com/latest/dex/tokens/)

## 4. System Workflows 

### A. Detection & Analysis Workflow (Tracker -> AI)
1.  **Tracker:** Asynchronously polls the Dexscreener API for new tokens.
2.  **Base Filtering:** Drops tokens if `dexId` is not `pump`, if liquidity is < $10,000, or if token age < 1 minute (to avoid initial graduation dumps).
3.  **Context Injection:** Passes surviving token data (5m volume, dev holding %, top 10 holders %) to the Hermes AI agent.
4.  **AI Evaluation:** The Gemini LLM evaluates the metrics using a System Prompt enriched by daily RAG reflections. Returns a score (0-100).
5.  **Decision:** If the score > 80, the AI triggers the `execute_trade` tool.

### B. Execution Workflow (Toggle Paper/Real)
*   **If `TRADING_MODE=PAPER`:** 
    Bypasses blockchain. Calls `database.insert_trade()`, deducts the dummy balance, and records the entry price in the SQLite database.
*   **If `TRADING_MODE=REAL`:** 
    1. Extracts Keypair from `PRIVATE_KEY_BASE58`.
    2. Constructs a `Swap` instruction to the Pump Swap smart contract using `solders`.
    3. Constructs a Jito Tip `Transfer` instruction (0.001 SOL).
    4. Compiles into a `VersionedTransaction` and POSTs the JSON payload to the Jito Block Engine.
    5. Records the live trade into SQLite upon confirmation.

### C. Continuous Learning Workflow (Self-Reflection RAG)
1.  A local scheduler runs daily at 00:00.
2.  Fetches all `CLOSED` trades from the `trade_history` table (initial metrics + final PnL).
3.  Sends the data to Gemini API with the prompt: *"Analyze this trading history. Identify patterns causing losses and wins. Output 3 new strict trading rules for tomorrow's filter."*
4.  Saves the AI's response to the `ai_rules` table.
5.  Injects these rules into the Hermes System Prompt upon the next startup.

---

## 5. Directory Structure
```text
ai_meme_bot/
├── .env                  # Environment configurations
├── database.db           # SQLite database (Auto-generated)
├── main.py               # Application orchestrator
├── core/
│   ├── __init__.py
│   ├── database.py       # SQLite CRUD operations
│   ├── tracker.py        # Dexscreener API polling
│   └── execution.py      # Transaction engine (Paper & Real)
└── agent/
    ├── __init__.py
    ├── hermes_bot.py     # Telegram interface & AI Agent configuration
    ├── tools.py          # MCP / Function declarations for AI
    └── reflection.py     # RAG Daily evaluation engine

## 6. Module Specifications & Implementation Details

### `core/database.py` (State Management)
*   **Functions:** `init_db()`, `update_dummy_balance()`, `insert_trade()`, `update_trade_status()`, `get_recent_trades()`.
*   **Required Tables:**
    *   `wallet`: `id`, `balance` (Default: 100.0).
    *   `trade_history`: `id`, `token_address`, `buy_price`, `sell_price`, `pnl`, `status` (OPEN/CLOSED), `timestamp`.
    *   `ai_rules`: `id`, `rules_text`, `date`.

### `core/tracker.py` (Data Ingestion)
*   **Logic:** Use `aiohttp` to fetch `https://api.dexscreener.com/latest/dex/tokens/{address}`. Apply JSON filters (`dexId == 'pump'`). Return structured dictionaries for AI consumption.

### `core/execution.py` (Transaction Engine)
*   **Logic:** Contains the `execute_trade(token_address, action, amount)` function. Must implement strict `if/else` branching based on `TRADING_MODE`. Real execution must utilize `solana-py` and `solders` for transaction serialization.

### `agent/tools.py` (AI Tools/MCP)
*   **Logic:** Wrap core functions with comprehensive Python type-hinting and docstrings so the Gemini model understands how to invoke them. 
*   **Tools:** `get_current_balance()`, `analyze_token(token_address)`, `trigger_buy(token_address)`.

### `agent/reflection.py` (RAG Engine)
*   **Logic:** `generate_daily_rules()` function that queries closed trades, interfaces with the `google-generativeai` SDK, and updates the `ai_rules` table.

### `agent/hermes_bot.py` (Telegram & AI Brain)
*   **Logic:** Uses `python-telegram-bot` (`ApplicationBuilder`). Implements handlers: `/start`, `/status`, `/auto_on`, `/auto_off`.
*   **System Prompt Injection:** Dynamically fetches rules from `ai_rules` and injects them into the Gemini model's system instructions.

### `main.py` (Orchestrator)
*   **Logic:** The asynchronous entry point. Calls `init_db()`, creates background tasks for `tracker.py`, starts Telegram polling, and schedules the `reflection.py` cron job.

---

## 7. Installation & Setup Guide

**Step 1: Virtual Environment**
```bash
mkdir ai_meme_bot && cd ai_meme_bot
python3 -m venv venv
source venv/bin/activate

**Step 2: Core Dependencies**
pip install solana solders aiohttp websockets python-dotenv aiosqlite python-telegram-bot google-generativeai pydantic


## 8. Development Phases & Roadmap

*   **Phase 1: Database & Paper Trading Core**
    Implement `database.py` and the simulated branch of `execution.py`. Validate dummy balance deductions and PnL calculations locally.
*   **Phase 2: Blockchain Integration**
    Implement the `REAL` branch in `execution.py`. Construct Solana Keypairs, Anchor instruction payloads, and Jito Bundle POST requests.
*   **Phase 3: Tracker & AI Integration**
    Build `tracker.py` to ingest Dexscreener data. Configure `hermes_bot.py` to route this data to Gemini and handle Telegram `/commands`. Connect AI tool-calling to `execution.py`.
*   **Phase 4: Self-Reflection Engine**
    Build `reflection.py` to run automated nightly analysis on the SQLite database and update the AI's prompt rules.
*   **Phase 5: VPS Deployment**
    Transfer codebase to an Ubuntu VPS (US East/Tokyo). Secure with UFW. Run via `systemd` or `pm2` for 24/7 uptime.

---

## 9. Testing Tools & Methodologies

*   **Unit Testing (`pytest`):** Verify PnL math in `database.py` and validate transaction payload structures offline without broadcasting.
*   **API Mocking:** Feed hardcoded rugpull JSON data into the AI to ensure the prompt correctly rejects malicious tokens.
*   **Dry Run Execution:** Utilize Helius RPC `simulateTransaction` endpoint to validate Jito Bundles before executing real SOL transfers.

---

## 10. Codex Strict Instructions

1.  Apply `async/await` patterns across all networking and database modules.
2.  Implement robust `try/except` blocks inside `execution.py` to prevent application crashes during RPC timeouts or RPC rate limits.
3.  Generate the directory tree and code exactly as specified above without deviating from the `TRADING_MODE` toggle architecture.