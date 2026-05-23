#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"

find_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  for candidate in python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  return 1
}

if [[ ! -x ".venv/bin/python" ]]; then
  if ! PYTHON_BIN="$(find_python)"; then
    cat >&2 <<'EOF'
Python 3.11+ is required.

On macOS with Homebrew, install one and run again:
  brew install python@3.12
EOF
    exit 1
  fi

  echo "Creating .venv with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
DEPS_STAMP="$ROOT_DIR/.venv/.ai_meme_bot_deps_ready"

if ! "$VENV_PYTHON" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  cat >&2 <<'EOF'
Existing .venv uses Python older than 3.11.

Recreate it with a newer interpreter, for example:
  python3.12 -m venv --clear .venv
  ./run.sh
EOF
  exit 1
fi

if [[ ! -f "$DEPS_STAMP" || "pyproject.toml" -nt "$DEPS_STAMP" ]]; then
  echo "Preparing Python dependencies"
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
  "$VENV_PYTHON" -m pip install -e ".[dev]"
  touch "$DEPS_STAMP"
fi

if [[ ! -f ".env" ]]; then
  cp ai_meme_bot/.env.example .env
  cat <<'EOF'
Created .env from ai_meme_bot/.env.example.
Fill AI_BASE_URL, AI_API_KEY, AI_MODEL, and TELEGRAM_BOT_TOKEN, then run ./run.sh again.
EOF
  exit 0
fi

if grep -Eq '^(AI_BASE_URL=https://provider\.example/v1|AI_API_KEY=replace-me|AI_MODEL=replace-me|TELEGRAM_BOT_TOKEN=replace-me)$' .env; then
  cat >&2 <<'EOF'
.env still contains placeholder AI or Telegram values.
Set AI_BASE_URL, AI_API_KEY, AI_MODEL, and TELEGRAM_BOT_TOKEN before starting the bot.
EOF
  exit 1
fi

echo "Starting paper trading bot"
exec "$VENV_PYTHON" -m ai_meme_bot.main
