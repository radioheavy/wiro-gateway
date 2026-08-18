#!/usr/bin/env bash
# wiro-gateway one-line installer.
# Usage:  curl -fsSL https://raw.githubusercontent.com/radioheavy/wiro-gateway/main/install.sh | bash
#
# What it does:
#   1. Picks a Python (3.10+) on PATH.
#   2. pip-installs wiro-gateway from this GitHub repo (so `wiro-gateway` is on PATH).
#   3. Hands off to `wiro-gateway install`, which:
#        - writes .env (asks for your Wiro API key + secret),
#        - installs a systemd --user service so the gateway survives logouts,
#        - registers a `wiro` profile in ~/.codex/ and drops `codexw` in ~/.local/bin,
#        - health-checks the gateway and prints next steps.
set -euo pipefail

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[install] ERROR: $PY not found. Install Python 3.10+ first." >&2
  exit 1
fi
if ! "$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
  echo "[install] ERROR: Python 3.10+ required (found $($PY -V 2>&1))." >&2
  exit 1
fi

REPO="${WIRO_GATEWAY_REPO:-https://github.com/radioheavy/wiro-gateway.git}"
echo "[install] installing wiro-gateway from $REPO ..."

# pip-install from git. Use --user so it lands in ~/.local (no sudo, no venv).
# Fall back to --break-system-packages on Debian/Ubuntu where PEP 668 blocks system installs.
if "$PY" -m pip install --user "git+${REPO}" 2>/dev/null; then
  :
elif "$PY" -m pip install --user --break-system-packages "git+${REPO}"; then
  :
else
  echo "[install] pip install failed. If you have an externally-managed Python, try:" >&2
  echo "    python3 -m venv ~/.wiro-gateway-venv" >&2
  echo "    ~/.wiro-gateway-venv/bin/pip install git+${REPO}" >&2
  echo "    ~/.wiro-gateway-venv/bin/wiro-gateway install" >&2
  exit 1
fi

# Make sure ~/.local/bin is on PATH for the rest of this script.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v wiro-gateway >/dev/null 2>&1; then
  echo "[install] pip install OK but 'wiro-gateway' is not on PATH." >&2
  echo "Add this to your shell rc and re-run:  export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  exit 1
fi

# Hand off to the in-CLI wizard. Pass any flags through.
echo "[install] launching the in-CLI installer (it will ask for your Wiro key + secret) ..."
exec wiro-gateway install "$@"
