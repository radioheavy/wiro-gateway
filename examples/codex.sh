#!/usr/bin/env bash
# Source this file (or copy lines) before running codex:
#   . examples/codex.sh && codex "explain this repo"
# Or export inline:
#   eval "$(wiro-gateway printenv | python -c 'import sys,json,os;d=json.loads(sys.stdin.read().split("?")[0].split("=>")[0] if False else sys.stdin.read());print("\n".join(f"export {k}={v}" for k,v in d["openai"].items()))')"
set -euo pipefail
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8765}"
export OPENAI_BASE_URL="http://${GATEWAY_HOST}:${GATEWAY_PORT}/v1"
export OPENAI_API_KEY="wiro"   # any non-empty value; the gateway doesn't check it
echo "[codex] OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "[codex] now run:  codex \"<your prompt>\""
