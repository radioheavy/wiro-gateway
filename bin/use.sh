#!/usr/bin/env bash
# Print env vars to point Codex or Claude Code at the running wiro-gateway.
# Usage:
#   eval "$(bin/use.sh codex)"
#   eval "$(bin/use.sh claude)"
set -euo pipefail
target="${1:-both}"
host="${GATEWAY_HOST:-127.0.0.1}"
port="${GATEWAY_PORT:-8765}"
base="http://${host}:${port}"
case "$target" in
  codex)
    echo "export OPENAI_BASE_URL=\"${base}/v1\""
    echo "export OPENAI_API_KEY=\"wiro\""
    ;;
  claude)
    echo "export ANTHROPIC_BASE_URL=\"${base}\""
    echo "export ANTHROPIC_AUTH_TOKEN=\"wiro\""
    ;;
  both|*)
    echo "export OPENAI_BASE_URL=\"${base}/v1\""
    echo "export OPENAI_API_KEY=\"wiro\""
    echo "export ANTHROPIC_BASE_URL=\"${base}\""
    echo "export ANTHROPIC_AUTH_TOKEN=\"wiro\""
    ;;
esac
