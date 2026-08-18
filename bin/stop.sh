#!/usr/bin/env bash
# Stop the wiro-gateway server (uses /tmp/wiro-gateway.pid, then any wiro-gateway process).
set -euo pipefail
PID=/tmp/wiro-gateway.pid
if [ -f "$PID" ]; then
  pid=$(cat "$PID")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 0.3
    kill -9 "$pid" 2>/dev/null || true
    echo "[stop] killed pid $pid"
  fi
  : > "$PID"
fi
# Fallback: kill any leftover wiro-gateway/uvicorn process
pkill -f "wiro-gateway start" 2>/dev/null || true
pkill -f "wiro_gateway.server:app" 2>/dev/null || true
echo "[stop] done"
