#!/usr/bin/env bash
# Start the wiro-gateway server in the background. Writes PID to /tmp/wiro-gateway.pid.
# Logs to /tmp/wiro-gateway.log. Re-running is a no-op if already healthy.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=/tmp/wiro-gateway.log
PID=/tmp/wiro-gateway.pid
URL="http://127.0.0.1:${GATEWAY_PORT:-8765}/healthz"

if curl -sS -m 1 "$URL" > /dev/null 2>&1; then
  echo "[start] already running and healthy at $URL"
  exit 0
fi

# Need .env in cwd or the env vars exported
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

if [ -z "${WIRO_API_KEY:-}" ] || [ -z "${WIRO_API_SECRET:-}" ]; then
  echo "[start] ERROR: WIRO_API_KEY / WIRO_API_SECRET not set. Edit .env first." >&2
  exit 1
fi

# Detach: new session + nohup + & + disown
setsid nohup wiro-gateway start > "$LOG" 2>&1 < /dev/null &
PID_VAL=$!
disown || true
echo "$PID_VAL" > "$PID"

# Wait for healthz (max 15s)
for i in $(seq 1 30); do
  if curl -sS -m 1 "$URL" > /dev/null 2>&1; then
    echo "[start] up after ${i}*0.5s   pid=$PID_VAL   log=$LOG"
    exit 0
  fi
  sleep 0.5
done

echo "[start] FAILED to come up. Tail of $LOG:" >&2
tail -20 "$LOG" >&2
exit 1
