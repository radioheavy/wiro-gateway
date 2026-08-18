#!/usr/bin/env bash
# Install wiro-gateway as a systemd USER service. Survives terminal close,
# optional reboot (with lingering enabled). One-shot.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${GATEWAY_PORT:-8765}"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="$SERVICE_DIR/wiro-gateway.service"
LOG=/tmp/wiro-gateway.log

# 1. Validate .env
if [ ! -f .env ]; then
  echo "[install] ERROR: .env not found in $(pwd). Copy .env.example to .env and fill it." >&2
  exit 1
fi
set -a; . ./.env; set +a
if [ -z "${WIRO_API_KEY:-}" ] || [ -z "${WIRO_API_SECRET:-}" ]; then
  echo "[install] ERROR: WIRO_API_KEY / WIRO_API_SECRET missing in .env." >&2
  exit 1
fi

# 2. Find the absolute wiro-gateway binary
WG_BIN="$(command -v wiro-gateway || true)"
if [ -z "$WG_BIN" ]; then
  for p in "$HOME/.local/bin/wiro-gateway" "/usr/local/bin/wiro-gateway" "/usr/bin/wiro-gateway"; do
    [ -x "$p" ] && WG_BIN="$p" && break
  done
fi
if [ -z "$WG_BIN" ]; then
  echo "[install] ERROR: wiro-gateway not on PATH. Run: pip install -e ." >&2
  exit 1
fi
echo "[install] binary: $WG_BIN"

# 3. Write service unit
mkdir -p "$SERVICE_DIR"
cat > "$SERVICE_FILE" << UNIT
[Unit]
Description=wiro-gateway (Qwen3.8-27B-Uncensored -> OpenAI/Anthropic compatible API on 127.0.0.1:$PORT)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$(pwd)
EnvironmentFile=$(pwd)/.env
ExecStart=$WG_BIN start --host 127.0.0.1 --port $PORT
Restart=on-failure
RestartSec=2
StandardOutput=append:$LOG
StandardError=append:$LOG

[Install]
WantedBy=default.target
UNIT
echo "[install] wrote $SERVICE_FILE"

# 4. Reload and start
systemctl --user daemon-reload
systemctl --user enable --now wiro-gateway.service
sleep 0.5

# 5. Wait for healthz
URL="http://127.0.0.1:$PORT/healthz"
for i in $(seq 1 30); do
  if curl -sS -m 1 "$URL" > /dev/null 2>&1; then
    echo "[install] up after ${i}*0.5s"
    break
  fi
  sleep 0.5
done

if ! curl -sS -m 1 "$URL" > /dev/null 2>&1; then
  echo "[install] FAILED. Tail of $LOG:" >&2
  tail -30 "$LOG" >&2
  echo "[install] hint: 'systemctl --user status wiro-gateway.service'" >&2
  exit 1
fi

# 6. Print usage
cat << USAGE

============================================================
 wiro-gateway is live and managed by systemd --user
============================================================
 Health:   http://127.0.0.1:$PORT/healthz
 Logs:     $LOG
 Control:  systemctl --user {start|stop|restart|status} wiro-gateway
 Disable:  systemctl --user disable --now wiro-gateway.service

 To survive logout/reboot, enable lingering once:
   sudo loginctl enable-linger $USER

 For Codex:
   eval "\$(./bin/use.sh codex)"
   codex "<your prompt>"

 For Claude Code:
   eval "\$(./bin/use.sh claude)"
   claude "<your prompt>"
============================================================
USAGE
