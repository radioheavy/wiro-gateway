#!/usr/bin/env bash
# Remove the wiro-gateway systemd user service.
set -euo pipefail
systemctl --user disable --now wiro-gateway.service 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/wiro-gateway.service"
systemctl --user daemon-reload
echo "[uninstall] done. Server is stopped. .env and source files left untouched."
