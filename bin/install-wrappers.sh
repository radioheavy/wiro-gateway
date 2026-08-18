#!/usr/bin/env bash
# Symlink codexw / qwen-codex / codex-wiro into ~/.local/bin/ so they work anywhere.
# Idempotent: re-creates the wiro provider in ~/.codex/config.toml and the
# wiro profile in ~/.codex/wiro.config.toml (Codex 0.147+ moved profiles
# out of the main config).
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${DEST:-$HOME/.local/bin}"
mkdir -p "$DEST"

# 1) Symlinks
for name in qwen-codex codexw codex-wiro; do
  ln -sf "$(pwd)/bin/qwen-codex" "$DEST/$name"
  echo "[wrappers] $DEST/$name -> bin/qwen-codex"
done

# 2) Codex provider (stays in main config.toml)
CODEX_CFG="$HOME/.codex/config.toml"
if [ -f "$CODEX_CFG" ] && ! grep -q "^\[model_providers\.wiro\]" "$CODEX_CFG"; then
  cat >> "$CODEX_CFG" << 'TOML'

# --- wiro-gateway: Qwen3.8-27B-Uncensored via local wiro-gateway ---
# Added by ~/Masaüstü/wiro-gateway/bin/install-wrappers.sh
# Provider lives here; the `wiro` profile lives in ~/.codex/wiro.config.toml and is
# activated by `codex --profile wiro` (or the `codexw` wrapper).
[model_providers.wiro]
name = "Wiro (Qwen3.8-27B-Uncensored)"
base_url = "http://127.0.0.1:8765/v1"
api_key = "wiro"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
requires_openai_auth = false
TOML
  echo "[wrappers] added [model_providers.wiro] to $CODEX_CFG"
fi

# 3) Codex profile (in a dedicated file)
# Append/update pattern: preserve any [projects.*], [history], [notice], etc.
# blocks that Codex itself has written; only manage the wiro-gateway keys we
# own. Idempotent: re-running never duplicates lines.
CODEX_PROF="$HOME/.codex/wiro.config.toml"
# Honour a user-supplied default effort, falling back to "none" (legacy
# v0.1.0 behaviour: thinking off, chat-profile sampling).
DEFAULT_EFFORT="${WIRO_DEFAULT_REASONING_EFFORT:-${CODEX_DEFAULT_REASONING_EFFORT:-none}}"

touch "$CODEX_PROF"

# Strip any previously-written wiro-gateway header + managed keys so the file
# is idempotent across re-installs (we never touch user-added blocks).
CODEX_PROF_TMP="$(mktemp)"
python3 - "$CODEX_PROF" "$CODEX_PROF_TMP" <<'PYEOF'
import re, sys
src_path, dst_path = sys.argv[1], sys.argv[2]
with open(src_path, "r", encoding="utf-8") as f:
    text = f.read()

managed_keys = (
    "default_reasoning_effort",
    "model_reasoning_effort",
)
# Drop the wiro-gateway auto-generated banner + the managed lines.
text = re.sub(r"(?ms)^# wiro-gateway profile[^\n]*\n(# Auto-generated[^\n]*\n)?", "", text)
text = re.sub(r"(?m)^\s*(" + "|".join(managed_keys) + r")\s*=.*\n", "", text)
# Drop the wiro-gateway-managed legacy [model] block (v0.2+ removed it, but
# older installs may still have it). Only remove the lines we wrote, never
# any [model] block the user added themselves.
text = re.sub(
    r"(?ms)^# Codex 0\.147\+ nested shape[^\n]*\n\\[model\\]\s*\nreasoning = \\{ effort = \"[^\"]*\" \\}\s*\n",
    "",
    text,
)
text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
with open(dst_path, "w", encoding="utf-8") as f:
    f.write(text)
PYEOF

# Now append our managed block (banner + keys + nested [model].reasoning).
{
  echo ""
  echo "# wiro-gateway profile (loaded by \`codex --profile wiro\`) -- managed by"
  echo "# ~/Masaüstü/wiro-gateway/bin/install-wrappers.sh. Re-runs are idempotent."
  echo "# To change reasoning effort, edit this file OR set WIRO_DEFAULT_REASONING_EFFORT"
  echo "# in the gateway .env and re-run wiro-gateway install."
  echo "# Supported: none / low / medium / high."
  echo ""
  echo "default_reasoning_effort = \"$DEFAULT_EFFORT\""
  echo "model = \"qwen/qwen3-8-27b-uncensored\""
  echo "model_provider = \"wiro\""
  echo "model_reasoning_effort = \"$DEFAULT_EFFORT\""
  echo ""
  echo "# To switch effort from the Codex REPL: /reasoning-effort <none|low|medium|high>"
  echo "# (Codex CLI 0.147+ writes the choice into the request body, gateway reads"
  echo "#  body[\"reasoning\"][\"effort\"] and applies the Wiro preset.)"
} >> "$CODEX_PROF_TMP"
mv "$CODEX_PROF_TMP" "$CODEX_PROF"
echo "[wrappers] updated $CODEX_PROF (default_reasoning_effort=$DEFAULT_EFFORT, preserved other blocks)"

# 4) PATH hint
case ":$PATH:" in
  *":$DEST:"*) echo "[wrappers] $DEST already on PATH. Try 'codexw' in a new shell." ;;
  *) cat << HINT
[wrappers] NOTE: $DEST is not on your PATH.
  Add this to your ~/.bashrc (or ~/.zshrc):
      export PATH="$HOME/.local/bin:\$PATH"
HINT
  ;;
esac
