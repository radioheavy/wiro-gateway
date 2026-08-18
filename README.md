# wiro-gateway

> Use [Qwen 3.8 27B Uncensored](https://wiro.ai/models/qwen/qwen3-8-27b-uncensored)
> — an open-source 27B-parameter model hosted on [Wiro](https://wiro.ai) —
> as a drop-in backend for **Codex CLI** (and a few other clients).
>
> One command to install, one command to use.

```
$ codexw
workdir: /home/you/some-project
model:   qwen/qwen3-8-27b-uncensored
provider: wiro
user
refactor this file please
codex
...
```

## What is this?

A small local HTTP gateway that speaks **OpenAI Chat Completions**, **OpenAI
Responses** (with SSE streaming), and **Anthropic Messages**. It turns each
incoming request into a task on Wiro's `Run/<model>` endpoint, polls
`Task/Detail` until the 262k-context Qwen 3.8 27B model finishes, and sends the
answer back in the shape the client expects.

```
┌──────────────┐     OpenAI Responses / Chat     ┌──────────────────┐  HMAC + form   ┌────────────┐
│  Codex CLI   │ ──────────────────────────────▶ │  wiro-gateway    │ ─────────────▶ │  Wiro API  │
│  (codexw)    │ ◀─── SSE / JSON streaming ────── │  (127.0.0.1)     │ ◀─ task/poll ── │  Qwen3.8   │
└──────────────┘                                 └──────────────────┘                └────────────┘
```

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/radioheavy/wiro-gateway/main/install.sh | bash
```

That's it. The installer will:

1. `pip install` the package from this GitHub repo (no sudo needed, lives in `~/.local`).
2. Ask you to paste your **Wiro API key** and **API secret** (one-time prompt).
3. Install a `systemd --user` service so the gateway stays up across logouts.
4. Register a `wiro` profile in `~/.codex/` and symlink `codexw` into `~/.local/bin/`.
5. Health-check the gateway and print a "you are good to go" message.

### Pre-requisites

- **Python 3.10+** (most Linux distros have it).
- **Codex CLI** installed and on your `$PATH`. Install it with:
  ```bash
  npm i -g @openai/codex
  ```
- A Wiro account. Sign up free at [wiro.ai](https://wiro.ai), go to **Dashboard →
  Projects → New Project**, pick **Signature-Based**, then copy the API key +
  secret. (You'll paste them once during install.)

## Use it

Open any project directory and run:

```bash
codexw                  # interactive Codex REPL, Qwen-backed
codexw exec "refactor this file"   # one-shot, non-interactive
```

`codexw` is a thin wrapper that:

1. Health-checks the local gateway and starts the systemd service if needed.
2. Runs `codex --profile wiro`, where the `wiro` profile is defined in
   `~/.codex/wiro.config.toml` (model + provider) and `~/.codex/config.toml`
   (the `[model_providers.wiro]` block).

Your existing Codex default model (OpenAI, MiniMax, …) stays untouched — you
switch with `codexw` (Qwen) vs `codex` (whatever you had).

## Endpoints exposed

| Method | Path                  | Purpose                                       |
|--------|-----------------------|-----------------------------------------------|
| GET    | `/`                   | Service info                                  |
| GET    | `/healthz`            | Liveness probe                                |
| GET    | `/v1/models`          | OpenAI/Codex-shaped model list                |
| POST   | `/v1/chat/completions`| OpenAI Chat Completions (non-streaming)       |
| POST   | `/v1/responses`       | OpenAI Responses API (SSE streaming + JSON)   |
| POST   | `/v1/messages`        | Anthropic Messages (non-streaming)            |

The gateway polls `Task/Detail` every `WIRO_POLL_INTERVAL_SEC` (default `1s`)
and times out after `WIRO_POLL_TIMEOUT_SEC` (default `600s`). Wiro does not
stream LLM tokens incrementally, so the gateway "fakes" streaming by
chunking the full response into SSE `output_text.delta` events with a small
delay between them — Codex renders this as a normal progress bar.

## Manage

| What                                  | How                                                  |
|---------------------------------------|------------------------------------------------------|
| Service status / logs                 | `systemctl --user status wiro-gateway`  ·  `journalctl --user -u wiro-gateway -f`  ·  `tail -f /tmp/wiro-gateway.log` |
| Restart                               | `systemctl --user restart wiro-gateway`              |
| Stop / disable                        | `systemctl --user stop wiro-gateway`                 |
| Run ad-hoc, no systemd                | `wiro-gateway start` (Ctrl+C to stop)                |
| Verify credentials + Wiro reachability| `wiro-gateway doctor`                                |
| Print env vars (advanced)             | `wiro-gateway printenv`                              |
| **Uninstall everything**              | `wiro-gateway uninstall --purge --yes`               |
| Survive reboots (one-shot)            | `sudo loginctl enable-linger $USER`                  |

## Config

`wiro-gateway init` writes a `.env` next to the current directory (or the path
you pass). Edit the values any time; `systemctl --user restart wiro-gateway`
picks them up.

```env
WIRO_API_KEY=...
WIRO_API_SECRET=...
WIRO_MODEL=qwen/qwen3-8-27b-uncensored        # default
WIRO_API_BASE=https://api.wiro.ai/v1
GATEWAY_HOST=127.0.0.1
GATEWAY_PORT=8765
WIRO_POLL_INTERVAL_SEC=1.0
WIRO_POLL_TIMEOUT_SEC=600
WIRO_DEFAULT_REASONING_EFFORT=none            # none | low | medium | high
WIRO_PRESET_OVERRIDES_SAMPLING=true           # preset wins over request body
```

You can also override at request time (per Codex prompt) by setting these envs
in your shell before launching Codex — they take precedence over the gateway's
defaults and are passed through to Wiro:

| Request field (Codex)   | Wiro parameter      | Default (chat) |
|-------------------------|---------------------|----------------|
| `temperature`           | `temperature`       | 0.7            |
| `top_p`                 | `top_p`             | 0.8            |
| `top_k`                 | `top_k`             | 20             |
| `max_output_tokens`     | `max_tokens`        | 4096           |

## Reasoning effort (Wiro Qwen 3.8 27B)

The Wiro Qwen 3.8 27B Uncensored backend supports a `reasoning_effort` form
field (low / medium / high) plus an `enableThinking` toggle — together they
control how much chain-of-thought the model produces. The gateway maps every
client request into a Wiro preset so the answer quality matches the Wiro model
page:

| Effort   | `enableThinking` | `reasoning_effort` | temp / top_p / top_k | max_tokens | Use it for                                   |
|----------|------------------|--------------------|-----------------------|------------|----------------------------------------------|
| `none`   | `false`          | _(omitted)_        | 0.7 / 0.8 / 20        | 4096       | Snappy Q&A, autocompletes, no thinking       |
| `low`    | `true`           | `low`              | 0.6 / 0.95 / 20       | 8192       | Quick checks, small refactors                |
| `medium` | `true`           | `medium`           | 0.6 / 0.95 / 20       | 12288      | Multi-step edits, debugging, plan + implement|
| `high`   | `true`           | `high`             | 0.6 / 0.95 / 20       | 16384      | Architecture, hard bugs, long planning       |

### How the request is parsed

The gateway reads the requested effort from (first hit wins):

1. `body["reasoning"]["effort"]` — Codex Responses / OpenAI Responses shape.
2. `body["reasoning_effort"]` — flat OpenAI Chat Completions shape.
3. `body["thinking"]` — Claude Code shape (`type: "enabled" | "disabled" | "adaptive"`).
4. `body["reasoning_effort_level"]` — rare client variant.

Allowed values (aliases in parentheses): `none` (`off`, `disabled`),
`low` (`minimal`), `medium` (`med`), `high` (`max`, `maximum`). Anything
unrecognised falls back to `none` instead of 400ing.

### Per-request override vs. gateway default

The gateway default is `WIRO_DEFAULT_REASONING_EFFORT` in `.env` (default
`none`, preserves v0.1.0 behaviour). When a request supplies its own effort,
it always wins. When `WIRO_PRESET_OVERRIDES_SAMPLING=true` (default) the
chosen preset's Wiro-recommended sampling also wins over the request body's
`temperature`/`top_p`/etc. — flip it to `false` if you want the body values
to win and only the thinking toggle to change.

### Switching effort from the Codex REPL

Inside a `codexw` (Codex CLI) session:

```
/model qwen/qwen3-8-27b-uncensored            # pick the Wiro model
/reasoning-effort medium                      # switch thinking effort on the fly
```

Codex writes the choice into the request body, the gateway reads
`body["reasoning"]["effort"]`, and the next request goes out with
`enableThinking=true` and `reasoning_effort=medium`. The resolved config is
echoed back in the response under `x_wiro` so you can verify what was sent.

### Switching effort at the gateway (affects every request)

Edit `.env`:

```env
WIRO_DEFAULT_REASONING_EFFORT=medium
```

Then `wiro-gateway install` (re-runs `bin/install-wrappers.sh` and writes the
matching `default_reasoning_effort` into `~/.codex/wiro.config.toml`), or
manually edit `~/.codex/wiro.config.toml` to keep Codex's profile in sync.

## Security

- The gateway listens on **`127.0.0.1` only**. Do not expose it to a network —
  your Wiro secret would leak. The installer never binds to `0.0.0.0`.
- The gateway accepts any `OPENAI_API_KEY` / `ANTHROPIC_AUTH_TOKEN` because it
  is a local single-user tool. If you need to share it on a LAN, add a token
  check in `wiro_gateway/server.py` first.
- The `.env` file is created with `chmod 600` automatically. The `install.sh`
  bootstrapper never prints your secret after the prompt.

## Project layout

```
wiro-gateway/
├── install.sh              # curl|bash one-shot installer
├── pyproject.toml
├── README.md
├── LICENSE                 # MIT
├── wiro_gateway/
│   ├── cli.py              # wiro-gateway init|install|doctor|start|...
│   ├── auth.py             # HMAC-SHA256 signing
│   ├── client.py           # Wiro Run + Task/Detail + polling
│   ├── extract.py          # Robust assistant-text extractor (+ XML wrapper strip)
│   ├── wiro_to_params.py   # Chat/Responses/Anthropic -> Wiro prompt
│   ├── openai_compat.py    # /v1/chat/completions + /v1/models
│   ├── responses_compat.py # /v1/responses (SSE streaming + JSON)
│   ├── anthropic_compat.py # /v1/messages
│   └── server.py           # FastAPI app
├── tests/                  # 22 unit tests, all offline
└── bin/
    ├── install-systemd.sh  # systemd user service (called by `wiro-gateway install`)
    ├── install-wrappers.sh # codexw / qwen-codex symlinks + codex profile
    ├── uninstall-systemd.sh
    ├── start.sh, stop.sh
    ├── qwen-codex          # the wrapper that powers `codexw`
    └── use.sh              # manual env override (legacy)
```

## Development

```bash
git clone https://github.com/radioheavy/wiro-gateway
cd wiro-gateway
pip install -e ".[dev]"
pytest tests/ -v        # 22 cases, all offline
```

## License

MIT — see [LICENSE](LICENSE).
