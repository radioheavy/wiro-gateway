"""Unit tests for the reasoning-effort mapping + integration with /v1/responses
and /v1/chat/completions. All tests are offline -- no real Wiro call."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from wiro_gateway.config import Settings
from wiro_gateway.reasoning import (
    PRESETS,
    SUPPORTED_EFFORTS,
    EffortLevel,
    apply_preset_to_wiro_params,
    describe_effort,
    extract_effort_from_body,
    normalise_effort,
    resolve_effort,
)
from wiro_gateway.server import create_app


# ---------------------------------------------------------------------------
# normalise_effort
# ---------------------------------------------------------------------------


def test_normalise_effort_canonical():
    assert normalise_effort("low") == "low"
    assert normalise_effort("medium") == "medium"
    assert normalise_effort("high") == "high"
    assert normalise_effort("none") == "none"


def test_normalise_effort_aliases():
    assert normalise_effort("MINIMAL") == "low"
    assert normalise_effort("off") == "none"
    assert normalise_effort("disabled") == "none"
    assert normalise_effort("max") == "high"
    assert normalise_effort("maximum") == "high"
    assert normalise_effort("med") == "medium"


def test_normalise_effort_unknown_returns_none():
    """Anything we don't understand falls back to none, not an exception."""
    assert normalise_effort("banana") == "none"
    assert normalise_effort("") == "none"
    assert normalise_effort(None) == "none"
    assert normalise_effort(42) == "none"


def test_normalise_effort_bool():
    """A true/false toggle is sometimes what clients send; map it sensibly."""
    assert normalise_effort(True) == "low"
    assert normalise_effort(False) == "none"


# ---------------------------------------------------------------------------
# extract_effort_from_body
# ---------------------------------------------------------------------------


def test_extract_from_codex_responses_shape():
    body = {"reasoning": {"effort": "high"}}
    assert extract_effort_from_body(body) == "high"


def test_extract_from_openai_flat_shape():
    body = {"reasoning_effort": "medium"}
    assert extract_effort_from_body(body) == "medium"


def test_extract_from_claude_thinking_shape():
    body = {"thinking": {"type": "enabled"}}
    assert extract_effort_from_body(body) == "low"  # 'enabled' isn't canonical -> none, but


def test_extract_from_empty_body_returns_none():
    assert extract_effort_from_body({}) == "none"
    assert extract_effort_from_body({"messages": []}) == "none"


def test_extract_priority_codex_wins_over_flat():
    body = {"reasoning": {"effort": "high"}, "reasoning_effort": "low"}
    assert extract_effort_from_body(body) == "high"


def test_extract_legacy_field():
    body = {"reasoning_effort_level": "medium"}
    assert extract_effort_from_body(body) == "medium"


# ---------------------------------------------------------------------------
# resolve_effort
# ---------------------------------------------------------------------------


def test_resolve_uses_request_over_default():
    body = {"reasoning": {"effort": "high"}}
    assert resolve_effort(body, "low") == "high"
    assert resolve_effort(body, "none") == "high"


def test_resolve_falls_back_to_default():
    assert resolve_effort({}, "medium") == "medium"
    assert resolve_effort({"reasoning_effort": "low"}, "high") == "low"


# ---------------------------------------------------------------------------
# apply_preset_to_wiro_params
# ---------------------------------------------------------------------------


def test_apply_preset_low_stamps_thinking_and_effort():
    out = apply_preset_to_wiro_params({}, PRESETS["low"], override_sampling=True)
    assert out["enableThinking"] == "true"
    assert out["reasoning_effort"] == "low"
    assert out["temperature"] == "0.6"
    assert out["top_p"] == "0.95"
    assert out["max_tokens"] == "8192"


def test_apply_preset_none_drops_reasoning_effort_and_keeps_thinking_off():
    out = apply_preset_to_wiro_params(
        {"enableThinking": "true", "reasoning_effort": "high", "temperature": "0.5"},
        PRESETS["none"],
        override_sampling=True,
    )
    assert out["enableThinking"] == "false"
    assert "reasoning_effort" not in out
    # Sampling reset to chat profile.
    assert out["temperature"] == "0.7"
    assert out["top_p"] == "0.8"
    assert out["top_k"] == "20"


def test_apply_preset_override_sampling_false_keeps_body_values():
    body = {"temperature": "0.3", "top_p": "0.5", "top_k": "7", "max_tokens": "999"}
    out = apply_preset_to_wiro_params(body, PRESETS["high"], override_sampling=False)
    assert out["enableThinking"] == "true"
    assert out["reasoning_effort"] == "high"
    assert out["temperature"] == "0.3"  # body wins
    assert out["top_p"] == "0.5"
    assert out["top_k"] == "7"
    assert out["max_tokens"] == "999"


def test_describe_effort_human_readable():
    assert "thinking off" in describe_effort("none")
    assert "thinking on" in describe_effort("high")
    assert "reasoning_effort=medium" in describe_effort("medium")


# ---------------------------------------------------------------------------
# Settings: env validation
# ---------------------------------------------------------------------------


def test_settings_default_effort_is_none():
    s = Settings(wiro_api_key="k", wiro_api_secret="s")
    assert s.wiro_default_reasoning_effort == "none"
    assert s.wiro_preset_overrides_sampling is True
    assert s.wiro_enable_thinking_on_effort is True


def test_settings_accepts_aliases_in_env():
    s = Settings(wiro_api_key="k", wiro_api_secret="s", wiro_default_reasoning_effort="minimal")
    assert s.wiro_default_reasoning_effort == "low"


def test_settings_rejects_garbage():
    with pytest.raises(Exception):
        Settings(wiro_api_key="k", wiro_api_secret="s", wiro_default_reasoning_effort="banana")


# ---------------------------------------------------------------------------
# /v1/responses integration (Codex wire shape)
# ---------------------------------------------------------------------------


class _FakeWiro:
    def __init__(self, text: str = "ok"):
        self._text = text
        self.last_params = None
        self.last_model = None

    async def aclose(self):
        return None

    async def run_until_done(self, model_path, params):
        self.last_model = model_path
        self.last_params = dict(params)
        return {
            "tasklist": [
                {
                    "id": "1",
                    "status": "task_postprocess_end",
                    "debugoutput": self._text,
                    "outputs": [],
                    "parameters": {},
                }
            ],
            "result": True,
        }


def _app(default_effort: str = "none", override_sampling: bool = True) -> tuple[object, _FakeWiro]:
    s = Settings(
        wiro_api_key="k", wiro_api_secret="s",
        wiro_model="qwen/qwen3-8-27b-uncensored",
        wiro_poll_interval_sec=0.01, wiro_poll_timeout_sec=2,
        wiro_default_reasoning_effort=default_effort,
        wiro_preset_overrides_sampling=override_sampling,
    )
    app = create_app(s)
    fake = _FakeWiro()
    app.state.wiro = fake
    return app, fake


@pytest.mark.asyncio
async def test_responses_default_effort_is_none_keeps_thinking_off():
    app, fake = _app(default_effort="none")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/responses", json={
            "model": "any", "input": "hi", "stream": False,
        })
    assert r.status_code == 200, r.text
    assert fake.last_params["enableThinking"] == "false"
    assert "reasoning_effort" not in fake.last_params
    # x_wiro is echoed back so the user can see the resolved effort.
    body = r.json()
    assert body["x_wiro"]["effort"] == "none"
    assert body["x_wiro"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_responses_explicit_effort_enables_thinking():
    app, fake = _app(default_effort="none")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/responses", json={
            "model": "any",
            "input": "hi",
            "stream": False,
            "reasoning": {"effort": "medium"},
        })
    assert r.status_code == 200, r.text
    assert fake.last_params["enableThinking"] == "true"
    assert fake.last_params["reasoning_effort"] == "medium"
    # Preset sampling is stamped on.
    assert fake.last_params["temperature"] == "0.6"
    assert fake.last_params["top_p"] == "0.95"
    assert fake.last_params["max_tokens"] == "12288"
    assert r.json()["x_wiro"]["effort"] == "medium"


@pytest.mark.asyncio
async def test_responses_default_effort_medium_applies_when_no_body_field():
    app, fake = _app(default_effort="medium")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/responses", json={
            "model": "any", "input": "hi", "stream": False,
        })
    assert r.status_code == 200, r.text
    assert fake.last_params["enableThinking"] == "true"
    assert fake.last_params["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_responses_request_body_sampling_wins_when_override_off():
    app, fake = _app(default_effort="high", override_sampling=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/responses", json={
            "model": "any",
            "input": "hi",
            "stream": False,
            "temperature": 0.42,
            "top_p": 0.11,
            "top_k": 7,
        })
    assert r.status_code == 200, r.text
    # Thinking toggles still on (high preset) but the body values win.
    assert fake.last_params["enableThinking"] == "true"
    assert fake.last_params["reasoning_effort"] == "high"
    assert fake.last_params["temperature"] == "0.42"
    assert fake.last_params["top_p"] == "0.11"
    assert fake.last_params["top_k"] == "7"


@pytest.mark.asyncio
async def test_responses_minimal_alias_maps_to_low():
    app, fake = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/responses", json={
            "model": "any",
            "input": "hi",
            "stream": False,
            "reasoning": {"effort": "minimal"},
        })
    assert r.status_code == 200, r.text
    assert fake.last_params["enableThinking"] == "true"
    assert fake.last_params["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# /v1/chat/completions integration (OpenAI Chat wire shape)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_legacy_compat_keeps_thinking_off():
    app, fake = _app(default_effort="none")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "any",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert r.status_code == 200, r.text
    assert fake.last_params["enableThinking"] == "false"
    assert "reasoning_effort" not in fake.last_params
    # Legacy v0.1.0 default chat sampling must still apply when default effort is none.
    assert fake.last_params["temperature"] == "0.7"
    assert fake.last_params["top_p"] == "0.8"


@pytest.mark.asyncio
async def test_chat_completions_flat_reasoning_effort_works():
    app, fake = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "any",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        })
    assert r.status_code == 200, r.text
    assert fake.last_params["enableThinking"] == "true"
    assert fake.last_params["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# /v1/messages integration (Anthropic wire shape)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_messages_thinking_enabled_maps_to_low():
    app, fake = _app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/messages", json={
            "model": "any",
            "max_tokens": 100,
            "system": "Be brief.",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled"},
        })
    assert r.status_code == 200, r.text
    assert fake.last_params["enableThinking"] == "true"
    assert "reasoning_effort" in fake.last_params


# ---------------------------------------------------------------------------
# /v1/models advertises reasoning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_models_lists_reasoning_efforts():
    """Strict Codex 0.147+ /v1/models shape.

    Mirrors Codex's `model_info_from_slug` fallback constructor. Every
    field is required -- we use `null` for the `Option<T>` ones. Top-level
    list must include both `data` and `models` arrays.
    """
    app, _ = _app(default_effort="medium")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list) and len(body["data"]) == 2
    assert body["data"] == body["models"]
    required = {
        "id", "slug", "object", "created", "owned_by",
        "display_name", "description",
        "default_reasoning_level", "supported_reasoning_levels",
        "shell_type", "visibility", "supported_in_api", "priority",
        "additional_speed_tiers", "service_tiers", "default_service_tier",
        "availability_nux", "upgrade", "model_messages",
        "include_skills_usage_instructions",
        "include_plugin_usage_instructions",
        "include_apps_usage_instructions",
        "supports_reasoning_summary_parameter", "default_reasoning_summary",
        "support_verbosity", "default_verbosity",
        "apply_patch_tool_type", "web_search_tool_type", "truncation_policy",
        "supports_image_detail_original", "supports_parallel_tool_calls",
        "context_window", "max_context_window", "auto_compact_token_limit",
        "comp_hash", "effective_context_window_percent",
        "experimental_supported_tools", "input_modalities",
        "used_fallback_model_metadata", "supports_search_tool",
        "use_responses_lite", "node_repl_auto_review_required",
        "node_repl_disabled", "auto_review_model_override",
        "model_specialty", "tool_mode", "multi_agent_version",
        "available_in_plans", "reasoning_summary_format",
    }
    for m in body["data"]:
        missing = required - set(m)
        assert not missing, f"missing fields: {missing}"
        # default_reasoning_level is a STRING (e.g. "medium"), not a struct.
        assert m["default_reasoning_level"] == "medium"
        assert isinstance(m["default_reasoning_level"], str)
        # supported_reasoning_levels is a list of {effort, description}, not bare strings.
        levels = m["supported_reasoning_levels"]
        assert [x["effort"] for x in levels] == list(SUPPORTED_EFFORTS)
        for x in levels:
            assert "description" in x
        # Other type assertions.
        assert m["object"] == "model"
        assert isinstance(m["supported_in_api"], bool)
        assert isinstance(m["priority"], int)
        assert isinstance(m["model_messages"], dict)
        assert isinstance(m["truncation_policy"], dict)
