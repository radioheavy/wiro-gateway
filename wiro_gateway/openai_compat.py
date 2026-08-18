"""OpenAI Chat Completions compatible endpoint.

Implements POST /v1/chat/completions with a non-streaming response. Codex (and
most OpenAI-compatible clients) work fine with non-streaming when `stream=false`,
which is the default.

Schema is intentionally minimal: we accept the common fields Codex / curl users
actually send, and silently ignore the rest. We do NOT advertise tool_calls,
function_call, logprobs, etc. -- the upstream model does not support them.

Reasoning effort is read from `body["reasoning_effort"]` (OpenAI's flat form)
or `body["reasoning"]["effort"]` (Codex Responses / shared shape). When
`WIRO_PRESET_OVERRIDES_SAMPLING=true` (the default) the Wiro-recommended
sampling for that effort wins over the request body's own temperature/top_p.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .extract import extract_assistant_text
from .reasoning import PRESETS, EffortLevel, apply_preset_to_wiro_params, resolve_effort
from .wiro_to_params import merge_messages

router = APIRouter()

# Sampling defaults that match Wiro's Qwen3-8-27B-Uncensored recommended settings
# for the chat profile (thinking off, temp 0.7, top_p 0.80, top_k 20).
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 4096


def _wiro_params_from_request(
    body: dict[str, Any],
    merged: tuple[str, str],
    settings: Settings,
) -> tuple[dict[str, Any], EffortLevel, dict[str, Any]]:
    """Build the Wiro Run params + return the resolved effort for logging."""
    system_prompt, prompt = merged
    effort = resolve_effort(body, settings.wiro_default_reasoning_effort)
    preset = PRESETS[effort]

    # Per-request body wins over preset unless we are explicitly overriding.
    override = settings.wiro_preset_overrides_sampling
    temperature = float(body.get("temperature", preset.temperature if override else DEFAULT_TEMPERATURE))
    top_p = float(body.get("top_p", preset.top_p if override else DEFAULT_TOP_P))
    top_k = int(body.get("top_k", preset.top_k if override else DEFAULT_TOP_K))
    max_tokens = int(
        body.get("max_tokens")
        or (preset.max_tokens if override else DEFAULT_MAX_TOKENS)
    )
    stop = body.get("stop")
    if isinstance(stop, str):
        stop_sequences = stop
    elif isinstance(stop, list) and stop:
        stop_sequences = ";".join(str(s) for s in stop)
    else:
        stop_sequences = ""

    params: dict[str, Any] = {
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "temperature": str(temperature),
        "top_p": str(top_p),
        "top_k": str(top_k),
        "repetition_penalty": "1.0",
        "length_penalty": "1",
        "max_tokens": str(max_tokens),
        "min_tokens": "0",
        "stop_sequences": stop_sequences,
        "seed": "0",
        "do_sample": "--do_sample",
    }
    params = apply_preset_to_wiro_params(params, preset, override_sampling=override)
    meta = {
        "effort": effort,
        "enable_thinking": preset.enable_thinking,
        "reasoning_effort": preset.reasoning_effort,
        "preset_overrode_sampling": override,
    }
    return params, effort, meta


def _openai_response(model: str, text: str, usage: dict[str, int]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        },
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    client = request.app.state.wiro

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")

    model = body.get("model") or settings.wiro_model
    # OpenAI accepts any model id; we always use the configured Wiro model path.
    # (We do not switch models per-request to keep secrets contained.)

    merged = merge_messages(messages, body.get("system"))
    wiro_params, effort, meta = _wiro_params_from_request(body, merged, settings)

    try:
        payload = await client.run_until_done(settings.wiro_model, wiro_params)
    except Exception as exc:
        # Surface Wiro failure to the client with a 502 so the user sees the cause.
        raise HTTPException(status_code=502, detail=f"Wiro error: {exc}")

    try:
        text = extract_assistant_text(payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not extract assistant text: {exc}")

    # We have no real token count from Wiro; estimate by chars/4 (good-enough hint).
    pt = sum(len(str(m.get("content", ""))) for m in messages) // 4
    ct = len(text) // 4
    resp = _openai_response(model, text, {"prompt_tokens": pt, "completion_tokens": ct})
    # Echo the resolved reasoning config back so Codex / curl users can see it
    # in the response (also helpful for debugging the gateway).
    resp["x_wiro"] = meta
    return JSONResponse(resp)


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """OpenAI-shaped /v1/models with the strict Codex 0.147+ schema.

    Codex uses strict serde with the `ModelInfo` struct from
    `codex_protocol::openai_models`. The shape below mirrors Codex's
    `model_info_from_slug` fallback constructor so even unknown model
    slugs parse cleanly. Every field is required; we use `null` for the
    `Option<T>` ones.
    """
    from .reasoning import PRESETS, SUPPORTED_EFFORTS

    settings: Settings = request.app.state.settings
    now = int(time.time())
    default_effort = settings.wiro_default_reasoning_effort

    # Each element of supported_reasoning_levels is {effort, description},
    # not a bare string -- Codex's struct ReasoningEffortPreset has both.
    levels = [
        {"effort": e, "description": PRESETS[e].description}
        for e in SUPPORTED_EFFORTS
    ]

    def _build_model(model_id: str) -> dict[str, Any]:
        """Build a strict Codex ModelInfo for one model.

        Defaults match Codex's `model_info_from_slug` so this passes the
        same serde parse path that an OpenAI model goes through.
        """
        return {
            # OpenAI /v1/models base fields.
            "id": model_id,
            "slug": model_id,
            "object": "model",
            "created": now,
            "owned_by": "wiro",
            "display_name": "Qwen 3.8 27B Uncensored (Wiro)",
            "description": (
                "Open-source 27B Qwen 3.8 Uncensored hosted on Wiro. "
                "262k context. Reasoning effort: none/low/medium/high. "
                "Configure with the gateway WIRO_DEFAULT_REASONING_EFFORT "
                ".env var or the Codex /reasoning-effort slash command."
            ),
            "default_reasoning_level": default_effort,
            "supported_reasoning_levels": levels,
            "shell_type": "default",
            "visibility": "none",
            "supported_in_api": True,
            "priority": 99,
            "additional_speed_tiers": [],
            "service_tiers": [],
            "default_service_tier": None,
            "availability_nux": None,
            "upgrade": None,
            "model_messages": {
                "instructions_template": (
                    "You are a coding agent running in the Wiro-gateway "
                    "local backend. Be precise, safe, and helpful."
                ),
                "instructions_variables": None,
                "approvals": None,
                "collaboration_modes": None,
                "auto_review": None,
                "permissions": None,
                "multi_agent": None,
                "token_budget": None,
                "guardian_v2": None,
            },
            "include_skills_usage_instructions": False,
            "include_plugin_usage_instructions": False,
            "include_apps_usage_instructions": False,
            "supports_reasoning_summary_parameter": True,
            "default_reasoning_summary": "none",
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": None,
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "supports_image_detail_original": False,
            "context_window": 272000,
            "max_context_window": 272000,
            "auto_compact_token_limit": None,
            "comp_hash": None,
            "effective_context_window_percent": 95,
            "experimental_supported_tools": [],
            "input_modalities": ["text"],
            "used_fallback_model_metadata": True,
            "supports_search_tool": False,
            "use_responses_lite": False,
            "node_repl_auto_review_required": False,
            "node_repl_disabled": False,
            "auto_review_model_override": None,
            "model_specialty": None,
            "tool_mode": None,
            "multi_agent_version": None,
            "supports_parallel_tool_calls": False,
            "available_in_plans": [],
            "reasoning_summary_format": "experimental",
        }

    base = _build_model(settings.wiro_model)
    alias = _build_model("wiro")
    models = [base, alias]
    return JSONResponse({"object": "list", "data": models, "models": models})


