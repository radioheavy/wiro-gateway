"""OpenAI Chat Completions compatible endpoint.

Implements POST /v1/chat/completions with a non-streaming response. Codex (and
most OpenAI-compatible clients) work fine with non-streaming when `stream=false`,
which is the default.

Schema is intentionally minimal: we accept the common fields Codex / curl users
actually send, and silently ignore the rest. We do NOT advertise tool_calls,
function_call, logprobs, etc. — the upstream model does not support them.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .extract import extract_assistant_text
from .wiro_to_params import merge_messages

router = APIRouter()

# Sampling defaults that match Wiro's Qwen3-8-27B-Uncensored recommended settings
# (chat profile: thinking off, temp 0.7, top_p 0.80, top_k 20).
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_MAX_TOKENS = 4096


def _wiro_params_from_request(body: dict[str, Any], merged: tuple[str, str]) -> dict[str, Any]:
    system_prompt, prompt = merged
    temperature = float(body.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(body.get("top_p", DEFAULT_TOP_P))
    top_k = int(body.get("top_k", DEFAULT_TOP_K))
    max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
    stop = body.get("stop")
    if isinstance(stop, str):
        stop_sequences = stop
    elif isinstance(stop, list) and stop:
        stop_sequences = ";".join(str(s) for s in stop)
    else:
        stop_sequences = ""

    return {
        "enableThinking": "false",  # Codex tooling wants snappy, deterministic-ish replies
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
    wiro_params = _wiro_params_from_request(body, merged)

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
    return JSONResponse(_openai_response(model, text, {"prompt_tokens": pt, "completion_tokens": ct}))


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    now = int(time.time())
    models = [
        {
            "id": settings.wiro_model,
            "slug": settings.wiro_model,
            "object": "model",
            "created": now,
            "owned_by": "wiro",
            "display_name": "Qwen 3.8 27B Uncensored (Wiro)",
            "description": "Open-source 27B chat model hosted on Wiro. 262k context.",
            "context_window": 262144,
            "max_output_tokens": 4096,
            "supported_features": ["text"],
        },
        {
            "id": "wiro",
            "slug": "wiro",
            "object": "model",
            "created": now,
            "owned_by": "wiro",
            "display_name": "Wiro default",
            "description": "Alias that resolves to the configured Wiro model.",
            "context_window": 262144,
            "max_output_tokens": 4096,
            "supported_features": ["text"],
        },
    ]
    # Both shapes (data + models) for compatibility with OpenAI Chat Completions
    # clients and Codex's strict Responses-API model-list parser.
    return JSONResponse({"object": "list", "data": models, "models": models})
