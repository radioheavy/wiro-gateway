"""Anthropic Messages API compatible endpoint.

Implements POST /v1/messages with a non-streaming response. Claude Code sets
`ANTHROPIC_BASE_URL` to the gateway; it then issues /v1/messages calls that we
translate to Wiro Run tasks and convert the assistant text back.

Schema is the minimum Claude Code actually sends:
  - model: str
  - messages: [{role, content}, ...]
  - system: str | [{type, text}, ...]   (top-level, not in messages)
  - max_tokens: int
  - temperature, top_p, top_k, stop_sequences
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

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20


def _system_to_string(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(system)


def _anthropic_response(model: str, text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": max(1, len(text) // 4),
        },
    }


@router.post("/v1/messages")
async def messages(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    client = request.app.state.wiro

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages_in = body.get("messages") or []
    if not isinstance(messages_in, list) or not messages_in:
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")

    system = _system_to_string(body.get("system"))
    model = body.get("model") or settings.wiro_model
    max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
    temperature = float(body.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(body.get("top_p", DEFAULT_TOP_P))
    top_k = int(body.get("top_k", DEFAULT_TOP_K))
    stop = body.get("stop_sequences")

    merged = merge_messages(messages_in, system)
    system_prompt, prompt = merged
    wiro_params = {
        "enableThinking": "false",
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "temperature": str(temperature),
        "top_p": str(top_p),
        "top_k": str(top_k),
        "repetition_penalty": "1.0",
        "length_penalty": "1",
        "max_tokens": str(max_tokens),
        "min_tokens": "0",
        "stop_sequences": ";".join(stop) if isinstance(stop, list) else (stop or ""),
        "seed": "0",
        "do_sample": "--do_sample",
    }

    try:
        payload = await client.run_until_done(settings.wiro_model, wiro_params)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Wiro error: {exc}")

    try:
        text = extract_assistant_text(payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not extract assistant text: {exc}")

    return JSONResponse(_anthropic_response(model, text))
