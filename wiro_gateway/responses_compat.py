"""OpenAI Responses API compatible endpoint with SSE streaming.

Codex (>=0.147) sends `stream: true` to /v1/responses. Wiro is async and returns
the full text in one shot, so we "fake" streaming: we wait for the full answer,
then emit OpenAI-style SSE events:

  response.created
  response.output_item.added
  response.content_part.added
  response.output_text.delta       (one or more)
  response.output_text.done
  response.content_part.done
  response.output_item.done
  response.completed
  [DONE]

If `stream: false` (or omitted), we return a single JSON response in the same
shape but with `object: "response"`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .extract import extract_assistant_text
from .wiro_to_params import merge_messages

router = APIRouter()
log = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8

# How many characters to send per delta. 16 chars ≈ a few words, which keeps
# the client progress bar smooth without flooding the wire.
DELTA_CHARS = 16


def _input_to_messages(input_: Any, instructions: str | None) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if isinstance(input_, str):
        if input_.strip():
            msgs.append({"role": "user", "content": input_})
    elif isinstance(input_, list):
        for item in input_:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or item.get("type") or "user"
            content = item.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        if p.get("type") in ("input_text", "output_text", "text"):
                            parts.append(str(p.get("text", "")))
                        elif "text" in p:
                            parts.append(str(p["text"]))
                content = "\n".join(parts)
            msgs.append({"role": role, "content": content or ""})
    elif isinstance(input_, dict):
        msgs.append({"role": input_.get("role", "user"), "content": str(input_.get("content", ""))})

    if instructions and instructions.strip():
        msgs.insert(0, {"role": "system", "content": instructions.strip()})
    return msgs


async def _run_wiro(client, model_path: str, messages: list[dict[str, Any]], body: dict[str, Any]) -> str:
    system_prompt, prompt = merge_messages(messages, None)
    wiro_params = {
        "enableThinking": "false",
        "prompt": prompt,
        "system_prompt": system_prompt or "",
        "temperature": str(float(body.get("temperature", DEFAULT_TEMPERATURE))),
        "top_p": str(float(body.get("top_p", DEFAULT_TOP_P))),
        "top_k": "20",
        "repetition_penalty": "1.0",
        "length_penalty": "1",
        "max_tokens": str(int(body.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)),
        "min_tokens": "0",
        "stop_sequences": "",
        "seed": "0",
        "do_sample": "--do_sample",
    }
    payload = await client.run_until_done(model_path, wiro_params)
    return extract_assistant_text(payload)


def _build_response(model: str, text: str, pt: int, ct: int) -> dict[str, Any]:
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created": int(time.time()),
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "output_text": text,
        "usage": {"input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct},
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def _stream_response(model: str, text: str, pt: int, ct: int) -> AsyncIterator[bytes]:
    rid = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    base = {
        "id": rid,
        "object": "response",
        "created": created,
        "model": model,
    }
    # 1. created
    yield _sse("response.created", {"type": "response.created", "response": {**base, "status": "in_progress", "output": []}})
    # 2. output_item.added
    yield _sse("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"type": "message", "role": "assistant", "status": "in_progress", "content": []},
    })
    # 3. content_part.added
    yield _sse("response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": ""},
    })
    # 4. deltas
    if text:
        for i in range(0, len(text), DELTA_CHARS):
            chunk = text[i : i + DELTA_CHARS]
            yield _sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": chunk,
            })
            # Tiny delay so the client renders incrementally.
            await asyncio.sleep(0.01)
    # 5. text done
    yield _sse("response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": text,
    })
    # 6. content_part.done
    yield _sse("response.content_part.done", {
        "type": "response.content_part.done",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": text},
    })
    # 7. output_item.done
    yield _sse("response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        },
    })
    # 8. response.completed
    final = {
        **base,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "output_text": text,
        "usage": {"input_tokens": pt, "output_tokens": ct, "total_tokens": pt + ct},
    }
    yield _sse("response.completed", {"type": "response.completed", "response": final})
    # 9. terminator
    yield b"data: [DONE]\n\n"


@router.post("/v1/responses")
async def responses(request: Request):
    settings: Settings = request.app.state.settings
    client = request.app.state.wiro

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model") or settings.wiro_model
    messages = _input_to_messages(body.get("input"), body.get("instructions"))
    if not messages:
        raise HTTPException(status_code=400, detail="`input` must be a non-empty string or list")

    try:
        text = await _run_wiro(client, settings.wiro_model, messages, body)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Wiro error: {exc}")

    pt = sum(len(str(m.get("content", ""))) for m in messages) // 4
    ct = max(1, len(text) // 4)

    if body.get("stream"):
        return StreamingResponse(
            _stream_response(model, text, pt, ct),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    return JSONResponse(_build_response(model, text, pt, ct))
