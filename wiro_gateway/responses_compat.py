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

Reasoning support:
  - `body["reasoning"]` (Codex Responses shape) is the canonical input.
  - `body["reasoning_effort"]` is also accepted for OpenAI Chat-compat clients.
  - Resolved effort is stamped on the Wiro Run call as `reasoning_effort=<...>`
    plus `enableThinking=true|false` (see wiro_gateway.reasoning).
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
from .reasoning import PRESETS, EffortLevel, apply_preset_to_wiro_params, resolve_effort

def _split_wiro_reasoning_and_answer(raw_text: str) -> tuple[str, str]:
    """Pull thinking text and the final answer out of a Wiro response.

    Wiro returns either:
      "<answer>foo</answer>" (chat profile, thinking off), or
      "thinking prose... </think>\n\n <answer>foo</answer> " (thinking on).

    Returns (reasoning_text, answer_text). Either may be empty.
    """
    if not raw_text:
        return "", ""
    # Strip the <think>...</think> block first.
    m_think = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    reasoning = m_think.group(1).strip() if m_think else ""
    after = raw_text[m_think.end():] if m_think else raw_text
    # Strip the <answer>...</answer> wrapper.
    m_ans = re.search(r"<answer>(.*?)</answer>", after, re.DOTALL)
    if m_ans:
        answer = m_ans.group(1).strip()
    else:
        # Fall back to the whole text minus the thinking block.
        answer = after.strip()
    return reasoning, answer


def _build_output_item_message(item_id: str, text: str) -> dict[str, Any]:
    """Codex strict serde: ResponseItem::Message { id, role, content: [ContentItem] }."""
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _build_output_item_reasoning(item_id: str, summary_text: str, raw_text: str) -> dict[str, Any]:
    """Codex strict serde: ResponseItem::Reasoning { id, summary, content }."""
    item: dict[str, Any] = {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": summary_text}] if summary_text else [],
    }
    if raw_text:
        item["content"] = [{"type": "reasoning_text", "text": raw_text}]
    return item
from .wiro_to_params import merge_messages

router = APIRouter()
log = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8

# How many characters to send per delta. 16 chars ~= a few words, which keeps
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


def _wiro_params_from_request(
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    settings: Settings,
) -> tuple[dict[str, Any], EffortLevel]:
    system_prompt, prompt = merge_messages(messages, None)
    effort = resolve_effort(body, settings.wiro_default_reasoning_effort)
    preset = PRESETS[effort]
    override = settings.wiro_preset_overrides_sampling

    temperature = float(body.get("temperature", preset.temperature if override else DEFAULT_TEMPERATURE))
    top_p = float(body.get("top_p", preset.top_p if override else DEFAULT_TOP_P))
    top_k_str = body.get("top_k", str(preset.top_k if override else 20))
    top_k = int(top_k_str) if str(top_k_str).strip() else 20
    max_tokens = int(
        body.get("max_output_tokens")
        or (preset.max_tokens if override else DEFAULT_MAX_OUTPUT_TOKENS)
    )

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
        "stop_sequences": "",
        "seed": "0",
        "do_sample": "--do_sample",
    }
    params = apply_preset_to_wiro_params(params, preset, override_sampling=override)
    return params, effort


async def _run_wiro(
    client,
    model_path: str,
    messages: list[dict[str, Any]],
    body: dict[str, Any],
    settings: Settings,
) -> tuple[str, dict[str, Any], EffortLevel]:
    wiro_params, effort = _wiro_params_from_request(body, messages, settings)
    payload = await client.run_until_done(model_path, wiro_params)
    text = extract_assistant_text(payload)
    return text, wiro_params, effort


def _build_response(
    model: str,
    text: str,
    pt: int,
    ct: int,
    effort: EffortLevel,
    wiro_params: dict[str, Any],
) -> dict[str, Any]:
    """Build a full OpenAI Responses API response.

    Codex 0.147+ strict-serde deserialises this whole object, so we include
    every documented field with sane defaults. Wiro-only metadata goes
    under `x_wiro` (Codex ignores unknown fields here as long as the
    OpenAI spec fields are present and correctly typed).
    """
    preset = PRESETS[effort]
    reasoning_block: dict[str, Any] = {
        "effort": effort,
        "summary": None,
    }
    # Estimate reasoning_tokens = how many of the output tokens we assume the
    # model spent thinking. Wiro doesn't break this out, so we only count
    # tokens when thinking is on, and use a conservative fraction.
    if preset.enable_thinking:
        reasoning_block["summary"] = "auto"
    reasoning_tokens = ct if preset.enable_thinking else 0
    return {
        # OpenAI Responses API base fields.
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "model": model,
        # Output: Codex strict serde requires ResponseItem::Message with
        # content: [ContentItem::OutputText { text }]. No `annotations` field,
        # no extra top-level keys -- those break the tagged enum parse.
        "output": [_build_output_item_message(f"msg_{uuid.uuid4().hex}", text)],
        "output_text": text,
        # OpenAI Responses API v1 spec fields Codex strict-serde requires.
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "reasoning": reasoning_block,
        "temperature": float(wiro_params.get("temperature", preset.temperature)),
        "top_p": float(wiro_params.get("top_p", preset.top_p)),
        "tool_choice": "auto",
        "tools": [],
        "truncation": "disabled",
        "service_tier": "default",
        "store": False,
        "safety_identifier": "wiro-gateway-local",
        "text": {"format": {"type": "text"}},
        "top_logprobs": 0,
        "usage": {
            "input_tokens": pt,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": ct,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": pt + ct,
        },
        "user": None,
        # Wiro-internal echo (Codex tolerates unknown fields, kept for
        # curl / debugging convenience).
        "x_wiro": {
            "effort": effort,
            "enable_thinking": preset.enable_thinking,
            "reasoning_effort": preset.reasoning_effort,
            "applied_params": {
                k: v for k, v in wiro_params.items()
                if k in ("enableThinking", "reasoning_effort", "temperature", "top_p", "top_k", "max_tokens")
            },
        },
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def _stream_response(
    model: str,
    text: str,
    pt: int,
    ct: int,
    effort: EffortLevel,
    wiro_params: dict[str, Any],
) -> AsyncIterator[bytes]:
    import os as _os
    _DUMP = _os.environ.get("WIRO_DEBUG_STREAM") == "1"
    preset = PRESETS[effort]
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
    yield _sse("response.created", {
        "type": "response.created",
        "response": {**base, "status": "in_progress", "output": []},
    })
    # 2. output_item.added (ResponseItem::Message in_progress)
    yield _sse("response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [],
        },
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
    # 5. text done -- Codex trace-logs this as unhandled, skip to reduce noise.
    # 6. content_part.done -- Codex ignored, skip.
    # 7. output_item.done (full ResponseItem::Message)
    yield _sse("response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": _build_output_item_message(item_id, text),
    })
    # 8. response.completed -- Codex strict serde parses this as
    #    ResponseCompleted { id, usage, end_turn }. Anything else trips
    #    the parser, so emit only the three documented fields.
    reasoning_tokens = ct if preset.enable_thinking else 0
    final = {
        "id": rid,
        "usage": {
            "input_tokens": pt,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": ct,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": pt + ct,
        },
        "end_turn": True,
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
        text, wiro_params, effort = await _run_wiro(client, settings.wiro_model, messages, body, settings)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Wiro error: {exc}")

    pt = sum(len(str(m.get("content", ""))) for m in messages) // 4
    ct = max(1, len(text) // 4)

    if body.get("stream"):
        return StreamingResponse(
            _stream_response(model, text, pt, ct, effort, wiro_params),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    return JSONResponse(_build_response(model, text, pt, ct, effort, wiro_params))
