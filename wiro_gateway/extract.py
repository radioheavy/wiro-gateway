"""Extract the assistant's text from a completed Wiro LLM task.

The Wiro docs only show a generic image-to-image example whose `outputs[]` has
`url` (CDN PNG). For LLM tasks we need to be defensive: the assistant text may
appear in any of `debugoutput`, one of the `outputs[]` entries as `content` /
inline data, or as a `text` field. We try a few common shapes and fall back to
a CDN fetch if the only `outputs[]` we have is a URL whose `contenttype` looks
text-like. Finally, we strip common XML wrappers like `<answer>...</answer>`
that several open-source chat models (Qwen 3.8 included) add around their reply.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Matches a clean wrapper like <answer>...</answer> around the entire text.
_XML_WRAPPER_RE = re.compile(r"^<([A-Za-z_][A-Za-z0-9_-]*)>(.*)</\1>\s*$", re.DOTALL)


def extract_assistant_text(task_detail: dict[str, Any]) -> str:
    """Return the assistant text from a successful Wiro task detail response.

    Raises ValueError if we cannot locate any text. The caller should treat that
    as a hard failure rather than silently returning an empty string.
    """
    tasklist = task_detail.get("tasklist") or []
    if not tasklist:
        raise ValueError("tasklist is empty in Wiro response")
    task = tasklist[0]

    # 1. The most common Wiro LLM output field.
    debug = (task.get("debugoutput") or "").strip()
    if debug:
        return _strip_xml_wrappers(debug)

    # 2. `outputs[].content` — may be plain text or a JSON-encoded payload.
    for out in task.get("outputs") or []:
        content = out.get("content")
        if isinstance(content, str) and content.strip():
            stripped = content.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    t = _find_text_in_obj(parsed)
                    if t:
                        return _strip_xml_wrappers(t)
                except json.JSONDecodeError:
                    pass
            return _strip_xml_wrappers(content)
        if isinstance(content, dict):
            t = _find_text_in_obj(content)
            if t:
                return _strip_xml_wrappers(t)

    # 3. `outputs[].text`
    for out in task.get("outputs") or []:
        t = out.get("text")
        if isinstance(t, str) and t.strip():
            return _strip_xml_wrappers(t)

    # 4. `outputs[].url` pointing at a text/plain or text/html CDN asset.
    for out in task.get("outputs") or []:
        url = out.get("url")
        ctype = (out.get("contenttype") or "").lower()
        if url and ("text" in ctype or out.get("name", "").endswith((".txt", ".md"))):
            try:
                with httpx.Client(timeout=30.0) as c:
                    r = c.get(url)
                    r.raise_for_status()
                    return _strip_xml_wrappers(r.text)
            except Exception as exc:  # pragma: no cover - network path
                log.warning("Failed to fetch text from output url %s: %s", url, exc)

    # 5. `parameters` echoed back sometimes carries a `result` field (legacy).
    params = task.get("parameters") or {}
    if isinstance(params, dict):
        for k in ("result", "text", "output", "response"):
            v = params.get(k)
            if isinstance(v, str) and v.strip():
                return _strip_xml_wrappers(v)

    raise ValueError(
        f"Could not locate assistant text in Wiro task detail. "
        f"status={task.get('status')!r} debugoutput={(task.get('debugoutput') or '')[:120]!r} "
        f"outputs={task.get('outputs')!r}"
    )


def _strip_xml_wrappers(text: str) -> str:
    """Remove a leading/trailing XML wrapper like <answer>...</answer>.

    Several open-source chat models wrap their reply in such tags. We only strip
    the wrapper when it is a clean pair around the whole text; if the tags
    appear mid-text or are mismatched we leave the response alone so we do not
    eat legitimate content.
    """
    s = text.strip()
    if not (s.startswith("<") and s.endswith(">")):
        return text
    m = _XML_WRAPPER_RE.match(s)
    if not m:
        return text
    inner = m.group(2).strip()
    return inner if inner else text


def _find_text_in_obj(obj: Any) -> str | None:
    """Walk a JSON-like object looking for a likely assistant text leaf."""
    if isinstance(obj, dict):
        for k in ("text", "content", "message", "response", "answer"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for v in obj.values():
            r = _find_text_in_obj(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_text_in_obj(v)
            if r:
                return r
    return None
