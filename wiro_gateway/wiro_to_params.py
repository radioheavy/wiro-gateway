"""Convert OpenAI/Anthropic-style chat messages into Wiro's prompt + system_prompt.

Wiro's qwen3-8-27b-uncensored Run endpoint takes a single `prompt` (text) plus
an optional `system_prompt`. We concatenate the multi-turn conversation into a
plain-text transcript that Qwen reliably interprets:

  [System]
  <system_prompt>

  [User]
  <msg1>

  [Assistant]
  <msg2>
  ...

We also try to detect tool / function-call output and skip it for the first
cut, since the model isn't a tool-calling model in Wiro's listing.
"""

from __future__ import annotations

from typing import Any, Iterable

Role = str
Message = dict[str, Any]


def _content_to_text(content: Any) -> str:
    """Coerce OpenAI / Anthropic content shapes into a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(part.get("text", "")))
                elif "text" in part:
                    parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def merge_messages(messages: Iterable[Message], system: str | None) -> tuple[str, str]:
    """Return (system_prompt, prompt) ready for the Wiro Run call."""
    sys_parts: list[str] = []
    if system:
        sys_parts.append(system.strip())

    convo: list[str] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        if role in ("system",):
            text = _content_to_text(m.get("content"))
            if text.strip():
                sys_parts.append(text.strip())
            continue
        if role in ("assistant", "model", "ai"):
            label = "Assistant"
        elif role in ("user", "human"):
            label = "User"
        elif role in ("tool", "function", "tool_result"):
            # Tool output - keep the textual parts so the model can see them.
            text = _content_to_text(m.get("content"))
            if text.strip():
                convo.append(f"[Tool]\n{text.strip()}\n")
            continue
        else:
            label = role.capitalize() or "User"
        text = _content_to_text(m.get("content"))
        if not text.strip():
            continue
        convo.append(f"[{label}]\n{text.strip()}\n")

    system_prompt = "\n\n".join(sys_parts).strip()
    prompt = "\n".join(convo).strip()
    if not prompt:
        # Wiro requires some text; we fall back to a friendly nudge so the call
        # still returns *something* rather than 400.
        prompt = "Hello."
    # Ask the model to reply as the assistant only.
    if "[Assistant]" not in prompt and not prompt.endswith("\n"):
        prompt += "\n"
    prompt += "[Assistant]\n"
    return system_prompt, prompt
