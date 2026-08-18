"""Reasoning-effort mapping for the Wiro Qwen3.8-27B-Uncensored backend.

Codex (>=0.147) and Claude Code both let the user pick a *reasoning effort*
(low / medium / high) from the UI. Wiro's Qwen 3.8 27B exposes that as a
`reasoning_effort` form field plus an `enableThinking` toggle. This module
translates "what the client asked for" into the exact Wiro params we should
send on the next Run call, and applies the Wiro-recommended sampling preset
for that effort (different temperatures, top_p, top_k) so the answer quality
matches what the Wiro model page advertises.

The defaults preserve the v0.1.0 behaviour: when no effort is requested we
leave `enableThinking=false` and use the chat profile (temp 0.7, top_p 0.8,
top_k 20). The user opts into thinking simply by setting
`WIRO_DEFAULT_REASONING_EFFORT=medium` in the gateway .env, or by passing
`reasoning: { effort: "high" }` in the request body -- both paths land here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

EffortLevel = Literal["none", "low", "medium", "high"]

# Order matters: "minimal" is an alias Codex sometimes sends; we normalise it
# to "low" so the Wiro preset table is the only source of truth.
_EFFORT_ALIASES: dict[str, EffortLevel] = {
    "": "none",
    "none": "none",
    "off": "none",
    "disabled": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "med": "medium",
    "high": "high",
    "max": "high",
    "maximum": "high",
}

SUPPORTED_EFFORTS: tuple[EffortLevel, ...] = ("none", "low", "medium", "high")


@dataclass(frozen=True)
class EffortPreset:
    """Wiro-recommended sampling for a given reasoning effort.

    Numbers are taken from the Wiro model page for qwen/qwen3-8-27b-uncensored.
    "none" is the chat profile (thinking disabled); "low/medium/high" turn on
    `enableThinking` and tweak the sampler to keep thinking text coherent.
    """

    enable_thinking: bool
    reasoning_effort: str  # "" when not sent (keeps the Wiro payload clean)
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int  # baseline; client can still override via body field
    description: str


# Source of truth: Wiro model page "Recommended settings" + Qwen 3.8 27B
# chat-template notes. Update these in one place if Wiro rotates the values.
PRESETS: dict[EffortLevel, EffortPreset] = {
    "none": EffortPreset(
        enable_thinking=False,
        reasoning_effort="",
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=4096,
        description="Chat profile -- fast, no chain-of-thought (legacy v0.1.0 default).",
    ),
    "low": EffortPreset(
        enable_thinking=True,
        reasoning_effort="low",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=8192,
        description="Light reasoning -- quick checks, short plans.",
    ),
    "medium": EffortPreset(
        enable_thinking=True,
        reasoning_effort="medium",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=12288,
        description="Balanced reasoning -- multi-step edits, refactors, debugging.",
    ),
    "high": EffortPreset(
        enable_thinking=True,
        reasoning_effort="high",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
        description="Deep reasoning -- architecture, hard bugs, long planning.",
    ),
}


def normalise_effort(raw: Any) -> EffortLevel:
    """Coerce any incoming effort value into a known EffortLevel.

    Accepts None, "", "low"/"medium"/"high", plus common aliases
    (minimal/off/disabled/max). Anything unrecognised returns "none" so the
    gateway still answers instead of 400ing on bad input.
    """
    if raw is None:
        return "none"
    if isinstance(raw, bool):
        return "low" if raw else "none"
    s = str(raw).strip().lower()
    return _EFFORT_ALIASES.get(s, "none")


def extract_effort_from_body(body: dict[str, Any]) -> EffortLevel:
    """Pull the requested effort out of an OpenAI/Anthropic/Responses body.

    Lookup order (first hit wins):
      1. `body["reasoning"]["effort"]`              (Codex Responses, Claude thinking)
      2. `body["reasoning_effort"]`                 (OpenAI Chat Completions)
      3. `body["thinking"]["effort"]`               (Claude Code thinking block)
      4. `body["reasoning_effort_level"]`           (rare client variant)
    """
    if not isinstance(body, dict):
        return "none"
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        eff = normalise_effort(reasoning.get("effort"))
        # Treat an explicit "none" payload as a valid request to disable.
        if "effort" in reasoning:
            return eff
    direct = body.get("reasoning_effort")
    if direct is not None:
        return normalise_effort(direct)
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        # Claude Code's native thinking block: {"type": "enabled" | "disabled" | "adaptive"}.
        # Map to our effort ladder so the Wiro preset applies.
        t = thinking.get("type")
        if t == "enabled":
            return "low"
        if t == "adaptive":
            return "medium"
        if t == "disabled":
            return "none"
        return normalise_effort(thinking.get("effort"))
    legacy = body.get("reasoning_effort_level")
    if legacy is not None:
        return normalise_effort(legacy)
    return "none"


def resolve_effort(body: dict[str, Any], default: str) -> EffortLevel:
    """Decide which effort a request should use, with explicit override wins."""
    requested = extract_effort_from_body(body)
    if requested != "none":
        return requested
    return normalise_effort(default)


def apply_preset_to_wiro_params(
    params: dict[str, Any],
    preset: EffortPreset,
    *,
    override_sampling: bool = True,
) -> dict[str, Any]:
    """Stamp the preset's sampling + thinking toggles onto a Wiro params dict.

    If `override_sampling` is False, only the thinking toggles are touched --
    useful when the request body supplied its own temperature/top_p and we
    should not clobber it.
    """
    out = dict(params)
    out["enableThinking"] = "true" if preset.enable_thinking else "false"
    if preset.reasoning_effort:
        out["reasoning_effort"] = preset.reasoning_effort
    else:
        out.pop("reasoning_effort", None)
    if override_sampling:
        out["temperature"] = str(preset.temperature)
        out["top_p"] = str(preset.top_p)
        out["top_k"] = str(preset.top_k)
        out["max_tokens"] = str(preset.max_tokens)
    return out


def describe_effort(effort: EffortLevel) -> str:
    """Short human label, e.g. 'medium (thinking on, temp 0.6)'."""
    p = PRESETS[effort]
    if not p.enable_thinking:
        return f"{effort} (thinking off, temp {p.temperature})"
    return f"{effort} (thinking on, temp {p.temperature}, reasoning_effort={p.reasoning_effort})"


def all_presets() -> Iterable[tuple[EffortLevel, EffortPreset]]:
    """Yield (level, preset) in display order."""
    for level in SUPPORTED_EFFORTS:
        yield level, PRESETS[level]
