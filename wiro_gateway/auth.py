"""HMAC-SHA256 request signing for Wiro's signature-based auth.

Wiro docs: SIGNATURE = HMAC-SHA256(key=API_KEY, message=API_SECRET + NONCE)
Headers:   x-api-key, x-nonce, x-signature
The nonce is a unix timestamp (or any random integer); a fresh nonce per request
prevents replay.
"""

from __future__ import annotations

import hashlib
import hmac
import time


def make_signature(api_key: str, api_secret: str, nonce: str | int | None = None) -> tuple[str, str]:
    """Return (nonce, signature) for a Wiro request.

    The signature is HMAC-SHA256 of (api_secret + nonce) keyed by api_key.
    """
    if nonce is None:
        nonce = str(int(time.time() * 1000))  # ms precision; the doc only requires uniqueness
    msg = (api_secret + str(nonce)).encode("utf-8")
    sig = hmac.new(api_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return str(nonce), sig


def auth_headers(api_key: str, api_secret: str) -> dict[str, str]:
    """Return the three headers every Wiro request needs."""
    nonce, sig = make_signature(api_key, api_secret)
    return {
        "x-api-key": api_key,
        "x-nonce": nonce,
        "x-signature": sig,
    }
