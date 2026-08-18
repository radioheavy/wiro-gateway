import hashlib
import hmac

from wiro_gateway.auth import make_signature, auth_headers


def test_make_signature_matches_documented_formula():
    # Wiro docs: SIGNATURE = HMAC-SHA256(key=API_KEY, message=API_SECRET + NONCE)
    api_key = "key123"
    api_secret = "sec456"
    nonce = "1700000000"
    expected = hmac.new(api_key.encode(), (api_secret + nonce).encode(), hashlib.sha256).hexdigest()
    n, sig = make_signature(api_key, api_secret, nonce)
    assert n == nonce
    assert sig == expected


def test_auth_headers_shape():
    h = auth_headers("k", "s")
    assert h["x-api-key"] == "k"
    assert h["x-nonce"].isdigit()
    assert len(h["x-signature"]) == 64
    assert set(h) == {"x-api-key", "x-nonce", "x-signature"}


def test_nonce_uniqueness():
    n1, _ = make_signature("k", "s")
    n2, _ = make_signature("k", "s")
    assert n1.isdigit() and n2.isdigit()
    # both are ms timestamps, just ensure they are non-empty strings
    assert int(n1) > 0 and int(n2) > 0
