import pytest
from httpx import ASGITransport, AsyncClient

from wiro_gateway.config import Settings
from wiro_gateway.server import create_app


class _FakeWiro:
    def __init__(self, text: str = "pong"):
        self._text = text
        self.last_params = None
        self.last_model = None

    async def aclose(self):
        return None

    async def run_until_done(self, model_path, params):
        self.last_model = model_path
        self.last_params = params
        return {
            "tasklist": [
                {
                    "id": "1",
                    "status": "task_postprocess_end",
                    "debugoutput": self._text,
                    "outputs": [],
                    "parameters": {},
                }
            ],
            "result": True,
        }


@pytest.mark.asyncio
async def test_openai_chat_completions():
    s = Settings(
        wiro_api_key="k", wiro_api_secret="s",
        wiro_model="qwen/qwen3-8-27b-uncensored",
        wiro_poll_interval_sec=0.01, wiro_poll_timeout_sec=2,
    )
    app = create_app(s)
    fake = _FakeWiro(text="Hello from Wiro!")
    app.state.wiro = fake
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "any",
            "messages": [{"role": "user", "content": "Hi"}],
        })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello from Wiro!"
    assert body["model"] == "any"
    assert fake.last_model == "qwen/qwen3-8-27b-uncensored"
    assert "Hi" in fake.last_params["prompt"]
    assert fake.last_params["enableThinking"] == "false"


@pytest.mark.asyncio
async def test_anthropic_messages():
    s = Settings(
        wiro_api_key="k", wiro_api_secret="s",
        wiro_model="qwen/qwen3-8-27b-uncensored",
        wiro_poll_interval_sec=0.01, wiro_poll_timeout_sec=2,
    )
    app = create_app(s)
    fake = _FakeWiro(text="Anthropic reply")
    app.state.wiro = fake
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/messages", json={
            "model": "any",
            "max_tokens": 100,
            "system": "You are concise.",
            "messages": [{"role": "user", "content": "Hello"}],
        })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "Anthropic reply"
    assert "You are concise." in fake.last_params["system_prompt"]
    assert "Hello" in fake.last_params["prompt"]


@pytest.mark.asyncio
async def test_healthz():
    s = Settings(wiro_api_key="k", wiro_api_secret="s")
    app = create_app(s)
    app.state.wiro = _FakeWiro()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_openai_empty_messages_400():
    s = Settings(wiro_api_key="k", wiro_api_secret="s")
    app = create_app(s)
    app.state.wiro = _FakeWiro()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 400
