"""FastAPI application factory for wiro-gateway."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .anthropic_compat import router as anthropic_router
from .client import WiroClient
from .config import Settings, get_settings
from .openai_compat import router as openai_router
from .responses_compat import router as responses_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    app.state.wiro = WiroClient(settings)
    log.info(
        "wiro-gateway %s listening on http://%s:%s model=%s",
        __version__, settings.gateway_host, settings.gateway_port, settings.wiro_model,
    )
    try:
        yield
    finally:
        await app.state.wiro.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="wiro-gateway",
        version=__version__,
        description="Local gateway: Wiro Qwen3.8-27B-Uncensored -> OpenAI / Anthropic compatible APIs.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.get("/")
    async def root():
        return {
            "name": "wiro-gateway",
            "version": __version__,
            "model": settings.wiro_model,
            "endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/models", "/healthz"],
        }

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "model": settings.wiro_model}

    app.include_router(openai_router)
    app.include_router(anthropic_router)
    app.include_router(responses_router)
    return app


app = create_app()
