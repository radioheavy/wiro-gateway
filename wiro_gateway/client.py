"""Async HTTP client for the Wiro Run + Task/Detail endpoints.

Wiro is async: `Run/<model>` returns a `taskid` + `socketaccesstoken`, then we
poll `Task/Detail` until status reaches a terminal state. We do not currently
use the WebSocket channel because the model is text and Wiro does not stream
LLM tokens incrementally (debugoutput only appears once the worker finishes).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .auth import auth_headers
from .config import Settings

log = logging.getLogger(__name__)

TERMINAL_STATUSES = {"task_postprocess_end", "task_cancel", "task_killed"}


class WiroError(RuntimeError):
    """Raised for any Wiro API failure that the gateway cannot recover from."""

    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class WiroClient:
    def __init__(self, settings: Settings, *, http: httpx.AsyncClient | None = None):
        self.s = settings
        self._http = http or httpx.AsyncClient(timeout=60.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- low level ----------------------------------------------------

    async def _post_form(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.s.wiro_api_base}{path}"
        headers = auth_headers(self.s.wiro_api_key, self.s.wiro_api_secret)
        # Wiro accepts form-encoded bodies (application/x-www-form-urlencoded).
        # httpx will set the right Content-Type with the proper charset for us.
        r = await self._http.post(url, data=data, headers=headers)
        if r.status_code >= 500:
            raise WiroError(f"Wiro 5xx for {path}: {r.text}", status_code=r.status_code)
        try:
            payload = r.json()
        except Exception as exc:
            raise WiroError(
                f"Non-JSON Wiro response for {path}: {r.text[:200]}", status_code=r.status_code
            ) from exc

        if r.status_code == 401:
            raise WiroError("Wiro auth failed (401). Check WIRO_API_KEY / WIRO_API_SECRET.", payload=payload)
        if r.status_code == 403:
            raise WiroError("Wiro signature/permission failure (403).", payload=payload)
        if r.status_code == 429:
            raise WiroError("Wiro rate limited (429). Slow down and retry.", status_code=429, payload=payload)

        if payload.get("result") is False:
            errs = payload.get("errors") or []
            msg = errs[0].get("message") if errs and isinstance(errs[0], dict) else str(errs)
            raise WiroError(f"Wiro result=false: {msg}", payload=payload)

        return payload

    # --- public -------------------------------------------------------

    async def run_model(self, model_path: str, params: dict[str, Any]) -> tuple[str, str]:
        """Start a Run task. Returns (taskid, socketaccesstoken)."""
        payload = await self._post_form(f"/Run/{model_path}", params)
        taskid = payload.get("taskid")
        token = payload.get("socketaccesstoken")
        if not taskid or not token:
            raise WiroError(
                f"Run response missing taskid/token: keys={list(payload.keys())}", payload=payload
            )
        return str(taskid), str(token)

    async def get_task_detail(self, *, taskid: str | None = None, tasktoken: str | None = None) -> dict[str, Any]:
        if not taskid and not tasktoken:
            raise ValueError("Either taskid or tasktoken is required")
        data: dict[str, Any] = {}
        if taskid:
            data["taskid"] = taskid
        if tasktoken:
            data["tasktoken"] = tasktoken
        return await self._post_form("/Task/Detail", data)

    async def run_until_done(self, model_path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Start a Run and poll until terminal. Returns the final Task/Detail payload."""
        taskid, token = await self.run_model(model_path, params)
        log.info("Wiro task started: id=%s model=%s", taskid, model_path)
        deadline = asyncio.get_event_loop().time() + self.s.wiro_poll_timeout_sec
        interval = max(0.1, self.s.wiro_poll_interval_sec)
        attempt = 0
        while True:
            attempt += 1
            if asyncio.get_event_loop().time() > deadline:
                raise WiroError(f"Wiro task {taskid} timed out after {self.s.wiro_poll_timeout_sec}s")
            try:
                payload = await self.get_task_detail(taskid=taskid, tasktoken=token)
            except WiroError as exc:
                if exc.status_code == 429:
                    log.warning("Wiro 429 during poll, sleeping 5s")
                    await asyncio.sleep(5.0)
                    continue
                raise
            task = (payload.get("tasklist") or [{}])[0]
            status = task.get("status")
            log.debug("poll #%s task=%s status=%s", attempt, taskid, status)
            if status in TERMINAL_STATUSES:
                if status != "task_postprocess_end":
                    raise WiroError(
                        f"Wiro task {taskid} ended in non-success status={status!r}: "
                        f"debugerror={task.get('debugerror')!r}",
                        payload=payload,
                    )
                return payload
            await asyncio.sleep(interval)
