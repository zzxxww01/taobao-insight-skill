# -*- coding: utf-8 -*-
"""Raw CDP browser manager.

This module replaces the previous Playwright-over-CDP implementation with a
native DevTools Protocol implementation, following the same launch/connect
pattern used by the baoyu skill browser flow:
1) launch Chrome with --remote-debugging-port
2) poll /json/version for webSocketDebuggerUrl
3) connect over WebSocket and operate page targets via Target.attachToTarget
"""

from __future__ import annotations

import asyncio
import ast
import atexit
import base64
import json
import os
import platform
import re
import signal
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx

from tools.browser_hardening import (
    DEFAULT_ACCEPT_LANGUAGE,
    DEFAULT_LOCALE,
    DEFAULT_TIMEZONE_ID,
    build_browser_launch_args,
    load_stealth_script_source,
)
from tools import utils


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _find_available_port(start_port: int = 9222, max_attempts: int = 200) -> int:
    port = max(1024, int(start_port))
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"Cannot allocate debug port from {start_port} after {max_attempts} attempts")


def _find_chrome_executable(override_path: str = "") -> str:
    override = (override_path or "").strip()
    if override and os.path.isfile(override):
        return override

    candidates: list[str] = []
    system = platform.system()
    if system == "Windows":
        candidates = [
            os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe"),
        ]
    elif system == "Darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ]
    else:
        candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
            "/usr/bin/microsoft-edge",
            "/usr/bin/microsoft-edge-stable",
        ]

    for path in candidates:
        if os.path.isfile(path):
            return path
    raise RuntimeError(
        "Chrome/Edge executable not found. Set CUSTOM_BROWSER_PATH to a valid executable path."
    )


def _parse_http_debug_base(cdp_url: str, fallback_port: int = 9222) -> str:
    value = (cdp_url or "").strip()
    if not value:
        return f"http://127.0.0.1:{fallback_port}"
    if value.startswith("ws://") or value.startswith("wss://"):
        raise ValueError("websocket endpoint does not have an HTTP debug base")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    lowered = value.lower()
    if lowered.endswith("/json/version"):
        return value[: -len("/json/version")]
    if lowered.endswith("/json"):
        return value[: -len("/json")]
    if lowered.endswith("/json/list"):
        return value[: -len("/json/list")]
    return value.rstrip("/")


async def _wait_ws_url_from_debug_port(port: int, timeout_sec: int = 30) -> str:
    deadline = asyncio.get_running_loop().time() + max(5, int(timeout_sec))
    last_error = ""
    endpoint = f"http://127.0.0.1:{int(port)}/json/version"
    while asyncio.get_running_loop().time() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(endpoint, timeout=3.0)
            if resp.status_code == 200:
                payload = resp.json()
                ws_url = str(payload.get("webSocketDebuggerUrl", "")).strip()
                if ws_url:
                    return ws_url
                last_error = "missing webSocketDebuggerUrl"
            else:
                last_error = f"http {resp.status_code}"
        except Exception as exc:  # pragma: no cover - network failures are runtime-specific
            last_error = str(exc)
        await asyncio.sleep(0.2)
    raise RuntimeError(f"Chrome debug port not ready on {port}: {last_error}")


async def _resolve_ws_url(cdp_url: str) -> str:
    value = (cdp_url or "").strip()
    if not value:
        raise RuntimeError("empty cdp endpoint")
    if value.startswith(("ws://", "wss://")):
        return value
    base = _parse_http_debug_base(value)
    endpoint = base + "/json/version"
    async with httpx.AsyncClient() as client:
        resp = await client.get(endpoint, timeout=5.0)
    if resp.status_code != 200:
        raise RuntimeError(f"CDP endpoint is unavailable: {endpoint} -> HTTP {resp.status_code}")
    payload = resp.json()
    ws_url = str(payload.get("webSocketDebuggerUrl", "")).strip()
    if not ws_url:
        raise RuntimeError(f"CDP endpoint has no webSocketDebuggerUrl: {endpoint}")
    return ws_url


class CdpConnection:
    """Browser-level CDP connection with session-aware command routing."""

    def __init__(self, ws: Any, timeout_sec: float = 30.0):
        self._ws = ws
        self._timeout_sec = max(5.0, float(timeout_sec))
        self._next_id = 0
        self._pending: Dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._event_handlers: Dict[str, set[Callable[[dict[str, Any]], None]]] = {}
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._closed = False

    @classmethod
    async def connect(cls, ws_url: str, timeout_sec: float = 30.0) -> "CdpConnection":
        import websockets

        ws = await websockets.connect(
            ws_url,
            open_timeout=max(5.0, float(timeout_sec)),
            ping_interval=20,
        )
        conn = cls(ws=ws, timeout_sec=timeout_sec)
        conn._reader_task = asyncio.create_task(conn._reader_loop())
        return conn

    async def _reader_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                msg_id = msg.get("id")
                if isinstance(msg_id, int):
                    fut = self._pending.pop(msg_id, None)
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
                    continue
                method = str(msg.get("method", "") or "")
                if not method:
                    continue
                handlers = self._event_handlers.get(method, set())
                if not handlers:
                    continue
                for handler in list(handlers):
                    try:
                        handler(msg)
                    except Exception:
                        continue
        except Exception:
            pass
        finally:
            self._closed = True
            pending = list(self._pending.values())
            self._pending.clear()
            for fut in pending:
                if not fut.done():
                    fut.set_exception(RuntimeError("CDP connection closed"))

    async def send(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        *,
        session_id: str = "",
        timeout_sec: float = 15.0,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("CDP connection is closed")
        self._next_id += 1
        msg_id = self._next_id
        payload: dict[str, Any] = {"id": msg_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[msg_id] = fut

        await self._ws.send(json.dumps(payload, ensure_ascii=False))

        try:
            response = await asyncio.wait_for(
                fut, timeout=max(1.0, float(timeout_sec))
            )
        except Exception:
            self._pending.pop(msg_id, None)
            raise

        if "error" in response:
            err = response.get("error")
            raise RuntimeError(f"CDP {method} failed: {err}")
        return dict(response.get("result", {}) or {})

    def on(self, method: str, handler: Callable[[dict[str, Any]], None]) -> None:
        handlers = self._event_handlers.setdefault(method, set())
        handlers.add(handler)

    def off(self, method: str, handler: Callable[[dict[str, Any]], None]) -> None:
        handlers = self._event_handlers.get(method)
        if not handlers:
            return
        handlers.discard(handler)
        if not handlers:
            self._event_handlers.pop(method, None)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.close()
        except Exception:
            pass
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except Exception:
                pass
            self._reader_task = None
        pending = list(self._pending.values())
        self._pending.clear()
        for fut in pending:
            if not fut.done():
                fut.set_exception(RuntimeError("CDP connection closed"))

    @property
    def is_connected(self) -> bool:
        return not self._closed


class _RawCdpMouse:
    def __init__(self, page: "RawCdpPage"):
        self._page = page

    async def wheel(self, delta_x: int, delta_y: int) -> None:
        _ = delta_x
        await self._page.evaluate(f"window.scrollBy(0, {int(delta_y)}); true")


def _parse_has_text_selector(selector: str) -> tuple[str, str] | None:
    matched = re.match(
        r"^(?P<base>.*?):has-text\((?P<literal>'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")\)\s*$",
        str(selector or "").strip(),
    )
    if not matched:
        return None
    base_selector = str(matched.group("base") or "").strip() or "*"
    try:
        text_value = ast.literal_eval(matched.group("literal"))
    except Exception:
        return None
    return base_selector, str(text_value or "")


def _build_node_lookup_expression(
    *,
    selector: str = "",
    text: str = "",
    exact: bool = False,
) -> str:
    if text:
        label_json = json.dumps(text, ensure_ascii=False)
        exact_json = "true" if exact else "false"
        return f"""
(() => {{
  const label = {label_json};
  const exact = {exact_json};
  const clickableSelector = 'a,button,[role="tab"],[role="link"],[role="button"],[onclick],label';
  const poolSelector = 'a,button,[role="tab"],[role="link"],[role="button"],[onclick],li,span,div,p,label';
  const isVisible = (node) => {{
    if (!node) return false;
    const style = window.getComputedStyle(node);
    if (!style || style.visibility === 'hidden' || style.display === 'none') return false;
    const rect = node.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }};
  const nodes = Array.from(document.querySelectorAll(poolSelector));
  const candidates = [];
  for (const node of nodes) {{
    const textValue = String(node.innerText || node.textContent || '').replace(/\\s+/g, ' ').trim();
    if (!textValue) continue;
    if (exact ? textValue !== label : !textValue.includes(label)) continue;
    if (!isVisible(node)) continue;
    const clickable = node.closest ? (node.closest(clickableSelector) || node) : node;
    candidates.push({{
      target: clickable,
      exact: textValue === label ? 1 : 0,
      short: textValue.length <= Math.max(20, label.length + 10) ? 1 : 0,
      direct: clickable === node ? 1 : 0,
      length: textValue.length,
    }});
  }}
  candidates.sort((left, right) => {{
    if (left.exact !== right.exact) return right.exact - left.exact;
    if (left.short !== right.short) return right.short - left.short;
    if (left.direct !== right.direct) return right.direct - left.direct;
    return left.length - right.length;
  }});
  return candidates.length ? candidates[0].target : null;
}})()
"""

    has_text_selector = _parse_has_text_selector(selector)
    if has_text_selector is not None:
        base_selector, text_value = has_text_selector
        base_selector_json = json.dumps(base_selector, ensure_ascii=False)
        label_json = json.dumps(text_value, ensure_ascii=False)
        return f"""
(() => {{
  const baseSelector = {base_selector_json};
  const label = {label_json};
  const clickableSelector = 'a,button,[role="tab"],[role="link"],[role="button"],[onclick],label';
  const isVisible = (node) => {{
    if (!node) return false;
    const style = window.getComputedStyle(node);
    if (!style || style.visibility === 'hidden' || style.display === 'none') return false;
    const rect = node.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }};
  let nodes = [];
  try {{
    nodes = Array.from(document.querySelectorAll(baseSelector));
  }} catch (error) {{
    return null;
  }}
  const candidates = [];
  for (const node of nodes) {{
    const textValue = String(node.innerText || node.textContent || '').replace(/\\s+/g, ' ').trim();
    if (!textValue || !textValue.includes(label) || !isVisible(node)) continue;
    const clickable = node.closest ? (node.closest(clickableSelector) || node) : node;
    candidates.push({{
      target: clickable,
      short: textValue.length <= Math.max(20, label.length + 10) ? 1 : 0,
      direct: clickable === node ? 1 : 0,
      length: textValue.length,
    }});
  }}
  candidates.sort((left, right) => {{
    if (left.short !== right.short) return right.short - left.short;
    if (left.direct !== right.direct) return right.direct - left.direct;
    return left.length - right.length;
  }});
  return candidates.length ? candidates[0].target : null;
}})()
"""

    selector_value = str(selector or "").strip()
    if selector_value.startswith("text="):
        return _build_node_lookup_expression(text=selector_value[5:], exact=False)

    selector_json = json.dumps(selector_value, ensure_ascii=False)
    return f"""
(() => {{
  try {{
    return document.querySelector({selector_json});
  }} catch (error) {{
    return null;
  }}
}})()
"""


class _RawCdpLocator:
    def __init__(
        self,
        page: "RawCdpPage",
        selector: str = "",
        *,
        text: str = "",
        exact: bool = False,
    ):
        self._page = page
        self._selector = selector
        self._text = text
        self._exact = bool(exact)

    @property
    def first(self) -> "_RawCdpLocator":
        return self

    def _node_lookup_expression(self) -> str:
        return _build_node_lookup_expression(
            selector=self._selector,
            text=self._text,
            exact=self._exact,
        )

    async def _run_on_node(
        self,
        action_body: str,
        *,
        timeout: int = 1500,
    ) -> Any:
        deadline = asyncio.get_running_loop().time() + max(0.2, float(timeout) / 1000.0)
        lookup = self._node_lookup_expression()
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                result = await self._page.evaluate(
                    f"""
(() => {{
  const node = {lookup};
  if (!node) return {{ found: false }};
  {action_body}
}})()
"""
                )
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.05)
                continue
            if isinstance(result, dict) and result.get("found"):
                return result.get("value")
            await asyncio.sleep(0.05)

        target = self._text or self._selector or "<unknown>"
        if last_error is not None:
            raise RuntimeError(f"locator action failed for {target}: {last_error}")
        raise RuntimeError(f"locator node not found: {target}")

    async def click(self, timeout: int = 1500) -> None:
        await self._run_on_node(
            """
  node.scrollIntoView({ block: "center", inline: "center" });
  if (typeof node.click === "function") node.click();
  return { found: true, value: true };
""",
            timeout=timeout,
        )

    async def evaluate(self, expression: str, timeout: int = 1500) -> Any:
        expression_json = json.dumps(str(expression or ""), ensure_ascii=False)
        return await self._run_on_node(
            f"""
  const source = {expression_json};
  const fn = eval(source);
  return {{ found: true, value: fn(node) }};
""",
            timeout=timeout,
        )

    async def get_attribute(self, name: str, timeout: int = 1500) -> Any:
        name_json = json.dumps(str(name or ""), ensure_ascii=False)
        return await self._run_on_node(
            f"""
  return {{ found: true, value: node.getAttribute({name_json}) }};
""",
            timeout=timeout,
        )

    async def scroll_into_view_if_needed(self, timeout: int = 1500) -> None:
        await self._run_on_node(
            """
  node.scrollIntoView({ block: "center", inline: "center" });
  return { found: true, value: true };
""",
            timeout=timeout,
        )


class _RawCdpResponse:
    def __init__(
        self,
        *,
        page: "RawCdpPage",
        request_id: str,
        response_url: str,
        status: int = 0,
    ):
        self._page = page
        self._request_id = str(request_id or "")
        self.url = str(response_url or "")
        self.status = int(status or 0)

    async def text(self) -> str:
        if not self._request_id:
            return ""
        last_error: Exception | None = None
        for _ in range(6):
            try:
                payload = await self._page.conn.send(
                    "Network.getResponseBody",
                    {"requestId": self._request_id},
                    session_id=self._page.session_id,
                    timeout_sec=10.0,
                )
                text = str(payload.get("body", "") or "")
                if payload.get("base64Encoded"):
                    return base64.b64decode(text).decode("utf-8", errors="replace")
                return text
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.2)
        if last_error is not None:
            raise last_error
        return ""


class RawCdpPage:
    """Playwright-like page facade backed by raw CDP session."""

    def __init__(
        self,
        manager: "CDPBrowserManager",
        target_id: str,
        session_id: str,
    ):
        self.manager = manager
        self.conn = manager.connection
        self.target_id = target_id
        self.session_id = session_id
        self._closed = False
        self._last_url = ""
        self.mouse = _RawCdpMouse(self)
        self._page_event_handlers: dict[str, set[Callable[..., Any]]] = {}
        self._network_response_handler_registered = False

    async def initialize(self, user_agent: str = "") -> None:
        await self.conn.send("Page.enable", session_id=self.session_id)
        await self.conn.send("Runtime.enable", session_id=self.session_id)
        await self.conn.send("DOM.enable", session_id=self.session_id)
        await self.conn.send("Network.enable", session_id=self.session_id)
        try:
            await self.conn.send(
                "Network.setExtraHTTPHeaders",
                {"headers": {"Accept-Language": DEFAULT_ACCEPT_LANGUAGE}},
                session_id=self.session_id,
            )
        except Exception:
            pass
        if user_agent:
            params = {
                "userAgent": user_agent,
                "acceptLanguage": DEFAULT_ACCEPT_LANGUAGE,
                "platform": "Windows",
            }
            try:
                await self.conn.send(
                    "Network.setUserAgentOverride",
                    params,
                    session_id=self.session_id,
                )
            except Exception:
                pass
        for method, params in (
            ("Emulation.setLocaleOverride", {"locale": DEFAULT_LOCALE}),
            ("Emulation.setTimezoneOverride", {"timezoneId": DEFAULT_TIMEZONE_ID}),
        ):
            try:
                await self.conn.send(method, params, session_id=self.session_id)
            except Exception:
                continue
        if self.manager._stealth_script_source:
            await self.conn.send(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": self.manager._stealth_script_source},
                session_id=self.session_id,
            )
        await self._sync_url()

    @property
    def url(self) -> str:
        return self._last_url

    @property
    def frames(self) -> list[Any]:
        return []

    @property
    def main_frame(self) -> "RawCdpPage":
        return self

    def locator(self, selector: str) -> _RawCdpLocator:
        return _RawCdpLocator(self, selector)

    def get_by_text(self, text: str, exact: bool = False) -> _RawCdpLocator:
        return _RawCdpLocator(self, text=text, exact=exact)

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        handlers = self._page_event_handlers.setdefault(str(event or ""), set())
        handlers.add(callback)
        if event == "response" and not self._network_response_handler_registered:
            self.conn.on("Network.responseReceived", self._dispatch_network_response)
            self._network_response_handler_registered = True

    def _dispatch_network_response(self, msg: dict[str, Any]) -> None:
        if self._closed:
            return
        if str(msg.get("sessionId", "") or "") != self.session_id:
            return
        handlers = self._page_event_handlers.get("response", set())
        if not handlers:
            return
        params = dict(msg.get("params", {}) or {})
        request_id = str(params.get("requestId", "") or "")
        response = dict(params.get("response", {}) or {})
        response_event = _RawCdpResponse(
            page=self,
            request_id=request_id,
            response_url=str(response.get("url", "") or ""),
            status=int(response.get("status", 0) or 0),
        )
        for handler in list(handlers):
            try:
                handler(response_event)
            except Exception:
                continue

    async def _sync_url(self) -> None:
        try:
            value = await self.evaluate("location.href || ''")
            self._last_url = str(value or "")
        except Exception:
            pass

    async def evaluate(self, expression: str, return_by_value: bool = True) -> Any:
        result = await self.conn.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": bool(return_by_value),
                "awaitPromise": True,
            },
            session_id=self.session_id,
        )
        if isinstance(result.get("exceptionDetails"), dict):
            raise RuntimeError(f"CDP evaluate exception: {result['exceptionDetails']}")
        value = dict(result.get("result", {}) or {})
        if return_by_value:
            return value.get("value")
        return value

    async def eval_on_selector_all(self, selector: str, expression: str) -> Any:
        selector_json = json.dumps(selector, ensure_ascii=False)
        return await self.evaluate(
            f"""
(() => {{
  const nodes = Array.from(document.querySelectorAll({selector_json}));
  const mapper = {expression};
  return mapper(nodes);
}})()
"""
        )

    async def goto(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout: int = 120_000,
    ) -> None:
        _ = wait_until
        await self.conn.send(
            "Page.navigate",
            {"url": str(url or "")},
            session_id=self.session_id,
            timeout_sec=max(10.0, float(timeout) / 1000.0),
        )
        await self._wait_document_ready(timeout_ms=int(timeout))
        await self._sync_url()

    async def _wait_document_ready(self, timeout_ms: int = 30_000) -> None:
        deadline = asyncio.get_running_loop().time() + max(1.0, float(timeout_ms) / 1000.0)
        while asyncio.get_running_loop().time() < deadline:
            try:
                ready_state = await self.evaluate("document.readyState || ''")
            except Exception:
                ready_state = ""
            if ready_state in {"interactive", "complete"}:
                return
            await asyncio.sleep(0.25)

    async def title(self) -> str:
        await self._sync_url()
        value = await self.evaluate("document.title || ''")
        return str(value or "")

    async def inner_text(self, selector: str) -> str:
        await self._sync_url()
        selector_json = json.dumps(selector, ensure_ascii=False)
        value = await self.evaluate(
            f"""
(() => {{
  const node = document.querySelector({selector_json});
  return node ? String(node.innerText || node.textContent || '') : '';
}})()
"""
        )
        return str(value or "")

    async def content(self) -> str:
        await self._sync_url()
        value = await self.evaluate(
            "document.documentElement ? document.documentElement.outerHTML : ''"
        )
        return str(value or "")

    async def wait_for_timeout(self, ms: int) -> None:
        await asyncio.sleep(max(0.0, float(ms) / 1000.0))
        await self._sync_url()

    async def bring_to_front(self) -> None:
        await self.conn.send("Page.bringToFront", session_id=self.session_id)
        await self._sync_url()

    async def screenshot(self, path: str, full_page: bool = False) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if full_page:
            metrics = await self.evaluate(
                """
(() => {
  const body = document.body;
  const html = document.documentElement;
  const width = Math.max(
    body ? body.scrollWidth : 0,
    html ? html.scrollWidth : 0,
    body ? body.offsetWidth : 0,
    html ? html.offsetWidth : 0,
    html ? html.clientWidth : 0
  );
  const height = Math.max(
    body ? body.scrollHeight : 0,
    html ? html.scrollHeight : 0,
    body ? body.offsetHeight : 0,
    html ? html.offsetHeight : 0,
    html ? html.clientHeight : 0
  );
  return { width, height, dpr: window.devicePixelRatio || 1 };
})()
"""
            )
            width = int(max(1, min(8000, float((metrics or {}).get("width", 1440)))))
            height = int(max(1, min(16000, float((metrics or {}).get("height", 2200)))))
            dpr = float((metrics or {}).get("dpr", 1.0) or 1.0)
            await self.conn.send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "mobile": False,
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": max(1.0, dpr),
                },
                session_id=self.session_id,
            )

        try:
            payload = await self.conn.send(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "fromSurface": True,
                    "captureBeyondViewport": bool(full_page),
                },
                session_id=self.session_id,
                timeout_sec=60.0,
            )
            data = str(payload.get("data", "") or "").strip()
            if not data:
                raise RuntimeError("empty screenshot payload")
            output.write_bytes(base64.b64decode(data))
        finally:
            if full_page:
                try:
                    await self.conn.send(
                        "Emulation.clearDeviceMetricsOverride",
                        session_id=self.session_id,
                    )
                except Exception:
                    pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._network_response_handler_registered:
            try:
                self.conn.off("Network.responseReceived", self._dispatch_network_response)
            except Exception:
                pass
            self._network_response_handler_registered = False
        try:
            await self.conn.send("Target.closeTarget", {"targetId": self.target_id})
        except Exception:
            pass

    def is_closed(self) -> bool:
        return bool(self._closed)


class RawCdpContext:
    """Minimal BrowserContext-like facade."""

    def __init__(self, manager: "CDPBrowserManager"):
        self.manager = manager
        self.pages: list[RawCdpPage] = []

    async def new_page(self) -> RawCdpPage:
        page = await self.manager._create_page(url="about:blank")
        self.pages.append(page)
        return page

    async def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
        if not cookies:
            return
        session_id = await self.manager._ensure_session()
        normalized: list[dict[str, Any]] = []
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            name = str(cookie.get("name", "")).strip()
            value = str(cookie.get("value", ""))
            if not name:
                continue
            item: dict[str, Any] = {
                "name": name,
                "value": value,
            }
            for key in ("url", "domain", "path", "secure", "httpOnly", "sameSite", "expires"):
                if key in cookie:
                    item[key] = cookie[key]
            normalized.append(item)
        if not normalized:
            return
        await self.manager.connection.send(
            "Network.setCookies",
            {"cookies": normalized},
            session_id=session_id,
        )

    async def cookies(self, urls: list[str] | None = None) -> list[dict[str, Any]]:
        session_id = await self.manager._ensure_session()
        params: dict[str, Any] = {}
        if urls:
            params["urls"] = [str(u) for u in urls if u]
        payload = await self.manager.connection.send(
            "Network.getCookies",
            params,
            session_id=session_id,
        )
        rows = payload.get("cookies", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return []

    async def storage_state(self, path: str = "") -> dict[str, Any]:
        payload = {
            "cookies": await self.cookies(),
            "origins": [],
        }
        if path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return payload

    async def close(self) -> None:
        pages = list(self.pages)
        self.pages.clear()
        for page in pages:
            try:
                await page.close()
            except Exception:
                continue


class CDPBrowserManager:
    """Native CDP browser manager compatible with existing call sites."""

    def __init__(
        self,
        custom_browser_path: str = "",
        debug_port: int = 9222,
        save_login_state: bool = True,
        user_data_dir_template: str = "taobao_cdp_profile",
        auto_close_browser: bool = False,
        browser_launch_timeout: int = 30,
        safe_mode: bool = True,
        cdp_url: str = "",
    ):
        self.browser_process: Optional[subprocess.Popen] = None
        self.browser_context: Optional[RawCdpContext] = None
        self.connection: Optional[CdpConnection] = None
        self.debug_port: Optional[int] = None
        self.ws_url: str = ""

        self.custom_browser_path = custom_browser_path
        self.config_debug_port = int(debug_port)
        self.save_login_state = bool(save_login_state)
        self.user_data_dir_template = str(user_data_dir_template or "taobao_cdp_profile")
        self.auto_close_browser = bool(auto_close_browser)
        self.browser_launch_timeout = max(5, int(browser_launch_timeout))
        self.safe_mode = bool(safe_mode)
        self.cdp_url = str(cdp_url or "").strip()
        self._cleanup_registered = False
        self._stealth_script_source = ""

    def _register_cleanup_handlers(self) -> None:
        if self._cleanup_registered:
            return

        def _sync_cleanup() -> None:
            if self.browser_process and self.browser_process.poll() is None:
                try:
                    self.browser_process.terminate()
                except Exception:
                    pass

        atexit.register(_sync_cleanup)

        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _signal_handler(signum: int, frame: Any) -> None:
            _ = frame
            if self.browser_process and self.browser_process.poll() is None:
                try:
                    self.browser_process.terminate()
                except Exception:
                    pass
            if signum == signal.SIGINT:
                if prev_sigint == signal.default_int_handler:
                    raise KeyboardInterrupt
                if callable(prev_sigint):
                    prev_sigint(signum, frame)
            raise SystemExit(0)

        if prev_sigint in (signal.default_int_handler, signal.SIG_DFL):
            signal.signal(signal.SIGINT, _signal_handler)
        if prev_sigterm == signal.SIG_DFL:
            signal.signal(signal.SIGTERM, _signal_handler)

        self._cleanup_registered = True

    def _resolve_user_data_dir(self) -> str:
        if not self.save_login_state:
            return ""
        path = self.user_data_dir_template
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(os.getcwd()) / "browser_data" / path
        candidate.mkdir(parents=True, exist_ok=True)
        return str(candidate.resolve())

    async def _launch_browser_process(self, browser_path: str, headless: bool = False) -> None:
        start_port = 9330 if self.safe_mode else self.config_debug_port
        self.debug_port = _find_available_port(start_port=start_port)
        user_data_dir = self._resolve_user_data_dir()

        args = [
            browser_path,
            f"--remote-debugging-port={self.debug_port}",
        ]
        args.extend(build_browser_launch_args(headless=headless))
        if user_data_dir:
            args.append(f"--user-data-dir={user_data_dir}")

        utils.logger.info("[CDPBrowserManager] launching browser path=%s", browser_path)
        utils.logger.info("[CDPBrowserManager] debug_port=%s safe_mode=%s", self.debug_port, self.safe_mode)
        self.browser_process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if _is_windows() else 0,
        )
        self.ws_url = await _wait_ws_url_from_debug_port(
            int(self.debug_port),
            timeout_sec=self.browser_launch_timeout,
        )

    async def _connect_browser_level(self) -> None:
        if not self.ws_url:
            raise RuntimeError("empty websocket debugger url")
        self.connection = await CdpConnection.connect(
            self.ws_url,
            timeout_sec=max(10.0, float(self.browser_launch_timeout)),
        )

    async def _create_page(self, url: str = "about:blank", user_agent: str = "") -> RawCdpPage:
        if self.connection is None:
            raise RuntimeError("CDP connection is not initialized")
        create_result = await self.connection.send(
            "Target.createTarget",
            {"url": str(url or "about:blank")},
            timeout_sec=20.0,
        )
        target_id = str(create_result.get("targetId", "")).strip()
        if not target_id:
            raise RuntimeError("failed to create CDP target")
        attach_result = await self.connection.send(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
            timeout_sec=20.0,
        )
        session_id = str(attach_result.get("sessionId", "")).strip()
        if not session_id:
            raise RuntimeError("failed to attach CDP target")

        page = RawCdpPage(manager=self, target_id=target_id, session_id=session_id)
        await page.initialize(user_agent=user_agent)
        return page

    async def _ensure_session(self) -> str:
        if self.browser_context is None:
            raise RuntimeError("CDP browser context is not initialized")
        for page in self.browser_context.pages:
            if not page.is_closed():
                return page.session_id
        page = await self._create_page(url="about:blank")
        self.browser_context.pages.append(page)
        return page.session_id

    async def launch_and_connect(
        self,
        playwright: Any = None,
        playwright_proxy: Optional[Dict[str, Any]] = None,
        user_agent: Optional[str] = None,
        headless: bool = False,
    ) -> RawCdpContext:
        _ = playwright
        _ = playwright_proxy
        try:
            if self.cdp_url:
                self.ws_url = await _resolve_ws_url(self.cdp_url)
                utils.logger.info("[CDPBrowserManager] connecting to existing CDP endpoint: %s", self.cdp_url)
            else:
                browser_path = _find_chrome_executable(self.custom_browser_path)
                await self._launch_browser_process(browser_path=browser_path, headless=headless)

            await self._connect_browser_level()
            self.browser_context = RawCdpContext(manager=self)
            first_page = await self._create_page(
                url="about:blank",
                user_agent=user_agent or "",
            )
            self.browser_context.pages.append(first_page)
            self._register_cleanup_handlers()
            return self.browser_context
        except Exception as exc:
            utils.logger.error("[CDPBrowserManager] launch_and_connect failed: %s", exc)
            await self.cleanup(force=True)
            raise

    async def add_stealth_script(self, script_path: str = "") -> None:
        self._stealth_script_source = load_stealth_script_source(script_path)
        if not self._stealth_script_source:
            return
        try:
            if self.browser_context:
                for page in self.browser_context.pages:
                    if page.is_closed():
                        continue
                    try:
                        await self.connection.send(
                            "Page.addScriptToEvaluateOnNewDocument",
                            {"source": self._stealth_script_source},
                            session_id=page.session_id,
                        )
                    except Exception:
                        continue
            utils.logger.info("[CDPBrowserManager] stealth script registered")
        except Exception as exc:
            utils.logger.warning("[CDPBrowserManager] add_stealth_script failed: %s", exc)

    async def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
        if self.browser_context is None:
            return
        await self.browser_context.add_cookies(cookies)

    async def get_cookies(self) -> list[dict[str, Any]]:
        if self.browser_context is None:
            return []
        return await self.browser_context.cookies()

    async def cleanup(self, force: bool = False) -> None:
        try:
            if self.browser_context is not None:
                try:
                    await self.browser_context.close()
                except Exception:
                    pass
                self.browser_context = None

            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                self.connection = None
        finally:
            should_close_process = bool(force or self.auto_close_browser)
            if should_close_process and self.browser_process is not None:
                try:
                    self.browser_process.terminate()
                except Exception:
                    pass
                self.browser_process = None

    def is_connected(self) -> bool:
        return self.connection is not None and self.connection.is_connected

    async def get_browser_info(self) -> Dict[str, Any]:
        return {
            "debug_port": self.debug_port,
            "ws_url": self.ws_url,
            "contexts_count": 1 if self.browser_context is not None else 0,
            "is_connected": self.is_connected(),
        }
