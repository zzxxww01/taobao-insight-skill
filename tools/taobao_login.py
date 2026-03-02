# -*- coding: utf-8 -*-
"""Taobao login handler."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Tuple

from . import utils
from .login_rules import (
    LOGIN_COOKIE_NAMES,
    TAOBAO_COOKIE_URLS,
    decide_login_page,
    is_search_result_url,
    looks_like_search_content,
)


@dataclass
class LoginDecision:
    is_login_page: bool
    has_login_cookie: bool
    has_search_dom: bool
    reason: str
    url: str
    timestamp: float


@dataclass
class LoginHandleResult:
    ok: bool
    reason: str
    final_state: str
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    elapsed_sec: float = 0.0


class TaobaoLogin:
    """Handle Taobao login interception and QR wait flow."""

    def __init__(
        self,
        browser_context: Any,
        context_page: Any,
        login_timeout_sec: int = 300,
    ):
        self.browser_context = browser_context
        self.context_page = context_page
        self.login_timeout_sec = max(30, int(login_timeout_sec))
        self._login_cookie_names = set(LOGIN_COOKIE_NAMES)
        self._decision_trace: list[dict[str, Any]] = []
        self._started_at: float = 0.0

    @staticmethod
    def _is_search_url(url: str) -> bool:
        return is_search_result_url(url)

    def _append_trace(
        self, state: str, decision: LoginDecision | None = None, note: str = ""
    ) -> None:
        event: dict[str, Any] = {
            "state": state,
            "note": note,
            "ts": round(time.time(), 3),
        }
        if self._started_at > 0:
            event["elapsed_sec"] = round(
                max(0.0, time.monotonic() - self._started_at),
                3,
            )
        if decision is not None:
            event.update(
                {
                    "is_login_page": bool(decision.is_login_page),
                    "has_login_cookie": bool(decision.has_login_cookie),
                    "has_search_dom": bool(decision.has_search_dom),
                    "decision_reason": decision.reason,
                    "url": decision.url,
                    "decision_ts": round(decision.timestamp, 3),
                }
            )
        self._decision_trace.append(event)

    def _finalize(self, ok: bool, reason: str, final_state: str) -> LoginHandleResult:
        elapsed_sec = 0.0
        if self._started_at > 0:
            elapsed_sec = round(
                max(0.0, time.monotonic() - self._started_at),
                3,
            )
        return LoginHandleResult(
            ok=bool(ok),
            reason=str(reason or ""),
            final_state=str(final_state or ""),
            decision_trace=list(self._decision_trace),
            elapsed_sec=elapsed_sec,
        )

    async def _has_login_cookie(self) -> bool:
        try:
            cookies = await self.browser_context.cookies(TAOBAO_COOKIE_URLS)
        except Exception:
            return False
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            name = str(cookie.get("name", "")).strip()
            domain = str(cookie.get("domain", "")).lower()
            if name in self._login_cookie_names and (
                "taobao.com" in domain or "tmall.com" in domain
            ):
                return True
        return False

    async def _login_cookie_fingerprint(self) -> str:
        try:
            cookies = await self.browser_context.cookies(TAOBAO_COOKIE_URLS)
        except Exception:
            return ""
        parts: list[str] = []
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            name = str(cookie.get("name", "")).strip()
            if name not in self._login_cookie_names:
                continue
            domain = str(cookie.get("domain", "")).lower()
            if "taobao.com" not in domain and "tmall.com" not in domain:
                continue
            value = str(cookie.get("value", ""))
            parts.append(f"{name}={value}")
        parts.sort()
        return "|".join(parts)

    async def _has_search_dom(self, body_text: str) -> bool:
        item_link_count = 0
        try:
            raw_count = await self.context_page.eval_on_selector_all(
                "a[href*='item.htm?id=']",
                "nodes => nodes.length",
            )
            item_link_count = int(raw_count or 0)
        except Exception:
            item_link_count = 0
        if item_link_count >= 3:
            return True
        return looks_like_search_content(body_text)

    async def _read_page_snapshot(self) -> Tuple[str, str, str]:
        current_url = (getattr(self.context_page, "url", "") or "").strip()
        try:
            title = (await self.context_page.title()) or ""
        except Exception:
            title = ""
        try:
            body_text = (await self.context_page.inner_text("body")) or ""
        except Exception:
            body_text = ""
        return current_url, title, body_text

    async def _evaluate_login_decision(self) -> LoginDecision:
        current_url, title, body_text = await self._read_page_snapshot()
        has_login_cookie = await self._has_login_cookie()
        has_search_dom = False
        if self._is_search_url(current_url):
            has_search_dom = await self._has_search_dom(body_text)
        is_login_page, reason = decide_login_page(
            current_url=current_url,
            title=title,
            body_text=body_text,
            has_login_cookie=has_login_cookie,
            has_search_dom=has_search_dom,
        )
        return LoginDecision(
            is_login_page=is_login_page,
            has_login_cookie=has_login_cookie,
            has_search_dom=has_search_dom,
            reason=reason,
            url=current_url,
            timestamp=time.time(),
        )

    async def _wait_for_login(self, initial_url: str) -> LoginHandleResult:
        _ = initial_url
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.login_timeout_sec
        baseline_cookie_fp = await self._login_cookie_fingerprint()

        while loop.time() < deadline:
            decision = await self._evaluate_login_decision()
            self._append_trace("WAIT_QR_FROZEN", decision, note="waiting for qr scan")
            current_cookie_fp = await self._login_cookie_fingerprint()
            cookie_changed = bool(current_cookie_fp) and current_cookie_fp != baseline_cookie_fp

            if decision.is_login_page and cookie_changed:
                self._append_trace(
                    "COOKIE_CONFIRMED",
                    decision,
                    note="cookie changed while still on login page (no auto navigation)",
                )
                return self._finalize(
                    True,
                    "cookie_changed_on_login_page",
                    "COOKIE_CONFIRMED",
                )

            if decision.has_login_cookie and not decision.is_login_page:
                self._append_trace(
                    "SUCCESS",
                    decision,
                    note="cookie+page confirmed login success",
                )
                return self._finalize(
                    True,
                    "cookie_and_page_confirmed",
                    "SUCCESS",
                )

            if not decision.is_login_page:
                self._append_trace(
                    "SUCCESS",
                    decision,
                    note="page left login wall without cookie signal",
                )
                return self._finalize(True, "page_left_login", "SUCCESS")

            await asyncio.sleep(1.0)

        self._append_trace("TIMEOUT", note="qr login timeout")
        return self._finalize(False, "qr_login_timeout", "TIMEOUT")

    async def handle_login_interception(
        self,
        bring_to_front: bool = True,
    ) -> LoginHandleResult:
        initial_url = getattr(self.context_page, "url", "")
        utils.logger.warning("[TaobaoLogin] Login page detected at: %s", initial_url)
        utils.logger.warning(
            "[TaobaoLogin] Please scan QR code to login (timeout: %ss)",
            self.login_timeout_sec,
        )
        utils.logger.info(
            "[TaobaoLogin] QR wait mode enabled: page navigation/refresh is frozen until login succeeds or times out"
        )

        if bring_to_front:
            try:
                await self.context_page.bring_to_front()
                utils.logger.info("[TaobaoLogin] Browser window brought to front")
            except Exception as exc:
                utils.logger.warning("[TaobaoLogin] Failed to bring window to front: %s", exc)

        try:
            result = await self._wait_for_login(initial_url)
            if not result.ok:
                utils.logger.error(
                    "[TaobaoLogin] Login timeout after %ss",
                    self.login_timeout_sec,
                )
                return result

            wait_redirect_sec = 5
            utils.logger.info(
                "[TaobaoLogin] Login successful, waiting %ss for redirect...",
                wait_redirect_sec,
            )
            await asyncio.sleep(wait_redirect_sec)
            return self._finalize(True, result.reason, result.final_state)
        except Exception as exc:
            utils.logger.error("[TaobaoLogin] Login handler failed: %s", exc)
            self._append_trace("FAILED", note=f"exception: {type(exc).__name__}: {exc}")
            return self._finalize(False, f"exception:{type(exc).__name__}", "FAILED")

    async def check_and_handle_login(self) -> LoginHandleResult:
        self._decision_trace = []
        self._started_at = time.monotonic()
        try:
            decision = await self._evaluate_login_decision()
            if decision.is_login_page:
                self._append_trace("LOGIN_REQUIRED", decision, note="detected login wall")
                utils.logger.info("[TaobaoLogin] Detected Taobao login page, initiating login handler...")
                return await self.handle_login_interception()
            self._append_trace("IDLE", decision, note="login not required")
            return self._finalize(True, "login_not_required", "SUCCESS")
        except Exception as exc:
            self._append_trace("FAILED", note=f"exception: {type(exc).__name__}: {exc}")
            return self._finalize(False, f"exception:{type(exc).__name__}", "FAILED")

    async def check_and_handle_login_bool(self) -> bool:
        result = await self.check_and_handle_login()
        return bool(result.ok)
