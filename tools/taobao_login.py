# -*- coding: utf-8 -*-
"""Taobao login handler, and shared _BaseLogin for all platform login handlers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

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
    cookie_fingerprint_before: str = ""
    cookie_fingerprint_after: str = ""
    cookie_changed: bool = False


class _BaseLogin:
    """Shared login-handler logic; subclasses supply platform specifics.

    Subclasses must define:
        _cookie_urls          – list of URLs to request cookies from
        _cookie_domain_tokens – tuple of domain substrings that count as
                                "logged-in" cookie domains
        _search_dom_selector  – CSS selector whose node count signals a
                                rendered search-result page
        _log_prefix           – e.g. "[TaobaoLogin]"
        _wait_trace_note      – note string for the WAIT trace event
        _success_trace_note   – note string for the SUCCESS trace event
        _cookie_success_note  – note for "page left after cookie" case
        _page_left_note       – note for "page left without cookie" case
        _timeout_reason       – reason string written on timeout
        _timeout_trace_note   – note for TIMEOUT trace event
        _login_required_note  – note for LOGIN_REQUIRED trace event
        _login_detected_log   – info-level log message on login detection

    And implement:
        _is_search_url(url) -> bool           (static method)
        _has_search_dom(body_text) -> bool    (async)
    """

    # ---- platform-specific attributes (override in subclass) ----
    _cookie_urls: list[str] = []
    _cookie_domain_tokens: tuple[str, ...] = ()
    _search_dom_selector: str = ""
    _log_prefix: str = "[Login]"
    _wait_trace_note: str = "waiting for login"
    _cookie_updated_note: str = "cookie changed; keep page frozen until unblocked"
    _success_cookie_note: str = "cookie+page confirmed login success"
    _success_cookie_reason: str = "cookie_and_page_confirmed"
    _page_left_after_cookie_note: str = "page left after cookie confirmation"
    _page_left_note: str = "page left without cookie signal"
    _page_left_after_cookie_reason: str = "page_left_after_cookie"
    _page_left_reason: str = "page_left"
    _timeout_reason: str = "login_timeout"
    _timeout_trace_note: str = "login timeout"
    _login_required_note: str = "detected login wall"
    _login_detected_log: str = "Detected login page, initiating handler..."

    def __init__(
        self,
        browser_context: Any,
        context_page: Any,
        login_timeout_sec: int = 300,
    ):
        self.browser_context = browser_context
        self.context_page = context_page
        self.login_timeout_sec = max(30, int(login_timeout_sec))
        self._login_cookie_names: set[str] = set()
        self._decision_trace: list[dict[str, Any]] = []
        self._started_at: float = 0.0

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

    def _finalize(
        self,
        ok: bool,
        reason: str,
        final_state: str,
        *,
        cookie_fingerprint_before: str = "",
        cookie_fingerprint_after: str = "",
    ) -> LoginHandleResult:
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
            cookie_fingerprint_before=str(cookie_fingerprint_before or ""),
            cookie_fingerprint_after=str(cookie_fingerprint_after or ""),
            cookie_changed=bool(
                (cookie_fingerprint_before or cookie_fingerprint_after)
                and cookie_fingerprint_before != cookie_fingerprint_after
            ),
        )

    async def _has_login_cookie(self) -> bool:
        try:
            cookies = await self.browser_context.cookies(self._cookie_urls)
        except Exception:
            return False
        for cookie in cookies:
            if not isinstance(cookie, dict):
                continue
            name = str(cookie.get("name", "")).strip()
            domain = str(cookie.get("domain", "")).lower()
            if name in self._login_cookie_names and any(
                t in domain for t in self._cookie_domain_tokens
            ):
                return True
        return False

    async def _login_cookie_fingerprint(self) -> str:
        try:
            cookies = await self.browser_context.cookies(self._cookie_urls)
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
            if not any(t in domain for t in self._cookie_domain_tokens):
                continue
            value = str(cookie.get("value", ""))
            parts.append(f"{name}={value}")
        parts.sort()
        return "|".join(parts)

    async def _read_page_snapshot(self) -> tuple[str, str, str]:
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

    # --- to be overridden by subclasses ---

    @staticmethod
    def _is_search_url(url: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def _has_search_dom(self, body_text: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def _evaluate_login_decision(self) -> LoginDecision:
        current_url, title, body_text = await self._read_page_snapshot()
        has_login_cookie = await self._has_login_cookie()
        has_search_dom = False
        if self._is_search_url(current_url):
            has_search_dom = await self._has_search_dom(body_text)
        is_login_page, reason = self._decide_login_page(
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

    def _decide_login_page(
        self,
        *,
        current_url: str,
        title: str,
        body_text: str,
        has_login_cookie: bool,
        has_search_dom: bool,
    ) -> tuple[bool, str]:  # pragma: no cover
        raise NotImplementedError

    async def _wait_for_login(self, initial_url: str) -> LoginHandleResult:
        _ = initial_url
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.login_timeout_sec
        initial_cookie_fp = await self._login_cookie_fingerprint()
        baseline_cookie_fp = initial_cookie_fp
        cookie_confirmed = False

        while loop.time() < deadline:
            decision = await self._evaluate_login_decision()
            self._append_trace("WAIT", decision, note=self._wait_trace_note)
            current_cookie_fp = await self._login_cookie_fingerprint()
            cookie_changed = bool(current_cookie_fp) and current_cookie_fp != baseline_cookie_fp

            if cookie_changed:
                cookie_confirmed = True
                baseline_cookie_fp = current_cookie_fp
                self._append_trace(
                    "COOKIE_UPDATED",
                    decision,
                    note=self._cookie_updated_note,
                )
                if decision.is_login_page:
                    await asyncio.sleep(1.0)
                    continue

            if decision.has_login_cookie and not decision.is_login_page:
                self._append_trace("SUCCESS", decision, note=self._success_cookie_note)
                return self._finalize(
                    True,
                    self._success_cookie_reason,
                    "SUCCESS",
                    cookie_fingerprint_before=initial_cookie_fp,
                    cookie_fingerprint_after=current_cookie_fp,
                )

            if not decision.is_login_page:
                self._append_trace(
                    "SUCCESS",
                    decision,
                    note=(
                        self._page_left_after_cookie_note
                        if cookie_confirmed
                        else self._page_left_note
                    ),
                )
                return self._finalize(
                    True,
                    self._page_left_after_cookie_reason if cookie_confirmed else self._page_left_reason,
                    "SUCCESS",
                    cookie_fingerprint_before=initial_cookie_fp,
                    cookie_fingerprint_after=current_cookie_fp,
                )

            await asyncio.sleep(1.0)

        self._append_trace("TIMEOUT", note=self._timeout_trace_note)
        return self._finalize(
            False,
            self._timeout_reason,
            "TIMEOUT",
            cookie_fingerprint_before=initial_cookie_fp,
            cookie_fingerprint_after=await self._login_cookie_fingerprint(),
        )

    async def handle_login_interception(
        self,
        bring_to_front: bool = True,
    ) -> LoginHandleResult:
        initial_url = getattr(self.context_page, "url", "")
        utils.logger.warning("%s Login page detected at: %s", self._log_prefix, initial_url)
        utils.logger.warning(
            "%s Please complete login in the browser (timeout: %ss)",
            self._log_prefix,
            self.login_timeout_sec,
        )
        utils.logger.info(
            "%s Wait mode enabled: frozen until login succeeds or times out",
            self._log_prefix,
        )

        if bring_to_front:
            try:
                await self.context_page.bring_to_front()
                utils.logger.info("%s Browser window brought to front", self._log_prefix)
            except Exception as exc:
                utils.logger.warning(
                    "%s Failed to bring window to front: %s", self._log_prefix, exc
                )

        try:
            result = await self._wait_for_login(initial_url)
            if not result.ok:
                utils.logger.error(
                    "%s Login timeout after %ss",
                    self._log_prefix,
                    self.login_timeout_sec,
                )
                return result

            wait_redirect_sec = 5
            utils.logger.info(
                "%s Login successful, waiting %ss for redirect...",
                self._log_prefix,
                wait_redirect_sec,
            )
            await asyncio.sleep(wait_redirect_sec)
            return self._finalize(
                True,
                result.reason,
                result.final_state,
                cookie_fingerprint_before=result.cookie_fingerprint_before,
                cookie_fingerprint_after=result.cookie_fingerprint_after,
            )
        except Exception as exc:
            utils.logger.error("%s Login handler failed: %s", self._log_prefix, exc)
            self._append_trace("FAILED", note=f"exception: {type(exc).__name__}: {exc}")
            return self._finalize(False, f"exception:{type(exc).__name__}", "FAILED")

    async def check_and_handle_login(self) -> LoginHandleResult:
        self._decision_trace = []
        self._started_at = time.monotonic()
        try:
            decision = await self._evaluate_login_decision()
            if decision.is_login_page:
                self._append_trace("LOGIN_REQUIRED", decision, note=self._login_required_note)
                utils.logger.info("%s %s", self._log_prefix, self._login_detected_log)
                return await self.handle_login_interception()
            self._append_trace("IDLE", decision, note="login not required")
            return self._finalize(True, "login_not_required", "SUCCESS")
        except Exception as exc:
            self._append_trace("FAILED", note=f"exception: {type(exc).__name__}: {exc}")
            return self._finalize(False, f"exception:{type(exc).__name__}", "FAILED")

    async def check_and_handle_login_bool(self) -> bool:
        result = await self.check_and_handle_login()
        return bool(result.ok)


class TaobaoLogin(_BaseLogin):
    """Handle Taobao/Tmall login interception and QR wait flow."""

    _cookie_urls = list(TAOBAO_COOKIE_URLS)
    _cookie_domain_tokens = ("taobao.com", "tmall.com")
    _search_dom_selector = "a[href*='item.htm?id=']"
    _log_prefix = "[TaobaoLogin]"
    _wait_trace_note = "waiting for qr scan"
    _cookie_updated_note = "cookie changed; keep page frozen until it leaves login wall"
    _success_cookie_note = "cookie+page confirmed login success"
    _success_cookie_reason = "cookie_and_page_confirmed"
    _page_left_after_cookie_note = "page left login wall after cookie confirmation"
    _page_left_note = "page left login wall without cookie signal"
    _page_left_after_cookie_reason = "page_left_login_after_cookie"
    _page_left_reason = "page_left_login"
    _timeout_reason = "qr_login_timeout"
    _timeout_trace_note = "qr login timeout"
    _login_required_note = "detected login wall"
    _login_detected_log = "Detected Taobao login page, initiating login handler..."

    def __init__(
        self,
        browser_context: Any,
        context_page: Any,
        login_timeout_sec: int = 300,
    ):
        super().__init__(browser_context, context_page, login_timeout_sec)
        self._login_cookie_names = set(LOGIN_COOKIE_NAMES)

    @staticmethod
    def _is_search_url(url: str) -> bool:
        return is_search_result_url(url)

    async def _has_search_dom(self, body_text: str) -> bool:
        item_link_count = 0
        try:
            raw_count = await self.context_page.eval_on_selector_all(
                self._search_dom_selector,
                "nodes => nodes.length",
            )
            item_link_count = int(raw_count or 0)
        except Exception:
            item_link_count = 0
        if item_link_count >= 3:
            return True
        return looks_like_search_content(body_text)

    def _decide_login_page(
        self,
        *,
        current_url: str,
        title: str,
        body_text: str,
        has_login_cookie: bool,
        has_search_dom: bool,
    ) -> tuple[bool, str]:
        return decide_login_page(
            current_url=current_url,
            title=title,
            body_text=body_text,
            has_login_cookie=has_login_cookie,
            has_search_dom=has_search_dom,
        )
