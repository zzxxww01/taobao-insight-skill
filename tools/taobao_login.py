# -*- coding: utf-8 -*-
"""Taobao Login Handler

Handles login detection and waiting for QR code scan when Taobao blocks the crawler.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Tuple

from playwright.async_api import BrowserContext, Page

from tools import utils


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
    """Handle Taobao login interception"""

    def __init__(
        self,
        browser_context: BrowserContext,
        context_page: Page,
        login_timeout_sec: int = 300,
    ):
        self.browser_context = browser_context
        self.context_page = context_page
        self.login_timeout_sec = max(30, int(login_timeout_sec))
        self._login_cookie_names = {"cookie2", "unb", "_tb_token_", "_m_h5_tk"}
        self._decision_trace: list[dict[str, Any]] = []
        self._started_at: float = 0.0

    @staticmethod
    def _is_search_url(url: str) -> bool:
        lowered = (url or "").lower()
        return (
            "s.taobao.com/search" in lowered
            or "list.tmall.com/search_product.htm" in lowered
        )

    def _append_trace(
        self, state: str, decision: LoginDecision | None = None, note: str = ""
    ) -> None:
        event: dict[str, Any] = {
            "state": state,
            "note": note,
            "ts": round(time.time(), 3),
        }
        if self._started_at > 0:
            event["elapsed_sec"] = round(max(0.0, time.monotonic() - self._started_at), 3)
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
            elapsed_sec = round(max(0.0, time.monotonic() - self._started_at), 3)
        return LoginHandleResult(
            ok=bool(ok),
            reason=str(reason or ""),
            final_state=str(final_state or ""),
            decision_trace=list(self._decision_trace),
            elapsed_sec=elapsed_sec,
        )

    async def _has_login_cookie(self) -> bool:
        try:
            cookies = await self.browser_context.cookies(
                [
                    "https://www.taobao.com",
                    "https://s.taobao.com",
                    "https://detail.tmall.com",
                    "https://www.tmall.com",
                ]
            )
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
            cookies = await self.browser_context.cookies(
                [
                    "https://www.taobao.com",
                    "https://s.taobao.com",
                    "https://detail.tmall.com",
                    "https://www.tmall.com",
                ]
            )
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

        search_markers = ("人付款", "已售", "综合排序", "销量", "筛选", "店铺", "价格")
        has_search_markers = sum(1 for marker in search_markers if marker in body_text) >= 2
        return has_search_markers

    async def _read_page_snapshot(self) -> Tuple[str, str, str]:
        """Read the latest page snapshot safely."""
        current_url = (self.context_page.url or "").strip()
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
        """Evaluate current page and cookies to decide login state."""
        current_url, title, body_text = await self._read_page_snapshot()
        url_lower = current_url.lower()
        title_lower = title.lower()
        body_lower = body_text.lower()

        has_login_cookie = await self._has_login_cookie()
        has_search_dom = False
        is_login_page = False
        reason = "non_login_default"

        if "login.taobao.com" in url_lower or "member/login" in url_lower:
            is_login_page = True
            reason = "url_login"
        elif any(token in url_lower for token in ("captcha", "punish", "x5sec", "_____tmd_____")):
            is_login_page = True
            reason = "url_antibot"
        else:
            strong_login_indicators = (
                "扫码登录",
                "请扫码登录",
                "扫一扫登录",
                "账号密码登录",
                "短信登录",
                "手机验证码登录",
                "忘记密码",
                "免费注册",
                "请登录",
            )
            hits = sum(
                1 for token in strong_login_indicators if token in body_text or token in title
            )

            if self._is_search_url(current_url):
                has_search_dom = await self._has_search_dom(body_text)
                if has_search_dom:
                    is_login_page = False
                    reason = "search_surface_ready"
                elif hits >= 2 and ("登录" in body_text or "login" in body_lower):
                    is_login_page = True
                    reason = "search_login_wall"
                elif has_login_cookie:
                    is_login_page = False
                    reason = "cookie_present_on_search"
                else:
                    is_login_page = False
                    reason = "search_surface_unknown"
            else:
                if hits >= 2:
                    is_login_page = True
                    reason = "content_login_markers"
                elif "login" in title_lower and ("taobao" in title_lower or "tmall" in title_lower):
                    is_login_page = True
                    reason = "title_login"
                elif "扫码" in body_text and "登录" in body_text:
                    is_login_page = True
                    reason = "qr_login_prompt"
                elif "请登录" in body_text and ("taobao" in body_lower or "tmall" in body_lower):
                    is_login_page = True
                    reason = "login_prompt"
                elif has_login_cookie:
                    is_login_page = False
                    reason = "cookie_present_non_login"
                else:
                    is_login_page = False
                    reason = "non_login_default"

        return LoginDecision(
            is_login_page=is_login_page,
            has_login_cookie=has_login_cookie,
            has_search_dom=has_search_dom,
            reason=reason,
            url=current_url,
            timestamp=time.time(),
        )

    async def _is_login_page(self) -> bool:
        """Check if current page is a Taobao login/verification page."""
        try:
            decision = await self._evaluate_login_decision()
            return bool(decision.is_login_page)
        except Exception:
            return False

    async def _wait_for_login(self, initial_url: str) -> LoginHandleResult:
        """
        Wait for user to complete QR code login

        Args:
            initial_url: The URL that triggered the login page

        Returns:
            LoginHandleResult
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.login_timeout_sec
        baseline_cookie_fp = await self._login_cookie_fingerprint()

        while loop.time() < deadline:
            decision = await self._evaluate_login_decision()
            self._append_trace("WAIT_QR_FROZEN", decision, note="waiting for qr scan")
            current_cookie_fp = await self._login_cookie_fingerprint()
            cookie_changed = bool(current_cookie_fp) and current_cookie_fp != baseline_cookie_fp

            # Keep waiting mode frozen: no page.goto/refresh during QR login.
            # If cookie changes while still on login page, treat as login success and
            # let caller perform one recovery navigation afterwards.
            if decision.is_login_page and cookie_changed:
                self._append_trace(
                    "COOKIE_CONFIRMED",
                    decision,
                    note="cookie changed while still on login page (no auto navigation)",
                )
                return self._finalize(True, "cookie_changed_on_login_page", "COOKIE_CONFIRMED")

            if decision.has_login_cookie and not decision.is_login_page:
                self._append_trace(
                    "SUCCESS", decision, note="cookie+page confirmed login success"
                )
                return self._finalize(
                    True,
                    "cookie_and_page_confirmed",
                    "SUCCESS",
                )

            if not decision.is_login_page:
                self._append_trace(
                    "SUCCESS", decision, note="page left login wall without cookie signal"
                )
                return self._finalize(True, "page_left_login", "SUCCESS")
            await asyncio.sleep(1)
        self._append_trace("TIMEOUT", note="qr login timeout")
        return self._finalize(False, "qr_login_timeout", "TIMEOUT")

    async def handle_login_interception(
        self, bring_to_front: bool = True
    ) -> LoginHandleResult:
        """
        Handle login interception - pause and wait for QR code scan

        Args:
            bring_to_front: Whether to bring the browser window to front

        Returns:
            LoginHandleResult
        """
        initial_url = self.context_page.url
        utils.logger.warning(f"[TaobaoLogin] Login page detected at: {initial_url}")
        utils.logger.warning(f"[TaobaoLogin] Please scan QR code to login (timeout: {self.login_timeout_sec}s)")
        utils.logger.info("[TaobaoLogin] QR wait mode enabled: page navigation/refresh is frozen until login succeeds or times out")

        if bring_to_front:
            try:
                # Bring page to front
                await self.context_page.bring_to_front()
                utils.logger.info("[TaobaoLogin] Browser window brought to front")
            except Exception as e:
                utils.logger.warning(f"[TaobaoLogin] Failed to bring window to front: {e}")

        try:
            result = await self._wait_for_login(initial_url)
            if not result.ok:
                utils.logger.error(f"[TaobaoLogin] Login timeout after {self.login_timeout_sec}s")
                return result

            wait_redirect_sec = 5
            utils.logger.info(f"[TaobaoLogin] Login successful, waiting {wait_redirect_sec}s for redirect...")
            await asyncio.sleep(wait_redirect_sec)
            return self._finalize(True, result.reason, result.final_state)

        except Exception as exc:
            utils.logger.error(f"[TaobaoLogin] Login handler failed: {exc}")
            self._append_trace("FAILED", note=f"exception: {type(exc).__name__}: {exc}")
            return self._finalize(False, f"exception:{type(exc).__name__}", "FAILED")

    async def check_and_handle_login(self) -> LoginHandleResult:
        """
        Check if on login page and handle if needed

        Returns:
            LoginHandleResult
        """
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
        """Backward-compatible bool wrapper for legacy callsites."""
        result = await self.check_and_handle_login()
        return bool(result.ok)
