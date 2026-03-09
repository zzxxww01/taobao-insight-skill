# -*- coding: utf-8 -*-
"""JD login/risk handler."""

from __future__ import annotations

from typing import Any

from .jd_login_rules import (
    JD_COOKIE_URLS,
    LOGIN_COOKIE_NAMES,
    decide_login_page,
    is_search_result_url,
    looks_like_search_content,
)
from .taobao_login import LoginDecision, LoginHandleResult, _BaseLogin


class JDLogin(_BaseLogin):
    """Handle JD login interception and manual risk verification flow."""

    _cookie_urls = list(JD_COOKIE_URLS)
    _cookie_domain_tokens = ("jd.com",)
    _search_dom_selector = "li.gl-item, .gl-item[data-sku], .j-sku-item"
    _log_prefix = "[JDLogin]"
    _wait_trace_note = "waiting for jd verification"
    _cookie_updated_note = "cookie changed; keep page frozen until it leaves blocked state"
    _success_cookie_note = "cookie+page confirmed verification success"
    _success_cookie_reason = "cookie_and_page_confirmed"
    _page_left_after_cookie_note = "page left blocked state after cookie confirmation"
    _page_left_note = "page left blocked state without cookie signal"
    _page_left_after_cookie_reason = "page_left_blocked_state_after_cookie"
    _page_left_reason = "page_left_blocked_state"
    _timeout_reason = "verification_timeout"
    _timeout_trace_note = "jd verification timeout"
    _login_required_note = "detected jd blocked page"
    _login_detected_log = "Detected JD blocked page, initiating recovery..."

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
