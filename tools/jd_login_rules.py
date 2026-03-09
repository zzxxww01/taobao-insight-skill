# -*- coding: utf-8 -*-
"""Centralized JD login/risk detection and wait helpers."""

from __future__ import annotations

import time
from typing import Any


LOGIN_COOKIE_NAMES = {"thor", "pin", "pinId", "TrackID", "unick"}
JD_COOKIE_URLS = [
    "https://www.jd.com",
    "https://search.jd.com",
    "https://item.jd.com",
]

LOGIN_URL_TOKENS = (
    "passport.jd.com",
    "passport-login",
    "new/login.aspx",
    "uc.jd.com",
)
ANTI_BOT_URL_TOKENS = (
    "risk_handler",
    "jdr_shields",
    "captcha",
    "verifycode",
    "slide",
)


def _expand_markers(*tokens: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        value = str(token or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


LOGIN_MARKERS = _expand_markers(
    "\u626b\u7801\u767b\u5f55",
    "\u8bf7\u767b\u5f55",
    "\u8d26\u53f7\u767b\u5f55",
    "\u8d26\u6237\u767b\u5f55",
    "\u5bc6\u7801\u767b\u5f55",
    "\u77ed\u4fe1\u767b\u5f55",
    "\u624b\u673a\u9a8c\u8bc1\u7801\u767b\u5f55",
    "\u4eac\u4e1c\u767b\u5f55",
    "jd login",
)

SEARCH_MARKERS = _expand_markers(
    "\u7efc\u5408\u6392\u5e8f",
    "\u9500\u91cf",
    "\u8bc4\u4ef7",
    "\u81ea\u8425",
    "\u65d7\u8230\u5e97",
    "\u914d\u9001\u81f3",
    "search.jd.com",
    "\u4eac\u4e1c",
)

ANTI_BOT_MARKERS = _expand_markers(
    "\u4eac\u4e1c\u9a8c\u8bc1",
    "\u5b89\u5168\u9a8c\u8bc1",
    "JDR_shields",
    "risk_handler",
    "\u62d6\u52a8\u6ed1\u5757",
    "\u8bf7\u5b8c\u6210\u5b89\u5168\u9a8c\u8bc1",
    "ipaas-floor-app",
)

_LOGIN_TITLE_MARKERS = _expand_markers(
    "\u767b\u5f55",
    "\u4eac\u4e1c",
)
_CAPTCHA_TITLE_MARKERS = _expand_markers(
    "\u9a8c\u8bc1",
    "\u5b89\u5168",
    "captcha",
)

# Pre-computed token sets used in decide_login_page hot path.
_LOGIN_KEYWORD_MARKERS = _expand_markers("登录", "login")
_QR_SCAN_MARKERS = _expand_markers("扫码", "登录")
_JD_TITLE_LOGIN_MARKERS = _expand_markers("jd login", "京东登录")


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    haystack = str(text or "")
    haystack_lower = haystack.lower()
    return any(token in haystack or token.lower() in haystack_lower for token in tokens)


def is_search_result_url(url: str) -> bool:
    lower = (url or "").lower()
    return "search.jd.com/search" in lower


def looks_like_search_content(body_text: str) -> bool:
    body = body_text or ""
    body_lower = body.lower()
    hits = sum(1 for token in SEARCH_MARKERS if token in body or token.lower() in body_lower)
    return hits >= 2


def count_login_marker_hits(body_text: str, title: str) -> int:
    body = body_text or ""
    title_text = title or ""
    body_lower = body.lower()
    title_lower = title_text.lower()
    return sum(
        1
        for token in LOGIN_MARKERS
        if token in body or token in title_text or token.lower() in body_lower or token.lower() in title_lower
    )


def detect_non_product_page(current_url: str, title: str, body_text: str) -> str:
    url_lower = (current_url or "").lower()
    title_text = title or ""
    body = body_text or ""

    if any(token in url_lower for token in LOGIN_URL_TOKENS):
        return "redirected to JD login page"
    if any(token in url_lower for token in ANTI_BOT_URL_TOKENS):
        return "redirected to JD anti-bot verification page"
    if _contains_any(body, ANTI_BOT_MARKERS) or _contains_any(title_text, ANTI_BOT_MARKERS):
        return "JD anti-bot validation detected"

    if is_search_result_url(url_lower):
        has_item_link = "item.jd.com/" in body.lower()
        if has_item_link or looks_like_search_content(body):
            return ""

    login_hits = count_login_marker_hits(body, title_text)
    if login_hits >= 2:
        return "JD login page content detected"
    if _contains_any(title_text, _LOGIN_TITLE_MARKERS) and _contains_any(title_text, _CAPTCHA_TITLE_MARKERS):
        return "JD login page title detected"
    return ""


def decide_login_page(
    current_url: str,
    title: str,
    body_text: str,
    *,
    has_login_cookie: bool,
    has_search_dom: bool,
) -> tuple[bool, str]:
    url_lower = (current_url or "").lower()
    body = body_text or ""
    title_text = title or ""

    if any(token in url_lower for token in LOGIN_URL_TOKENS):
        return True, "url_login"
    if any(token in url_lower for token in ANTI_BOT_URL_TOKENS):
        return True, "url_antibot"
    if _contains_any(body, ANTI_BOT_MARKERS) or _contains_any(title_text, ANTI_BOT_MARKERS):
        return True, "content_antibot"

    hits = count_login_marker_hits(body, title_text)
    if is_search_result_url(current_url):
        if has_search_dom:
            return False, "search_surface_ready"
        if hits >= 2 and _contains_any(body, _LOGIN_KEYWORD_MARKERS):
            return True, "search_login_wall"
        if has_login_cookie:
            return False, "cookie_present_on_search"
        return False, "search_surface_unknown"

    if hits >= 2:
        return True, "content_login_markers"
    if _contains_any(title_text, _JD_TITLE_LOGIN_MARKERS):
        return True, "title_login"
    if _contains_any(body, _QR_SCAN_MARKERS):
        return True, "qr_login_prompt"
    if has_login_cookie:
        return False, "cookie_present_non_login"
    return False, "non_login_default"


async def wait_until_page_unblocked(
    page: Any,
    *,
    timeout_sec: int,
    stage: str,
) -> None:
    deadline = time.time() + max(30, int(timeout_sec))
    try:
        await page.bring_to_front()
    except Exception:
        pass

    while time.time() < deadline:
        current_url = getattr(page, "url", "") or ""
        try:
            title = await page.title()
        except Exception:
            title = ""
        try:
            body_text = await page.inner_text("body")
        except Exception:
            body_text = ""
        reason = detect_non_product_page(current_url, title, body_text)
        if not reason:
            return
        try:
            await page.wait_for_timeout(2000)
        except Exception:
            break
    raise RuntimeError(f"manual login timeout in {stage}, still blocked by login/anti-bot")
