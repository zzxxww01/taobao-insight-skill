# -*- coding: utf-8 -*-
"""Centralized Taobao/Tmall login detection and wait helpers."""

from __future__ import annotations

import time
from typing import Any


LOGIN_COOKIE_NAMES = {"cookie2", "unb", "_tb_token_", "_m_h5_tk"}
TAOBAO_COOKIE_URLS = [
    "https://www.taobao.com",
    "https://s.taobao.com",
    "https://detail.tmall.com",
    "https://www.tmall.com",
]

LOGIN_URL_TOKENS = ("login.taobao.com", "member/login")
ANTI_BOT_URL_TOKENS = ("captcha", "punish", "x5sec", "_____tmd_____")

# Keep keywords explicit and centralized for future tuning.
LOGIN_MARKERS = (
    "扫码登录",
    "请扫码登录",
    "扫一扫登录",
    "账号密码登录",
    "密码登录",
    "短信登录",
    "手机验证码登录",
    "忘记密码",
    "免费注册",
    "请登录",
    "登录淘宝",
    "登录天猫",
    "taobao login",
    "tmall login",
)

SEARCH_MARKERS = (
    "人付款",
    "已售",
    "综合排序",
    "销量",
    "筛选",
    "收货地",
    "店铺",
    "价格",
    "taobao",
    "tmall",
)


def is_search_result_url(url: str) -> bool:
    lower = (url or "").lower()
    if "s.taobao.com/search" in lower:
        return True
    if "s.taobao.com" in lower and ("q=" in lower or "search" in lower or "sort=" in lower):
        return True
    if "list.tmall.com/search_product.htm" in lower:
        return True
    if "list.tmall.com" in lower and "q=" in lower:
        return True
    return False


def looks_like_search_content(body_text: str) -> bool:
    body = body_text or ""
    body_lower = body.lower()
    hits = sum(1 for token in SEARCH_MARKERS if token in body or token in body_lower)
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
    title_lower = (title or "").lower()
    body = body_text or ""
    body_lower = body.lower()

    if any(token in url_lower for token in LOGIN_URL_TOKENS):
        return "redirected to Taobao login page"
    if any(token in url_lower for token in ANTI_BOT_URL_TOKENS):
        return "redirected to anti-bot verification page"
    if "rgv587" in body_lower or "fail_sys_user_validate" in body_lower:
        return "anti-bot validation detected in response"

    if is_search_result_url(url_lower):
        has_item_link = (
            "item.taobao.com/item.htm?id=" in body_lower
            or "detail.tmall.com/item.htm?id=" in body_lower
        )
        if has_item_link or looks_like_search_content(body):
            return ""

    login_hits = count_login_marker_hits(body, title)
    if login_hits >= 2:
        return "login page content detected"
    if ("登录" in (title or "") and ("淘宝" in (title or "") or "天猫" in (title or ""))) or (
        "login" in title_lower and ("taobao" in title_lower or "tmall" in title_lower)
    ):
        return "login page title detected"
    if "captcha" in title_lower:
        return "captcha title detected"
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
    title_lower = (title or "").lower()
    body = body_text or ""
    body_lower = body.lower()

    if any(token in url_lower for token in LOGIN_URL_TOKENS):
        return True, "url_login"
    if any(token in url_lower for token in ANTI_BOT_URL_TOKENS):
        return True, "url_antibot"

    hits = count_login_marker_hits(body, title)
    if is_search_result_url(current_url):
        if has_search_dom:
            return False, "search_surface_ready"
        if hits >= 2 and ("登录" in body or "login" in body_lower):
            return True, "search_login_wall"
        if has_login_cookie:
            return False, "cookie_present_on_search"
        return False, "search_surface_unknown"

    if hits >= 2:
        return True, "content_login_markers"
    if "login" in title_lower and ("taobao" in title_lower or "tmall" in title_lower):
        return True, "title_login"
    if "扫码" in body and "登录" in body:
        return True, "qr_login_prompt"
    if "请登录" in body and ("taobao" in body_lower or "tmall" in body_lower):
        return True, "login_prompt"
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

