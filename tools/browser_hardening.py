# -*- coding: utf-8 -*-
"""Shared browser hardening helpers for CDP and persistent modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_VIEWPORT = {"width": 1920, "height": 1080}
DEFAULT_LOCALE = "zh-CN"
DEFAULT_TIMEZONE_ID = "Asia/Shanghai"
DEFAULT_ACCEPT_LANGUAGE = "zh-CN,zh;q=0.9,en;q=0.8"

_COMMON_BROWSER_ARGS = [
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-features=TranslateUI",
    "--disable-ipc-flooding-protection",
    "--disable-hang-monitor",
    "--disable-prompt-on-repost",
    "--disable-sync",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--exclude-switches=enable-automation",
    "--disable-infobars",
    f"--lang={DEFAULT_LOCALE}",
]

_HEADLESS_BROWSER_ARGS = [
    "--headless=new",
    "--disable-gpu",
]


def build_browser_launch_args(*, headless: bool) -> list[str]:
    args = list(_COMMON_BROWSER_ARGS)
    if headless:
        args.extend(_HEADLESS_BROWSER_ARGS)
    else:
        args.append("--start-maximized")
    return args


def build_persistent_context_kwargs(*, headless: bool, user_agent: str) -> dict[str, Any]:
    return {
        "headless": bool(headless),
        "args": build_browser_launch_args(headless=headless),
        "ignore_default_args": ["--enable-automation"],
        "viewport": dict(DEFAULT_VIEWPORT),
        "channel": "chrome",
        "user_agent": user_agent or None,
        "locale": DEFAULT_LOCALE,
        "timezone_id": DEFAULT_TIMEZONE_ID,
    }


def default_stealth_script_path() -> Path:
    return (Path(__file__).resolve().parent.parent / "libs" / "stealth.min.js").resolve()


def load_stealth_script_source(script_path: str = "") -> str:
    path = Path(script_path).resolve() if script_path else default_stealth_script_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")
