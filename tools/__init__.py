# -*- coding: utf-8 -*-
"""Tools package for Taobao Market Research"""

from .cdp_browser import CDPBrowserManager
from .browser_launcher import BrowserLauncher
from .browser_manager import GlobalBrowserManager, get_global_browser_manager, cleanup_global_browser
from .taobao_login import LoginDecision, LoginHandleResult, TaobaoLogin
from .login_rules import LOGIN_COOKIE_NAMES, detect_non_product_page, wait_until_page_unblocked
from . import utils

__all__ = [
    "CDPBrowserManager",
    "BrowserLauncher",
    "GlobalBrowserManager",
    "get_global_browser_manager",
    "cleanup_global_browser",
    "LoginDecision",
    "LoginHandleResult",
    "TaobaoLogin",
    "LOGIN_COOKIE_NAMES",
    "detect_non_product_page",
    "wait_until_page_unblocked",
    "utils",
]
