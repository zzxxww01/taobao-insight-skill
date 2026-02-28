# -*- coding: utf-8 -*-
"""Utility functions for Taobao Market Research"""

import logging

# Initialize logging
def init_logging_config():
    level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    _logger = logging.getLogger("TaobaoInsight")
    _logger.setLevel(level)

    # Disable httpx INFO level logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return _logger


logger = init_logging_config()


def convert_cookies(cookies_list):
    """
    Convert Playwright cookies to dict format

    Args:
        cookies_list: List of cookies from browser_context.cookies()

    Returns:
        tuple: (cookies_list, cookies_dict)
    """
    if not cookies_list:
        return [], {}

    cookies_dict = {}
    for cookie in cookies_list:
        name = cookie.get('name', '')
        value = cookie.get('value', '')
        if name:
            cookies_dict[name] = value

    return cookies_list, cookies_dict


def convert_str_cookie_to_dict(cookie_str: str) -> dict:
    """
    Convert cookie string to dict

    Args:
        cookie_str: Cookie string like "name1=value1; name2=value2"

    Returns:
        dict: Cookie dict
    """
    if not cookie_str:
        return {}

    cookies = {}
    for item in cookie_str.split(';'):
        item = item.strip()
        if '=' in item:
            name, value = item.split('=', 1)
            cookies[name.strip()] = value.strip()

    return cookies
