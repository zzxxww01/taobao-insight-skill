# -*- coding: utf-8 -*-
# Adapted for Taobao Market Research from MediaCrawler project
# Original: https://github.com/NanmiCoder/MediaCrawler

import os
import asyncio
import socket
import httpx
import signal
import atexit
from typing import Optional, Dict, Any
from playwright.async_api import Browser, BrowserContext, Playwright

from tools.browser_launcher import BrowserLauncher
from tools import utils


class CDPBrowserManager:
    """
    CDP browser manager, responsible for launching and managing browsers connected via CDP
    Adapted for Taobao Market Research
    """

    def __init__(
        self,
        custom_browser_path: str = "",
        debug_port: int = 9222,
        save_login_state: bool = True,
        user_data_dir_template: str = "taobao_cdp_profile",
        auto_close_browser: bool = False,  # 默认不关闭，保护用户浏览器
        browser_launch_timeout: int = 30,
        safe_mode: bool = True,  # 安全模式：避开常用端口
    ):
        self.launcher = BrowserLauncher()
        self.browser: Optional[Browser] = None
        self.browser_context: Optional[BrowserContext] = None
        self.debug_port: Optional[int] = None

        # Configuration
        self.custom_browser_path = custom_browser_path
        self.config_debug_port = debug_port
        self.save_login_state = save_login_state
        self.user_data_dir_template = user_data_dir_template
        self.auto_close_browser = auto_close_browser
        self.browser_launch_timeout = browser_launch_timeout
        self.safe_mode = safe_mode

        self._cleanup_registered = False

    def _register_cleanup_handlers(self):
        """Register cleanup handlers"""
        if self._cleanup_registered:
            return

        def sync_cleanup():
            if self.launcher and self.launcher.browser_process:
                utils.logger.info("[CDPBrowserManager] atexit: Cleaning up browser process")
                self.launcher.cleanup()

        atexit.register(sync_cleanup)

        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def signal_handler(signum, frame):
            utils.logger.info(f"[CDPBrowserManager] Received signal {signum}, cleaning up")
            if self.launcher and self.launcher.browser_process:
                self.launcher.cleanup()
            if signum == signal.SIGINT:
                if prev_sigint == signal.default_int_handler:
                    return prev_sigint(signum, frame)
                raise KeyboardInterrupt
            raise SystemExit(0)

        install_sigint = prev_sigint in (signal.default_int_handler, signal.SIG_DFL)
        install_sigterm = prev_sigterm == signal.SIG_DFL

        if install_sigint:
            signal.signal(signal.SIGINT, signal_handler)
        if install_sigterm:
            signal.signal(signal.SIGTERM, signal_handler)

        self._cleanup_registered = True
        utils.logger.info("[CDPBrowserManager] Cleanup handlers registered")

    async def launch_and_connect(
        self,
        playwright: Playwright,
        playwright_proxy: Optional[Dict] = None,
        user_agent: Optional[str] = None,
        headless: bool = False,
    ) -> BrowserContext:
        """Launch browser and connect via CDP"""
        try:
            browser_path = await self._get_browser_path()
            # Safe mode: use higher port range to avoid conflicts with user's browser
            start_port = 9330 if self.safe_mode else self.config_debug_port
            self.debug_port = self.launcher.find_available_port(start_port)
            utils.logger.info(f"[CDPBrowserManager] Using port: {self.debug_port} (safe_mode={self.safe_mode})")
            await self._launch_browser(browser_path, headless)
            self._register_cleanup_handlers()
            await self._connect_via_cdp(playwright)
            browser_context = await self._create_browser_context(playwright_proxy, user_agent)
            self.browser_context = browser_context
            return browser_context
        except Exception as e:
            utils.logger.error(f"[CDPBrowserManager] CDP browser launch failed: {e}")
            await self.cleanup()
            raise

    async def _get_browser_path(self) -> str:
        """Get browser path"""
        if self.custom_browser_path and os.path.isfile(self.custom_browser_path):
            utils.logger.info(f"[CDPBrowserManager] Using custom browser: {self.custom_browser_path}")
            return self.custom_browser_path

        browser_paths = self.launcher.detect_browser_paths()
        if not browser_paths:
            raise RuntimeError(
                "No browser found. Please install Chrome/Edge or set CUSTOM_BROWSER_PATH"
            )

        browser_path = browser_paths[0]
        browser_name, browser_version = self.launcher.get_browser_info(browser_path)
        utils.logger.info(f"[CDPBrowserManager] Detected: {browser_name} ({browser_version})")
        return browser_path

    async def _launch_browser(self, browser_path: str, headless: bool):
        """Launch browser process"""
        user_data_dir = None
        if self.save_login_state:
            user_data_dir = os.path.join(
                os.getcwd(),
                "browser_data",
                self.user_data_dir_template,
            )
            os.makedirs(user_data_dir, exist_ok=True)
            utils.logger.info(f"[CDPBrowserManager] User data dir: {user_data_dir}")

        self.launcher.browser_process = self.launcher.launch_browser(
            browser_path=browser_path,
            debug_port=self.debug_port,
            headless=headless,
            user_data_dir=user_data_dir,
        )

        if not self.launcher.wait_for_browser_ready(self.debug_port, self.browser_launch_timeout):
            raise RuntimeError(f"Browser failed to start within {self.browser_launch_timeout}s")

        await asyncio.sleep(1)

    async def _connect_via_cdp(self, playwright: Playwright):
        """Connect to browser via CDP"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{self.debug_port}/json/version", timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    ws_url = data.get("webSocketDebuggerUrl")
                    if not ws_url:
                        raise RuntimeError("webSocketDebuggerUrl not found")
                else:
                    raise RuntimeError(f"HTTP {response.status_code}")
        except Exception as e:
            utils.logger.error(f"[CDPBrowserManager] Failed to get WebSocket URL: {e}")
            raise

        utils.logger.info(f"[CDPBrowserManager] Connecting via CDP: {ws_url}")
        self.browser = await playwright.chromium.connect_over_cdp(ws_url)

        if self.browser.is_connected():
            utils.logger.info("[CDPBrowserManager] Connected successfully")
        else:
            raise RuntimeError("CDP connection failed")

    async def _create_browser_context(
        self, playwright_proxy: Optional[Dict] = None, user_agent: Optional[str] = None
    ) -> BrowserContext:
        """Create or get browser context"""
        if not self.browser:
            raise RuntimeError("Browser not connected")

        contexts = self.browser.contexts

        if contexts:
            browser_context = contexts[0]
            utils.logger.info("[CDPBrowserManager] Using existing context")
        else:
            context_options = {
                "viewport": {"width": 1920, "height": 1080},
                "accept_downloads": True,
            }

            if user_agent:
                context_options["user_agent"] = user_agent
                utils.logger.info(f"[CDPBrowserManager] Setting UA: {user_agent}")

            if playwright_proxy:
                utils.logger.warning("[CDPBrowserManager] Proxy may not work in CDP mode")

            browser_context = await self.browser.new_context(**context_options)
            utils.logger.info("[CDPBrowserManager] Created new context")

        return browser_context

    async def add_stealth_script(self, script_path: str = None):
        """Add anti-detection script"""
        if script_path is None:
            script_path = os.path.join(os.path.dirname(__file__), "..", "libs", "stealth.min.js")

        if self.browser_context and os.path.exists(script_path):
            try:
                await self.browser_context.add_init_script(path=script_path)
                utils.logger.info(f"[CDPBrowserManager] Added stealth script: {script_path}")
            except Exception as e:
                utils.logger.warning(f"[CDPBrowserManager] Failed to add stealth: {e}")

    async def add_cookies(self, cookies: list):
        """Add cookies"""
        if self.browser_context:
            try:
                await self.browser_context.add_cookies(cookies)
                utils.logger.info(f"[CDPBrowserManager] Added {len(cookies)} cookies")
            except Exception as e:
                utils.logger.warning(f"[CDPBrowserManager] Failed to add cookies: {e}")

    async def get_cookies(self) -> list:
        """Get current cookies"""
        if self.browser_context:
            try:
                return await self.browser_context.cookies()
            except Exception as e:
                utils.logger.warning(f"[CDPBrowserManager] Failed to get cookies: {e}")
        return []

    async def cleanup(self, force: bool = False):
        """Cleanup resources"""
        try:
            if self.browser_context:
                try:
                    await self.browser_context.close()
                    utils.logger.info("[CDPBrowserManager] Context closed")
                except Exception:
                    utils.logger.debug("[CDPBrowserManager] Context already closed")
                finally:
                    self.browser_context = None

            if self.browser:
                try:
                    if self.browser.is_connected():
                        await self.browser.close()
                        utils.logger.info("[CDPBrowserManager] Browser disconnected")
                except Exception:
                    utils.logger.debug("[CDPBrowserManager] Browser already disconnected")
                finally:
                    self.browser = None

            if force or self.auto_close_browser:
                if self.launcher and self.launcher.browser_process:
                    self.launcher.cleanup()
            else:
                utils.logger.info("[CDPBrowserManager] Browser kept running")

        except Exception as e:
            utils.logger.error(f"[CDPBrowserManager] Cleanup error: {e}")

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.browser is not None and self.browser.is_connected()

    async def get_browser_info(self) -> Dict[str, Any]:
        """Get browser info"""
        if not self.browser:
            return {}
        try:
            return {
                "version": self.browser.version,
                "contexts_count": len(self.browser.contexts),
                "debug_port": self.debug_port,
                "is_connected": self.is_connected(),
            }
        except Exception as e:
            utils.logger.warning(f"[CDPBrowserManager] Failed to get info: {e}")
            return {}
