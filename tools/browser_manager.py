# -*- coding: utf-8 -*-
"""Global Browser Manager for Taobao Market Research

Implements single browser instance + multiple pages pattern.
"""

import asyncio
from typing import Optional, Any
from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from tools.cdp_browser import CDPBrowserManager
from tools import utils


class GlobalBrowserManager:
    """
    Global browser manager that maintains a single browser instance
    and creates multiple pages as needed.
    """

    _instance: Optional["GlobalBrowserManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.playwright: Optional[Playwright] = None
        self.browser_context: Optional[BrowserContext] = None
        self.cdp_manager: Optional[CDPBrowserManager] = None
        self._is_cdp_mode: bool = False
        self._headless: bool = False
        self._initialized: bool = False
        # Lock to serialize access to browser context
        self._context_lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls) -> "GlobalBrowserManager":
        """Get or create the singleton instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def initialize(
        self,
        browser_mode: str = "cdp",
        headless: bool = False,
        cdp_url: str = "",
        custom_browser_path: str = "",
        debug_port: int = 9222,
        save_login_state: bool = True,
        user_data_dir: str = "taobao_cdp_profile",
        auto_close_browser: bool = False,
        user_agent: str = "",
    ) -> BrowserContext:
        """
        Initialize the global browser instance

        Returns:
            BrowserContext: The shared browser context
        """
        if self._initialized:
            utils.logger.info("[GlobalBrowserManager] Already initialized, returning existing context")
            return self.browser_context

        # Lock to prevent concurrent initialization
        async with self._lock:
            if self._initialized:
                utils.logger.info("[GlobalBrowserManager] Already initialized (double-check), returning existing context")
                return self.browser_context

            self._headless = headless
            self._is_cdp_mode = (browser_mode == "cdp")

            utils.logger.info(f"[GlobalBrowserManager] Initializing browser (mode={browser_mode}, headless={headless})")

            try:
                self.playwright = await async_playwright().start()

                if self._is_cdp_mode:
                    # Use CDP mode with CDPBrowserManager
                    self.cdp_manager = CDPBrowserManager(
                        custom_browser_path=custom_browser_path,
                        debug_port=debug_port,
                        save_login_state=save_login_state,
                        user_data_dir_template=user_data_dir,
                        auto_close_browser=auto_close_browser,
                        safe_mode=True,  # 启用安全模式，避开常用端口
                    )

                    if cdp_url:
                        # Connect to existing CDP browser
                        utils.logger.info(f"[GlobalBrowserManager] Connecting to existing CDP: {cdp_url}")
                        self.browser_context = await self._connect_to_existing_cdp(cdp_url)
                    else:
                        # Launch new CDP browser
                        utils.logger.info("[GlobalBrowserManager] Launching new CDP browser")
                        self.browser_context = await self.cdp_manager.launch_and_connect(
                            playwright=self.playwright,
                            user_agent=user_agent or None,
                            headless=headless,
                        )

                    # Add stealth script
                    await self.cdp_manager.add_stealth_script()

                else:
                    # Use persistent context mode
                    from pathlib import Path
                    user_data_path = Path(user_data_dir).resolve()
                    user_data_path.mkdir(parents=True, exist_ok=True)

                    utils.logger.info(f"[GlobalBrowserManager] Using persistent context: {user_data_path}")

                    args = [
                        "--disable-blink-features=AutomationControlled",
                        "--exclude-switches=enable-automation",
                        "--disable-infobars",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-blink-features=AutomationControlled",
                    ]

                    self.browser_context = await self.playwright.chromium.launch_persistent_context(
                        user_data_dir=str(user_data_path),
                        headless=headless,
                        args=args,
                        viewport={"width": 1920, "height": 1080},
                        channel="chrome",
                        user_agent=user_agent or None,
                    )

                    # Add stealth script
                    stealth_path = Path(__file__).parent.parent / "libs" / "stealth.min.js"
                    if stealth_path.exists():
                        await self.browser_context.add_init_script(path=str(stealth_path))
                        utils.logger.info("[GlobalBrowserManager] Added stealth.min.js")

                self._initialized = True
                utils.logger.info("[GlobalBrowserManager] Browser initialized successfully")

                # Verify context is valid before returning
                if self.browser_context is None:
                    raise RuntimeError("Browser context is None after initialization")

                utils.logger.info(f"[GlobalBrowserManager] Context valid check: pages={len(self.browser_context.pages)}")

                return self.browser_context

            except Exception as e:
                utils.logger.error(f"[GlobalBrowserManager] Failed to initialize: {e}")
                await self.cleanup()
                raise

    async def _connect_to_existing_cdp(self, cdp_url: str) -> BrowserContext:
        """Connect to an existing CDP browser"""
        # Try to get WebSocket URL from CDP endpoint
        import httpx

        # Extract port from URL
        port = 9222
        if ":9222" in cdp_url:
            port = int(cdp_url.split(":")[-1].split("/")[0])

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{port}/json/version", timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    ws_url = data.get("webSocketDebuggerUrl")
                    if ws_url:
                        utils.logger.info(f"[GlobalBrowserManager] Got WS URL: {ws_url}")
                        browser = await self.playwright.chromium.connect_over_cdp(ws_url)
                        contexts = browser.contexts
                        if contexts:
                            utils.logger.info("[GlobalBrowserManager] Using existing CDP context")
                            return contexts[0]
                        else:
                            utils.logger.info("[GlobalBrowserManager] Creating new CDP context")
                            return await browser.new_context()
        except Exception as e:
            utils.logger.warning(f"[GlobalBrowserManager] Failed to get CDP info: {e}")

        # Fallback to direct connection
        utils.logger.info(f"[GlobalBrowserManager] Connecting directly to: {cdp_url}")
        browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
        contexts = browser.contexts
        if contexts:
            return contexts[0]
        return await browser.new_context()

    async def get_page(self, url: Optional[str] = None) -> Page:
        """
        Get a page from the browser context

        Args:
            url: Optional URL to navigate to

        Returns:
            Page: A page (new or existing)
        """
        if not self._initialized or not self.browser_context:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        # Use lock to prevent concurrent page creation issues
        async with self._context_lock:
            # Try to find an existing page
            pages = self.browser_context.pages
            if pages:
                page = pages[0]
            else:
                page = await self.browser_context.new_page()

            if url:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)

            return page

    async def new_page(self) -> Page:
        """Create a new page with lock protection"""
        if not self._initialized or not self.browser_context:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        async with self._context_lock:
            try:
                # Try to reuse existing pages instead of creating new ones
                pages = self.browser_context.pages
                if pages:
                    # Reuse the first page
                    page = pages[0]
                    utils.logger.info(f"[GlobalBrowserManager] Reusing existing page: {page}")
                    # Navigate to blank if needed
                    if page.url != "about:blank":
                        try:
                            await page.goto("about:blank", wait_until="domcontentloaded", timeout=5000)
                        except Exception:
                            pass
                    return page

                # If no pages exist, try to create a new one
                utils.logger.info("[GlobalBrowserManager] No existing pages, creating new page")
                page = await self.browser_context.new_page()
                utils.logger.info(f"[GlobalBrowserManager] Successfully created new page: {page}")
                return page
            except Exception as e:
                # Log the error for debugging
                utils.logger.error(f"[GlobalBrowserManager] Failed to create/reuse page: {e}")
                utils.logger.error(f"[GlobalBrowserManager] Context: {self.browser_context}")
                utils.logger.error(f"[GlobalBrowserManager] Context._impl_obj: {getattr(self.browser_context, '_impl_obj', 'N/A')}")
                if hasattr(self.browser_context, '_impl_obj') and self.browser_context._impl_obj:
                    browser = getattr(self.browser_context._impl_obj, '_browser', None)
                    utils.logger.error(f"[GlobalBrowserManager] Browser: {browser}")
                    if browser:
                        utils.logger.error(f"[GlobalBrowserManager] Browser.is_connected(): {browser.is_connected()}")
                raise

    async def cleanup(self):
        """Cleanup browser resources"""
        utils.logger.info("[GlobalBrowserManager] Cleaning up...")

        if self.cdp_manager:
            await self.cdp_manager.cleanup(force=True)
            self.cdp_manager = None

        if self.browser_context:
            try:
                await self.browser_context.close()
            except Exception:
                pass
            self.browser_context = None

        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
            self.playwright = None

        self._initialized = False
        utils.logger.info("[GlobalBrowserManager] Cleanup complete")

    @property
    def is_initialized(self) -> bool:
        """Check if browser is initialized"""
        return self._initialized and self.browser_context is not None


# Singleton instance
_global_manager: Optional[GlobalBrowserManager] = None


async def get_global_browser_manager() -> GlobalBrowserManager:
    """Get the global browser manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = await GlobalBrowserManager.get_instance()
    return _global_manager


async def cleanup_global_browser():
    """Cleanup the global browser manager"""
    global _global_manager
    if _global_manager is None:
        _global_manager = GlobalBrowserManager._instance
    if _global_manager:
        await _global_manager.cleanup()
        _global_manager = None
