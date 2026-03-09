# -*- coding: utf-8 -*-
"""Global browser manager for marketplace workflows.

Both Taobao and JD use the same browser strategy here:
1. raw CDP first
2. fallback to Playwright persistent context on CDP init failure
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import Playwright, async_playwright

from tools import utils
from tools.browser_hardening import (
    build_persistent_context_kwargs,
    default_stealth_script_path,
)
from tools.cdp_browser import CDPBrowserManager


class GlobalBrowserManager:
    """Maintain a shared browser context and issue pages on demand."""

    _instance: Optional["GlobalBrowserManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.playwright: Optional[Playwright] = None
        self.browser_context: Optional[Any] = None
        self.cdp_manager: Optional[CDPBrowserManager] = None
        self._is_cdp_mode = False
        self._headless = False
        self._initialized = False
        self._context_lock = asyncio.Lock()
        self._config_signature: tuple[Any, ...] | None = None

    @classmethod
    async def get_instance(cls) -> "GlobalBrowserManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def _initialize_persistent_context(
        self,
        *,
        headless: bool,
        user_data_dir: str,
        user_agent: str,
    ) -> Any:
        self.playwright = await async_playwright().start()
        profile = Path(user_data_dir).resolve()
        profile.mkdir(parents=True, exist_ok=True)
        launch_kwargs = build_persistent_context_kwargs(
            headless=headless,
            user_agent=user_agent,
        )
        try:
            context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile),
                **launch_kwargs,
            )
        except Exception as exc:
            fallback = (profile.parent / f"{profile.name}_runtime").resolve()
            fallback.mkdir(parents=True, exist_ok=True)
            utils.logger.warning(
                "[GlobalBrowserManager] profile launch failed %s, fallback to %s: %s",
                profile,
                fallback,
                exc,
            )
            context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(fallback),
                **launch_kwargs,
            )
        stealth = default_stealth_script_path()
        if stealth.exists():
            await context.add_init_script(path=str(stealth))
        return context

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
    ) -> Any:
        async with self._lock:
            self._headless = bool(headless)
            requested_mode = str(browser_mode or "cdp").strip().lower()
            if requested_mode not in {"cdp", "persistent"}:
                requested_mode = "cdp"
            resolved_user_data_dir = str(Path(user_data_dir).resolve())
            requested_signature = (
                requested_mode,
                bool(headless),
                resolved_user_data_dir,
                str(cdp_url or "").strip(),
                str(custom_browser_path or "").strip(),
            )
            if self._initialized:
                if self._config_signature == requested_signature:
                    utils.logger.info("[GlobalBrowserManager] already initialized (double-check)")
                    return self.browser_context
                utils.logger.info(
                    "[GlobalBrowserManager] config changed, reinitializing browser context"
                )
                await self.cleanup()

            self._is_cdp_mode = (requested_mode == "cdp")
            utils.logger.info(
                "[GlobalBrowserManager] initializing mode=%s headless=%s",
                requested_mode,
                headless,
            )
            try:
                if requested_mode == "cdp":
                    try:
                        self.cdp_manager = CDPBrowserManager(
                            custom_browser_path=custom_browser_path,
                            debug_port=debug_port,
                            save_login_state=save_login_state,
                            user_data_dir_template=user_data_dir,
                            auto_close_browser=auto_close_browser,
                            safe_mode=True,
                            cdp_url=cdp_url or "",
                        )
                        self.browser_context = await self.cdp_manager.launch_and_connect(
                            playwright=None,
                            user_agent=user_agent or None,
                            headless=headless,
                        )
                        await self.cdp_manager.add_stealth_script()
                        self._is_cdp_mode = True
                    except Exception as exc:
                        utils.logger.warning(
                            "[GlobalBrowserManager] CDP init failed, fallback to persistent mode: %s",
                            exc,
                        )
                        if self.cdp_manager is not None:
                            try:
                                await self.cdp_manager.cleanup(force=True)
                            except Exception:
                                pass
                            self.cdp_manager = None
                        self._is_cdp_mode = False
                        self.browser_context = await self._initialize_persistent_context(
                            headless=headless,
                            user_data_dir=resolved_user_data_dir,
                            user_agent=user_agent,
                        )
                else:
                    self._is_cdp_mode = False
                    self.browser_context = await self._initialize_persistent_context(
                        headless=headless,
                        user_data_dir=resolved_user_data_dir,
                        user_agent=user_agent,
                    )

                if self.browser_context is None:
                    raise RuntimeError("browser context is None")
                self._initialized = True
                self._config_signature = requested_signature
                utils.logger.info(
                    "[GlobalBrowserManager] initialized pages=%s",
                    len(getattr(self.browser_context, "pages", []) or []),
                )
                return self.browser_context
            except Exception:
                await self.cleanup()
                raise

    async def get_page(self, url: Optional[str] = None) -> Any:
        if not self._initialized or self.browser_context is None:
            raise RuntimeError("Browser not initialized")

        def _is_target_closed_error(exc: Exception) -> bool:
            text = str(exc).lower()
            return "target page" in text and "closed" in text

        async with self._context_lock:
            pages = [
                p for p in list(getattr(self.browser_context, "pages", []) or [])
                if not p.is_closed()
            ]
            page = pages[0] if pages else await self.browser_context.new_page()
            if url:
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                except Exception as exc:
                    if not _is_target_closed_error(exc):
                        raise
                    try:
                        if not page.is_closed():
                            await page.close()
                    except Exception:
                        pass
                    page = await self.browser_context.new_page()
                    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            return page

    async def new_page(self) -> Any:
        if not self._initialized or self.browser_context is None:
            raise RuntimeError("Browser not initialized")
        async with self._context_lock:
            return await self.browser_context.new_page()

    async def cleanup(self) -> None:
        utils.logger.info("[GlobalBrowserManager] cleanup")

        if self.cdp_manager is not None:
            await self.cdp_manager.cleanup(force=True)
            self.cdp_manager = None

        if self.browser_context is not None:
            try:
                await self.browser_context.close()
            except Exception:
                pass
            self.browser_context = None

        if self.playwright is not None:
            try:
                await self.playwright.stop()
            except Exception:
                pass
            self.playwright = None

        self._initialized = False
        self._config_signature = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized and self.browser_context is not None

    @property
    def current_mode(self) -> str:
        return "cdp" if self._is_cdp_mode else "persistent"


_global_manager: Optional[GlobalBrowserManager] = None


async def get_global_browser_manager() -> GlobalBrowserManager:
    global _global_manager
    instance = await GlobalBrowserManager.get_instance()
    if _global_manager is None:
        _global_manager = instance
    return _global_manager


async def cleanup_global_browser() -> None:
    global _global_manager
    if _global_manager is None:
        _global_manager = GlobalBrowserManager._instance
    if _global_manager is not None:
        await _global_manager.cleanup()
        _global_manager = None
