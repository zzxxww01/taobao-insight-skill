"""Web scraping and data extraction for Taobao/Tmall."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import statistics
import sys
import threading
import urllib.request
from html import unescape as html_unescape
from pathlib import Path
from typing import Any
from urllib.parse import (
    quote_plus,
    urlparse,
)

# Global lock to serialize Playwright access across threads
PLAYWRIGHT_LOCK = threading.Lock()

# Import new browser management tools
# Add parent directory to sys.path to import tools module
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from tools import (
    LoginHandleResult,
    TaobaoLogin,
    get_global_browser_manager,
)
from tools.login_rules import (
    detect_non_product_page,
    is_search_result_url,
)

from config import (
    ANTI_BOT_MARKERS,
    BANNED_SHOP_MARKER_RE,
    BRAND_RE,
    LINE_BREAK_RE,
    NON_WORD_RE,
    SEARCH_BLOCK_HINT,
    SHOP_NAME_RE,
    SKU_MAP_RE,
)
from data import (
    ItemDetail,
    Storage,
    UrlRecord,
    clean_text,
    extract_candidate_item_urls,
    is_official_shop,
    normalize_brand_name,
    normalize_item_title,
    normalize_url,
    now_iso,
    parse_price_values,
    parse_sales_to_int,
    read_text_utf8_best,
)

LOG = logging.getLogger("taobao_insight")


def detect_antibot_signal(*texts: str) -> str:
    merged = "\n".join(t for t in texts if t)
    if not merged:
        return ""
    merged_lower = merged.lower()
    for marker in ANTI_BOT_MARKERS:
        if marker.lower() in merged_lower:
            return marker
    return ""


def _build_login_recovery_event(
    *,
    source: str,
    stage: str,
    blocked_reason: str,
    current_url: str,
    login_result: LoginHandleResult,
    browser_mode: str,
    storage_state_file: Path,
    user_data_dir: Path,
    item_id: str = "",
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "source": str(source or ""),
        "stage": str(stage or ""),
        "ok": bool(login_result.ok),
        "blocked_reason": str(blocked_reason or ""),
        "final_state": str(login_result.final_state or ""),
        "reason": str(login_result.reason or ""),
        "elapsed_sec": round(float(login_result.elapsed_sec), 3),
        "url": str(current_url or ""),
        "browser_mode": str(browser_mode or ""),
        "storage_state_file": str(storage_state_file),
        "user_data_dir": str(user_data_dir),
        "cookie_fingerprint_before": str(login_result.cookie_fingerprint_before or ""),
        "cookie_fingerprint_after": str(login_result.cookie_fingerprint_after or ""),
        "cookie_changed": bool(login_result.cookie_changed),
        "decision_trace": list(login_result.decision_trace),
        "updated_at": now_iso(),
    }
    if item_id:
        event["item_id"] = str(item_id)
    return event


def _read_storage_state_cookies(storage_state_file: Path) -> list[dict[str, Any]]:
    if not storage_state_file.exists():
        return []
    try:
        payload = json.loads(storage_state_file.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Failed to read storage state %s: %s", storage_state_file, exc)
        return []
    cookies = payload.get("cookies", []) if isinstance(payload, dict) else []
    return [row for row in cookies if isinstance(row, dict)]


def _cookie_header_for_domains(
    storage_state_file: Path,
    *,
    allowed_domain_tokens: tuple[str, ...],
) -> str:
    cookie_pairs: list[str] = []
    for cookie in _read_storage_state_cookies(storage_state_file):
        domain = str(cookie.get("domain", "")).lower()
        if not any(token in domain for token in allowed_domain_tokens):
            continue
        name = str(cookie.get("name", "")).strip()
        value = str(cookie.get("value", "")).strip()
        if not name:
            continue
        cookie_pairs.append(f"{name}={value}")
    return "; ".join(cookie_pairs)


async def _browser_fetch_text_via_page(
    page: Any,
    target_url: str,
    *,
    accept: str,
    max_len: int = 1_200_000,
) -> str:
    script = f"""
(async () => {{
  try {{
    const response = await fetch({json.dumps(str(target_url or ""), ensure_ascii=False)}, {{
      method: "GET",
      credentials: "include",
      cache: "no-store",
      headers: {{
        "Accept": {json.dumps(str(accept or "*/*"), ensure_ascii=False)}
      }}
    }});
    const text = await response.text();
    return String(text || "").slice(0, {int(max_len)});
  }} catch (error) {{
    return "";
  }}
}})()
"""
    try:
        value = await page.evaluate(script)
    except Exception:
        return ""
    return str(value or "")


async def _browser_user_agent(page: Any) -> str:
    try:
        value = await page.evaluate("navigator.userAgent || ''")
    except Exception:
        return ""
    return str(value or "").strip()


def _http_fetch_text_with_cookie_header(
    target_url: str,
    *,
    user_agent: str,
    referer: str,
    accept: str,
    cookie_header: str,
    timeout_sec: int = 20,
    max_len: int = 1_200_000,
) -> str:
    headers = {
        "User-Agent": user_agent or "Mozilla/5.0",
        "Referer": referer,
        "Accept": accept,
    }
    if cookie_header:
        headers["Cookie"] = cookie_header
    request = urllib.request.Request(target_url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return str(response.read().decode("utf-8", errors="replace"))[:max_len]


def is_item_detail_url(url: str) -> bool:
    lower = (url or "").lower()
    return "detail.tmall.com/item.htm" in lower or "item.taobao.com/item.htm" in lower


def load_url_lines(path: str | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"item URL file not found: {path}")
    lines: list[str] = []
    for line in read_text_utf8_best(file_path).splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            lines.append(value)
    return lines


def _default_user_data_dir(profile_name: str = "taobao_insight_profile") -> Path:
    appdata = os.getenv("APPDATA", "")
    if appdata:
        return Path(appdata) / profile_name
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / profile_name
    return Path.home() / ".config" / profile_name


def _cdp_http_base(cdp_url: str) -> str:
    value = (cdp_url or "").strip()
    if not value:
        raise ValueError("empty cdp url")
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/")
    if value.startswith("ws://") or value.startswith("wss://"):
        parsed = urlparse(value)
        scheme = "http" if parsed.scheme == "ws" else "https"
        if not parsed.netloc:
            raise ValueError(f"invalid cdp websocket url: {cdp_url}")
        return f"{scheme}://{parsed.netloc}"
    raise ValueError(f"unsupported cdp endpoint: {cdp_url}")


def _read_json_url(url: str, timeout_sec: float = 10.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = response.read()
    text = payload.decode("utf-8", errors="replace")
    return json.loads(text)


def list_cdp_pages(cdp_url: str) -> list[dict[str, str]]:
    base = _cdp_http_base(cdp_url)
    payload = _read_json_url(f"{base}/json/list", timeout_sec=10.0)
    pages: list[dict[str, str]] = []
    if not isinstance(payload, list):
        return pages
    for row in payload:
        if not isinstance(row, dict):
            continue
        if str(row.get("type", "")).lower() != "page":
            continue
        pages.append(
            {
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "")),
                "url": str(row.get("url", "")),
                "webSocketDebuggerUrl": str(row.get("webSocketDebuggerUrl", "")),
            }
        )
    return pages


class SearchClient:
    DEFAULT_STORAGE_STATE_FILE = "taobao_storage_state.json"
    DEFAULT_USER_DATA_DIR_NAME = "taobao_insight_profile"
    PLATFORM_LABEL = "Taobao"
    LOGIN_HANDLER_CLS = TaobaoLogin
    DETECT_NON_PRODUCT_PAGE_FN = staticmethod(detect_non_product_page)

    def __init__(
        self,
        headless: bool = False,
        browser_mode: str = "cdp",
        cdp_url: str = "",
        manual_wait_seconds: int = 0,
        storage_state_file: str | Path | None = None,
        user_data_dir: str | Path | None = None,
        manual_login_timeout_sec: int = 300,
    ) -> None:
        self.headless = headless
        self.browser_mode = (browser_mode or "cdp").strip().lower()
        if self.browser_mode not in {"cdp", "persistent"}:
            self.browser_mode = "cdp"
        self.cdp_url = cdp_url.strip()
        self.manual_wait_seconds = max(0, int(manual_wait_seconds))
        self.storage_state_file = (
            Path(storage_state_file).resolve()
            if storage_state_file
            else Path("data") / self.DEFAULT_STORAGE_STATE_FILE
        )
        self.storage_state_file = self.storage_state_file.resolve()
        self.user_data_dir = (
            Path(user_data_dir).resolve()
            if user_data_dir
            else _default_user_data_dir(self.DEFAULT_USER_DATA_DIR_NAME).resolve()
        )
        self.manual_login_timeout_sec = max(30, int(manual_login_timeout_sec))

        # Global browser manager reference.
        self._global_browser_manager: Any = None
        # Structured login recovery events for diagnostics/output.
        self.login_recovery_events: list[dict[str, Any]] = []
        self._restored_storage_state_context_ids: set[int] = set()

    def _effective_browser_mode(self) -> str:
        if self._global_browser_manager is not None:
            mode = getattr(self._global_browser_manager, "current_mode", "")
            if mode:
                return str(mode)
        return self.browser_mode

    def _detect_non_product_page(self, current_url: str, title: str, body_text: str) -> str:
        detector = getattr(self, "DETECT_NON_PRODUCT_PAGE_FN", detect_non_product_page)
        return str(detector(current_url, title, body_text) or "")

    @staticmethod
    def _records_from_candidate_urls(urls: list[str], top_n: int) -> list[UrlRecord]:
        records: list[UrlRecord] = []
        seen: set[str] = set()
        for url in urls:
            rec = normalize_url(url)
            if not rec or rec.item_id in seen:
                continue
            seen.add(rec.item_id)
            records.append(rec)
            if len(records) >= top_n:
                break
        return records

    @staticmethod
    def _records_from_cards(
        cards: list[dict[str, Any]],
        top_n: int,
        search_sort: str,
        official_only: bool,
    ) -> list[UrlRecord]:
        ranked: list[tuple[int, int, UrlRecord]] = []
        seen_item_ids: set[str] = set()
        for index, card in enumerate(cards, start=1):
            if not isinstance(card, dict):
                continue
            href = str(card.get("href") or card.get("url") or "").strip()
            if not href:
                continue
            rec = normalize_url(href)
            if not rec or rec.item_id in seen_item_ids:
                continue
            title = normalize_item_title(str(card.get("title", "")), max_len=160)
            shop_name = clean_text(str(card.get("shop_name", "")), max_len=120)
            sales_text = clean_text(str(card.get("sales_text", "")), max_len=40)
            card_text = clean_text(str(card.get("card_text", "")), max_len=400)

            if BANNED_SHOP_MARKER_RE.search(shop_name) or BANNED_SHOP_MARKER_RE.search(
                card_text
            ):
                continue

            official = is_official_shop(shop_name, card_text)
            if official_only and not official:
                continue
            rec.search_rank = index
            rec.title = title
            rec.shop_name = shop_name
            rec.sales_text = sales_text
            rec.is_official_store = official
            seen_item_ids.add(rec.item_id)
            sales_value = parse_sales_to_int(sales_text or card_text)
            ranked.append((index, sales_value, rec))

        if search_sort == "sales":
            ranked.sort(key=lambda item: (-item[1], item[0]))
        else:
            ranked.sort(key=lambda item: item[0])
        return [item[2] for item in ranked[:top_n]]

    async def _persist_storage_state_async(self, context: Any) -> None:
        self.storage_state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            await context.storage_state(path=str(self.storage_state_file))
        except Exception:
            return

    async def _restore_storage_state_async(self, context: Any) -> None:
        context_key = id(context)
        if context_key in self._restored_storage_state_context_ids:
            return
        normalized = _read_storage_state_cookies(self.storage_state_file)
        if not normalized:
            return
        try:
            await context.add_cookies(normalized)
            self._restored_storage_state_context_ids.add(context_key)
            LOG.info(
                "Restored %s cookies from storage state %s",
                len(normalized),
                self.storage_state_file,
            )
        except Exception as exc:
            LOG.warning("Failed to restore cookies from %s: %s", self.storage_state_file, exc)

    async def _read_page_snapshot_async(self, page: Any) -> tuple[str, str, str]:
        current_url = page.url or ""
        try:
            title = await page.title()
        except Exception:
            title = ""
        try:
            body_text = await page.inner_text("body")
        except Exception:
            body_text = ""
        return current_url, title, body_text

    async def _handle_login_if_needed_async(
        self, page: Any, context: Any, stage: str
    ) -> bool:
        current_url, title, body_text = await self._read_page_snapshot_async(page)
        reason = self._detect_non_product_page(current_url, title, body_text)
        if not reason:
            return False

        LOG.warning("[%s] Detected blocked page during %s: %s", self.PLATFORM_LABEL, stage, reason)
        login_handler = self.LOGIN_HANDLER_CLS(
            browser_context=context,
            context_page=page,
            login_timeout_sec=self.manual_login_timeout_sec,
        )
        login_result: LoginHandleResult = await login_handler.check_and_handle_login()
        self.login_recovery_events.append(
            _build_login_recovery_event(
                source="search",
                stage=stage,
                blocked_reason=reason,
                current_url=current_url,
                login_result=login_result,
                browser_mode=self._effective_browser_mode(),
                storage_state_file=self.storage_state_file,
                user_data_dir=self.user_data_dir,
            )
        )
        if not login_result.ok:
            raise RuntimeError(
                f"{self.PLATFORM_LABEL} login timeout in {stage}, still blocked: "
                f"{reason} ({login_result.reason})"
            )

        # Persist cookies/state immediately after successful login recovery.
        await self._persist_storage_state_async(context)
        return True

    async def _ensure_search_surface_async(
        self,
        page: Any,
        context: Any,
        target_url: str,
        stage: str,
    ) -> None:
        # First pass: handle login/captcha interception.
        login_handled = await self._handle_login_if_needed_async(page, context, stage)
        if login_handled:
            recovered_url, recovered_title, recovered_body = await self._read_page_snapshot_async(page)
            recovered_reason = self._detect_non_product_page(
                recovered_url,
                recovered_title,
                recovered_body,
            )
            if recovered_reason or not is_search_result_url(recovered_url):
                await page.goto(target_url, wait_until="domcontentloaded", timeout=120_000)
                await page.wait_for_timeout(max(3000, self.manual_wait_seconds * 1000))

        current_url, _, _ = await self._read_page_snapshot_async(page)
        if is_item_detail_url(current_url):
            LOG.warning(
                "Search stage landed on item detail page (%s); navigating back to search",
                current_url,
            )
            await page.goto(target_url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(3000, self.manual_wait_seconds * 1000))
            current_url, _, _ = await self._read_page_snapshot_async(page)

        if not is_search_result_url(current_url):
            LOG.warning(
                "Unexpected search page URL (%s), forcing navigation to target search URL",
                current_url,
            )
            await page.goto(target_url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(3000, self.manual_wait_seconds * 1000))

        # Final pass: fail fast if still blocked or still not on search page.
        await self._handle_login_if_needed_async(page, context, f"{stage}-post-nav")
        final_url, _, _ = await self._read_page_snapshot_async(page)
        if is_item_detail_url(final_url):
            raise RuntimeError(
                "search phase is still on item detail page after recovery; "
                "please keep the browser on search results while crawling"
            )
        if not is_search_result_url(final_url):
            raise RuntimeError(
                f"search phase failed to stay on Taobao search page (current: {final_url})"
            )

    async def _search_with_global_browser_async(
        self,
        keyword: str,
        top_n: int,
        search_url: str | None = None,
        search_sort: str = "page",
        official_only: bool = False,
    ) -> tuple[list[UrlRecord], str]:
        """Search using GlobalBrowserManager - doesn't create separate Playwright instance"""
        # Ensure global browser manager is initialized
        if self._global_browser_manager is None:
            self._global_browser_manager = await get_global_browser_manager()

        target_url = (
            search_url or f"https://s.taobao.com/search?q={quote_plus(keyword)}"
        )
        fallback_target_url = ""
        if not search_url and "s.taobao.com/search" in target_url:
            fallback_target_url = (
                f"https://list.tmall.com/search_product.htm?q={quote_plus(keyword)}"
            )
        candidate_urls: list[str] = []
        card_records: list[dict[str, Any]] = []
        resource_urls: list[str] = []
        response_snippets: list[str] = []
        response_tasks: set[asyncio.Task[Any]] = set()
        page_text = ""
        page_html = ""

        # Get browser context from global manager
        context = self._global_browser_manager.browser_context
        if context is None:
            # Initialize if needed
            context = await self._global_browser_manager.initialize(
                browser_mode=self.browser_mode,
                headless=self.headless,
                cdp_url=self.cdp_url,
                user_data_dir=str(self.user_data_dir),
            )
        await self._restore_storage_state_async(context)

        page = None
        owns_page = False
        try:
            # Under raw CDP, prefer an isolated tab per search run to avoid cross-task interference.
            if getattr(self._global_browser_manager, "current_mode", "") == "cdp":
                page = await self._global_browser_manager.new_page()
                owns_page = True
            else:
                page = await self._global_browser_manager.get_page()

            # Setup response capture
            async def _capture_response(resp: Any) -> None:
                url_lower = (resp.url or "").lower()
                if "taobao.com" not in url_lower and "tmall.com" not in url_lower:
                    return
                if not any(
                    k in url_lower for k in ("h5api", "search", "item", "recommend")
                ):
                    return
                try:
                    text = await resp.text()
                except Exception:
                    return
                if text:
                    response_snippets.append(text[:120000])

            def _on_response(resp: Any) -> None:
                task = asyncio.create_task(_capture_response(resp))
                response_tasks.add(task)
                task.add_done_callback(lambda done: response_tasks.discard(done))

            page.on("response", _on_response)

            # Navigate to search page. In some environments s.taobao.com can time out
            # while Tmall search is still reachable, so keep a single fallback.
            navigation_errors: list[str] = []
            for candidate_target_url in [target_url, fallback_target_url]:
                if not candidate_target_url:
                    continue
                try:
                    await page.goto(
                        candidate_target_url,
                        wait_until="domcontentloaded",
                        timeout=120_000,
                    )
                    if candidate_target_url != target_url:
                        LOG.warning(
                            "Primary Taobao search URL timed out or failed; fallback to %s",
                            candidate_target_url,
                        )
                        target_url = candidate_target_url
                    break
                except Exception as exc:
                    navigation_errors.append(
                        f"{candidate_target_url} -> {type(exc).__name__}: {exc}"
                    )
            else:
                raise RuntimeError(
                    "failed to open any Taobao search surface: "
                    + " | ".join(navigation_errors)
                )
            await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))

            await self._ensure_search_surface_async(
                page=page,
                context=context,
                target_url=target_url,
                stage="search-initial-global",
            )

            # Scroll to load more results
            scroll_rounds = max(4, 4 + (self.manual_wait_seconds // 3))
            for _ in range(scroll_rounds):
                await page.mouse.wheel(0, 2400)
                await page.wait_for_timeout(1200)

            await self._ensure_search_surface_async(
                page=page,
                context=context,
                target_url=target_url,
                stage="search-post-scroll-global",
            )

            # Extract card data
            try:
                card_records = await page.evaluate(r"""
                    () => {
                      const nodes = Array.from(document.querySelectorAll('a[id^="item_id_"], a[href*="item.htm?id="]'));
                      const results = [];
                      const seen = new Set();
                      for (const node of nodes) {
                        const href = (node.href || node.getAttribute('href') || '').trim();
                        if (!href || !/item\\.htm\\?/i.test(href)) continue;
                        const match = href.match(/[?&]id=(\\d{6,})/i);
                        if (!match) continue;
                        const itemId = match[1];
                        if (seen.has(itemId)) continue;
                        seen.add(itemId);
                        const card = node.closest('[id^="item_id_"]') || node;
                        const cardText = ((card && card.innerText) || '').replace(/\\s+/g, ' ').trim();
                        let titleNode = null;
                        if (card) {
                          titleNode = card.querySelector('[class*="title"], [class*="Title"], h3, h4');
                        }
                        const title = ((titleNode && titleNode.textContent) || node.textContent || '').replace(/\\s+/g, ' ').trim();
                        let shopNode = null;
                        if (card) {
                          shopNode = card.querySelector('[class*="shopName"], [class*="shop"], [class*="store"]');
                        }
                        const shopName = ((shopNode && shopNode.textContent) || '').replace(/\\s+/g, ' ').trim();
                        const salesMatch = cardText.match(/\d+(?:\.\d+)?\s*万?\+?\s*(?:人付款|已售)/);
                        results.push({
                          href,
                          title,
                          shop_name: shopName,
                          sales_text: salesMatch ? salesMatch[0] : '',
                          card_text: cardText
                        });
                        if (results.length >= 260) break;
                      }
                      return results;
                    }
                """)
            except Exception:
                card_records = []

            # Extract URLs
            hrefs = await page.eval_on_selector_all("a[href]", "nodes => nodes.map(n => n.href)")
            candidate_urls.extend([str(v) for v in hrefs if v])

            try:
                resource_urls_raw = await page.evaluate(
                    "() => performance.getEntriesByType('resource').map(e => e.name)"
                )
                resource_urls = [str(v) for v in resource_urls_raw if isinstance(v, str)]
            except Exception:
                resource_urls = []
            candidate_urls.extend(resource_urls)

            page_html = await page.content()
            page_text = await page.inner_text("body")
            candidate_urls.extend(
                extract_candidate_item_urls(page_html, limit=top_n * 50)
            )
            candidate_urls.extend(
                extract_candidate_item_urls(page_text, limit=top_n * 50)
            )

            await page.wait_for_timeout(1000)
            if response_tasks:
                await asyncio.gather(*list(response_tasks), return_exceptions=True)
            for snippet in response_snippets:
                candidate_urls.extend(
                    extract_candidate_item_urls(snippet, limit=top_n * 50)
                )
            await self._persist_storage_state_async(context)

        finally:
            if owns_page and page is not None:
                try:
                    await page.close()
                except Exception:
                    pass

        # Process records
        records = SearchClient._records_from_cards(
            card_records,
            top_n=top_n,
            search_sort=search_sort,
            official_only=official_only,
        )
        if not records:
            records = SearchClient._records_from_candidate_urls(candidate_urls, top_n)
        anti_bot_signal = detect_antibot_signal(
            page_text,
            page_html,
            "\n".join(resource_urls[:100]),
            "\n".join(response_snippets[:20]),
        )
        return records, anti_bot_signal

    def search_top_items(
        self,
        keyword: str,
        top_n: int,
        search_url: str | None = None,
        search_sort: str = "page",
        official_only: bool = False,
    ) -> list[UrlRecord]:
        """Single-path search via unified browser backend."""
        try:
            records, anti_bot_signal = asyncio.run(
                self._search_with_global_browser_async(
                    keyword=keyword,
                    top_n=top_n,
                    search_url=search_url,
                    search_sort=search_sort,
                    official_only=official_only,
                )
            )
            if records:
                return records
            if anti_bot_signal:
                raise RuntimeError(f"{SEARCH_BLOCK_HINT} (detected: {anti_bot_signal})")
            raise RuntimeError(
                "no Taobao item URL found from search; "
                "ensure you are logged in and the search page loads correctly"
            )
        except Exception as exc:
            raise RuntimeError(f"Search failed: {exc}")


class Crawler:
    DEFAULT_STORAGE_STATE_FILE = "taobao_storage_state.json"
    DEFAULT_USER_DATA_DIR_NAME = "taobao_insight_profile"
    PLATFORM_LABEL = "Taobao"
    LOGIN_HANDLER_CLS = TaobaoLogin
    DETECT_NON_PRODUCT_PAGE_FN = staticmethod(detect_non_product_page)

    def __init__(
        self,
        storage: Storage,
        headless: bool,
        browser_mode: str = "cdp",
        cdp_url: str = "",
        manual_wait_seconds: int = 0,
        storage_state_file: str | Path | None = None,
        user_data_dir: str | Path | None = None,
        manual_login_timeout_sec: int = 300,
    ) -> None:
        self.storage = storage
        self.headless = headless
        self.browser_mode = (browser_mode or "cdp").strip().lower()
        if self.browser_mode not in {"cdp", "persistent"}:
            self.browser_mode = "cdp"
        self.cdp_url = cdp_url.strip()
        self.manual_wait_seconds = max(0, int(manual_wait_seconds))
        self.storage_state_file = (
            Path(storage_state_file).resolve()
            if storage_state_file
            else Path("data") / self.DEFAULT_STORAGE_STATE_FILE
        )
        self.storage_state_file = self.storage_state_file.resolve()
        self.user_data_dir = (
            Path(user_data_dir).resolve()
            if user_data_dir
            else _default_user_data_dir(self.DEFAULT_USER_DATA_DIR_NAME).resolve()
        )
        self.manual_login_timeout_sec = max(30, int(manual_login_timeout_sec))
        self._global_browser_manager: Any = None
        # Structured login recovery events for diagnostics/output.
        self.login_recovery_events: list[dict[str, Any]] = []
        self._restored_storage_state_context_ids: set[int] = set()

    def _effective_browser_mode(self) -> str:
        if self._global_browser_manager is not None:
            mode = getattr(self._global_browser_manager, "current_mode", "")
            if mode:
                return str(mode)
        return self.browser_mode

    def _detect_non_product_page(self, current_url: str, title: str, body_text: str) -> str:
        detector = getattr(self, "DETECT_NON_PRODUCT_PAGE_FN", detect_non_product_page)
        return str(detector(current_url, title, body_text) or "")

    @staticmethod
    def _price_from_text(raw: str) -> float | None:
        text = (raw or "").strip()
        if not text:
            return None
        try:
            value = float(text)
        except Exception:
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _is_promotional_sku_name(sku_name: str) -> bool:
        text = str(sku_name or "").strip()
        if not text:
            return False
        promo_tokens = (
            "\u9886\u5238",
            "\u7acb\u51cf",
            "\u4f18\u60e0",
            "\u6743\u76ca",
            "\u8054\u7cfb\u5ba2\u670d",
            "\u8d60\u54c1",
            "\u5165\u4f1a",
            "\u54a8\u8be2",
            "\u6d3b\u52a8",
        )
        return any(token in text for token in promo_tokens)

    @staticmethod
    def _refine_price_values(values: list[float]) -> list[float]:
        normalized = sorted(
            {
                float(v)
                for v in values
                if isinstance(v, (int, float)) and 0 < float(v) < 100000
            }
        )
        if not normalized:
            return []
        if len(normalized) >= 2:
            max_value = normalized[-1]
            filtered_micro = [
                v for v in normalized if not (v < 1 and max_value >= 5 and v <= max_value * 0.05)
            ]
            if filtered_micro:
                normalized = filtered_micro
        if len(normalized) >= 2:
            median_value = float(statistics.median(normalized))
            if median_value >= 30:
                filtered = [v for v in normalized if v >= median_value * 0.25]
                if filtered:
                    normalized = filtered
        return normalized

    @classmethod
    def _prices_from_skus(cls, skus: list[dict[str, str]]) -> list[float]:
        regular: list[float] = []
        promo: list[float] = []
        for row in skus:
            value = cls._price_from_text(str(row.get("price", "")))
            if value is None:
                continue
            sku_name = str(row.get("sku_name", ""))
            if cls._is_promotional_sku_name(sku_name):
                promo.append(value)
            else:
                regular.append(value)
        preferred = regular or promo
        return cls._refine_price_values(preferred)

    @staticmethod
    def _normalize_image_url(url: str) -> str:
        value = html_unescape((url or "").strip())
        if not value:
            return ""
        url_match = re.search(r"url\((?P<u>[^)]+)\)", value, flags=re.IGNORECASE)
        if url_match:
            value = str(url_match.group("u") or "").strip()
        value = value.strip("\"'")
        value = re.split(r"[\"'\s);]+", value, maxsplit=1)[0].strip()
        value = value.replace("\\/", "/")
        if value.startswith("//"):
            value = "https:" + value
        if value and not value.lower().startswith(("http://", "https://")):
            return ""
        return value

    @staticmethod
    def _is_noise_image(url: str, alt: str = "", cls: str = "") -> bool:
        text = f"{url} {alt} {cls}".lower()
        bad_tokens = (
            "logo",
            "tmall",
            "taobao",
            "icon",
            "avatar",
            "service",
            "aliyun",
            "sprite",
            "shop-head",
            "alihealth",
        )
        return any(token in text for token in bad_tokens)

    @staticmethod
    def _is_non_product_image_context(text: str) -> bool:
        value = clean_text(str(text or ""), max_len=240).lower()
        if not value:
            return False
        bad_tokens = (
            "88vip",
            "会员",
            "礼券",
            "礼遇",
            "优惠",
            "券后",
            "顺丰",
            "速达",
            "退货",
            "无理由",
            "运费险",
            "售后",
            "物流",
            "直供",
            "原装正品",
            "假一赔",
            "服务保障",
        )
        return any(token in value for token in bad_tokens)

    @staticmethod
    def _is_noise_detail_text(text: str) -> bool:
        raw = str(text or "").replace("\r", " ").replace("\n", " ").strip()
        if not raw:
            return True
        value = re.sub(r"\s+", " ", raw)[:240]
        lowered = value.lower()
        if any(token in lowered for token in ("tmall.com", "taobao.com", "http://", "https://")):
            return True
        if "�" in value:
            return True
        hard_tokens = (
            "我的淘宝",
            "购物车",
            "收藏夹",
            "免费开店",
            "千牛卖家中心",
            "帮助中心",
            "联系客服",
            "网站导航",
            "登录",
            "注册",
            "language",
            "无障碍",
            "亲，请登录",
            "天猫超市",
            "淘宝网首页",
        )
        if any(token in value for token in hard_tokens):
            return True
        nav_tokens = (
            "首页",
            "店铺",
            "客服",
            "收藏",
            "优惠券",
            "会员",
            "活动",
            "规则",
            "物流",
            "售后",
            "退货",
        )
        nav_hit_count = sum(1 for token in nav_tokens if token in value)
        if nav_hit_count >= 3:
            return True
        if len(value) <= 16 and nav_hit_count >= 2:
            return True
        return False

    @classmethod
    def _is_relevant_detail_dom_image_row(cls, row: dict[str, Any]) -> bool:
        src = cls._normalize_image_url(str(row.get("src", "") or row.get("content", "")))
        if not src:
            return False
        alt = str(row.get("alt", "") or "")
        css = str(row.get("class_name", "") or "")
        context_text = str(row.get("context_text", "") or "")
        lower = src.lower()
        width = int(float(row.get("width", 0) or 0))
        height = int(float(row.get("height", 0) or 0))
        if cls._is_noise_image(src, alt, css):
            return False
        if lower.endswith("/s.gif") or lower.endswith("s.gif"):
            return False
        if any(token in lower for token in ("service", "refund", "coupon", "vip")):
            return False
        if width and height and min(width, height) < 180:
            return False
        if width and height and width / max(height, 1) >= 3.2:
            return False
        if cls._is_non_product_image_context(" ".join([alt, css, context_text])):
            return False
        return True

    @staticmethod
    def _is_tail_banner_image_row(row: dict[str, Any]) -> bool:
        width = int(float(row.get("width", 0) or 0))
        height = int(float(row.get("height", 0) or 0))
        if not width or not height:
            return False
        return width >= 600 and (width / max(height, 1) >= 3.2 or height <= 260)

    @classmethod
    def _is_tail_noise_image_row(cls, row: dict[str, Any]) -> bool:
        src = cls._normalize_image_url(str(row.get("src", "") or row.get("content", "")))
        alt = str(row.get("alt", "") or "")
        css = str(row.get("class_name", "") or "")
        context_text = str(row.get("context_text", "") or "")
        lower = src.lower()
        width = int(float(row.get("width", 0) or 0))
        height = int(float(row.get("height", 0) or 0))
        ratio = width / max(height, 1) if width and height else 0.0
        if cls._is_non_product_image_context(" ".join([alt, css, context_text])):
            return True
        if any(
            token in lower
            for token in (
                "service",
                "refund",
                "vip",
                "wuying",
                "barrier",
                "accessible",
                "green",
                "carbon",
            )
        ):
            return True
        if width and height and min(width, height) < 180:
            return True
        if width >= 240 and height <= 320 and ratio >= 2.0:
            return True
        return False

    @classmethod
    def _trim_non_product_tail_image_rows(
        cls, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(rows) < 4:
            return rows
        trim_index: int | None = None
        start = max(0, len(rows) - 8)
        for idx in range(start, len(rows)):
            suffix = rows[idx:]
            banner_count = sum(1 for row in suffix if cls._is_tail_banner_image_row(row))
            if len(suffix) >= 4 and banner_count >= max(3, len(suffix) - 1):
                trim_index = idx
                break
        if trim_index is None:
            trimmed_rows = rows
        else:
            if trim_index > 0 and len(rows) - trim_index >= 4:
                prev = rows[trim_index - 1]
                prev_width = int(float(prev.get("width", 0) or 0))
                prev_height = int(float(prev.get("height", 0) or 0))
                prev_ratio = prev_width / max(prev_height, 1) if prev_width and prev_height else 0.0
                if (
                    prev_width >= 600
                    and 600 <= prev_height <= 1000
                    and 0.7 <= prev_ratio <= 1.4
                ):
                    trim_index -= 1
            trimmed_rows = rows[:trim_index]
        end = len(trimmed_rows)
        trimmed_tail = 0
        while end > 0 and cls._is_tail_noise_image_row(trimmed_rows[end - 1]):
            end -= 1
            trimmed_tail += 1
        if trimmed_tail >= 2:
            return trimmed_rows[:end]
        return trimmed_rows

    @classmethod
    def _pick_main_image(cls, images: list[dict[str, Any]]) -> str:
        ranked: list[tuple[int, str]] = []
        for image in images:
            src = cls._normalize_image_url(str(image.get("src", "")))
            if not src:
                continue
            alt = str(image.get("alt", ""))
            css = str(image.get("class_name", ""))
            width = int(float(image.get("width", 0) or 0))
            height = int(float(image.get("height", 0) or 0))
            if cls._is_noise_image(src, alt, css):
                continue
            lower = src.lower()
            if not any(
                token in lower
                for token in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    "alicdn.com",
                    "tbcdn.cn",
                )
            ):
                continue
            score = 0
            if width >= 600 and height >= 600:
                score += 6
            elif width >= 360 and height >= 360:
                score += 4
            elif width >= 220 and height >= 220:
                score += 2
            if "imgextra" in lower or "bao/uploaded" in lower:
                score += 4
            if "main" in lower or "primary" in lower:
                score += 2
            ranked.append((score, src))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1] if ranked else ""

    @classmethod
    def _score_candidate_main_image(
        cls, candidate: str, origin: str, position: int
    ) -> int:
        value = cls._normalize_image_url(candidate)
        if not value:
            return -999
        lower = value.lower()
        score = 0
        if origin == "payload":
            score += 3
        if value.startswith("http://") or value.startswith("https://"):
            score += 8
        else:
            score += 2
        if any(
            token in lower
            for token in (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                "img.alicdn",
                "alicdn.com",
                "tbcdn",
            )
        ):
            score += 4
        if "detail_full" in lower and lower.endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):
            score += 3
        if "imgextra" in lower or "bao/uploaded" in lower:
            score += 3
        if cls._is_noise_image(value):
            score -= 8
        if any(
            token in lower
            for token in (
                "avatar",
                "logo",
                "icon",
                "service",
                "shop-head",
                "seller",
                "_50x50",
                "_80x80",
            )
        ):
            score -= 8
        score -= min(position, 10)
        return score

    @staticmethod
    def _extract_shop_name(html: str, body_text: str) -> str:
        match = SHOP_NAME_RE.search(html)
        if match:
            return clean_text(match.group("name"), max_len=120)
        for line in LINE_BREAK_RE.split(body_text):
            snippet = clean_text(line, max_len=120)
            if not snippet:
                continue
            if (
                "\u65d7\u8230\u5e97" in snippet
                or "\u5b98\u65b9\u5e97" in snippet
                or "\u4e13\u5356\u5e97" in snippet
            ):
                return snippet
        return ""

    @staticmethod
    def _extract_brand(title: str, html: str) -> str:
        match = BRAND_RE.search(html)
        if match:
            return clean_text(match.group("brand"), max_len=80)
        text = clean_text(title, max_len=120)
        if not text:
            return ""
        token = NON_WORD_RE.split(text)[0]
        return clean_text(token, max_len=40)

    @staticmethod
    def _extract_prices(text: str, html: str) -> list[float]:
        prices: list[float] = []
        for token in re.findall(r"[¥￥]\s*(\d{1,6}(?:\.\d{1,2})?)", text):
            try:
                value = float(token)
            except ValueError:
                continue
            if 0 < value < 100000:
                prices.append(value)

        for token in re.findall(
            r'"(?:price|promotionPrice|reservePrice|skuPrice|salePrice)"\s*:\s*"?(?P<v>\d{1,6}(?:\.\d{1,2})?)"?',
            html,
            flags=re.IGNORECASE,
        ):
            try:
                value = float(token)
            except ValueError:
                continue
            if 0 < value < 100000:
                prices.append(value)

        if not prices:
            return []
        prices = sorted(set(prices))
        low_band = [value for value in prices if value <= 5000]
        return low_band if low_band else prices

    @staticmethod
    def _extract_detail_blocks(text: str, images: list[str]) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []
        noise_markers = (
            "\u6dd8\u5b9d",
            "\u5929\u732b",
            "\u9996\u9875",
            "\u767b\u5f55",
            "\u8d2d\u7269\u8f66",
            "\u5ba2\u670d",
            "\u4e3e\u62a5",
            "\u6536\u85cf",
            "\u5206\u4eab",
            "\u5e97\u94fa",
            "\u65fa\u65fa",
        )
        seen_text: set[str] = set()
        chunks = [clean_text(raw, max_len=180) for raw in LINE_BREAK_RE.split(text)]
        chunks = [chunk for chunk in chunks if chunk]
        if len(chunks) <= 2:
            for fragment in re.split(r"[\u3002\uff1b;!\uff01?\uff1f]", text):
                candidate = clean_text(fragment, max_len=160)
                if candidate:
                    chunks.append(candidate)
        text_index = 1
        for line in chunks:
            if len(line) < 8:
                continue
            if sum(1 for token in noise_markers if token in line) >= 2:
                continue
            if line in seen_text:
                continue
            seen_text.add(line)
            blocks.append(
                {
                    "source_type": "text",
                    "source_ref": f"text_{text_index}",
                    "content": line,
                }
            )
            text_index += 1
            if text_index > 40:
                break
        if text_index == 1:
            fallback = clean_text(text, max_len=160)
            if fallback:
                blocks.append(
                    {"source_type": "text", "source_ref": "text_1", "content": fallback}
                )
                text_index = 2
        seen_img: set[str] = set()
        image_index = 1
        for url in images:
            normalized = Crawler._normalize_image_url(url)
            if not normalized or normalized in seen_img:
                continue
            seen_img.add(normalized)
            blocks.append(
                {
                    "source_type": "image",
                    "source_ref": f"image_{image_index}",
                    "content": normalized,
                }
            )
            image_index += 1
            if image_index > 18:
                break
        return blocks

    def _build_item_detail_from_raw(
        self,
        item_id: str,
        crawl_time: str,
        title: str,
        html: str,
        body_text: str,
        image_meta: list[dict[str, Any]],
        image_urls: list[str],
        detail_blocks_override: list[dict[str, str]] | None = None,
        visit_chain: list[dict[str, str]] | None = None,
    ) -> ItemDetail:
        image_urls = [str(u) for u in image_urls if isinstance(u, str)]
        main_image_url = self._pick_main_image(image_meta)
        if not main_image_url:
            candidates = [self._normalize_image_url(url) for url in image_urls]
            main_image_url = next(
                (url for url in candidates if url and not self._is_noise_image(url)), ""
            )

        text = clean_text(body_text, max_len=36000)
        prices = self._extract_prices(text, html)
        if not prices:
            prices = parse_price_values(text)
        prices = self._refine_price_values(prices)
        skus: list[dict[str, str]] = []
        seen_sku: set[str] = set()
        for match in SKU_MAP_RE.finditer(html):
            sku_id = match.group("sku_id")
            if sku_id in seen_sku:
                continue
            seen_sku.add(sku_id)
            skus.append(
                {"sku_id": sku_id, "sku_name": "SKU", "price": match.group("price")}
            )
            if len(skus) >= 30:
                break
        if not skus and prices:
            skus.append(
                {
                    "sku_id": f"{item_id}001",
                    "sku_name": "default-sku",
                    "price": f"{prices[0]:.2f}",
                }
            )
        sku_prices = self._prices_from_skus(skus)
        if sku_prices:
            prices = self._refine_price_values(prices + sku_prices) or sku_prices

        detail_images: list[str] = []
        for url_candidate in image_urls[:200]:
            normalized = self._normalize_image_url(url_candidate)
            if not normalized or self._is_noise_image(normalized):
                continue
            detail_images.append(normalized)
        shop_name = self._extract_shop_name(html, text)
        brand = normalize_brand_name(
            self._extract_brand(title, html), title=title, shop_name=shop_name
        )
        detail_text_source = "\n".join(filter(None, [title, shop_name, brand, text]))
        detail_blocks = detail_blocks_override or self._extract_detail_blocks(
            detail_text_source, detail_images
        )
        citations = [
            block["source_ref"] for block in detail_blocks if block.get("source_ref")
        ]
        detail_text = "\n".join(
            [
                block["content"]
                for block in detail_blocks
                if block.get("source_type") == "text"
            ]
        )
        if not detail_text:
            detail_text = text
        detail_summary_parts: list[str] = []
        for block in detail_blocks:
            source_ref = str(block.get("source_ref", "")).strip()
            content = clean_text(str(block.get("content", "")), max_len=120)
            if not source_ref or not content:
                continue
            detail_summary_parts.append(f"{content}[{source_ref}]")
            if len(detail_summary_parts) >= 6:
                break
        detail_summary = "; ".join(detail_summary_parts)

        return ItemDetail(
            item_id=item_id,
            title=normalize_item_title(title, max_len=120),
            main_image_url=main_image_url or "",
            shop_name=clean_text(shop_name, max_len=120),
            brand=normalize_brand_name(brand, title=title, shop_name=shop_name),
            prices=prices,
            skus=skus,
            detail_summary=detail_summary,
            detail_text=detail_text,
            detail_blocks=detail_blocks,
            citations=citations,
            crawl_time=crawl_time,
            visit_chain=list(visit_chain or []),
            error="",
        )

    async def _persist_storage_state_async(self, context: Any) -> None:
        self.storage_state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            await context.storage_state(path=str(self.storage_state_file))
        except Exception:
            return

    async def _restore_storage_state_async(self, context: Any) -> None:
        context_key = id(context)
        if context_key in self._restored_storage_state_context_ids:
            return
        normalized = _read_storage_state_cookies(self.storage_state_file)
        if not normalized:
            return
        try:
            await context.add_cookies(normalized)
            self._restored_storage_state_context_ids.add(context_key)
            LOG.info(
                "Restored %s cookies from storage state %s",
                len(normalized),
                self.storage_state_file,
            )
        except Exception as exc:
            LOG.warning("Failed to restore cookies from %s: %s", self.storage_state_file, exc)

    async def _read_page_snapshot_async(self, page: Any) -> tuple[str, str, str]:
        current_url = page.url or ""
        try:
            title = await page.title()
        except Exception:
            title = ""
        try:
            body_text = await page.inner_text("body")
        except Exception:
            body_text = ""
        return current_url, title, body_text

    async def _handle_login_if_needed_async(
        self,
        page: Any,
        context: Any,
        *,
        stage: str,
        item_id: str,
    ) -> bool:
        current_url, title, body_text = await self._read_page_snapshot_async(page)
        reason = self._detect_non_product_page(current_url, title, body_text)
        if not reason:
            return False

        LOG.warning(
            "[%s] Detected blocked page during %s for item %s: %s",
            self.PLATFORM_LABEL,
            stage,
            item_id,
            reason,
        )
        login_handler = self.LOGIN_HANDLER_CLS(
            browser_context=context,
            context_page=page,
            login_timeout_sec=self.manual_login_timeout_sec,
        )
        login_result: LoginHandleResult = await login_handler.check_and_handle_login()
        self.login_recovery_events.append(
            _build_login_recovery_event(
                source="crawl",
                stage=stage,
                blocked_reason=reason,
                current_url=current_url,
                login_result=login_result,
                browser_mode=self._effective_browser_mode(),
                storage_state_file=self.storage_state_file,
                user_data_dir=self.user_data_dir,
                item_id=item_id,
            )
        )
        if not login_result.ok:
            raise RuntimeError(
                f"{self.PLATFORM_LABEL} login timeout in {stage} for item {item_id}: "
                f"{reason} ({login_result.reason})"
            )

        await self._persist_storage_state_async(context)
        return True

    async def _try_open_detail_tab(self, page: Any) -> None:
        selectors = [
            "text=\u56fe\u6587\u8be6\u60c5",
            "a:has-text('\u56fe\u6587\u8be6\u60c5')",
            "li:has-text('\u56fe\u6587\u8be6\u60c5')",
            "[role='tab']:has-text('\u56fe\u6587\u8be6\u60c5')",
            "text=\u5546\u54c1\u8be6\u60c5",
            "a:has-text('\u5546\u54c1\u8be6\u60c5')",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                await locator.click(timeout=1500)
                await page.wait_for_timeout(800)
                return
            except Exception:
                continue

    async def _scroll_detail_section(self, page: Any, rounds: int = 6) -> None:
        for _ in range(max(2, rounds)):
            try:
                await page.mouse.wheel(0, 1800)
                await page.wait_for_timeout(700)
            except Exception:
                return

    async def _extract_detail_payload_from_target(
        self, target: Any
    ) -> dict[str, Any]:
        script = """
() => {
  const selectors = [
    '#description',
    '#J_DivItemDesc',
    '.descV8-container',
    '.descV8-content',
    '.desc-content',
    '.detail-content',
    '.tb-detail-bd',
    '[id*="desc"]',
    '[class*="desc"]'
  ];
  let root = null;
  for (const sel of selectors) {
    const found = document.querySelector(sel);
    if (found) { root = found; break; }
  }
  if (!root) root = document.body;

  const texts = [];
  const textSeen = new Set();
  for (const node of root.querySelectorAll('h1,h2,h3,h4,p,li,span,div,strong,b')) {
    const raw = String(node.innerText || node.textContent || '').replace(/\\s+/g, ' ').trim();
    if (!raw) continue;
    if (raw.length < 6 || raw.length > 140) continue;
    if (!/[\\u4e00-\\u9fff]/.test(raw)) continue;
    if (textSeen.has(raw)) continue;
    textSeen.add(raw);
    texts.push(raw);
    if (texts.length >= 40) break;
  }

  const readAttr = (node, attrs) => {
    for (const attr of attrs) {
      const v = (node.getAttribute && node.getAttribute(attr)) || '';
      if (v) return String(v).trim();
    }
    return '';
  };
  const images = [];
  const imageSeen = new Set();
  const imageRows = [];
  for (const img of root.querySelectorAll('img')) {
    let src = readAttr(img, ['src', 'data-src', 'data-ks-lazyload', 'data-lazy-src', 'data-original']) || String(img.currentSrc || '').trim();
    if (!src) continue;
    if (src.startsWith('//')) src = 'https:' + src;
    if (!/^https?:\\/\\//i.test(src)) continue;
    if (imageSeen.has(src)) continue;
    imageSeen.add(src);
    const rect = img.getBoundingClientRect();
    let contextText = '';
    let parent = img;
    for (let depth = 0; depth < 4 && parent; depth += 1) {
      parent = parent.parentElement;
      if (!parent) break;
      const raw = String(parent.innerText || parent.textContent || '').replace(/\\s+/g, ' ').trim();
      if (raw && raw.length >= 6) {
        contextText = raw.slice(0, 200);
        break;
      }
    }
    images.push(src);
    imageRows.push({
      src,
      alt: String(img.alt || '').trim(),
      class_name: String(img.className || '').trim(),
      width: Number(img.naturalWidth || img.width || rect.width || 0),
      height: Number(img.naturalHeight || img.height || rect.height || 0),
      context_text: contextText
    });
    if (images.length >= 40) break;
  }
  return { texts, images, image_rows: imageRows };
}
"""
        try:
            payload = await target.evaluate(script)
        except Exception:
            return {"texts": [], "images": [], "image_rows": []}
        if not isinstance(payload, dict):
            return {"texts": [], "images": [], "image_rows": []}
        texts = [str(v) for v in payload.get("texts", []) if isinstance(v, str)]
        images = [str(v) for v in payload.get("images", []) if isinstance(v, str)]
        image_rows = [v for v in payload.get("image_rows", []) if isinstance(v, dict)]
        return {"texts": texts, "images": images, "image_rows": image_rows}

    async def _collect_detail_blocks_from_page(
        self, page: Any, item_id: str, image_dir: Path
    ) -> list[dict[str, str]]:
        try:
            await self._try_open_detail_tab(page)
            await self._scroll_detail_section(page, rounds=8)
        except Exception:
            pass

        screenshot_path = ""
        try:
            image_dir.mkdir(parents=True, exist_ok=True)
            detail_path = image_dir / "detail_full.png"
            await page.screenshot(path=str(detail_path), full_page=True)
            if detail_path.exists() and detail_path.stat().st_size > 0:
                screenshot_path = str(detail_path.resolve())
        except Exception:
            screenshot_path = ""

        texts: list[str] = []
        images: list[str] = []
        image_rows: list[dict[str, Any]] = []
        payload = await self._extract_detail_payload_from_target(page)
        texts.extend(payload.get("texts", []))
        images.extend(payload.get("images", []))
        image_rows.extend(payload.get("image_rows", []))

        for frame in page.frames:
            if frame == page.main_frame:
                continue
            frame_url = str(getattr(frame, "url", "") or "").lower()
            if frame_url and not any(
                token in frame_url
                for token in ("desc", "detail", "img.alicdn", "taobao", "tmall")
            ):
                continue
            frame_payload = await self._extract_detail_payload_from_target(frame)
            texts.extend(frame_payload.get("texts", []))
            images.extend(frame_payload.get("images", []))
            image_rows.extend(frame_payload.get("image_rows", []))

        cleaned_texts: list[str] = []
        seen_text: set[str] = set()
        for raw in texts:
            text = clean_text(str(raw or ""), max_len=140)
            if len(text) < 6:
                continue
            if not re.search(r"[\u4e00-\u9fff]", text):
                continue
            if self._is_noise_detail_text(text):
                continue
            if text in seen_text:
                continue
            seen_text.add(text)
            cleaned_texts.append(text)
            if len(cleaned_texts) >= 20:
                break

        cleaned_images: list[str] = []
        seen_image: set[str] = set()
        candidate_image_rows: list[dict[str, Any]] = image_rows or [
            {"src": raw, "alt": "", "class_name": "", "width": 0, "height": 0, "context_text": ""}
            for raw in images
        ]
        candidate_image_rows = self._trim_non_product_tail_image_rows(candidate_image_rows)
        for row in candidate_image_rows:
            normalized = self._normalize_image_url(str(row.get("src", "") or row.get("content", "")))
            if not normalized or normalized in seen_image:
                continue
            if not self._is_relevant_detail_dom_image_row(row):
                continue
            seen_image.add(normalized)
            cleaned_images.append(normalized)
            if len(cleaned_images) >= 20:
                break

        blocks: list[dict[str, str]] = []
        for idx, text in enumerate(cleaned_texts, start=1):
            blocks.append(
                {"source_type": "text", "source_ref": f"text_{idx}", "content": text}
            )
        image_candidates: list[str] = []
        if screenshot_path:
            image_candidates.append(screenshot_path)
        image_candidates.extend(cleaned_images)
        for idx, image_url in enumerate(image_candidates, start=1):
            blocks.append(
                {
                    "source_type": "image",
                    "source_ref": f"image_{idx}",
                    "content": image_url,
                }
            )
        return blocks

    def crawl(self, workbook_id: str, item_id: str, url: str) -> ItemDetail:
        """Single-path crawl via unified browser backend."""
        # Use global lock to serialize Playwright access across threads
        # This prevents connection corruption when multiple crawlers run concurrently
        with PLAYWRIGHT_LOCK:
            try:
                return asyncio.run(
                    self.crawl_async_global(
                        url=url,
                        item_id=item_id,
                        workbook_id=workbook_id,
                    )
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                LOG.error("crawl failed for item %s: %s", item_id, error)
                return ItemDetail(
                    item_id=item_id,
                    title="",
                    main_image_url="",
                    shop_name="",
                    brand="",
                    prices=[],
                    skus=[],
                    detail_summary="",
                    detail_text="",
                    detail_blocks=[],
                    citations=[],
                    crawl_time=now_iso(),
                    error=error,
                )

    async def crawl_async_global(
        self, url: str, item_id: str, workbook_id: str
    ) -> ItemDetail:
        """Async crawl using the shared browser manager in a single event loop."""

        image_dir = self.storage.images_dir / workbook_id / item_id
        image_dir.mkdir(parents=True, exist_ok=True)
        crawl_time = now_iso()

        html = ""
        body_text = ""
        title = ""
        image_urls: list[str] = []
        image_meta: list[dict[str, Any]] = []
        current_url = ""
        detail_blocks: list[dict[str, str]] = []
        visit_chain: list[dict[str, str]] = []

        def _is_retryable_navigation_error(message: str) -> bool:
            text = (message or "").lower()
            return any(
                token in text
                for token in (
                    "net::err_aborted",
                    "net::err_connection_reset",
                    "navigation interrupted",
                    "timeout",
                    "target page, context or browser has been closed",
                )
            )

        max_nav_attempts = 3
        for nav_attempt in range(1, max_nav_attempts + 1):
            page = None
            try:
                # Get the global browser manager (should already be initialized)
                global_manager = await get_global_browser_manager()
                if not global_manager.is_initialized:
                    await global_manager.initialize(
                        browser_mode=self.browser_mode,
                        headless=self.headless,
                        cdp_url=self.cdp_url,
                        user_data_dir=str(self.user_data_dir),
                    )
                browser_context = global_manager.browser_context
                if browser_context is None:
                    raise RuntimeError("global browser context is unavailable")
                await self._restore_storage_state_async(browser_context)

                # Each crawl should use an isolated page to avoid cross-task navigation interruption.
                page = await global_manager.new_page()

                async def _capture_visit(stage: str, note: str = "") -> None:
                    current_visit_url, current_visit_title, _ = await self._read_page_snapshot_async(page)
                    if not current_visit_url:
                        return
                    event = {"stage": stage, "url": str(current_visit_url)}
                    title_text = clean_text(str(current_visit_title or ""), max_len=120)
                    note_text = clean_text(str(note or ""), max_len=160)
                    if title_text:
                        event["title"] = title_text
                    if note_text:
                        event["note"] = note_text
                    if visit_chain and visit_chain[-1].get("stage") == event["stage"] and visit_chain[-1].get("url") == event["url"]:
                        return
                    visit_chain.append(event)

                # Navigate to the item page
                await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))
                await _capture_visit("item_initial", note="initial goto")

                # Check for non-product page (login required, blocked, etc.)
                login_handled = await self._handle_login_if_needed_async(
                    page=page,
                    context=browser_context,
                    stage="item-initial",
                    item_id=item_id,
                )
                if login_handled:
                    recovered_url, recovered_title, recovered_body = await self._read_page_snapshot_async(page)
                    recovered_reason = self._detect_non_product_page(
                        recovered_url,
                        recovered_title,
                        recovered_body,
                    )
                    recovered_rec = normalize_url(recovered_url)
                    target_rec = normalize_url(url)
                    same_item_surface = bool(
                        recovered_rec
                        and target_rec
                        and recovered_rec.platform == target_rec.platform
                        and recovered_rec.item_id == target_rec.item_id
                    )
                    if not recovered_reason and same_item_surface:
                        await _capture_visit(
                            "item_after_login_recovery",
                            note="reused post-login detail page without reload",
                        )
                    else:
                        await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                        await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))
                        await _capture_visit(
                            "item_after_login_recovery",
                            note="reloaded after login recovery",
                        )
                        retry_url, retry_title, retry_body = await self._read_page_snapshot_async(page)
                        retry_reason = self._detect_non_product_page(
                            retry_url,
                            retry_title,
                            retry_body,
                        )
                        if retry_reason:
                            raise RuntimeError(
                                f"taobao item page blocked after login recovery: {retry_reason}. "
                                f"Please ensure you are logged in and complete QR verification."
                            )

                # Scroll down to load all content
                scroll_rounds = max(4, 4 + (self.manual_wait_seconds // 4))
                for _ in range(scroll_rounds):
                    await page.mouse.wheel(0, 1800)
                    await page.wait_for_timeout(1200)

                # Take full page screenshot
                _full_page_path = str(image_dir / "full_page.png")
                for _shot_attempt in range(2):
                    try:
                        await page.screenshot(path=_full_page_path, full_page=True)
                        break
                    except Exception as _shot_exc:
                        if _shot_attempt == 0:
                            LOG.warning(
                                "full_page screenshot failed for item %s (attempt 1), retrying: %s",
                                item_id,
                                _shot_exc,
                            )
                            await page.wait_for_timeout(1500)
                        else:
                            LOG.warning(
                                "full_page screenshot failed for item %s (attempt 2), skipping screenshot: %s",
                                item_id,
                                _shot_exc,
                            )

                # Collect detail blocks (product description, images, etc.)
                detail_blocks = await self._collect_detail_blocks_from_page(
                    page, item_id=item_id, image_dir=image_dir
                )

                # Extract page content
                current_url = page.url
                html = await page.content()
                body_text = await page.inner_text("body")
                title = await page.title()
                await _capture_visit("item_ready", note="detail page ready for extraction")

                # Extract image metadata
                image_meta = await page.eval_on_selector_all(
                    "img",
                    """
                    nodes => nodes.map(n => {
                      const rect = n.getBoundingClientRect();
                      return {
                        src: (n.currentSrc || n.src || '').trim(),
                        alt: (n.alt || '').trim(),
                        class_name: String(n.className || ''),
                        width: Number(n.naturalWidth || n.width || rect.width || 0),
                        height: Number(n.naturalHeight || n.height || rect.height || 0)
                      };
                    }).filter(v => v.src)
                    """,
                )
                image_urls = [
                    str(v.get("src", "")) for v in image_meta if isinstance(v, dict)
                ]

                # Final check for blocked page
                blocked_reason = detect_non_product_page(current_url, title, body_text)
                if blocked_reason:
                    raise RuntimeError(blocked_reason)

                # Persist storage state (cookies, local storage)
                await self._persist_storage_state_async(browser_context)

                # Build and return ItemDetail from raw data
                return self._build_item_detail_from_raw(
                    item_id=item_id,
                    crawl_time=crawl_time,
                    title=title,
                    html=html,
                    body_text=body_text,
                    image_meta=image_meta,
                    image_urls=image_urls,
                    detail_blocks_override=detail_blocks or None,
                    visit_chain=visit_chain,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                can_retry = nav_attempt < max_nav_attempts and _is_retryable_navigation_error(error)
                if can_retry:
                    LOG.warning(
                        "crawl_async_global retry for item %s (%s/%s): %s",
                        item_id,
                        nav_attempt,
                        max_nav_attempts,
                        error,
                    )
                    await asyncio.sleep(min(1.5 * nav_attempt, 4.0))
                    continue
                LOG.error("crawl_async_global failed for item %s: %s", item_id, error)
                return ItemDetail(
                    item_id=item_id,
                    title="",
                    main_image_url="",
                    shop_name="",
                    brand="",
                    prices=[],
                    skus=[],
                    detail_summary="",
                    detail_text="",
                    detail_blocks=[],
                    citations=[],
                    crawl_time=crawl_time,
                    visit_chain=visit_chain,
                    error=error,
                )
            finally:
                if page is not None and not page.is_closed():
                    try:
                        await page.close()
                    except Exception:
                        pass

        return ItemDetail(
            item_id=item_id,
            title="",
            main_image_url="",
            shop_name="",
            brand="",
            prices=[],
            skus=[],
            detail_summary="",
            detail_text="",
            detail_blocks=[],
            citations=[],
            crawl_time=crawl_time,
            error="RuntimeError: crawl_async_global exhausted retries",
        )
