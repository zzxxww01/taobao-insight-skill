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


def _default_user_data_dir() -> Path:
    appdata = os.getenv("APPDATA", "")
    if appdata:
        return Path(appdata) / "taobao_insight_profile"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "taobao_insight_profile"
    return Path.home() / ".config" / "taobao_insight_profile"


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
            else Path("data/taobao_storage_state.json").resolve()
        )
        self.user_data_dir = (
            Path(user_data_dir).resolve()
            if user_data_dir
            else _default_user_data_dir().resolve()
        )
        self.manual_login_timeout_sec = max(30, int(manual_login_timeout_sec))

        # Global browser manager reference.
        self._global_browser_manager: Any = None
        # Structured login recovery events for diagnostics/output.
        self.login_recovery_events: list[dict[str, Any]] = []

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
        reason = detect_non_product_page(current_url, title, body_text)
        if not reason:
            return False

        LOG.warning("Detected blocked page during %s: %s", stage, reason)
        login_handler = TaobaoLogin(
            browser_context=context,
            context_page=page,
            login_timeout_sec=self.manual_login_timeout_sec,
        )
        login_result: LoginHandleResult = await login_handler.check_and_handle_login()
        self.login_recovery_events.append(
            {
                "source": "search",
                "stage": stage,
                "ok": bool(login_result.ok),
                "blocked_reason": reason,
                "final_state": login_result.final_state,
                "reason": login_result.reason,
                "elapsed_sec": round(float(login_result.elapsed_sec), 3),
                "url": current_url,
                "decision_trace": login_result.decision_trace,
                "updated_at": now_iso(),
            }
        )
        if not login_result.ok:
            raise RuntimeError(
                f"Login timeout in {stage}, still blocked: {reason} ({login_result.reason})"
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

        page = None
        try:
            # Get a page using the global manager (has lock protection)
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

            # Navigate to search page
            await page.goto(target_url, wait_until="domcontentloaded", timeout=120_000)
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
                card_records = await page.evaluate("""
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
            # Don't close page or context - let GlobalBrowserManager manage lifecycle
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
            else Path("data/taobao_storage_state.json").resolve()
        )
        self.user_data_dir = (
            Path(user_data_dir).resolve()
            if user_data_dir
            else _default_user_data_dir().resolve()
        )
        self.manual_login_timeout_sec = max(30, int(manual_login_timeout_sec))
        self._global_browser_manager: Any = None
        # Structured login recovery events for diagnostics/output.
        self.login_recovery_events: list[dict[str, Any]] = []

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
            error="",
        )

    async def _persist_storage_state_async(self, context: Any) -> None:
        self.storage_state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            await context.storage_state(path=str(self.storage_state_file))
        except Exception:
            return

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
        reason = detect_non_product_page(current_url, title, body_text)
        if not reason:
            return False

        LOG.warning("Detected blocked page during %s for item %s: %s", stage, item_id, reason)
        login_handler = TaobaoLogin(
            browser_context=context,
            context_page=page,
            login_timeout_sec=self.manual_login_timeout_sec,
        )
        login_result: LoginHandleResult = await login_handler.check_and_handle_login()
        self.login_recovery_events.append(
            {
                "source": "crawl",
                "stage": stage,
                "item_id": item_id,
                "ok": bool(login_result.ok),
                "blocked_reason": reason,
                "final_state": login_result.final_state,
                "reason": login_result.reason,
                "elapsed_sec": round(float(login_result.elapsed_sec), 3),
                "url": current_url,
                "decision_trace": login_result.decision_trace,
                "updated_at": now_iso(),
            }
        )
        if not login_result.ok:
            raise RuntimeError(
                f"Login timeout in {stage} for item {item_id}: "
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
    ) -> dict[str, list[str]]:
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
  for (const img of root.querySelectorAll('img')) {
    let src = readAttr(img, ['src', 'data-src', 'data-ks-lazyload', 'data-lazy-src', 'data-original']) || String(img.currentSrc || '').trim();
    if (!src) continue;
    if (src.startsWith('//')) src = 'https:' + src;
    if (!/^https?:\\/\\//i.test(src)) continue;
    if (imageSeen.has(src)) continue;
    imageSeen.add(src);
    images.push(src);
    if (images.length >= 40) break;
  }
  return { texts, images };
}
"""
        try:
            payload = await target.evaluate(script)
        except Exception:
            return {"texts": [], "images": []}
        if not isinstance(payload, dict):
            return {"texts": [], "images": []}
        texts = [str(v) for v in payload.get("texts", []) if isinstance(v, str)]
        images = [str(v) for v in payload.get("images", []) if isinstance(v, str)]
        return {"texts": texts, "images": images}

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
        payload = await self._extract_detail_payload_from_target(page)
        texts.extend(payload.get("texts", []))
        images.extend(payload.get("images", []))

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

        cleaned_texts: list[str] = []
        seen_text: set[str] = set()
        for raw in texts:
            text = clean_text(str(raw or ""), max_len=140)
            if len(text) < 6:
                continue
            if not re.search(r"[\u4e00-\u9fff]", text):
                continue
            if text in seen_text:
                continue
            seen_text.add(text)
            cleaned_texts.append(text)
            if len(cleaned_texts) >= 20:
                break

        cleaned_images: list[str] = []
        seen_image: set[str] = set()
        for raw in images:
            normalized = self._normalize_image_url(raw)
            if not normalized or normalized in seen_image:
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

                # Each crawl should use an isolated page to avoid cross-task navigation interruption.
                page = await global_manager.new_page()

                # Navigate to the item page
                await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))

                # Check for non-product page (login required, blocked, etc.)
                login_handled = await self._handle_login_if_needed_async(
                    page=page,
                    context=browser_context,
                    stage="item-initial",
                    item_id=item_id,
                )
                if login_handled:
                    await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                    await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))
                    retry_url, retry_title, retry_body = await self._read_page_snapshot_async(page)
                    retry_reason = detect_non_product_page(retry_url, retry_title, retry_body)
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
