"""Web scraping and data extraction for JD.com."""

from __future__ import annotations

import asyncio
import logging
import re
from html import unescape as html_unescape
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

from tools.browser_manager import get_global_browser_manager
from tools.jd_login import JDLogin
from tools.jd_login_rules import detect_non_product_page, is_search_result_url

from config import (
    ANTI_BOT_MARKERS,
    BANNED_SHOP_MARKER_RE,
    BRAND_RE,
    SEARCH_BLOCK_HINT,
    SHOP_NAME_RE,
)
from data import (
    ItemDetail,
    UrlRecord,
    clean_text,
    extract_candidate_item_urls,
    extract_json_object,
    is_official_shop,
    normalize_brand_name,
    normalize_item_title,
    normalize_url,
    now_iso,
)
from scraper import Crawler as BaseCrawler
from scraper import SearchClient as BaseSearchClient
from scraper import (
    _browser_fetch_text_via_page,
    _browser_user_agent,
    _cookie_header_for_domains,
    _http_fetch_text_with_cookie_header,
)

LOG = logging.getLogger("taobao_insight")

# JD shows review/comment counts ("评论" / "评价") instead of "已售" / "人付款".
# Define a separate regex so the shared SALES_TEXT_RE in config.py stays clean
# for Taobao/Tmall and doesn't risk misidentifying review counts as sales.
_JD_SALES_TEXT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(万)?\+?\s*(?:已售|评论|评价|人购买|人好评)",
    re.IGNORECASE,
)


def _jd_parse_sales_to_int(text: str) -> int:
    source = (text or "").strip()
    if not source:
        return 0
    match = _JD_SALES_TEXT_RE.search(source)
    if not match:
        return 0
    try:
        number = float(match.group(1))
    except ValueError:
        return 0
    if match.group(2):
        number *= 10000.0
    return int(number)


def detect_jd_antibot_signal(*texts: str) -> str:
    merged = "\n".join(t for t in texts if t)
    if not merged:
        return ""
    merged_lower = merged.lower()
    for marker in ANTI_BOT_MARKERS:
        if marker.lower() in merged_lower:
            return marker
    return ""


def is_jd_item_detail_url(url: str) -> bool:
    return "item.jd.com/" in (url or "").lower()


def parse_jd_search_cards_from_html(html: str) -> list[dict[str, str]]:
    if not html:
        return []

    def _pick_match(block: str, patterns: tuple[str, ...]) -> re.Match[str] | None:
        for pattern in patterns:
            match = re.search(pattern, block, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match
        return None

    cards: list[dict[str, str]] = []
    root_pattern = re.compile(
        r'<(?:li|div)\b[^>]*class="[^"]*(?:gl-item|j-sku-item|plugin_goodsCardWrapper)[^"]*"[^>]*>',
        flags=re.IGNORECASE,
    )
    starts = [match.start() for match in root_pattern.finditer(html)]
    if not starts:
        return []
    starts.append(len(html))
    blocks = [html[starts[index] : starts[index + 1]] for index in range(len(starts) - 1)]

    for block in blocks:
        sku_match = re.search(r'data-sku="(?P<sku>\d{6,})"', block, flags=re.IGNORECASE)
        href_match = re.search(
            r'href="(?P<href>(?:https?:)?//item\.jd\.com/\d+\.html[^"]*)"',
            block,
            flags=re.IGNORECASE,
        )
        href = ""
        if href_match:
            href = html_unescape(href_match.group("href"))
        elif sku_match:
            href = f"https://item.jd.com/{sku_match.group('sku')}.html"
        if not href:
            continue
        title_match = _pick_match(
            block,
            (
                r'<span[^>]*class="[^"]*_text_[^"]*"[^>]*>(?P<title>.*?)</span>',
                r'<(?:div|em)[^>]*class="[^"]*p-name[^"]*"[^>]*>(?P<title>.*?)</(?:div|em)>',
                r'<em>(?P<title>.*?)</em>',
                r'<(?:div|span|a)[^>]*class="[^"]*(?:_wrapper_|goods_title|p-name)[^"]*"[^>]*title="(?P<title>[^"]{4,300})"[^>]*>',
                r'<(?:div|span|a)[^>]*title="(?P<title>[^"]{4,300})"[^>]*class="[^"]*(?:_wrapper_|goods_title|p-name)[^"]*"[^>]*>',
            ),
        )
        shop_match = _pick_match(
            block,
            (
                r'<(?:span|div)[^>]*class="[^"]*_name_[^"]*"[^>]*>\s*<span[^>]*>(?P<shop>.*?)</span>',
                r'<(?:div|span)[^>]*class="[^"]*p-shop[^"]*"[^>]*>.*?<a[^>]*title="(?P<shop>[^"]+)"',
                r'<(?:div|span)[^>]*class="[^"]*p-shop[^"]*"[^>]*>.*?<a[^>]*>(?P<shop>.*?)</a>',
            ),
        )
        sales_match = _pick_match(
            block,
            (
                r'<(?:div|span)[^>]*class="[^"]*_goods_volume_container_[^"]*"[^>]*>(?P<sales>.*?)</(?:div|span)>',
                r'<(?:div|span)[^>]*class="[^"]*_goods_volume_[^"]*"[^>]*>(?P<sales>.*?)</(?:div|span)>',
                r'<(?:div|strong|a)[^>]*class="[^"]*(?:p-commit|p-sales|comments|rate)[^"]*"[^>]*>(?P<sales>.*?)</(?:div|strong|a)>',
            ),
        )
        card_text = re.sub(r"<[^>]+>", " ", block)
        card_text = clean_text(html_unescape(card_text), max_len=500)
        cards.append(
            {
                "href": href,
                "title": clean_text(
                    html_unescape(re.sub(r"<[^>]+>", " ", title_match.group("title")))
                    if title_match
                    else "",
                    max_len=180,
                ),
                "shop_name": clean_text(
                    html_unescape(re.sub(r"<[^>]+>", " ", shop_match.group("shop")))
                    if shop_match
                    else "",
                    max_len=120,
                ),
                "sales_text": clean_text(
                    html_unescape(re.sub(r"<[^>]+>", " ", sales_match.group("sales")))
                    if sales_match
                    else "",
                    max_len=60,
                ),
                "card_text": card_text,
            }
        )
    return cards


class JDSearchClient(BaseSearchClient):
    DEFAULT_STORAGE_STATE_FILE = "jd_storage_state.json"
    DEFAULT_USER_DATA_DIR_NAME = "jd_insight_profile"
    PLATFORM_LABEL = "JD"
    LOGIN_HANDLER_CLS = JDLogin
    DETECT_NON_PRODUCT_PAGE_FN = staticmethod(detect_non_product_page)

    @staticmethod
    def _records_from_candidate_urls(urls: list[str], top_n: int) -> list[UrlRecord]:
        records: list[UrlRecord] = []
        seen: set[str] = set()
        for url in urls:
            rec = normalize_url(url, default_platform="jd", allowed_platforms={"jd"})
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
            rec = normalize_url(href, default_platform="jd", allowed_platforms={"jd"})
            if not rec or rec.item_id in seen_item_ids:
                continue
            title = normalize_item_title(str(card.get("title", "")), max_len=160)
            shop_name = clean_text(str(card.get("shop_name", "")), max_len=120)
            sales_text = clean_text(str(card.get("sales_text", "")), max_len=60)
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
            sales_value = _jd_parse_sales_to_int(sales_text or card_text)
            ranked.append((index, sales_value, rec))

        if search_sort == "sales":
            ranked.sort(key=lambda item: (-item[1], item[0]))
        else:
            ranked.sort(key=lambda item: item[0])
        return [item[2] for item in ranked[:top_n]]

    async def _extract_search_surface_async(
        self,
        page: Any,
        *,
        top_n: int,
    ) -> tuple[str, str, list[dict[str, Any]], list[str]]:
        try:
            page_html = await page.content()
        except Exception:
            page_html = ""
        try:
            page_text = await page.inner_text("body")
        except Exception:
            page_text = ""

        try:
            card_records = await page.evaluate(
                """
() => {
  const text = (node) => String((node && (node.innerText || node.textContent)) || '')
    .replace(/\\s+/g, ' ')
    .trim();
  const attr = (node, names) => {
    if (!node) return '';
    for (const name of names) {
      const value = node.getAttribute && node.getAttribute(name);
      if (value) return String(value).trim();
    }
    return '';
  };
  const selectors = [
    'li.gl-item',
    '.gl-item[data-sku]',
    '.j-sku-item',
    '.plugin_goodsCardWrapper[data-sku]',
    '[data-sku].plugin_goodsCardWrapper',
    '[data-sku][data-point-id]',
  ];
  const roots = [];
  const seen = new Set();
  for (const selector of selectors) {
    for (const node of document.querySelectorAll(selector)) {
      if (!node || seen.has(node)) continue;
      seen.add(node);
      roots.push(node);
    }
  }
  return roots.slice(0, 200).map((root) => {
    const sku = attr(root, ['data-sku', 'data-skuid', 'sku']);
    const link = root.querySelector('a[href*="item.jd.com/"], a[href*="//item.jd.com/"]');
    let href = attr(link, ['href']);
    if (!href && sku) {
      href = `https://item.jd.com/${sku}.html`;
    } else if (href.startsWith('//')) {
      href = `https:${href}`;
    }
    const titleNode = root.querySelector(
      '.p-name em, .p-name-type-2 em, .p-name a em, [class*="goods_title"] [class*="_text_"], [class*="_text_"][title], [class*="_wrapper_"][title], em'
    );
    const shopNode = root.querySelector(
      '.p-shop a[title], .p-shop a, [class*="_shopFloor_"] [class*="_name_"] span, [class*="_shopFloor_"] [class*="_name_"], [class*="shop"] a[title], [class*="shop"] a'
    );
    const salesNode = root.querySelector(
      '.p-commit a, .p-commit strong, .p-sales, [class*="commit"] a, [class*="_goods_volume_container_"], [class*="_goods_volume_"]'
    );
    return {
      href,
      title: attr(titleNode, ['title']) || text(titleNode),
      shop_name: attr(shopNode, ['title']) || text(shopNode),
      sales_text: text(salesNode),
      card_text: text(root),
    };
  }).filter((row) => row.href || row.title || row.card_text);
}
""",
            )
        except Exception:
            card_records = []

        if not isinstance(card_records, list):
            card_records = []
        if not card_records:
            card_records = parse_jd_search_cards_from_html(page_html)

        try:
            hrefs = await page.eval_on_selector_all(
                "a[href]",
                "nodes => nodes.map(node => node.href || node.getAttribute('href') || '')",
            )
        except Exception:
            hrefs = []

        candidate_urls: list[str] = []
        candidate_urls.extend(
            extract_candidate_item_urls(
                page_html,
                limit=top_n * 50,
                default_platform="jd",
                allowed_platforms={"jd"},
            )
        )
        candidate_urls.extend(
            extract_candidate_item_urls(
                page_text,
                limit=top_n * 50,
                default_platform="jd",
                allowed_platforms={"jd"},
            )
        )
        for raw_url in hrefs:
            if not raw_url:
                continue
            candidate_urls.extend(
                extract_candidate_item_urls(
                    str(raw_url),
                    limit=2,
                    default_platform="jd",
                    allowed_platforms={"jd"},
                )
            )
        for sku in re.findall(r'data-sku="(\d{6,})"', page_html, flags=re.IGNORECASE):
            candidate_urls.append(f"https://item.jd.com/{sku}.html")
        return page_html, page_text, card_records, candidate_urls

    async def _ensure_search_surface_async(
        self,
        page: Any,
        context: Any,
        target_url: str,
        stage: str,
    ) -> None:
        login_handled = await self._handle_login_if_needed_async(page, context, stage)
        if login_handled:
            await page.goto(target_url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(3000, self.manual_wait_seconds * 1000))

        current_url, _, _ = await self._read_page_snapshot_async(page)
        if is_jd_item_detail_url(current_url):
            LOG.warning(
                "JD search stage landed on item detail page (%s); navigating back to search",
                current_url,
            )
            await page.goto(target_url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(3000, self.manual_wait_seconds * 1000))
            current_url, _, _ = await self._read_page_snapshot_async(page)

        if not is_search_result_url(current_url):
            LOG.warning(
                "Unexpected JD search page URL (%s), forcing navigation to target search URL",
                current_url,
            )
            await page.goto(target_url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(3000, self.manual_wait_seconds * 1000))

        await self._handle_login_if_needed_async(page, context, f"{stage}-post-nav")
        final_url, _, _ = await self._read_page_snapshot_async(page)
        if is_jd_item_detail_url(final_url):
            raise RuntimeError(
                "search phase is still on item detail page after recovery; "
                "please keep the browser on JD search results while crawling"
            )
        if not is_search_result_url(final_url):
            raise RuntimeError(
                f"search phase failed to stay on JD search page (current: {final_url})"
            )

    async def _search_with_global_browser_async(
        self,
        keyword: str,
        top_n: int,
        search_url: str | None = None,
        search_sort: str = "page",
        official_only: bool = False,
    ) -> tuple[list[UrlRecord], str]:
        if self._global_browser_manager is None:
            self._global_browser_manager = await get_global_browser_manager()

        target_url = search_url or f"https://search.jd.com/Search?keyword={quote_plus(keyword)}&enc=utf-8"
        candidate_urls: list[str] = []
        card_records: list[dict[str, Any]] = []
        resource_urls: list[str] = []
        response_snippets: list[str] = []
        response_tasks: set[asyncio.Task[Any]] = set()
        page_text = ""
        page_html = ""

        context = self._global_browser_manager.browser_context
        if context is None:
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
            if getattr(self._global_browser_manager, "current_mode", "") == "cdp":
                page = await self._global_browser_manager.new_page()
                owns_page = True
            else:
                page = await self._global_browser_manager.get_page()

            async def _capture_response(resp: Any) -> None:
                url_lower = (resp.url or "").lower()
                if "jd.com" not in url_lower and "3.cn" not in url_lower:
                    return
                try:
                    text = await resp.text()
                except Exception:
                    return
                if text:
                    response_snippets.append(text[:120000])

            def _on_response(resp: Any) -> None:
                resource_urls.append(resp.url or "")
                task = asyncio.create_task(_capture_response(resp))
                response_tasks.add(task)
                task.add_done_callback(lambda done: response_tasks.discard(done))

            page.on("response", _on_response)
            await page.goto(target_url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))

            await self._ensure_search_surface_async(
                page=page,
                context=context,
                target_url=target_url,
                stage="search-initial-global",
            )

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
            page_html, page_text, card_records, candidate_urls = (
                await self._extract_search_surface_async(page, top_n=top_n)
            )
            if not card_records and not candidate_urls:
                current_url, current_title, current_body = await self._read_page_snapshot_async(page)
                blocked_reason = detect_non_product_page(
                    current_url,
                    current_title,
                    current_body,
                )
                if blocked_reason:
                    LOG.warning(
                        "JD search surface is empty and still blocked during %s: %s",
                        "search-empty-surface",
                        blocked_reason,
                    )
                    login_handled = await self._handle_login_if_needed_async(
                        page,
                        context,
                        "search-empty-surface",
                    )
                    if login_handled:
                        await page.goto(
                            target_url,
                            wait_until="domcontentloaded",
                            timeout=120_000,
                        )
                        await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))
                        await self._ensure_search_surface_async(
                            page=page,
                            context=context,
                            target_url=target_url,
                            stage="search-empty-surface-retry",
                        )
                        page_html, page_text, card_records, candidate_urls = (
                            await self._extract_search_surface_async(page, top_n=top_n)
                        )
        finally:
            if response_tasks:
                await asyncio.gather(*list(response_tasks), return_exceptions=True)
            if owns_page and page is not None:
                try:
                    await page.close()
                except Exception:
                    pass

        records = JDSearchClient._records_from_cards(
            card_records,
            top_n=top_n,
            search_sort=search_sort,
            official_only=official_only,
        )
        if not records:
            records = JDSearchClient._records_from_candidate_urls(candidate_urls, top_n)
        anti_bot_signal = detect_jd_antibot_signal(
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
                "no JD item URL found from search; "
                "ensure you are logged in and the search page loads correctly"
            )
        except Exception as exc:
            raise RuntimeError(f"JD search failed: {exc}")


class JDCrawler(BaseCrawler):
    DEFAULT_STORAGE_STATE_FILE = "jd_storage_state.json"
    DEFAULT_USER_DATA_DIR_NAME = "jd_insight_profile"
    PLATFORM_LABEL = "JD"
    LOGIN_HANDLER_CLS = JDLogin
    DETECT_NON_PRODUCT_PAGE_FN = staticmethod(detect_non_product_page)

    @staticmethod
    def _is_noise_image(url: str, alt: str = "", cls: str = "") -> bool:
        text = f"{url} {alt} {cls}".lower()
        bad_tokens = (
            "logo",
            "icon",
            "avatar",
            "service",
            "sprite",
            "shop-head",
            "_50x50",
            "_80x80",
            ".mp4",
            ".m3u8",
            "video-icon",
            "video-player",
            "/video/",
            "poster",
            "top-logo",
            "top-shop-icon",
            "/shaidan/",
            "default.image",
            "i.imageupload",
            "imagetools",
        )
        return any(token in text for token in bad_tokens)

    @staticmethod
    def _extract_shop_name(html: str, body_text: str) -> str:
        candidates: list[str] = []
        seen: set[str] = set()

        def _normalize_candidate(raw: str) -> str:
            value = re.sub(r"<[^>]+>", " ", html_unescape(str(raw or "")))
            value = clean_text(value, max_len=120)
            if not value:
                return ""
            matched_name = re.search(
                r"(?P<name>[A-Za-z0-9\u4e00-\u9fff·（）()\-]{2,80}(?:京东自营旗舰店|京东自营|官方旗舰店|旗舰店|专卖店|专营店))",
                value,
            )
            if matched_name:
                value = clean_text(matched_name.group("name"), max_len=80)
            if not value or len(value) > 80:
                return ""
            if BANNED_SHOP_MARKER_RE.search(value):
                return ""
            if any(
                token in value
                for token in (
                    "￥",
                    "价格",
                    "累计评价",
                    "降价通知",
                    "商品详情",
                    "大家评",
                    "买家评价",
                    "推荐",
                    "更多优惠",
                    "精选镇店好物",
                )
            ):
                return ""
            if not re.search(r"(京东自营|旗舰店|专卖店|专营店|店$)", value):
                return ""
            return value

        def _push(raw: str) -> None:
            value = _normalize_candidate(raw)
            if not value or value in seen:
                return
            if any(
                token in value
                for token in ("进入店铺", "进店逛逛", "联系客服", "关注店铺", "更多优惠")
            ):
                return
            seen.add(value)
            candidates.append(value)

        for match in SHOP_NAME_RE.finditer(html or ""):
            _push(match.group("name"))
        for pattern in (
            r'"(?:venderName|shopNameStr|storeName|mallName|vendorName)"\s*:\s*"(?P<name>[^"]{2,120})"',
            r'title="(?P<name>[^"]{2,120}(?:京东自营旗舰店|京东自营|官方旗舰店|旗舰店|专卖店|专营店))"',
            r">(?P<name>[^<]{2,120}(?:京东自营旗舰店|京东自营|官方旗舰店|旗舰店|专卖店|专营店))<",
        ):
            for match in re.finditer(pattern, html or "", flags=re.IGNORECASE):
                _push(match.group("name"))
        for pattern in (
            r'title="(?P<name>[^"]{2,80}(?:京东自营旗舰店|京东自营|官方旗舰店|旗舰店|专卖店|专营店))"',
            r">(?P<name>[^<]{2,80}(?:京东自营旗舰店|京东自营|官方旗舰店|旗舰店|专卖店|专营店))<",
        ):
            match = re.search(pattern, html, flags=re.IGNORECASE)
            if match:
                _push(match.group("name"))
        for line in body_text.splitlines():
            snippet = clean_text(html_unescape(line), max_len=120)
            if re.search(r"(京东自营旗舰店|京东自营|官方旗舰店|旗舰店|专卖店|专营店)", snippet):
                _push(snippet)
        if candidates:
            candidates.sort(
                key=lambda value: (
                    "京东自营旗舰店" in value,
                    "京东自营" in value,
                    "官方旗舰店" in value,
                    "旗舰店" in value,
                    "专卖店" in value or "专营店" in value,
                    len(value),
                ),
                reverse=True,
            )
            return candidates[0]
        return ""

    @staticmethod
    def _extract_brand(title: str, html: str) -> str:
        for match in BRAND_RE.finditer(html or ""):
            brand_from_html = normalize_brand_name(
                match.group("brand"),
                title=title,
                shop_name="",
            )
            if brand_from_html:
                return brand_from_html
        shop_name = JDCrawler._extract_shop_name(html, "")
        if shop_name:
            brand_from_shop = normalize_brand_name("", title=title, shop_name=shop_name)
            if brand_from_shop:
                return brand_from_shop
        title_text = normalize_item_title(title, max_len=120)
        if not title_text:
            return ""
        token = re.split(r"(（|\(|\s)", title_text, maxsplit=1)[0]
        return clean_text(token, max_len=40)

    @classmethod
    def _resolve_shop_name(
        cls,
        candidate_values: list[str],
        html_and_text_pairs: list[tuple[str, str]],
    ) -> str:
        for value in candidate_values:
            normalized = cls._extract_shop_name("", str(value or ""))
            if normalized:
                return normalized
        for html, body_text in html_and_text_pairs:
            shop_name = cls._extract_shop_name(html, body_text)
            if shop_name:
                return shop_name
        return ""

    @classmethod
    def _resolve_brand(
        cls,
        title: str,
        shop_name: str,
        html_candidates: list[str],
    ) -> str:
        for html in html_candidates:
            brand = normalize_brand_name(
                cls._extract_brand(title, html),
                title=title,
                shop_name=shop_name,
            )
            if brand:
                return brand
        return normalize_brand_name("", title=title, shop_name=shop_name)

    @staticmethod
    def _merge_page_texts(*texts: str, max_lines: int = 240) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for raw in re.split(r"[\r\n]+", str(text or "")):
                value = clean_text(html_unescape(raw), max_len=260)
                if not value or value in seen:
                    continue
                seen.add(value)
                lines.append(value)
                if len(lines) >= max_lines:
                    return "\n".join(lines)
        return "\n".join(lines)

    @staticmethod
    def _extract_prices(text: str, html: str) -> list[float]:
        prices: list[float] = []
        for token in re.findall(r"[¥￥]\s*(\d{1,6}(?:\.\d{1,2})?)", text):
            try:
                prices.append(float(token))
            except ValueError:
                continue
        for token in re.findall(
            r'"(?:price|jdPrice|salePrice|pcPrice|p)"\s*:\s*"?(?P<v>\d{1,6}(?:\.\d{1,2})?)"?',
            html,
            flags=re.IGNORECASE,
        ):
            try:
                prices.append(float(token))
            except ValueError:
                continue
        if not prices:
            return []
        return JDCrawler._refine_price_values(prices)

    @staticmethod
    def _extract_color_size_payload(html: str) -> list[dict[str, Any]]:
        match = re.search(
            r"colorSize\s*:\s*(\[[\s\S]*?\])\s*,\s*warestatus",
            html,
            flags=re.IGNORECASE,
        )
        if not match:
            return []
        payload = extract_json_object(match.group(1))
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

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
            if bool(image.get("is_video")):
                continue
            if cls._is_noise_image(src, alt, css):
                continue
            if not cls._looks_like_product_image_url(src):
                continue
            lower = src.lower()
            if not any(
                token in lower
                for token in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".avif",
                    "360buyimg.com",
                    "jd.com",
                    "jdimg",
                    "/jfs/",
                )
            ):
                continue
            score = 0
            if width >= 700 and height >= 700:
                score += 6
            elif width >= 420 and height >= 420:
                score += 4
            elif width >= 220 and height >= 220:
                score += 2
            if any(
                token in lower
                for token in ("360buyimg.com", "jdimg", "/jfs/", "s800x800", "s450x450")
            ):
                score += 4
            css_lower = css.lower()
            if any(token in css_lower for token in ("spec-img", "jqzoom", "preview")):
                score += 2
            if bool(image.get("is_main")):
                score += 3
            ranked.append((score, src))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1] if ranked else ""

    @staticmethod
    def _looks_like_product_image_url(url: str) -> bool:
        lower = str(url or "").strip().lower()
        if not lower.startswith(("http://", "https://")):
            return False
        if any(token in lower for token in (".js", ".css", ".svg", ".woff", ".ttf", "linksubmit")):
            return False
        if any(
            token in lower
            for token in (
                "/shaidan/",
                "default.image",
                "i.imageupload",
                "avatar",
                "comment",
                "review",
                "imagetools",
            )
        ):
            return False
        if not any(
            token in lower
            for token in (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".avif",
                ".dpg",
                "360buyimg.com",
                "jdimg",
                "/jfs/",
                "pcpubliccms",
            )
        ):
            return False
        return True

    @classmethod
    def _normalize_price_text_candidates(
        cls,
        raw_values: list[str],
        *,
        title: str = "",
        html: str = "",
    ) -> list[str]:
        canonical: list[str] = []
        seen: set[str] = set()

        def _push(text: str) -> None:
            value = clean_text(str(text or ""), max_len=80)
            if not value or value in seen:
                return
            seen.add(value)
            canonical.append(value)

        prices = cls._extract_prices("\n".join(raw_values), html)
        for price in prices:
            _push(f"价格信息：￥{price:.2f}")
        for token in re.findall(
            r'"(?:price|jdPrice|salePrice|pcPrice|p)"\s*:\s*"?(?P<v>\d{1,6}(?:\.\d{1,2})?)"?',
            html or "",
            flags=re.IGNORECASE,
        ):
            try:
                numeric = float(token)
            except ValueError:
                continue
            if numeric > 0:
                _push(f"价格信息：￥{numeric:.2f}")

        for raw in raw_values:
            value = clean_text(html_unescape(str(raw or "")), max_len=160)
            if not value:
                continue
            if any(token in value for token in ("累计评价", "降价通知", "促销", "优惠券", "预估到手价")):
                continue
            matched = re.findall(r"\d{1,6}(?:\.\d{1,2})?", value)
            if matched:
                for token in matched:
                    try:
                        numeric = float(token)
                    except ValueError:
                        continue
                    if numeric > 0:
                        _push(f"价格信息：￥{numeric:.2f}")
            elif "￥" in value:
                _push(value)
        return canonical

    @staticmethod
    def _image_identity(url: str) -> str:
        value = str(url or "").strip()
        if not value:
            return ""
        lower = value.lower()
        if lower.startswith(("http://", "https://")):
            parsed = urlparse(lower)
            name = parsed.path.rsplit("/", 1)[-1]
            return name or parsed.path
        return lower.replace("\\", "/").rsplit("/", 1)[-1]

    @classmethod
    def _is_relevant_dom_image_row(cls, row: dict[str, Any]) -> bool:
        src = cls._normalize_image_url(str(row.get("src", "")))
        if not src:
            return False
        alt = str(row.get("alt", ""))
        css = str(row.get("class_name", ""))
        lower = f"{src} {alt} {css}".lower()
        width = int(float(row.get("width", 0) or 0))
        height = int(float(row.get("height", 0) or 0))
        is_main = bool(row.get("is_main"))
        if bool(row.get("is_video")):
            return False
        if cls._is_noise_image(src, alt, css):
            return False
        if not cls._looks_like_product_image_url(src):
            return False
        if any(
            token in lower
            for token in (
                "play",
                "arrow",
                "swiper",
                "slider",
                "mask",
                "button",
                "btn",
                "pager",
            )
        ) and not is_main:
            return False
        if "popshop" in lower and not is_main:
            return False
        if "imagetools" in lower and not is_main:
            if width and height and max(width, height) <= 256:
                return False
        if not is_main and width and height:
            if max(width, height) <= 120:
                return False
            if width * height < 90_000:
                return False
        return True

    @classmethod
    def _normalize_gallery_asset_url(cls, raw: str) -> str:
        value = str(raw or "").strip()
        if not value:
            return ""
        if value.startswith("//"):
            return "https:" + value
        if value.startswith("http://") or value.startswith("https://"):
            return value
        if value.startswith("/"):
            return urljoin("https://img10.360buyimg.com/", value)
        if value.startswith("jfs/") or value.startswith("jfs\\"):
            return f"https://img10.360buyimg.com/n1/{value.lstrip('/')}"
        return cls._normalize_image_url(value)

    @classmethod
    def _extract_gallery_image_urls(cls, html: str) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()

        def _push(raw: str) -> None:
            value = cls._normalize_gallery_asset_url(raw)
            if not value or value in seen:
                return
            if cls._is_noise_image(value):
                return
            seen.add(value)
            urls.append(value)

        for pattern in (
            r'<img\b[^>]*id=["\']spec-img["\'][^>]*(?:data-origin|src)=["\'](?P<src>[^"\']+)["\']',
            r'<img\b[^>]*(?:data-origin|src)=["\'](?P<src>[^"\']+)["\'][^>]*id=["\']spec-img["\']',
        ):
            for match in re.finditer(pattern, html or "", flags=re.IGNORECASE):
                _push(match.group("src"))

        image_list_match = re.search(
            r"imageList\s*:\s*(\[[\s\S]*?\])\s*,\s*(?:cat|forceAdUpdate|brand|pType)",
            html or "",
            flags=re.IGNORECASE,
        )
        if image_list_match:
            payload = extract_json_object(image_list_match.group(1))
            if isinstance(payload, list):
                for item in payload[:40]:
                    _push(str(item))

        return urls[:40]

    @staticmethod
    def _extract_jd_detail_api_url(html: str) -> str:
        match = re.search(
            r"desc\s*:\s*['\"](?P<url>[^'\"]+)['\"]",
            html or "",
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        raw = str(match.group("url") or "").strip()
        if not raw:
            return ""
        return urljoin("https://item.jd.com/", raw)

    @staticmethod
    def _is_noise_detail_text(text: str) -> bool:
        value = clean_text(html_unescape(str(text or "")), max_len=220)
        if not value:
            return True
        lowered = value.lower()
        if re.fullmatch(r"\d{1,2}:\d{2}(?:\s*/\s*\d{1,2}:\d{2})?", value):
            return True
        if re.fullmatch(r"[A-Za-z0-9_-]{8,}\.\.\.", value):
            return True
        if re.fullmatch(r"[A-Za-z0-9_-]{12,}", value):
            return True
        if re.fullmatch(r"\d{1,8}", value):
            return True
        if re.fullmatch(r"\d+(?:\.\d+)?万\+?", value):
            return True
        if re.fullmatch(r"[￥¥]?\d{1,6}(?:\.\d{1,2})?", value):
            return True
        if re.fullmatch(r"\d+(?:年|个月|月)", value):
            return True
        if re.fullmatch(r"[A-Za-z]{1,6}\d{4,}", value):
            return True
        if any(ch in value for ch in ("✅", "😍", "❤", "👍", "💄", "✨")):
            return True
        if "🈱" in value:
            return True
        if re.search(r"[\U0001F000-\U0001FAFF]", value):
            return True
        if "?" in value or "？" in value:
            return True
        if re.search(r"满\s*\d+\s*减\s*\d+", value):
            return True
        if re.search(r"(返|送)\s*\d+\s*京豆", value):
            return True
        if value.startswith("海外") and re.search(r"[A-Z]{2,}", value):
            return True
        if value in ("产品（注册/备案）名称", "批准文号/备案编号", "是否特殊化妆品"):
            return True
        if re.search(r"\d{1,2}:\d{2}", value) and any(
            token in lowered for token in ("video", "poster", "播放", "预览")
        ):
            return True
        if "我" in value and any(
            token in value
            for token in (
                "推荐",
                "回购",
                "惊艳",
                "心头爱",
                "断货王",
                "不愧是",
                "拿在手里",
                "上嘴",
                "以后会继续",
            )
        ):
            return True
        noise_tokens = (
            "京东首页",
            "购物车",
            "我的订单",
            "我的京东",
            "企业采购",
            "网站导航",
            "手机京东",
            "立即登录",
            "立即注册",
            "打开京东app",
            "帮助中心",
            "联系客服",
            "关注店铺",
            "分享",
            "举报",
            "返回顶部",
            "进店逛逛",
            "更多优惠",
            "精选镇店好物",
                "大家评",
                "售后保障",
                "买家评价",
                "买家赞",
                "赞不绝口",
                "问答",
                "提问",
                "回答",
                "请问",
                "查看更多",
                "哪个好看",
                "被种草",
                "入手后",
                "太喜欢",
                "很好闻",
                "爱不释手",
                "我觉得",
                "强烈推荐",
                "心头爱",
                "以后会继续回购",
                "累计评价",
                "京东JD.COM提供",
                "网购指南",
                "正品行货",
                "可再享",
                "京豆",
                "一键找同款",
                "比价更省心",
                "热卖榜",
                "店铺 商品详情 售后保障 推荐",
                "商品编号：",
            "商品毛重：",
        )
        if any(token in value for token in noise_tokens):
            return True
        if any(token in lowered for token in ("passport.jd.com", "chat.jd.com")):
            return True
        if len(value) < 6 and not re.search(r"[¥￥\d:：]", value):
            return True
        return False

    @classmethod
    def _append_text_candidate(
        cls, out: list[str], seen: set[str], raw_text: str, max_len: int = 220
    ) -> None:
        text = clean_text(html_unescape(str(raw_text or "")), max_len=max_len)
        if not text or text in seen:
            return
        if cls._is_noise_detail_text(text):
            return
        seen.add(text)
        out.append(text)

    @classmethod
    def _append_image_candidate(
        cls, out: list[str], seen: set[str], raw_url: str
    ) -> None:
        value = cls._normalize_image_url(str(raw_url or ""))
        image_key = cls._image_identity(value)
        if not value or not image_key or image_key in seen:
            return
        if cls._is_noise_image(value):
            return
        if not cls._looks_like_product_image_url(value):
            return
        seen.add(image_key)
        out.append(value)

    @classmethod
    def _extract_texts_and_images_from_html_fragment(
        cls, fragment_html: str, base_url: str
    ) -> tuple[list[str], list[str]]:
        fragment = str(fragment_html or "").strip()
        if not fragment:
            return [], []

        text_candidates: list[str] = []
        seen_text: set[str] = set()
        image_candidates: list[str] = []
        seen_image: set[str] = set()

        for match in re.finditer(
            r"""<img\b[^>]*(?:data-origin|data-lazyload|data-src|src)=["'](?P<src>[^"']+)["']""",
            fragment,
            flags=re.IGNORECASE,
        ):
            src = urljoin(base_url, str(match.group("src") or "").strip())
            cls._append_image_candidate(image_candidates, seen_image, src)

        text_source = re.sub(
            r"(?is)<(script|style)[^>]*>.*?</\1>",
            " ",
            fragment,
        )
        text_source = re.sub(r"(?i)<br\s*/?>", "\n", text_source)
        text_source = re.sub(
            r"(?i)</(p|div|li|dd|dt|tr|h1|h2|h3|h4|th|td|span|section)>",
            "\n",
            text_source,
        )
        text_source = re.sub(r"<[^>]+>", " ", text_source)
        for raw_line in re.split(r"[\n\r]+", html_unescape(text_source)):
            cls._append_text_candidate(text_candidates, seen_text, raw_line, max_len=220)

        return text_candidates[:80], image_candidates[:60]

    @classmethod
    def _extract_texts_and_images_from_desc_payload(
        cls, raw_payload: str, base_url: str
    ) -> tuple[list[str], list[str]]:
        payload_text = str(raw_payload or "").strip()
        if not payload_text:
            return [], []

        html_candidates: list[str] = []
        parsed = extract_json_object(payload_text)

        def _walk(value: Any, depth: int = 0) -> None:
            if depth > 6 or value is None:
                return
            if isinstance(value, str):
                snippet = value.strip()
                lower = snippet.lower()
                if len(snippet) >= 20 and any(
                    token in lower for token in ("<img", "<p", "<div", "<li", "<table")
                ):
                    html_candidates.append(snippet)
                return
            if isinstance(value, list):
                for item in value[:20]:
                    _walk(item, depth + 1)
                return
            if isinstance(value, dict):
                for item in list(value.values())[:40]:
                    _walk(item, depth + 1)

        if parsed is not None:
            _walk(parsed)
        if not html_candidates and "<" in payload_text and ">" in payload_text:
            html_candidates.append(payload_text)

        text_candidates: list[str] = []
        image_candidates: list[str] = []
        seen_text: set[str] = set()
        seen_image: set[str] = set()
        for fragment in html_candidates[:6]:
            texts, images = cls._extract_texts_and_images_from_html_fragment(
                fragment,
                base_url=base_url,
            )
            for text in texts:
                cls._append_text_candidate(text_candidates, seen_text, text, max_len=220)
            for image in images:
                cls._append_image_candidate(image_candidates, seen_image, image)
        return text_candidates[:80], image_candidates[:60]

    async def _query_texts_from_selectors_async(
        self,
        page: Any,
        selectors: list[str],
        *,
        limit_per_selector: int = 20,
        max_total: int = 220,
    ) -> list[str]:
        texts: list[str] = []
        seen: set[str] = set()
        for selector in selectors:
            try:
                values = await page.eval_on_selector_all(
                    selector,
                    (
                        "nodes => nodes.slice(0, %d).map("
                        "node => String((node.innerText || node.textContent || '')).replace(/\\s+/g, ' ').trim()"
                        ")"
                    )
                    % limit_per_selector,
                )
            except Exception:
                continue
            if not isinstance(values, list):
                continue
            for value in values:
                self._append_text_candidate(texts, seen, str(value), max_len=220)
                if len(texts) >= max_total:
                    return texts
        return texts

    async def _query_image_meta_from_selectors_async(
        self,
        page: Any,
        selectors: list[str],
        *,
        limit_per_selector: int = 24,
        max_total: int = 120,
    ) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        seen: set[str] = set()
        expression = (
            "nodes => nodes.slice(0, %d).map(node => {"
            "const attr = (names) => {"
            "  for (const name of names) {"
            "    const value = node.getAttribute && node.getAttribute(name);"
            "    if (value) return String(value).trim();"
            "  }"
            "  return '';"
            "};"
            "let src = attr(['data-origin', 'data-src', 'data-lazyload', 'src', 'poster']) || String(node.currentSrc || '').trim();"
            "if (src && src.startsWith('//')) src = 'https:' + src;"
            "try { if (src) src = new URL(src, location.href).href; } catch (error) {}"
            "const container = node.closest ? node.closest('.video, #v-video, .J-video-view, [class*=\"video\"], [id*=\"video\"]') : null;"
            "const cls = attr(['class']);"
            "const id = attr(['id']);"
            "const marker = [cls, id, String(container && container.className || ''), String(container && container.id || ''), src].join(' ').toLowerCase();"
            "return {"
            "  src,"
            "  alt: attr(['alt']),"
            "  class_name: cls,"
            "  width: Number(node.naturalWidth || node.width || 0),"
            "  height: Number(node.naturalHeight || node.height || 0),"
            "  is_main: id === 'spec-img' || !!(node.closest && node.closest('#spec-n1')),"
            "  is_video: /(^|\\W)(video|poster)(\\W|$)/.test(marker) || /\\.(mp4|m3u8)(\\?|$)/.test(src)"
            "};"
            "})"
        ) % limit_per_selector
        for selector in selectors:
            try:
                values = await page.eval_on_selector_all(selector, expression)
            except Exception:
                continue
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                src = self._normalize_image_url(str(item.get("src", "")))
                if not src or src in seen:
                    continue
                row = dict(item)
                row["src"] = src
                seen.add(src)
                images.append(row)
                if len(images) >= max_total:
                    return images
        return images

    async def _extract_jd_dom_detail_payload_async(self, page: Any) -> dict[str, Any]:
        html = ""
        try:
            html = await page.content()
        except Exception:
            html = ""

        texts = await self._query_texts_from_selectors_async(
            page,
            [
                ".sku-name",
                ".news",
                ".summary-price .p-price",
                ".summary-service",
                ".contact .name a[title]",
                ".contact .name a",
                ".J-hove-wrap .name a[title]",
                ".J-hove-wrap .name a",
                "#crumb-wrap .contact .name a[title]",
                "#crumb-wrap .contact .name a",
                ".choose-attrs .item",
                "#choose-attrs .item",
                "#choose-attrs .dd a",
                ".itemInfo-wrap .item",
                ".Ptable tr",
                ".Ptable-item dl",
                ".parameter2 li",
                ".package-list li",
                ".promise-list li",
                ".after-service li",
            ],
            limit_per_selector=24,
            max_total=220,
        )
        image_meta = await self._query_image_meta_from_selectors_async(
            page,
            [
                "#spec-img",
                "#spec-n1 img",
                "#spec-list img",
                ".Ptable img",
                ".parameter2 img",
            ],
            limit_per_selector=24,
            max_total=120,
        )
        images = [
            row["src"]
            for row in image_meta
            if row.get("src") and self._is_relevant_dom_image_row(row)
        ]
        meta_description_match = re.search(
            r'<meta[^>]+name="description"[^>]+content="(?P<content>[^"]+)"',
            html,
            flags=re.IGNORECASE,
        )
        meta_keywords_match = re.search(
            r'<meta[^>]+name="(?:keywords|Keywords)"[^>]+content="(?P<content>[^"]+)"',
            html,
            flags=re.IGNORECASE,
        )
        return {
            "texts": texts,
            "images": images,
            "meta_description": clean_text(
                html_unescape(meta_description_match.group("content"))
                if meta_description_match
                else "",
                max_len=500,
            ),
            "meta_keywords": clean_text(
                html_unescape(meta_keywords_match.group("content"))
                if meta_keywords_match
                else "",
                max_len=500,
            ),
        }

    async def _fetch_jd_desc_payload_text_async(
        self, page: Any, desc_url: str
    ) -> tuple[str, str]:
        if not desc_url:
            return "", ""
        target = urljoin(str(page.url or "https://item.jd.com/"), desc_url)
        browser_text = await _browser_fetch_text_via_page(
            page,
            target,
            accept="application/json,text/plain,*/*",
            max_len=1_200_000,
        )
        if browser_text:
            return target, browser_text

        browser_user_agent = await _browser_user_agent(page)
        cookie_header = _cookie_header_for_domains(
            self.storage_state_file,
            allowed_domain_tokens=("jd.com", "3.cn"),
        )

        def _load_text() -> str:
            return _http_fetch_text_with_cookie_header(
                target,
                user_agent=browser_user_agent or "Mozilla/5.0",
                referer=str(page.url or "https://item.jd.com/"),
                accept="application/json,text/plain,*/*",
                cookie_header=cookie_header,
                timeout_sec=20,
                max_len=1_200_000,
            )

        try:
            text = await asyncio.to_thread(_load_text)
        except Exception:
            return target, ""
        return target, text

    async def _collect_detail_blocks_from_page(
        self,
        page: Any,
        item_id: str,
        image_dir: Path,
        *,
        html: str = "",
        runtime_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        runtime_payload = runtime_payload or {}
        try:
            await self._try_open_detail_tab(page)
            await page.wait_for_timeout(1200)
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
            full_page_path = image_dir / "full_page.png"
            if full_page_path.exists() and full_page_path.stat().st_size > 0:
                screenshot_path = str(full_page_path.resolve())

        dom_payload = await self._extract_jd_dom_detail_payload_async(page)
        desc_url = self._extract_jd_detail_api_url(html)
        desc_url, desc_raw = await self._fetch_jd_desc_payload_text_async(page, desc_url)
        desc_texts, desc_images = self._extract_texts_and_images_from_desc_payload(
            desc_raw,
            base_url=desc_url or page.url,
        )

        text_candidates: list[str] = []
        seen_text: set[str] = set()
        image_candidates: list[str] = []
        seen_image: set[str] = set()

        for value in (
            runtime_payload.get("title", ""),
            runtime_payload.get("shop_name", ""),
            dom_payload.get("meta_description", ""),
            dom_payload.get("meta_keywords", ""),
        ):
            self._append_text_candidate(text_candidates, seen_text, str(value), max_len=220)

        normalized_price_texts = self._normalize_price_text_candidates(
            list(runtime_payload.get("price_texts", []) or []),
            title=str(runtime_payload.get("title", "") or ""),
            html=html,
        )
        for value in normalized_price_texts:
            self._append_text_candidate(
                text_candidates,
                seen_text,
                value,
                max_len=160,
            )
        for value in runtime_payload.get("sku_texts", []) or []:
            self._append_text_candidate(text_candidates, seen_text, value, max_len=180)

        color_size_payload = self._extract_color_size_payload(html)
        color_names: list[str] = []
        for item in color_size_payload[:12]:
            spec_name = clean_text(
                str(item.get("规格", "") or item.get("name", "")),
                max_len=40,
            )
            if spec_name:
                color_names.append(spec_name)
        if color_names:
            self._append_text_candidate(
                text_candidates,
                seen_text,
                "可选规格：" + "；".join(color_names),
                max_len=220,
            )

        meta_description_match = re.search(
            r'<meta[^>]+name="description"[^>]+content="(?P<content>[^"]+)"',
            html or "",
            flags=re.IGNORECASE,
        )
        if meta_description_match:
            self._append_text_candidate(
                text_candidates,
                seen_text,
                meta_description_match.group("content"),
                max_len=220,
            )

        for value in dom_payload.get("texts", []) or []:
            self._append_text_candidate(text_candidates, seen_text, value, max_len=220)
        for value in desc_texts:
            self._append_text_candidate(text_candidates, seen_text, value, max_len=220)

        if len(text_candidates) < 8:
            body_text = str(runtime_payload.get("page_text", "") or "")
            for raw in re.split(r"[\n\r]+", body_text):
                self._append_text_candidate(text_candidates, seen_text, raw, max_len=200)
                if len(text_candidates) >= 16:
                    break

        if screenshot_path:
            image_candidates.append(screenshot_path)
            seen_image.add(screenshot_path)
        for value in runtime_payload.get("image_urls", []) or []:
            self._append_image_candidate(image_candidates, seen_image, value)
        for value in self._extract_gallery_image_urls(html):
            self._append_image_candidate(image_candidates, seen_image, value)
        for value in dom_payload.get("images", []) or []:
            self._append_image_candidate(image_candidates, seen_image, value)
        for value in desc_images:
            self._append_image_candidate(image_candidates, seen_image, value)

        blocks: list[dict[str, str]] = []
        for index, text in enumerate(text_candidates[:60], start=1):
            blocks.append(
                {"source_type": "text", "source_ref": f"text_{index}", "content": text}
            )
        for index, image_url in enumerate(image_candidates[:20], start=1):
            blocks.append(
                {
                    "source_type": "image",
                    "source_ref": f"image_{index}",
                    "content": image_url,
                }
            )
        return blocks

    async def _extract_runtime_payload(self, page: Any) -> dict[str, Any]:
        html = ""
        try:
            html = await page.content()
        except Exception:
            html = ""
        try:
            page_text = await page.inner_text("body")
        except Exception:
            page_text = ""

        title_candidates = await self._query_texts_from_selectors_async(
            page,
            [".sku-name", "#name h1"],
            limit_per_selector=2,
            max_total=4,
        )
        shop_candidates = await self._query_texts_from_selectors_async(
            page,
            [
                ".contact .name a[title]",
                ".contact .name a",
                ".J-hove-wrap .name a[title]",
                ".J-hove-wrap .name a",
                "#crumb-wrap .contact .name a[title]",
                "#crumb-wrap .contact .name a",
            ],
            limit_per_selector=4,
            max_total=8,
        )
        price_texts = await self._query_texts_from_selectors_async(
            page,
            [
                ".summary-price .p-price",
                ".summary-price-wrap .price",
                ".summary-price-wrap [class*='price']",
                "#J-summary-price .p-price",
            ],
            limit_per_selector=8,
            max_total=12,
        )
        sku_texts = await self._query_texts_from_selectors_async(
            page,
            [
                "#choose-attrs .item",
                "#choose-attrs .dd a",
                ".choose-attrs .item",
                ".itemInfo-wrap .item",
                ".Ptable tr",
                ".parameter2 li",
            ],
            limit_per_selector=20,
            max_total=60,
        )
        image_meta = await self._query_image_meta_from_selectors_async(
            page,
            [
                "#spec-img",
                "#spec-n1 img",
                "#spec-list img",
                "#J-detail-content img",
                "#J-detail-top img",
                "#J-detail-bottom img",
            ],
            limit_per_selector=24,
            max_total=120,
        )
        gallery_urls = self._extract_gallery_image_urls(html)
        image_urls: list[str] = []
        seen_image_urls: set[str] = set()
        for row in image_meta:
            if not self._is_relevant_dom_image_row(row):
                continue
            value = self._normalize_image_url(str(row.get("src", "")))
            if value and value not in seen_image_urls:
                seen_image_urls.add(value)
                image_urls.append(value)
        for value in gallery_urls:
            normalized = self._normalize_image_url(value)
            if not normalized or normalized in seen_image_urls:
                continue
            if self._is_noise_image(normalized):
                continue
            if not self._looks_like_product_image_url(normalized):
                continue
            seen_image_urls.add(normalized)
            image_urls.append(normalized)

        title = title_candidates[0] if title_candidates else ""
        if not title:
            page_title = ""
            try:
                page_title = await page.title()
            except Exception:
                page_title = ""
            title = clean_text(
                re.sub(r"\s*-\s*京东\s*$", "", str(page_title or "")),
                max_len=180,
            )
        shop_name = shop_candidates[0] if shop_candidates else ""
        if not shop_name:
            shop_name = self._extract_shop_name(html, page_text)

        return {
            "title": title,
            "shop_name": shop_name,
            "price_texts": price_texts,
            "sku_texts": sku_texts,
            "page_text": page_text,
            "image_meta": image_meta,
            "image_urls": image_urls,
        }

    async def crawl_async_global(
        self, url: str, item_id: str, workbook_id: str
    ) -> ItemDetail:
        image_dir = self.storage.images_dir / workbook_id / item_id
        image_dir.mkdir(parents=True, exist_ok=True)
        crawl_time = now_iso()

        max_nav_attempts = 3
        for nav_attempt in range(1, max_nav_attempts + 1):
            page = None
            visit_chain: list[dict[str, str]] = []
            try:
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

                await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))
                await _capture_visit("item_initial", note="initial goto")

                login_handled = await self._handle_login_if_needed_async(
                    page=page,
                    context=browser_context,
                    stage="item-initial",
                    item_id=item_id,
                )
                if login_handled:
                    await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                    await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))
                    await _capture_visit("item_after_login_recovery", note="reloaded after login recovery")
                    retry_url, retry_title, retry_body = await self._read_page_snapshot_async(page)
                    retry_reason = detect_non_product_page(retry_url, retry_title, retry_body)
                    if retry_reason:
                        raise RuntimeError(
                            f"jd item page blocked after login recovery: {retry_reason}. "
                            f"Please ensure you are logged in and complete verification."
                        )

                current_url, title, body_text = await self._read_page_snapshot_async(page)
                await _capture_visit("item_ready", note="detail page ready for extraction")
                if not is_jd_item_detail_url(current_url):
                    raise RuntimeError(f"unexpected JD item detail URL: {current_url}")

                pre_runtime_payload = await self._extract_runtime_payload(page)
                try:
                    pre_html = await page.content()
                except Exception:
                    pre_html = ""

                scroll_rounds = max(4, 4 + (self.manual_wait_seconds // 4))
                for _ in range(scroll_rounds):
                    await page.mouse.wheel(0, 1800)
                    await page.wait_for_timeout(1200)

                try:
                    await self._try_open_detail_tab(page)
                    await page.wait_for_timeout(1200)
                    await self._scroll_detail_section(page, rounds=8)
                except Exception:
                    pass

                try:
                    screenshot_path = image_dir / "full_page.png"
                    await page.screenshot(path=str(screenshot_path), full_page=True)
                except Exception:
                    pass

                post_runtime_payload = await self._extract_runtime_payload(page)
                try:
                    post_html = await page.content()
                except Exception:
                    post_html = ""

                html = post_html if len(post_html) >= len(pre_html) else pre_html
                pre_page_text = str(pre_runtime_payload.get("page_text", "") or "")
                post_page_text = str(post_runtime_payload.get("page_text", "") or "")
                title = str(
                    post_runtime_payload.get("title")
                    or pre_runtime_payload.get("title")
                    or title
                    or ""
                )
                shop_name = self._resolve_shop_name(
                    [
                        str(post_runtime_payload.get("shop_name", "") or ""),
                        str(pre_runtime_payload.get("shop_name", "") or ""),
                    ],
                    [
                        (pre_html, pre_page_text),
                        (post_html, post_page_text),
                        (html, body_text),
                    ],
                )
                brand = self._resolve_brand(
                    title=title,
                    shop_name=shop_name,
                    html_candidates=[pre_html, post_html, html],
                )

                def _merge_text_lists(*groups: list[str]) -> list[str]:
                    out: list[str] = []
                    seen: set[str] = set()
                    for group in groups:
                        for value in group:
                            text_value = clean_text(str(value or ""), max_len=220)
                            if not text_value or text_value in seen:
                                continue
                            seen.add(text_value)
                            out.append(text_value)
                    return out

                def _merge_image_meta(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
                    out: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for group in groups:
                        for row in group:
                            if not isinstance(row, dict):
                                continue
                            src = self._normalize_image_url(str(row.get("src", "")))
                            if not src or src in seen:
                                continue
                            seen.add(src)
                            normalized = dict(row)
                            normalized["src"] = src
                            out.append(normalized)
                    return out

                def _merge_image_urls(*groups: list[str]) -> list[str]:
                    out: list[str] = []
                    seen: set[str] = set()
                    for group in groups:
                        for value in group:
                            normalized = self._normalize_image_url(str(value or ""))
                            if not normalized or normalized in seen:
                                continue
                            if self._is_noise_image(normalized):
                                continue
                            if not self._looks_like_product_image_url(normalized):
                                continue
                            seen.add(normalized)
                            out.append(normalized)
                    return out

                runtime_payload = {
                    "title": title,
                    "shop_name": shop_name,
                    "brand": brand,
                    "price_texts": _merge_text_lists(
                        post_runtime_payload.get("price_texts", []) or [],
                        pre_runtime_payload.get("price_texts", []) or [],
                    ),
                    "sku_texts": _merge_text_lists(
                        post_runtime_payload.get("sku_texts", []) or [],
                        pre_runtime_payload.get("sku_texts", []) or [],
                    ),
                    "page_text": self._merge_page_texts(
                        post_page_text,
                        pre_page_text,
                        body_text,
                    ),
                    "image_meta": _merge_image_meta(
                        post_runtime_payload.get("image_meta", []) or [],
                        pre_runtime_payload.get("image_meta", []) or [],
                    ),
                    "image_urls": _merge_image_urls(
                        post_runtime_payload.get("image_urls", []) or [],
                        pre_runtime_payload.get("image_urls", []) or [],
                        self._extract_gallery_image_urls(post_html),
                        self._extract_gallery_image_urls(pre_html),
                    ),
                }

                detail_blocks = await self._collect_detail_blocks_from_page(
                    page,
                    item_id,
                    image_dir,
                    html=html,
                    runtime_payload=runtime_payload,
                )
                runtime_text_parts = [
                    runtime_payload.get("title", ""),
                    runtime_payload.get("shop_name", ""),
                    "\n".join(runtime_payload.get("price_texts", []) or []),
                    "\n".join(runtime_payload.get("sku_texts", []) or []),
                    runtime_payload.get("page_text", ""),
                    body_text,
                ]
                body_text = "\n".join(part for part in runtime_text_parts if part)
                image_meta = runtime_payload.get("image_meta", []) or []
                image_urls = runtime_payload.get("image_urls", []) or []

                detail = self._build_item_detail_from_raw(
                    item_id=item_id,
                    crawl_time=crawl_time,
                    title=title,
                    html=html,
                    body_text=body_text,
                    image_meta=image_meta,
                    image_urls=image_urls,
                    detail_blocks_override=detail_blocks,
                    visit_chain=visit_chain,
                )
                detail.platform = "jd"

                has_price_block = any(
                    str(block.get("source_type", "")).lower() == "text"
                    and str(block.get("content", "")).startswith("价格信息：")
                    for block in detail.detail_blocks
                )
                if detail.prices and not has_price_block:
                    next_text_index = (
                        sum(
                            1
                            for block in detail.detail_blocks
                            if str(block.get("source_type", "")).lower() == "text"
                        )
                        + 1
                    )
                    detail.detail_blocks.append(
                        {
                            "source_type": "text",
                            "source_ref": f"text_{next_text_index}",
                            "content": f"价格信息：￥{detail.prices[0]:.2f}",
                        }
                    )

                color_size_payload = self._extract_color_size_payload(html)
                if color_size_payload:
                    skus: list[dict[str, str]] = []
                    seen_sku: set[str] = set()
                    default_price = f"{detail.prices[0]:.2f}" if detail.prices else ""
                    for item in color_size_payload:
                        sku_id_value = str(item.get("skuId", "") or "").strip()
                        sku_name = clean_text(
                            str(item.get("规格", "") or item.get("name", "") or "SKU"),
                            max_len=80,
                        )
                        if not sku_id_value or sku_id_value in seen_sku:
                            continue
                        seen_sku.add(sku_id_value)
                        skus.append(
                            {
                                "sku_id": sku_id_value,
                                "sku_name": sku_name or "SKU",
                                "price": default_price,
                            }
                        )
                        if len(skus) >= 40:
                            break
                    if skus:
                        detail.skus = skus
                        sku_prices = self._prices_from_skus(detail.skus)
                        if sku_prices:
                            detail.prices = self._refine_price_values(detail.prices + sku_prices)

                if not detail.shop_name:
                    detail.shop_name = clean_text(
                        str(runtime_payload.get("shop_name", "")),
                        max_len=120,
                    )
                if not detail.shop_name:
                    detail.shop_name = self._resolve_shop_name(
                        [],
                        [
                            (pre_html, pre_page_text),
                            (post_html, post_page_text),
                            (html, body_text),
                        ],
                    )
                if not detail.main_image_url:
                    gallery_urls = self._extract_gallery_image_urls(html)
                    detail.main_image_url = next(
                        (
                            self._normalize_image_url(url)
                            for url in gallery_urls
                            if url and not self._is_noise_image(url)
                        ),
                        "",
                    )
                detail.brand = normalize_brand_name(
                    str(runtime_payload.get("brand", "") or detail.brand),
                    title=detail.title or title,
                    shop_name=detail.shop_name,
                )
                detail.title = normalize_item_title(detail.title or title, max_len=120)
                if detail.detail_blocks:
                    merged_detail_text = "\n".join(
                        str(block.get("content", "") or "")
                        for block in detail.detail_blocks
                        if str(block.get("source_type", "")).lower() == "text"
                    ).strip()
                    if merged_detail_text:
                        detail.detail_text = merged_detail_text

                if page is not None and not page.is_closed():
                    try:
                        await page.close()
                    except Exception:
                        pass
                return detail
            except Exception as exc:
                message = str(exc).lower()
                retryable = any(
                    token in message
                    for token in (
                        "net::err_aborted",
                        "net::err_connection_reset",
                        "navigation interrupted",
                        "timeout",
                        "target page, context or browser has been closed",
                    )
                )
                if page is not None and not page.is_closed():
                    try:
                        await page.close()
                    except Exception:
                        pass
                if retryable and nav_attempt < max_nav_attempts:
                    LOG.warning(
                        "Retry JD crawl for item %s after navigation error (%s/%s): %s",
                        item_id,
                        nav_attempt,
                        max_nav_attempts,
                        exc,
                    )
                    await asyncio.sleep(1.0)
                    continue
                error = f"{type(exc).__name__}: {exc}"
                LOG.error("JD crawl failed for item %s: %s", item_id, error)
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
                    platform="jd",
                    visit_chain=visit_chain,
                    error=error,
                )

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
            platform="jd",
            error="JD crawl failed: unknown error",
        )
