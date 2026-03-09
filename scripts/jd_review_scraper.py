"""Standalone JD review crawler built on top of the existing JD crawler."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any

from data import clean_text, extract_json_object, normalize_brand_name, normalize_item_title, now_iso
from jd_scraper import JDCrawler
from review_common import (
    compose_text_parts,
    filter_and_limit_reviews,
    flatten_tags,
    flatten_text,
    format_review_datetime,
    iter_nested_dicts,
    looks_like_review_dict,
    mask_user_name,
    normalize_bool,
    normalize_int,
    parse_review_datetime,
    pick_first_non_empty,
    subtract_days,
    subtract_months,
    utc_now_local,
)
from review_models import ReviewItemResult, ReviewRecord
from tools.browser_manager import get_global_browser_manager
from tools.jd_login_rules import detect_non_product_page

LOG = logging.getLogger("taobao_insight")


class JDReviewCrawler(JDCrawler):
    async def _extract_review_request_context_async(
        self,
        page: Any,
        *,
        item_id: str,
    ) -> dict[str, str]:
        default_item_id = clean_text(str(item_id or ""), max_len=40)
        script = f"""
(() => {{
  const product = ((window.pageConfig || {{}}).product || {{}});
  const cat = Array.isArray(product.cat) ? product.cat : [];
  return {{
    sku: String(product.skuid || product.mainSkuId || {json.dumps(default_item_id, ensure_ascii=True)} || ''),
    mainSkuId: String(product.mainSkuId || product.skuid || {json.dumps(default_item_id, ensure_ascii=True)} || ''),
    shopId: String(product.shopId || product.venderId || ''),
    shopType: product.isPop ? '1' : '0',
    category: cat.length ? cat.join(',') : '',
    style: '1',
  }};
}})()
"""
        try:
            payload = await page.evaluate(script)
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        sku = clean_text(str(payload.get("sku") or default_item_id), max_len=40) or default_item_id
        main_sku_id = clean_text(
            str(payload.get("mainSkuId") or sku or default_item_id),
            max_len=40,
        ) or sku
        shop_id = clean_text(str(payload.get("shopId") or ""), max_len=40)
        shop_type = clean_text(str(payload.get("shopType") or "0"), max_len=8) or "0"
        category = clean_text(str(payload.get("category") or ""), max_len=80)
        style = clean_text(str(payload.get("style") or "1"), max_len=12) or "1"
        return {
            "sku": sku,
            "mainSkuId": main_sku_id,
            "shopId": shop_id,
            "shopType": shop_type,
            "category": category,
            "style": style,
        }

    async def _ensure_item_detail_ready_async(
        self,
        page: Any,
        context: Any,
        *,
        url: str,
        item_id: str,
        stage: str,
    ) -> bool:
        login_handled = await self._handle_login_if_needed_async(
            page=page,
            context=context,
            stage=stage,
            item_id=item_id,
        )
        if login_handled:
            await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(3500, self.manual_wait_seconds * 1000))

        current_url, title, body_text = await self._read_page_snapshot_async(page)
        blocked_reason = detect_non_product_page(current_url, title, body_text)
        if blocked_reason:
            raise RuntimeError(
                f"jd review page blocked after login recovery during {stage}: {blocked_reason}. "
                f"Please ensure you are logged in and complete verification."
            )
        if not self._looks_like_item_detail_url(current_url):
            raise RuntimeError(f"unexpected JD item detail URL during {stage}: {current_url}")
        return login_handled

    @staticmethod
    def _looks_like_item_detail_url(url: str) -> bool:
        lowered = str(url or "").lower()
        return "item.jd.com/" in lowered

    async def _click_review_entry_async(
        self,
        page: Any,
        selectors: list[str],
        *,
        wait_ms: int = 1200,
        scroll_before: bool = False,
    ) -> bool:
        if scroll_before:
            try:
                await page.mouse.wheel(0, 1400)
                await page.wait_for_timeout(800)
            except Exception:
                pass
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                await locator.scroll_into_view_if_needed(timeout=1500)
            except Exception:
                pass
            try:
                locator = page.locator(selector).first
                await locator.click(timeout=2000)
                await page.wait_for_timeout(max(500, wait_ms))
                return True
            except Exception:
                pass
            try:
                locator = page.locator(selector).first
                await locator.evaluate("(node) => node.click()")
                await page.wait_for_timeout(max(500, wait_ms))
                return True
            except Exception:
                continue
        return False

    async def _click_review_text_async(
        self,
        page: Any,
        labels: list[str],
        *,
        wait_ms: int = 1200,
        scroll_before: bool = False,
    ) -> bool:
        if scroll_before:
            try:
                await page.mouse.wheel(0, 1400)
                await page.wait_for_timeout(800)
            except Exception:
                pass
        for label in labels:
            try:
                locator_factory = getattr(page, "get_by_text", None)
                if callable(locator_factory):
                    locator = locator_factory(label, exact=False).first
                    await locator.scroll_into_view_if_needed(timeout=1500)
            except Exception:
                pass
            try:
                locator_factory = getattr(page, "get_by_text", None)
                if callable(locator_factory):
                    locator = locator_factory(label, exact=False).first
                    await locator.click(timeout=2000)
                    await page.wait_for_timeout(max(500, wait_ms))
                    return True
            except Exception:
                pass
            try:
                label_json = json.dumps(label, ensure_ascii=False)
                expression = (
                    "(() => {"
                    f"const label = {label_json};"
                    """
  const isVisible = (node) => {
    if (!node) return false;
    const style = window.getComputedStyle(node);
    if (!style || style.visibility === 'hidden' || style.display === 'none') return false;
    const rect = node.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const clickableSelector = 'a,button,[role="tab"],[role="link"],[role="button"],[onclick]';
  const nodes = Array.from(
    document.querySelectorAll(
      'a,button,[role="tab"],[role="link"],[role="button"],li,span,div,p'
    )
  );
  const candidates = [];
  for (const node of nodes) {
    const text = String(node.innerText || node.textContent || '').replace(/\\s+/g, ' ').trim();
    if (!text || !text.includes(label) || !isVisible(node)) continue;
    const clickable = node.closest ? (node.closest(clickableSelector) || node) : node;
    candidates.push({
      node,
      clickable,
      text,
      exact: text === label ? 1 : 0,
      short: text.length <= Math.max(16, label.length + 8) ? 1 : 0,
      clickableTag: clickable !== node ? 1 : 0,
      length: text.length,
    });
  }
  candidates.sort((left, right) => {
    if (left.exact !== right.exact) return right.exact - left.exact;
    if (left.short !== right.short) return right.short - left.short;
    if (left.clickableTag !== right.clickableTag) return right.clickableTag - left.clickableTag;
    return left.length - right.length;
  });
  for (const candidate of candidates.slice(0, 20)) {
    const target = candidate.clickable || candidate.node;
    target.scrollIntoView({ block: 'center', inline: 'center' });
    if (typeof target.click === 'function') target.click();
    return true;
  }
  return false;
})()
"""
                )
                clicked = await page.evaluate(expression)
                if clicked:
                    await page.wait_for_timeout(max(500, wait_ms))
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _is_review_response_url(url: str) -> bool:
        lowered = str(url or "").strip().lower()
        return any(
            token in lowered
            for token in (
                "club.jd.com",
                "comment",
                "productpagecomments",
                "commentsummary",
                "getcommentlistwithcard",
                "pc-club",
            )
        )

    async def _extract_product_metadata_async(self, page: Any) -> tuple[str, str, str]:
        runtime_payload = await self._extract_runtime_payload(page)
        current_url, fallback_title, body_text = await self._read_page_snapshot_async(page)
        try:
            html = await page.content()
        except Exception:
            html = ""
        title = normalize_item_title(
            str(runtime_payload.get("title") or fallback_title or ""),
            max_len=120,
        )
        shop_name = self._resolve_shop_name(
            [str(runtime_payload.get("shop_name") or "")],
            [(html, body_text)],
        )
        brand = self._resolve_brand(title, shop_name, [html, body_text, current_url])
        brand = normalize_brand_name(brand, title=title, shop_name=shop_name)
        return title, shop_name, brand

    async def _invoke_signed_client_action_async(
        self,
        page: Any,
        *,
        function_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        function_json = json.dumps(str(function_id or "").strip(), ensure_ascii=True)
        body_json = json.dumps(body, ensure_ascii=True, separators=(",", ":"))
        script = f"""
(async () => {{
  const functionId = {function_json};
  const bodyPayload = {body_json};
  const getCookie = (name) => {{
    const prefix = `${{name}}=`;
    const hit = String(document.cookie || '').split('; ').find((item) => item.startsWith(prefix));
    return hit ? decodeURIComponent(hit.slice(prefix.length)) : '';
  }};
  const getToken = async () => {{
    if (typeof window.getJsToken !== 'function') return '';
    return await new Promise((resolve) => {{
      let done = false;
      const finish = (value) => {{
        if (done) return;
        done = true;
        if (value && typeof value === 'object') {{
          resolve(String(value.jsToken || ''));
          return;
        }}
        resolve(String(value || ''));
      }};
      try {{
        window.getJsToken((value) => finish(value), 3000);
      }} catch (err) {{
        finish('');
      }}
      setTimeout(() => finish(''), 3200);
    }});
  }};
  const uuid = ((getCookie('__jda') || '').split('.')[1]) || '';
  const token = await getToken();
  let params = {{
    functionId,
    appid: 'pc-rate-qa',
    t: Date.now(),
    uuid,
    loginType: '3',
    'x-api-eid-token': token,
    client: 'pc',
    clientVersion: '1.0.0',
    body: JSON.stringify({{ requestSource: 'pc', ...bodyPayload }}),
  }};
  let signed = {{ ...params }};
  let signError = '';
  if (window.ParamsSign) {{
    try {{
      signed = await new window.ParamsSign({{ appId: '01a47', debug: false }}).sign(params);
    }} catch (err) {{
      signError = String((err && err.message) || err || '');
    }}
  }}
  const form = new URLSearchParams();
  for (const [key, value] of Object.entries(signed)) {{
    if (value === undefined || value === null) continue;
    form.append(key, String(value));
  }}
  const response = await fetch('https://api.m.jd.com/client.action', {{
    method: 'POST',
    credentials: 'include',
    headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
    body: form.toString(),
  }});
  const text = await response.text();
  let data = null;
  try {{
    data = JSON.parse(text);
  }} catch (err) {{
    data = {{
      code: String(response.status || ''),
      parse_error: String((err && err.message) || err || ''),
      raw: String(text || '').slice(0, 2000),
    }};
  }}
  return {{
    status: Number(response.status || 0),
    uuid,
    tokenLength: String(token || '').length,
    signError,
    data,
  }};
}})()
"""
        try:
            result = await page.evaluate(script)
        except Exception as exc:
            return {"status": 0, "data": {"code": "-1", "message": f"{type(exc).__name__}: {exc}"}}
        return result if isinstance(result, dict) else {}

    async def _fetch_comment_list_page_async(
        self,
        page: Any,
        *,
        item_id: str,
        page_num: int,
        page_size: int = 10,
        sort_type: str = "6",
    ) -> dict[str, Any] | None:
        context = await self._extract_review_request_context_async(page, item_id=item_id)
        sku = clean_text(str(context.get("sku") or item_id), max_len=40)
        main_sku_id = clean_text(str(context.get("mainSkuId") or sku), max_len=40) or sku
        category = clean_text(str(context.get("category") or ""), max_len=80)
        if not sku or not category:
            return None
        request_body = {
            "shopComment": 0,
            "sameComment": 0,
            "channel": None,
            "extInfo": {
                "aigc": "0",
                "isQzc": "0",
                "spuId": main_sku_id,
                "commentRate": "1",
                "needTopAlbum": "1",
                "bbtf": "",
                "userGroupComment": "1",
            },
            "num": str(max(1, int(page_size))),
            "pictureCommentType": "A",
            "scval": None,
            "shadowMainSku": "0",
            "shopType": str(context.get("shopType") or "0"),
            "shopId": str(context.get("shopId") or ""),
            "sku": sku,
            "category": category,
            "shieldCurrentComment": False,
            "pageSize": str(max(1, int(page_size))),
            "isFirstRequest": int(page_num) == 1,
            "style": str(context.get("style") or "1"),
            "isCurrentSku": True,
            "sortType": clean_text(str(sort_type or "6"), max_len=12) or "6",
            "tagId": "",
            "tagType": "",
            "type": "0",
            "pageNum": str(max(1, int(page_num))),
        }
        response = await self._invoke_signed_client_action_async(
            page,
            function_id="getCommentListPage",
            body=request_body,
        )
        payload = response.get("data")
        return payload if isinstance(payload, dict) else None

    def _extract_reviews_from_comment_list_payload(
        self,
        payload: dict[str, Any],
        *,
        item_id: str,
        item_url: str,
        title: str,
        shop_name: str,
        brand: str,
    ) -> list[ReviewRecord]:
        records: list[ReviewRecord] = []
        result = payload.get("result")
        if not isinstance(result, dict):
            return records
        floors = result.get("floors")
        if not isinstance(floors, list):
            return records
        for floor in floors:
            if not isinstance(floor, dict) or str(floor.get("mId") or "") != "commentlist-list":
                continue
            rows = floor.get("data")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                comment_row = row.get("commentInfo")
                if not isinstance(comment_row, dict):
                    continue
                record = self._record_from_api_dict(
                    comment_row,
                    item_id=item_id,
                    item_url=item_url,
                    title=title,
                    shop_name=shop_name,
                    brand=brand,
                )
                if record is not None:
                    records.append(record)
        return records

    @staticmethod
    def _extract_comment_list_max_page(payload: dict[str, Any]) -> int:
        result = payload.get("result")
        if not isinstance(result, dict):
            return 0
        page_info = result.get("pageInfo")
        if isinstance(page_info, dict):
            data = page_info.get("data")
            if isinstance(data, dict):
                try:
                    return max(0, int(str(data.get("maxPage") or "0").strip() or "0"))
                except ValueError:
                    return 0
        return 0

    @staticmethod
    def _reviews_older_than_cutoff(records: list[ReviewRecord], cutoff: Any) -> bool:
        parsed_values = [
            parse_review_datetime(record.comment_time)
            for record in records
            if isinstance(record, ReviewRecord)
        ]
        parsed_values = [value for value in parsed_values if value is not None]
        if not parsed_values:
            return False
        return all(value < cutoff for value in parsed_values[: min(5, len(parsed_values))])

    async def _collect_reviews_via_comment_api_async(
        self,
        *,
        page: Any,
        context: Any,
        url: str,
        item_id: str,
        title: str,
        shop_name: str,
        brand: str,
        cutoff: Any,
        limit: int,
    ) -> tuple[list[ReviewRecord], str, bool]:
        all_reviews: dict[str, ReviewRecord] = {}
        effective_limit = int(limit)
        max_pages_hint = 200 if effective_limit <= 0 else min(200, max(6, math.ceil(effective_limit / 10) + 6))
        api_supported = False
        stop_reason = "api_unavailable"
        max_page = 0

        for page_num in range(1, max_pages_hint + 1):
            await self._ensure_item_detail_ready_async(
                page=page,
                context=context,
                url=url,
                item_id=item_id,
                stage=f"review-api-page-{page_num}",
            )
            payload = await self._fetch_comment_list_page_async(
                page,
                item_id=item_id,
                page_num=page_num,
                page_size=10,
                sort_type="6",
            )
            if not isinstance(payload, dict):
                if page_num == 1:
                    return [], "api_unavailable", False
                stop_reason = "pagination_exhausted"
                break
            if str(payload.get("code") or "") != "0":
                if page_num == 1:
                    return [], "api_unavailable", False
                stop_reason = "pagination_exhausted"
                break

            api_supported = True
            page_records = self._extract_reviews_from_comment_list_payload(
                payload,
                item_id=item_id,
                item_url=url,
                title=title,
                shop_name=shop_name,
                brand=brand,
            )
            max_page = max(max_page, self._extract_comment_list_max_page(payload))

            for record in filter_and_limit_reviews(
                page_records + list(all_reviews.values()),
                cutoff=cutoff,
                limit=(effective_limit * 3) if effective_limit > 0 else 0,
            ):
                all_reviews[record.identity()] = record

            final_reviews = filter_and_limit_reviews(
                list(all_reviews.values()),
                cutoff=cutoff,
                limit=effective_limit,
            )
            if effective_limit > 0 and len(final_reviews) >= effective_limit:
                stop_reason = "limit_reached"
                return final_reviews, stop_reason, api_supported
            if self._reviews_older_than_cutoff(page_records, cutoff) and final_reviews:
                stop_reason = "reached_cutoff"
                return final_reviews, stop_reason, api_supported
            if not page_records:
                stop_reason = "no_more_review_pages" if page_num == 1 else "pagination_exhausted"
                break
            if max_page > 0 and page_num >= max_page:
                stop_reason = "pagination_exhausted"
                break

        final_reviews = filter_and_limit_reviews(
            list(all_reviews.values()),
            cutoff=cutoff,
            limit=effective_limit,
        )
        if final_reviews and stop_reason == "api_unavailable":
            stop_reason = "pagination_exhausted"
        if not final_reviews and api_supported and stop_reason == "api_unavailable":
            stop_reason = "no_reviews_found"
        return final_reviews, stop_reason, api_supported


    async def _open_review_surface_async(self, page: Any) -> bool:
        primary_opened = await self._click_review_text_async(
            page,
            [
                "\u5927\u5bb6\u8bc4",
                "\u5546\u54c1\u8bc4\u4ef7",
                "\u4e70\u5bb6\u8bc4\u4ef7",
                "\u7d2f\u8ba1\u8bc4\u4ef7",
            ],
            wait_ms=max(1200, self.manual_wait_seconds * 1000),
        )
        if not primary_opened:
            primary_opened = await self._click_review_entry_async(
                page,
                [
                    "#comment",
                    "[href*='#comment']",
                ],
                wait_ms=max(1200, self.manual_wait_seconds * 1000),
            )

        opened_all_reviews = await self._click_review_text_async(
            page,
            [
                "\u5168\u90e8\u8bc4\u4ef7",
                "\u67e5\u770b\u5168\u90e8\u8bc4\u4ef7",
                "\u8d5e\u4e0d\u7edd\u53e3",
            ],
            wait_ms=max(1500, self.manual_wait_seconds * 1000),
            scroll_before=True,
        )
        if not opened_all_reviews:
            opened_all_reviews = await self._click_review_entry_async(
                page,
                [
                    "a[href*='comment']",
                    "[data-anchor*='comment']",
                    "[href*='#comment']",
                ],
                wait_ms=max(1500, self.manual_wait_seconds * 1000),
                scroll_before=True,
            )

        if primary_opened or opened_all_reviews:
            try:
                await page.mouse.wheel(0, 1800)
                await page.wait_for_timeout(1200)
            except Exception:
                pass
            return True
        try:
            await page.mouse.wheel(0, 2200)
            await page.wait_for_timeout(1200)
        except Exception:
            pass
        return False

    async def _extract_dom_review_rows_async(self, page: Any) -> list[dict[str, Any]]:
        script = """
(() => {
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
  const pick = (root, selectors) => {
    for (const selector of selectors) {
      const found = root.querySelector(selector);
      if (found) return found;
    }
    return null;
  };
  const selectors = [
    '.jdc-pc-rate-card',
    '[class*="rate-card"]',
    '#comment .comment-item',
    '.comment-item',
    'li.comment-item',
    '[class*="comment-item"]',
    '[class*="CommentItem"]',
    '[data-testid*="comment"]'
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
  return roots.slice(0, 120).map((root, index) => {
    const contentNode = pick(root, [
      '.jdc-pc-rate-card-main-desc',
      '.comment-con',
      '.comment-content',
      '[class*="comment-con"]',
      '[class*="commentContent"]',
      'p'
    ]);
    const appendNode = pick(root, [
      '.comment-after',
      '.append-comment',
      '[class*="append"]',
      '[class*="after"]'
    ]);
    const timeNode = pick(root, [
      '.date.list',
      '.date',
      '.order-info',
      '.comment-time',
      '[class*="time"]',
      '.date'
    ]);
    const skuNode = pick(root, [
      '.info',
      '.jdc-pc-rate-card-info .info',
      '.comment-sku',
      '[class*="sku"]',
      '[class*="spec"]',
      '.order-info'
    ]);
    const userNode = pick(root, [
      '.jdc-pc-rate-card-nick',
      '.user-info',
      '.user-name',
      '[class*="user"]',
      '.name'
    ]);
    const levelNode = pick(root, [
      '.user-level',
      '[class*="level"]',
      '[class*="vip"]'
    ]);
    const idValue =
      attr(root, ['data-guid', 'data-id', 'data-comment-id', 'data-index', 'data-item-index'])
      || String(index + 1);
    const scoreNodes = root.querySelectorAll('.star, [class*="star"], [class*="score"], img[alt="star"]');
    const mediaNodes = Array.from(root.querySelectorAll('img, video'));
    const countNodes = Array.from(root.querySelectorAll('.jdc-count'))
      .map((node) => text(node))
      .filter(Boolean);
    return {
      comment_id: idValue,
      comment_text: text(contentNode) || text(root),
      append_text: text(appendNode),
      comment_time: text(timeNode),
      sku_info: text(skuNode),
      user_name: text(userNode),
      user_level: text(levelNode),
      rating: scoreNodes.length ? '5' : '',
      like_count: countNodes.length ? countNodes[countNodes.length - 1] : '',
      reply_count: countNodes.length > 1 ? countNodes[countNodes.length - 2] : '',
      has_images: root.querySelectorAll('img').length > 0,
      has_video: root.querySelectorAll('video, [class*="video"]').length > 0,
      raw_tags: mediaNodes.map((node) => text(node)).filter(Boolean),
      raw_source: text(root),
    };
  }).filter((row) => row.comment_text || row.append_text || row.comment_time);
})()
"""
        try:
            rows = await page.evaluate(script)
        except Exception:
            return []
        return rows if isinstance(rows, list) else []

    def _record_from_dom_row(
        self,
        row: dict[str, Any],
        *,
        item_id: str,
        item_url: str,
        title: str,
        shop_name: str,
        brand: str,
    ) -> ReviewRecord | None:
        text = flatten_text(row.get("comment_text"), max_len=1200)
        append_text = flatten_text(row.get("append_text"), max_len=600)
        if not text and not append_text:
            return None
        parsed_time = parse_review_datetime(row.get("comment_time"))
        return ReviewRecord(
            platform="jd",
            item_id=item_id,
            item_url=item_url,
            title=title,
            shop_name=shop_name,
            brand=brand,
            comment_id=clean_text(str(row.get("comment_id") or ""), max_len=120),
            comment_time=format_review_datetime(parsed_time),
            comment_text=text,
            rating=clean_text(str(row.get("rating") or ""), max_len=20),
            sku_info=clean_text(str(row.get("sku_info") or ""), max_len=200),
            user_name_masked=mask_user_name(row.get("user_name")),
            user_level=clean_text(str(row.get("user_level") or ""), max_len=60),
            is_anonymous=normalize_bool(row.get("is_anonymous")),
            is_append=bool(append_text),
            append_time="",
            append_text=append_text,
            like_count=normalize_int(row.get("like_count")),
            reply_count=normalize_int(row.get("reply_count")),
            has_images=normalize_bool(row.get("has_images")),
            has_video=normalize_bool(row.get("has_video")),
            raw_tags=flatten_tags(row.get("raw_tags")),
            raw_source={"source": "dom", "snippet": clean_text(str(row.get("raw_source") or ""), max_len=300)},
            collected_at=now_iso(),
        )

    def _record_from_api_dict(
        self,
        row: dict[str, Any],
        *,
        item_id: str,
        item_url: str,
        title: str,
        shop_name: str,
        brand: str,
    ) -> ReviewRecord | None:
        text = flatten_text(
            pick_first_non_empty(
                row,
                "content",
                "commentData",
                "commentText",
                "rateContent",
                "feedback",
                "text",
            ),
            max_len=1200,
        )
        append_payload = pick_first_non_empty(row, "appendComment", "afterComment", "append")
        append_text = ""
        append_time = ""
        if isinstance(append_payload, dict):
            append_text = flatten_text(
                pick_first_non_empty(
                    append_payload,
                    "content",
                    "commentData",
                    "commentText",
                    "rateContent",
                    "feedback",
                ),
                max_len=600,
            )
            append_time = format_review_datetime(
                parse_review_datetime(
                    pick_first_non_empty(
                        append_payload,
                        "creationTime",
                        "commentTime",
                        "rateDate",
                        "time",
                        "date",
                    )
                )
            )
        if not text and not append_text:
            return None

        comment_time = format_review_datetime(
            parse_review_datetime(
                pick_first_non_empty(
                    row,
                    "creationTime",
                    "commentTime",
                    "commentDate",
                    "rateDate",
                    "time",
                    "date",
                )
            )
        )
        sku_info = compose_text_parts(
            [
                row.get("referenceName"),
                row.get("productColor"),
                row.get("productSize"),
                row.get("skuInfo"),
                row.get("specs"),
                row.get("wareAttribute"),
            ],
            max_len=200,
        )
        images_value = pick_first_non_empty(row, "images", "imageList", "picList", "pictureInfoList")
        video_value = pick_first_non_empty(row, "videos", "video", "videoInfo")
        return ReviewRecord(
            platform="jd",
            item_id=item_id,
            item_url=item_url,
            title=title,
            shop_name=shop_name,
            brand=brand,
            comment_id=clean_text(
                str(
                    pick_first_non_empty(
                        row,
                        "commentId",
                        "id",
                        "guid",
                        "comment_id",
                    )
                    or ""
                ),
                max_len=120,
            ),
            comment_time=comment_time,
            comment_text=text,
            rating=clean_text(
                str(pick_first_non_empty(row, "score", "commentScore", "star", "starLevel") or ""),
                max_len=20,
            ),
            sku_info=sku_info,
            user_name_masked=mask_user_name(
                pick_first_non_empty(
                    row,
                    "displayUserNick",
                    "nickname",
                    "userNickName",
                    "nickName",
                    "nick",
                )
            ),
            user_level=clean_text(
                str(
                    pick_first_non_empty(
                        row,
                        "userLevelName",
                        "plusLevel",
                        "userLevel",
                        "memberLevel",
                    )
                    or ""
                ),
                max_len=60,
            ),
            is_anonymous=normalize_bool(
                pick_first_non_empty(row, "anonymousFlag", "anonymous", "anony", "isAnonymous")
            ),
            is_append=bool(append_text),
            append_time=append_time,
            append_text=append_text,
            like_count=normalize_int(
                pick_first_non_empty(row, "usefulVoteCount", "praiseCount", "likeCount")
            ),
            reply_count=normalize_int(pick_first_non_empty(row, "replyCount", "replyCnt")),
            has_images=isinstance(images_value, list) and len(images_value) > 0,
            has_video=bool(video_value),
            raw_tags=flatten_tags(
                pick_first_non_empty(row, "commentTags", "tagList", "tags", "labels")
            ),
            raw_source={"source": "api", "row": row},
            collected_at=now_iso(),
        )

    def _extract_reviews_from_payloads(
        self,
        payload_texts: list[str],
        *,
        item_id: str,
        item_url: str,
        title: str,
        shop_name: str,
        brand: str,
    ) -> list[ReviewRecord]:
        records: list[ReviewRecord] = []
        for payload_text in payload_texts:
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except Exception:
                payload = extract_json_object(payload_text)
            if payload is None:
                continue
            for row in iter_nested_dicts(payload):
                if not looks_like_review_dict(row):
                    continue
                record = self._record_from_api_dict(
                    row,
                    item_id=item_id,
                    item_url=item_url,
                    title=title,
                    shop_name=shop_name,
                    brand=brand,
                )
                if record is not None:
                    records.append(record)
        return records

    @staticmethod
    def _rows_older_than_cutoff(rows: list[dict[str, Any]], cutoff: Any) -> bool:
        parsed_values = [
            parse_review_datetime(row.get("comment_time"))
            for row in rows
            if isinstance(row, dict)
        ]
        parsed_values = [value for value in parsed_values if value is not None]
        if len(parsed_values) < 3:
            return False
        return all(value < cutoff for value in parsed_values[: min(5, len(parsed_values))])


    async def _advance_review_surface_async(self, page: Any) -> bool:
        if await self._click_review_text_async(
            page,
            ["\u4e0b\u4e00\u9875"],
            wait_ms=1800,
        ):
            return True
        next_selectors = [
            ".ui-pager-next",
            "a.ui-pager-next",
            "a.fp-next",
        ]
        for selector in next_selectors:
            try:
                locator = page.locator(selector).first
                classes = str(await locator.get_attribute("class") or "").lower()
                if "disabled" in classes:
                    continue
                await locator.click(timeout=1500)
                await page.wait_for_timeout(1800)
                return True
            except Exception:
                continue
        try:
            await page.mouse.wheel(0, 2200)
            await page.wait_for_timeout(1500)
            return True
        except Exception:
            return False

    async def collect_reviews_async(
        self,
        *,
        url: str,
        item_id: str,
        months: int,
        days: int,
        limit: int,
        target_name: str,
    ) -> ReviewItemResult:
        _ = target_name
        effective_days = max(0, int(days))
        effective_limit = int(limit)
        cutoff = subtract_days(utc_now_local(), effective_days) if effective_days > 0 else subtract_months(utc_now_local(), max(1, int(months)))
        login_event_start = len(self.login_recovery_events)
        page = None
        response_texts: list[str] = []
        response_tasks: set[asyncio.Task[Any]] = set()

        if self._global_browser_manager is None:
            self._global_browser_manager = await get_global_browser_manager()
        context = self._global_browser_manager.browser_context
        if context is None:
            context = await self._global_browser_manager.initialize(
                browser_mode=self.browser_mode,
                headless=self.headless,
                cdp_url=self.cdp_url,
                user_data_dir=str(self.user_data_dir),
            )
        await self._restore_storage_state_async(context)

        try:
            page = await self._global_browser_manager.new_page()

            async def _capture_response(resp: Any) -> None:
                if not self._is_review_response_url(getattr(resp, "url", "")):
                    return
                try:
                    text = await resp.text()
                except Exception:
                    return
                if text:
                    response_texts.append(text[:800000])

            def _on_response(resp: Any) -> None:
                task = asyncio.create_task(_capture_response(resp))
                response_tasks.add(task)
                task.add_done_callback(lambda done: response_tasks.discard(done))

            page.on("response", _on_response)
            await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(3500, self.manual_wait_seconds * 1000))
            await self._ensure_item_detail_ready_async(
                page=page,
                context=context,
                url=url,
                item_id=item_id,
                stage="review-item-initial",
            )

            title, shop_name, brand = await self._extract_product_metadata_async(page)
            api_reviews, api_stop_reason, api_supported = await self._collect_reviews_via_comment_api_async(
                page=page,
                context=context,
                url=url,
                item_id=item_id,
                title=title,
                shop_name=shop_name,
                brand=brand,
                cutoff=cutoff,
                limit=effective_limit,
            )
            if api_supported and api_reviews:
                await self._persist_storage_state_async(context)
                return ReviewItemResult(
                    platform="jd",
                    item_id=item_id,
                    item_url=url,
                    title=title,
                    shop_name=shop_name,
                    brand=brand,
                    reviews=api_reviews,
                    collected_count=len(api_reviews),
                    cutoff_time=cutoff.isoformat(timespec="seconds"),
                    stopped_reason=api_stop_reason,
                    login_recovery_events=self.login_recovery_events[login_event_start:],
                )

            await self._open_review_surface_async(page)
            login_handled = await self._ensure_item_detail_ready_async(
                page=page,
                context=context,
                url=url,
                item_id=item_id,
                stage="review-open-surface",
            )
            if login_handled:
                await self._open_review_surface_async(page)
                await self._ensure_item_detail_ready_async(
                    page=page,
                    context=context,
                    url=url,
                    item_id=item_id,
                    stage="review-open-surface-retry",
                )

            all_reviews: dict[str, ReviewRecord] = {}
            response_cursor = 0
            stale_rounds = 0
            stop_reason = "no_reviews_found"
            if effective_limit <= 0:
                max_rounds = 40
            else:
                max_rounds = max(4, min(12, (max(1, effective_limit) // 10) + 4))

            for _ in range(max_rounds):
                await page.wait_for_timeout(max(1600, self.manual_wait_seconds * 1000))
                login_handled = await self._ensure_item_detail_ready_async(
                    page=page,
                    context=context,
                    url=url,
                    item_id=item_id,
                    stage="review-loop",
                )
                if login_handled:
                    await self._open_review_surface_async(page)
                    await self._ensure_item_detail_ready_async(
                        page=page,
                        context=context,
                        url=url,
                        item_id=item_id,
                        stage="review-loop-retry",
                    )
                if response_tasks:
                    await asyncio.gather(*list(response_tasks), return_exceptions=True)
                dom_rows = await self._extract_dom_review_rows_async(page)
                records = self._extract_reviews_from_payloads(
                    response_texts[response_cursor:],
                    item_id=item_id,
                    item_url=url,
                    title=title,
                    shop_name=shop_name,
                    brand=brand,
                )
                response_cursor = len(response_texts)
                for row in dom_rows:
                    record = self._record_from_dom_row(
                        row,
                        item_id=item_id,
                        item_url=url,
                        title=title,
                        shop_name=shop_name,
                        brand=brand,
                    )
                    if record is not None:
                        records.append(record)
                before = len(all_reviews)
                for record in filter_and_limit_reviews(
                    records + list(all_reviews.values()),
                    cutoff=cutoff,
                    limit=(effective_limit * 3) if effective_limit > 0 else 0,
                ):
                    all_reviews[record.identity()] = record
                if len(all_reviews) == before:
                    stale_rounds += 1
                else:
                    stale_rounds = 0

                final_reviews = filter_and_limit_reviews(
                    list(all_reviews.values()),
                    cutoff=cutoff,
                    limit=effective_limit,
                )
                if effective_limit > 0 and len(final_reviews) >= effective_limit:
                    stop_reason = "limit_reached"
                    all_reviews = {record.identity(): record for record in final_reviews}
                    break
                if self._rows_older_than_cutoff(dom_rows, cutoff) and final_reviews:
                    stop_reason = "reached_cutoff"
                    all_reviews = {record.identity(): record for record in final_reviews}
                    break
                advanced = await self._advance_review_surface_async(page)
                if not advanced or stale_rounds >= 2:
                    stop_reason = "pagination_exhausted" if advanced else "no_more_review_pages"
                    break

            await self._persist_storage_state_async(context)
            return ReviewItemResult(
                platform="jd",
                item_id=item_id,
                item_url=url,
                title=title,
                shop_name=shop_name,
                brand=brand,
                reviews=filter_and_limit_reviews(
                    list(all_reviews.values()),
                    cutoff=cutoff,
                    limit=effective_limit,
                ),
                collected_count=len(filter_and_limit_reviews(list(all_reviews.values()), cutoff=cutoff, limit=effective_limit)),
                cutoff_time=cutoff.isoformat(timespec="seconds"),
                stopped_reason=stop_reason if stop_reason != "no_reviews_found" else api_stop_reason if api_supported else stop_reason,
                login_recovery_events=self.login_recovery_events[login_event_start:],
            )
        except Exception as exc:
            return ReviewItemResult(
                platform="jd",
                item_id=item_id,
                item_url=url,
                cutoff_time=cutoff.isoformat(timespec="seconds"),
                login_recovery_events=self.login_recovery_events[login_event_start:],
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if page is not None and not page.is_closed():
                try:
                    await page.close()
                except Exception:
                    pass
