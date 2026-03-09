"""Standalone Taobao/Tmall review crawler built on top of the existing crawler."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from data import clean_text, extract_json_object, normalize_brand_name, normalize_item_title, now_iso
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
from scraper import Crawler
from tools.browser_manager import get_global_browser_manager

LOG = logging.getLogger("taobao_insight")


class TaobaoReviewCrawler(Crawler):
    @staticmethod
    def _is_review_response_url(url: str) -> bool:
        lowered = str(url or "").strip().lower()
        return any(
            token in lowered
            for token in (
                "rate",
                "review",
                "feedback",
                "comment",
            )
        ) and any(domain in lowered for domain in ("taobao.com", "tmall.com", "alicdn.com"))

    async def _extract_product_metadata_async(self, page: Any) -> tuple[str, str, str]:
        _, fallback_title, body_text = await self._read_page_snapshot_async(page)
        try:
            html = await page.content()
        except Exception:
            html = ""
        title = normalize_item_title(fallback_title, max_len=120)
        shop_name = self._extract_shop_name(html, body_text)
        brand = normalize_brand_name(self._extract_brand(title, html), title=title, shop_name=shop_name)
        return title, shop_name, brand

    async def _open_review_surface_async(self, page: Any) -> bool:
        selectors = [
            "text=累计评论",
            "text=宝贝评价",
            "text=评价",
            "text=商品评价",
            "a:has-text('累计评论')",
            "a:has-text('宝贝评价')",
            "a:has-text('评价')",
            "[href*='review']",
            "[href*='rate']",
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                await locator.click(timeout=2000)
                await page.wait_for_timeout(max(1200, self.manual_wait_seconds * 1000))
                return True
            except Exception:
                continue
        try:
            await page.mouse.wheel(0, 2400)
            await page.wait_for_timeout(1200)
        except Exception:
            pass
        return False

    async def _extract_dom_review_rows_async(self, page: Any) -> list[dict[str, Any]]:
        script = """
() => {
  const text = (node) => String((node && (node.innerText || node.textContent)) || '')
    .replace(/\\s+/g, ' ')
    .trim();
  const pick = (root, selectors) => {
    for (const selector of selectors) {
      const found = root.querySelector(selector);
      if (found) return found;
    }
    return null;
  };
  const selectors = [
    '.rate-grid tr',
    '.tm-rate-fulltxt',
    '.review-list .review-item',
    '.tb-rev-item',
    '[class*="review-item"]',
    '[class*="rate-item"]'
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
      '.tm-rate-content',
      '.rate-content',
      '.review-content',
      '[class*="content"]',
      'p'
    ]);
    const appendNode = pick(root, [
      '.append',
      '.tm-append',
      '[class*="append"]'
    ]);
    const timeNode = pick(root, [
      '.tm-rate-date',
      '.rate-date',
      '[class*="date"]',
      '[class*="time"]'
    ]);
    const skuNode = pick(root, [
      '.tm-rate-sku',
      '.rate-sku',
      '[class*="sku"]',
      '[class*="auction"]'
    ]);
    const userNode = pick(root, [
      '.tm-rate-user-info',
      '.rate-user-info',
      '[class*="user"]',
      '[class*="nick"]'
    ]);
    return {
      comment_id: String(index + 1),
      comment_text: text(contentNode) || text(root),
      append_text: text(appendNode),
      comment_time: text(timeNode),
      sku_info: text(skuNode),
      user_name: text(userNode),
      has_images: root.querySelectorAll('img').length > 0,
      has_video: root.querySelectorAll('video, [class*="video"]').length > 0,
      raw_source: text(root),
    };
  }).filter((row) => row.comment_text || row.append_text || row.comment_time);
}
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
            platform="taobao",
            item_id=item_id,
            item_url=item_url,
            title=title,
            shop_name=shop_name,
            brand=brand,
            comment_id=clean_text(str(row.get("comment_id") or ""), max_len=120),
            comment_time=format_review_datetime(parsed_time),
            comment_text=text,
            rating="",
            sku_info=clean_text(str(row.get("sku_info") or ""), max_len=200),
            user_name_masked=mask_user_name(row.get("user_name")),
            user_level="",
            is_anonymous=False,
            is_append=bool(append_text),
            append_time="",
            append_text=append_text,
            like_count=normalize_int(row.get("like_count")),
            reply_count=normalize_int(row.get("reply_count")),
            has_images=normalize_bool(row.get("has_images")),
            has_video=normalize_bool(row.get("has_video")),
            raw_tags=[],
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
                "rateContent",
                "content",
                "feedback",
                "commentText",
                "text",
            ),
            max_len=1200,
        )
        append_payload = pick_first_non_empty(row, "appendComment", "append", "appendRate")
        append_text = ""
        append_time = ""
        if isinstance(append_payload, dict):
            append_text = flatten_text(
                pick_first_non_empty(
                    append_payload,
                    "content",
                    "rateContent",
                    "feedback",
                    "commentText",
                ),
                max_len=600,
            )
            append_time = format_review_datetime(
                parse_review_datetime(
                    pick_first_non_empty(
                        append_payload,
                        "time",
                        "date",
                        "commentTime",
                        "rateDate",
                    )
                )
            )
        if not text and not append_text:
            return None
        comment_time = format_review_datetime(
            parse_review_datetime(
                pick_first_non_empty(
                    row,
                    "rateDate",
                    "commentTime",
                    "date",
                    "time",
                )
            )
        )
        return ReviewRecord(
            platform="taobao",
            item_id=item_id,
            item_url=item_url,
            title=title,
            shop_name=shop_name,
            brand=brand,
            comment_id=clean_text(
                str(pick_first_non_empty(row, "rateId", "commentId", "id") or ""),
                max_len=120,
            ),
            comment_time=comment_time,
            comment_text=text,
            rating=clean_text(str(pick_first_non_empty(row, "rateScore", "score", "star") or ""), max_len=20),
            sku_info=compose_text_parts(
                [
                    row.get("auctionSku"),
                    row.get("skuInfo"),
                    row.get("spec"),
                ],
                max_len=200,
            ),
            user_name_masked=mask_user_name(
                pick_first_non_empty(row, "displayUserNick", "nick", "nickName", "userNick")
            ),
            user_level=clean_text(str(pick_first_non_empty(row, "memberLevel", "userLevel") or ""), max_len=60),
            is_anonymous=normalize_bool(
                pick_first_non_empty(row, "anonymous", "anonymousFlag", "isAnonymous")
            ),
            is_append=bool(append_text),
            append_time=append_time,
            append_text=append_text,
            like_count=normalize_int(pick_first_non_empty(row, "useful", "likeCount", "praiseCount")),
            reply_count=normalize_int(pick_first_non_empty(row, "replyCount", "replyCnt")),
            has_images=bool(pick_first_non_empty(row, "pics", "images", "photos")),
            has_video=bool(pick_first_non_empty(row, "video", "videos")),
            raw_tags=flatten_tags(pick_first_non_empty(row, "tags", "tagList", "labels")),
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
        next_selectors = [
            "button:has-text('下一页')",
            "a:has-text('下一页')",
            ".next-page",
            ".tm-rate-page .next",
            ".pagination-next",
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
            await page.mouse.wheel(0, 2400)
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

            login_handled = await self._handle_login_if_needed_async(
                page=page,
                context=context,
                stage="review-item-initial",
                item_id=item_id,
            )
            if login_handled:
                await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                await page.wait_for_timeout(max(3500, self.manual_wait_seconds * 1000))

            title, shop_name, brand = await self._extract_product_metadata_async(page)
            await self._open_review_surface_async(page)

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
                platform="taobao",
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
                stopped_reason=stop_reason,
                login_recovery_events=self.login_recovery_events[login_event_start:],
            )
        except Exception as exc:
            return ReviewItemResult(
                platform="taobao",
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
