"""Service layer for standalone review collection workflows."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

from data import now_iso

LOG = logging.getLogger("taobao_insight")

from review_common import (
    filter_and_limit_reviews,
    subtract_days,
    subtract_months,
    summarize_reason_counts,
    utc_now_local,
)
from review_export import ReviewExporter
from review_models import ReviewItemResult, ReviewRunSummary


class ReviewCollectionService:
    def __init__(self, crawler: Any, *, platform: str, data_dir: str | Path) -> None:
        self.crawler = crawler
        self.platform = str(platform or "").strip().lower()
        self.data_dir = Path(data_dir)

    async def run_async(
        self,
        *,
        target_name: str,
        targets: list[Any],
        months: int,
        days: int,
        limit: int,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if not targets:
            raise ValueError("at least one item URL/ID is required")
        effective_months = max(1, int(months))
        effective_days = max(0, int(days))
        effective_limit = int(limit)
        now = utc_now_local()
        cutoff = subtract_days(now, effective_days) if effective_days > 0 else subtract_months(now, effective_months)
        default_output_name = target_name
        if len(targets) == 1:
            candidate_item_id = str(getattr(targets[0], "item_id", "") or "").strip()
            if candidate_item_id:
                default_output_name = candidate_item_id
        effective_output_dir = (
            Path(output_dir)
            if output_dir
            else ReviewExporter.default_output_dir(
                data_dir=self.data_dir,
                platform=self.platform,
                target_name=default_output_name,
                months=effective_months,
                days=effective_days,
                limit=effective_limit,
            )
        )

        item_results: list[ReviewItemResult] = []
        errors: list[str] = []
        stopped_reasons: list[str] = []
        login_event_count = 0

        for record in targets:
            try:
                result = await self.crawler.collect_reviews_async(
                    url=record.normalized_url,
                    item_id=record.item_id,
                    months=effective_months,
                    days=effective_days,
                    limit=effective_limit,
                    target_name=target_name,
                )
            except Exception as exc:
                LOG.exception(
                    "collect_reviews_async failed for item_id=%s url=%s",
                    record.item_id,
                    record.normalized_url,
                )
                result = ReviewItemResult(
                    platform=self.platform,
                    item_id=record.item_id,
                    item_url=record.normalized_url,
                    cutoff_time=cutoff.isoformat(timespec="seconds"),
                    error=f"{type(exc).__name__}: {exc}",
                )
            result.reviews = filter_and_limit_reviews(
                result.reviews,
                cutoff=cutoff,
                limit=effective_limit,
            )
            result.collected_count = len(result.reviews)
            item_results.append(result)
            login_event_count += len(result.login_recovery_events)
            if result.error:
                errors.append(result.error)
            if result.stopped_reason:
                stopped_reasons.append(result.stopped_reason)

        total_reviews = sum(item.collected_count for item in item_results)
        success_items = sum(1 for item in item_results if not item.error)
        failed_items = max(0, len(item_results) - success_items)

        summary_payload = {
            "platform": self.platform,
            "target_name": target_name,
            "months": effective_months,
            "days": effective_days,
            "limit": effective_limit,
            "item_count": len(item_results),
            "success_items": success_items,
            "failed_items": failed_items,
            "total_reviews": total_reviews,
            "per_item_counts": [
                {
                    "platform": item.platform,
                    "item_id": item.item_id,
                    "count": item.collected_count,
                    "stopped_reason": item.stopped_reason,
                    "error": item.error,
                }
                for item in item_results
            ],
            "stopped_reasons": summarize_reason_counts(stopped_reasons),
            "top_errors": summarize_reason_counts(errors),
            "login_recovery_event_count": login_event_count,
            "created_at": now_iso(),
        }
        jsonl_path, csv_path, summary_json_path, summary_md_path = ReviewExporter.export(
            output_dir=effective_output_dir,
            item_results=item_results,
            summary_payload=summary_payload,
        )
        run_summary = ReviewRunSummary(
            platform=self.platform,
            target_name=target_name,
            months=effective_months,
            days=effective_days,
            limit=effective_limit,
            item_count=len(item_results),
            success_items=success_items,
            failed_items=failed_items,
            total_reviews=total_reviews,
            output_dir=str(Path(effective_output_dir).resolve()),
            jsonl_path=str(jsonl_path.resolve()),
            csv_path=str(csv_path.resolve()),
            run_summary_json_path=str(summary_json_path.resolve()),
            run_summary_md_path=str(summary_md_path.resolve()),
            per_item_counts=summary_payload["per_item_counts"],
            stopped_reasons=summary_payload["stopped_reasons"],
            top_errors=summary_payload["top_errors"],
            login_recovery_event_count=login_event_count,
            created_at=summary_payload["created_at"],
        )
        payload = run_summary.to_json_dict()
        payload["ok"] = failed_items == 0
        return payload
