"""Pipeline orchestration and CLI definition."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import shutil
import sys
import traceback
import uuid
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
import datetime as dt

from config import (
    load_simple_dotenv,
    DEFAULT_GROUP_CODE,
    MAX_TOP_N,
    CRAWL_WORKERS,
    LLM_WORKERS,
    LLM_WORKERS_MIN,
)
from data import (
    GroupService,
    ItemDetail,
    Storage,
    TaskTracker,
    UrlRecord,
    URLService,
    WorkbookService,
    clean_text,
    extract_json_object,
    normalize_url,
    normalize_brand_name,
    now_iso,
    preserve_multiline_text,
)
from scraper import (
    Crawler,
    SearchClient,
    list_cdp_pages,
    load_url_lines,
)
from analysis import (
    Analyzer,
    SellingPointExtractor,
)
from report import ReportGenerator

LOG = logging.getLogger("taobao_insight")


def _default_user_data_dir() -> str:
    from pathlib import Path

    appdata = os.getenv("APPDATA", "")
    if appdata:
        return str(Path(appdata) / "taobao_insight_profile")
    if sys.platform == "darwin":
        return str(
            Path.home() / "Library" / "Application Support" / "taobao_insight_profile"
        )
    return str(Path.home() / ".config" / "taobao_insight_profile")


class Pipeline:
    def __init__(
        self,
        storage: Storage,
        workbook_service: WorkbookService,
        group_service: GroupService,
        url_service: URLService,
        search_client: SearchClient,
        crawler: Crawler,
        extractor: SellingPointExtractor,
        analyzer: Analyzer,
        task_tracker: TaskTracker,
    ) -> None:
        self.storage = storage
        self.workbook_service = workbook_service
        self.group_service = group_service
        self.url_service = url_service
        self.search_client = search_client
        self.crawler = crawler
        self.extractor = extractor
        self.analyzer = analyzer
        self.task_tracker = task_tracker
        self.reporter = ReportGenerator(storage, workbook_service, analyzer)

    @staticmethod
    def _build_workbook_points_map(
        workbook_id: str, selling_points_payload: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        prefix = f"{workbook_id}:"
        for key, value in selling_points_payload.items():
            if not key.startswith(prefix) or not isinstance(value, dict):
                continue
            item_id = key.split(":", 1)[1]
            points = value.get("points", [])
            if isinstance(points, list):
                result[item_id] = [p for p in points if isinstance(p, dict)]
        return result

    @staticmethod
    def _derive_html_output_path(
        workbook_id: str,
        md_output_path: str | None,
        html_output_path: str | None,
        exports_dir: Path,
    ) -> Path | None:
        if html_output_path:
            return Path(html_output_path)
        if md_output_path:
            md_path = Path(md_output_path)
            if md_path.suffix:
                return md_path.with_suffix(".html")
            return md_path.parent / f"{md_path.name}.html"
        return None

    @staticmethod
    def _derive_md_output_path(
        workbook_id: str,
        md_output_path: str | None,
        html_output_path: str | None,
        exports_dir: Path,
    ) -> Path | None:
        if md_output_path:
            md_path = Path(md_output_path)
            if md_path.suffix:
                return md_path.with_suffix(".md")
            return md_path.parent / f"{md_path.name}.md"
        if html_output_path:
            html_path = Path(html_output_path)
            if html_path.suffix:
                return html_path.with_suffix(".md")
            return html_path.parent / f"{html_path.name}.md"
        return None

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        text = str(exc).lower()
        transient_tokens = (
            "429",
            "rate limit",
            "quota",
            "resource exhausted",
            "timeout",
            "timed out",
            "deadline",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "broken pipe",
            "502",
            "503",
            "504",
        )
        return any(token in text for token in transient_tokens)

    @staticmethod
    def _elapsed_seconds(started_at: float) -> float:
        return max(0.0, time.perf_counter() - started_at)

    @staticmethod
    def _effective_crawl_workers(requested_workers: int) -> int:
        try:
            requested = max(1, int(requested_workers))
        except (TypeError, ValueError):
            requested = CRAWL_WORKERS
        if requested > CRAWL_WORKERS:
            LOG.warning(
                "crawl_workers=%s configured; higher Taobao concurrency may increase anti-bot risk",
                requested,
            )
        return requested

    def _reset_login_recovery_events(self) -> None:
        if hasattr(self.search_client, "login_recovery_events"):
            self.search_client.login_recovery_events = []
        if hasattr(self.crawler, "login_recovery_events"):
            self.crawler.login_recovery_events = []

    def _collect_login_recovery_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        search_events = getattr(self.search_client, "login_recovery_events", [])
        crawl_events = getattr(self.crawler, "login_recovery_events", [])
        if isinstance(search_events, list):
            events.extend([e for e in search_events if isinstance(e, dict)])
        if isinstance(crawl_events, list):
            events.extend([e for e in crawl_events if isinstance(e, dict)])
        return events

    @staticmethod
    def _apply_search_metadata(row: dict[str, str], rec: UrlRecord) -> None:
        if rec.search_rank is not None:
            row["search_rank"] = str(rec.search_rank)
        if rec.sales_text:
            row["sales_text"] = rec.sales_text
        row["official_store"] = "1" if rec.is_official_store else "0"
        if rec.title and not row.get("title"):
            row["title"] = rec.title
        if rec.shop_name and not row.get("shop_name"):
            row["shop_name"] = rec.shop_name
        if rec.item_source_url:
            row["item_source_url"] = rec.item_source_url
        if rec.source_type:
            row["source_type"] = rec.source_type
        if not row.get("item_source_url"):
            row["item_source_url"] = row.get("normalized_url", "")
        if not row.get("source_type") and row.get("item_source_url", "").startswith(
            "http"
        ):
            row["source_type"] = "official"

    @staticmethod
    def _apply_crawl_detail(row: dict[str, str], detail: ItemDetail) -> None:
        row["title"] = detail.title
        row["main_image_url"] = detail.main_image_url
        row["shop_name"] = detail.shop_name
        row["brand"] = normalize_brand_name(
            detail.brand, title=detail.title, shop_name=detail.shop_name
        )
        row["price_min"] = f"{min(detail.prices):.2f}" if detail.prices else ""
        row["price_max"] = f"{max(detail.prices):.2f}" if detail.prices else ""
        row["sku_count"] = str(len(detail.skus))
        row["sku_list"] = "\n".join(
            [
                f"{s['sku_id']}|{clean_text(s['sku_name'], 60)}|{s['price']}"
                for s in detail.skus
            ]
        )
        row["detail_summary"] = detail.detail_summary
        row["crawl_time"] = detail.crawl_time
        row["process_status"] = "2"
        row["updated_at"] = now_iso()

    def _extract_points_for_detail(
        self, detail: ItemDetail
    ) -> tuple[str, list[dict[str, str]]]:
        item_context = {
            "title": detail.title,
            "brand": detail.brand,
            "shop_name": detail.shop_name,
        }
        detail_summary = self.extractor.summarize_detail(
            item_context=item_context,
            detail_blocks=detail.detail_blocks,
        )
        merged_summary = detail_summary or detail.detail_summary
        points = self.extractor.extract(
            item_context=item_context,
            detail_blocks=detail.detail_blocks,
            detail_summary=merged_summary,
        )
        return merged_summary, points

    def _run_logs_dir(self) -> Path:
        path = self.storage.data_dir / "run_logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_log_path(self, task_id: str) -> Path:
        return self._run_logs_dir() / f"{task_id}.jsonl"

    def _append_run_log(
        self,
        run_log_path: Path | None,
        stage: str,
        status: str = "info",
        **fields: Any,
    ) -> None:
        if run_log_path is None:
            return
        payload = {
            "ts": now_iso(),
            "stage": stage,
            "status": status,
            **fields,
        }
        try:
            with run_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            LOG.warning("failed to write run log %s: %s", run_log_path, exc)
        LOG.info("[run-stage] %s %s %s", stage, status, clean_text(str(fields), 260))

    @staticmethod
    def _normalize_failure_reason(reason: str) -> str:
        text = (reason or "").strip()
        lowered = text.lower()
        if "net::err_aborted" in lowered:
            return "Page.goto ERR_ABORTED"
        if "remoteprotocolerror" in lowered or "server disconnected" in lowered:
            return "Gemini/HTTP disconnected"
        if "servererror: 500" in lowered or "500 internal" in lowered:
            return "Gemini 500 INTERNAL"
        if "llm extraction failed" in lowered:
            return "LLM extraction failed"
        if "llm analyze failed" in lowered:
            return "LLM analyze failed"
        if "login" in lowered:
            return "Login/session issue"
        if "timeout" in lowered:
            return "Timeout"
        first_line = text.splitlines()[0] if text else ""
        return clean_text(first_line or "Unknown", max_len=100)

    @staticmethod
    def _safe_slug(text: str, fallback: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", (text or "").strip()).strip("-_")
        return slug or fallback

    def _write_run_summary(
        self,
        *,
        task_id: str,
        workbook_id: str,
        workbook_name: str,
        keyword: str,
        status: str,
        result_payload: dict[str, Any] | None,
        error_text: str = "",
        run_log_path: Path | None = None,
    ) -> tuple[dict[str, Any], Path]:
        task_payload = self.task_tracker.get(task_id)
        failures = task_payload.get("failures", [])
        if not isinstance(failures, list):
            failures = []

        total_items = int(task_payload.get("total_items", 0) or 0)
        failure_count = len(failures)
        success_count = max(0, total_items - failure_count)
        error_counter: Counter[str] = Counter(
            [self._normalize_failure_reason(str(f.get("reason", ""))) for f in failures if isinstance(f, dict)]
        )
        top_errors = [
            {"reason": reason, "count": count}
            for reason, count in error_counter.most_common(5)
        ]

        timings = (result_payload or {}).get("timings_sec", {})
        if not isinstance(timings, dict):
            timings = {}
        total_sec = float(timings.get("total", 0) or 0)
        final_conclusion = clean_text(
            str((result_payload or {}).get("final_conclusion", "")), max_len=500
        )
        market_summary = clean_text(
            str((result_payload or {}).get("market_summary", "")), max_len=500
        )
        key_conclusion = final_conclusion or market_summary or clean_text(error_text, 500)

        summary_payload = {
            "task_id": task_id,
            "workbook_id": workbook_id,
            "workbook_name": workbook_name,
            "keyword": keyword,
            "status": status,
            "created_at": task_payload.get("created_at", ""),
            "updated_at": task_payload.get("updated_at", ""),
            "total_items": total_items,
            "success_items": success_count,
            "failed_items": failure_count,
            "success_rate": round((success_count / total_items) * 100, 2) if total_items else 0.0,
            "timings_sec": {},
            "total_runtime_sec": round(total_sec, 3),
            "top_errors": top_errors,
            "key_conclusion": key_conclusion,
            "error": clean_text(error_text, 500) if error_text else "",
            "run_log_path": str(run_log_path) if run_log_path else "",
        }
        for key, value in timings.items():
            try:
                summary_payload["timings_sec"][key] = round(float(value), 3)
            except Exception:
                continue

        slug = self._safe_slug(workbook_name, task_id)
        summary_path = self.storage.exports_dir / f"{slug}-run-summary.md"
        lines = [
            "# 运行总结",
            "",
            f"- task_id: {task_id}",
            f"- workbook_id: {workbook_id}",
            f"- workbook_name: {workbook_name}",
            f"- keyword: {keyword}",
            f"- status: {status}",
            f"- created_at: {task_payload.get('created_at', '')}",
            f"- updated_at: {task_payload.get('updated_at', '')}",
            "",
            "## 运行时间",
            f"- total_runtime_sec: {summary_payload['total_runtime_sec']}",
        ]
        for key in ("search", "crawl", "llm_extract", "llm_analyze", "export"):
            if key in timings:
                lines.append(f"- {key}: {timings.get(key)}")
        lines.extend(
            [
                "",
                "## 成功率",
                f"- total_items: {total_items}",
                f"- success_items: {success_count}",
                f"- failed_items: {failure_count}",
                f"- success_rate: {summary_payload['success_rate']}%",
                "",
                "## 关键错误",
            ]
        )
        if top_errors:
            for idx, item in enumerate(top_errors, start=1):
                lines.append(f"{idx}. {item['reason']} ({item['count']})")
        else:
            lines.append("1. 无")
        lines.extend(
            [
                "",
                "## 关键结论",
                key_conclusion or "无",
            ]
        )
        if error_text:
            lines.extend(["", "## 运行异常", clean_text(error_text, 1000)])
        if run_log_path:
            lines.extend(["", "## 关键步骤日志", str(run_log_path)])
        summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return summary_payload, summary_path

    def _run_adaptive_stage(
        self,
        items: list[Any],
        worker_fn: Callable[[Any], Any],
        max_workers: int,
        min_workers: int,
        stage_name: str,
        max_retries: int = 2,
    ) -> list[Any]:
        if not items:
            return []
        upper = max(1, int(max_workers))
        lower = max(1, min(upper, int(min_workers)))
        current = upper
        pending: list[tuple[int, Any, int]] = [
            (index, item, 0) for index, item in enumerate(items)
        ]
        results: dict[int, Any] = {}

        while pending:
            batch = pending[:current]
            pending = pending[current:]
            if not batch:
                continue
            transient_hit = False
            pool_size = max(1, min(current, len(batch)))
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                future_to_job = {
                    executor.submit(worker_fn, item): (idx, item, retry)
                    for idx, item, retry in batch
                }
                for future in as_completed(future_to_job):
                    idx, item, retry = future_to_job[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        if self._is_transient_error(exc) and retry < max_retries:
                            pending.append((idx, item, retry + 1))
                            transient_hit = True
                            LOG.warning(
                                "Transient %s error (retry %s/%s): %s",
                                stage_name,
                                retry + 1,
                                max_retries,
                                exc,
                            )
                        else:
                            results[idx] = exc
                            LOG.error("Failed %s task: %s", stage_name, exc)
                            if self._is_transient_error(exc):
                                transient_hit = True

            if transient_hit and current > lower:
                current = lower
                LOG.warning(
                    "Downshifting %s concurrency to %s due to transient failures",
                    stage_name,
                    current,
                )
                time.sleep(1.0)
            elif not transient_hit and current < upper:
                current += 1

        return [results[i] for i in range(len(items))]

    def _build_market_outputs(
        self,
        workbook_id: str,
        workbook_name: str,
        workbook_rows: list[dict[str, str]],
        workbook_points_map: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, Any], str, str, str]:
        batch_summary_payload: dict[str, Any] = {}
        market: dict[str, Any] = {}
        final_conclusion_text = ""

        try:
            batch_summary_payload = self.analyzer.generate_batch_competitor_summary(
                workbook_rows, workbook_points_map
            )
        except Exception as exc:
            LOG.error("Failed batch competitor summary: %s", exc)
        batch_summary_text = preserve_multiline_text(
            batch_summary_payload.get("batch_competitor_summary_text", ""),
            max_len=4000,
        )

        try:
            market = self.analyzer.generate_market_summary(
                workbook_rows, workbook_points_map
            )
        except Exception as exc:
            LOG.error("Failed market summary: %s", exc)
            market = {}
        market_summary_text = preserve_multiline_text(
            market.get("summary_text", ""), max_len=4000
        )

        try:
            final_conclusion_text = self.analyzer.generate_final_conclusion(
                workbook_rows,
                batch_competitor_summary_text=batch_summary_text,
                market_summary_text=market_summary_text,
            )
        except Exception as exc:
            LOG.error("Failed final conclusion: %s", exc)
            final_conclusion_text = ""

        point_item_count = sum(
            1 for points in workbook_points_map.values() if isinstance(points, list) and points
        )
        priced_item_count = sum(
            1
            for row in workbook_rows
            if (row.get("price_min", "") or "").strip()
            or (row.get("price_max", "") or "").strip()
        )
        if not batch_summary_text:
            batch_summary_text = (
                f"样本共{len(workbook_rows)}个商品，其中{point_item_count}个商品提取到有效卖点。"
                "当前暂缺可用批量竞品总结数据。"
            )
        if not market_summary_text:
            market_summary_text = (
                f"样本共{len(workbook_rows)}个商品，{priced_item_count}个商品包含价格信息，"
                f"{point_item_count}个商品包含卖点证据。当前暂缺可用市场总结数据。"
            )
        if not final_conclusion_text:
            final_conclusion_text = (
                "本次样本暂缺可用分析数据，无法生成稳定结论。"
                "请重试任务并检查 Gemini 服务状态、网络稳定性与图文详情抓取结果。"
            )
        if not market.get("summary_text"):
            market["summary_text"] = market_summary_text

        point_freq: dict[str, int] = {}
        prices: list[float] = []
        for row in workbook_rows:
            for point in workbook_points_map.get(row.get("item_id", ""), []):
                point_text = clean_text(str(point.get("point", "")), max_len=80)
                if point_text:
                    point_freq[point_text] = point_freq.get(point_text, 0) + 1
            for field in ("price_min", "price_max"):
                raw = (row.get(field, "") or "").strip()
                if not raw:
                    continue
                try:
                    prices.append(float(raw))
                    break
                except ValueError:
                    continue
        q1, q2 = Analyzer._price_quantiles(prices)
        for row in workbook_rows:
            mapping = self.analyzer.build_row_market_mapping(row, point_freq, q1, q2)
            row["batch_competitor_summary_text"] = mapping["batch"] or batch_summary_text
            row["market_summary_text"] = mapping["market"] or market_summary_text
            row["final_conclusion_text"] = mapping["final"] or final_conclusion_text
            if market.get("market_tags"):
                row["market_tags"] = " | ".join(market["market_tags"])
            row["updated_at"] = now_iso()

        market_payload = self.storage.read_json(self.storage.market_report_json)
        market_payload[workbook_id] = {
            "workbook_id": workbook_id,
            "workbook_name": workbook_name,
            **market,
            "batch_competitor_summary_text": batch_summary_text,
            "final_conclusion_text": final_conclusion_text,
        }
        return (
            market_payload,
            batch_summary_text,
            market_summary_text,
            final_conclusion_text,
        )

    @staticmethod
    def _normalize_process_status(raw: Any) -> str:
        status = str(raw or "").strip()
        if status in {"1", "2", "3", "4"}:
            return status
        return "unknown"

    def _collect_export_readiness(
        self,
        workbook_id: str,
        ignore_running_task_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        rows = [r for r in self.storage.list_products() if r["workbook_id"] == workbook_id]
        if not rows:
            raise ValueError(f"no products found for workbook: {workbook_id}")

        status_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "unknown": 0}
        selling_points_count = 0
        analyzed_count = 0
        price_count = 0
        for row in rows:
            status = self._normalize_process_status(row.get("process_status", ""))
            status_counts[status] += 1
            if str(row.get("selling_points_text", "")).strip():
                selling_points_count += 1
            if str(row.get("competitor_analysis_text", "")).strip():
                analyzed_count += 1
            if str(row.get("price_min", "")).strip() or str(row.get("price_max", "")).strip():
                price_count += 1

        tasks = self.storage.list_tasks()
        ignore_ids = set(ignore_running_task_ids or set())
        running_task_ids = sorted(
            [
                task_id
                for task_id, task in tasks.items()
                if (
                    task.get("workbook_id") == workbook_id
                    and task.get("status") == "running"
                    and task_id not in ignore_ids
                )
            ]
        )

        market_report = self.storage.read_json(self.storage.market_report_json).get(
            workbook_id, {}
        )
        has_market_summary = bool(
            str(market_report.get("summary_text", "")).strip()
            or str(market_report.get("batch_competitor_summary_text", "")).strip()
            or str(market_report.get("final_conclusion_text", "")).strip()
        )

        return {
            "sample_count": len(rows),
            "status_counts": status_counts,
            "selling_points_count": selling_points_count,
            "analyzed_count": analyzed_count,
            "price_count": price_count,
            "running_task_ids": running_task_ids,
            "has_market_summary": has_market_summary,
        }

    def _assert_export_ready(
        self,
        workbook_id: str,
        allow_incomplete: bool = False,
        ignore_running_task_ids: set[str] | None = None,
    ) -> None:
        stats = self._collect_export_readiness(
            workbook_id, ignore_running_task_ids=ignore_running_task_ids
        )
        if allow_incomplete:
            if stats["running_task_ids"]:
                LOG.warning(
                    "allow-incomplete export for workbook %s while task(s) still running: %s",
                    workbook_id,
                    ",".join(stats["running_task_ids"]),
                )
            return

        if stats["running_task_ids"]:
            LOG.warning(
                "export requested while task(s) still marked running for workbook %s: %s",
                workbook_id,
                ",".join(stats["running_task_ids"]),
            )

        blockers: list[str] = []
        if stats["status_counts"]["1"] >= stats["sample_count"]:
            blockers.append(
                "all rows are process_status=1 (search-only data; detail crawl not finished)"
            )
        if stats["selling_points_count"] == 0:
            blockers.append("selling_points_text is empty for all rows")

        if blockers:
            status_view = stats["status_counts"]
            summary = (
                f"sample={stats['sample_count']}, status_1={status_view['1']}, "
                f"status_2={status_view['2']}, status_3={status_view['3']}, "
                f"status_4={status_view['4']}, selling_points={stats['selling_points_count']}, "
                f"analyzed={stats['analyzed_count']}, price={stats['price_count']}"
            )
            raise RuntimeError(
                "workbook is not ready for final export: "
                + "; ".join(blockers)
                + f" | {summary}. "
                + "Run full pipeline to completion or pass --allow-incomplete to force export."
            )
        if stats["analyzed_count"] == 0 and not stats["has_market_summary"]:
            LOG.warning(
                "exporting workbook %s without market summary/final conclusion "
                "(sample=%s, selling_points=%s)",
                workbook_id,
                stats["sample_count"],
                stats["selling_points_count"],
            )

    def export_html_and_md(
        self,
        workbook_id: str,
        output_path: str | None = None,
        html_output_path: str | None = None,
        allow_incomplete: bool = False,
        ignore_running_task_ids: set[str] | None = None,
    ) -> tuple[Path, Path]:
        self._assert_export_ready(
            workbook_id,
            allow_incomplete=allow_incomplete,
            ignore_running_task_ids=ignore_running_task_ids,
        )
        derived_md = self._derive_md_output_path(
            workbook_id=workbook_id,
            md_output_path=output_path,
            html_output_path=html_output_path,
            exports_dir=self.storage.exports_dir,
        )
        derived_html = self._derive_html_output_path(
            workbook_id=workbook_id,
            md_output_path=output_path,
            html_output_path=html_output_path,
            exports_dir=self.storage.exports_dir,
        )
        export_html_path = self.reporter.export_html(
            workbook_id=workbook_id,
            output_path=str(derived_html) if derived_html else None,
            md_output_path=str(derived_md) if derived_md else None,
            write_md_copy=True,
        )
        export_md_path = derived_md if derived_md else export_html_path.with_suffix(".md")
        return export_md_path, export_html_path

    def export_csv(
        self,
        workbook_id: str,
        output_path: str | None = None,
        allow_incomplete: bool = False,
    ) -> Path:
        export_md_path, _ = self.export_html_and_md(
            workbook_id=workbook_id,
            output_path=output_path,
            allow_incomplete=allow_incomplete,
        )
        return export_md_path

    def export_html(
        self,
        workbook_id: str,
        output_path: str | None = None,
        allow_incomplete: bool = False,
    ) -> Path:
        self._assert_export_ready(workbook_id, allow_incomplete=allow_incomplete)
        return self.reporter.export_html(workbook_id=workbook_id, output_path=output_path)

    def cleanup_workbook_data(self, workbook_id: str) -> None:
        """清理工作簿的中间产物，只保留报告文件（md 和 html）"""
        # 1. 从 products.csv 中删除该工作簿的产品记录
        products = self.storage.list_products()
        products = [p for p in products if p.get("workbook_id") != workbook_id]
        self.storage.save_products(products)

        # 2. 删除图片目录
        image_dir = self.storage.images_dir / workbook_id
        if image_dir.exists():
            shutil.rmtree(image_dir)

        # 3. 从 tasks.json 中删除任务记录
        tasks = self.storage.list_tasks()
        tasks_to_remove = [
            tid for tid, t in tasks.items() if t.get("workbook_id") == workbook_id
        ]
        for tid in tasks_to_remove:
            del tasks[tid]
        self.storage.save_tasks(tasks)

    async def run_keyword_async(
        self,
        keyword: str,
        top_n: int,
        workbook_name: str | None,
        search_url: str | None,
        search_sort: str,
        shop_filter_enabled: bool,
        output_path: str | None,
        html_output_path: str | None = None,
        input_urls: list[str] | None = None,
        crawl_workers: int = CRAWL_WORKERS,
        llm_workers: int = 24,
        llm_workers_min: int = 8,
    ) -> dict[str, Any]:
        """Async version of run_keyword with run summary output."""
        run_started_at = time.perf_counter()
        keyword = keyword.strip()
        if not keyword:
            raise ValueError("keyword cannot be empty")
        recovered_tasks = self.task_tracker.recover_orphaned_running(
            stale_after_minutes=60
        )
        if recovered_tasks:
            LOG.warning(
                "Recovered orphaned running tasks before new run: %s",
                ",".join(recovered_tasks),
            )
        self._reset_login_recovery_events()

        workbook_name = (
            workbook_name
            or f"{keyword}-top{top_n}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        workbook = self.workbook_service.create(workbook_name)
        workbook_id = workbook["workbook_id"]

        normalized_inputs: list[UrlRecord] = []
        seen_input_ids: set[str] = set()
        for raw_url in input_urls or []:
            rec = normalize_url(raw_url)
            if not rec or rec.item_id in seen_input_ids:
                continue
            seen_input_ids.add(rec.item_id)
            normalized_inputs.append(rec)

        search_started_at = time.perf_counter()
        if normalized_inputs:
            search_records = normalized_inputs[:top_n]
            LOG.info(
                "Use %s direct input item URLs; skip search step", len(search_records)
            )
        else:
            search_records, _anti_bot_signal = await self.search_client._search_with_global_browser_async(
                keyword=keyword,
                top_n=top_n,
                search_url=search_url,
                search_sort=search_sort,
                official_only=shop_filter_enabled,
            )
        search_elapsed_sec = self._elapsed_seconds(search_started_at)
        if not search_records:
            raise RuntimeError(
                "no Taobao item URL found; provide --item-url/--item-urls-file or use a real signed-in browser session"
            )

        self.url_service.add_records(
            workbook_id=workbook_id,
            records=search_records[:top_n],
            group_code=DEFAULT_GROUP_CODE,
        )

        task_id = self.task_tracker.create(
            workbook_id=workbook_id,
            keyword=keyword,
            total_items=min(top_n, len(search_records)),
        )
        run_log_path = self._run_log_path(task_id)
        effective_crawl_workers = self._effective_crawl_workers(crawl_workers)
        finished = 0
        self._append_run_log(
            run_log_path,
            "run-start",
            status="ok",
            task_id=task_id,
            workbook_id=workbook_id,
            keyword=keyword,
            top_n=top_n,
            search_elapsed_sec=round(search_elapsed_sec, 3),
            search_records=len(search_records),
            crawl_workers=effective_crawl_workers,
            llm_workers=max(1, int(llm_workers)),
        )

        crawl_elapsed_sec = 0.0
        llm_extract_elapsed_sec = 0.0
        llm_analyze_elapsed_sec = 0.0
        export_elapsed_sec = 0.0
        try:
            products = self.storage.list_products()
            product_map = {(r["workbook_id"], r["item_id"]): r for r in products}
            selling_points = self.storage.read_json(self.storage.selling_points_json)

            targets: list[tuple[UrlRecord, dict[str, str]]] = []
            for rec in search_records[:top_n]:
                row = product_map.get((workbook_id, rec.item_id))
                if not row:
                    continue
                self._apply_search_metadata(row, rec)
                targets.append((rec, row))
            self._append_run_log(
                run_log_path,
                "target-build",
                status="ok",
                target_count=len(targets),
            )

            crawl_started_at = time.perf_counter()
            llm_extract_started_at = time.perf_counter()
            semaphore = asyncio.Semaphore(max(1, effective_crawl_workers))

            async def crawl_single_item(rec: UrlRecord, row: dict) -> tuple[UrlRecord, dict[str, str], ItemDetail]:
                async with semaphore:
                    self._append_run_log(
                        run_log_path,
                        "crawl-item-start",
                        status="ok",
                        item_id=rec.item_id,
                    )
                    detail = await self.crawler.crawl_async_global(
                        url=rec.normalized_url,
                        item_id=rec.item_id,
                        workbook_id=workbook_id,
                    )
                    if detail.error:
                        self._append_run_log(
                            run_log_path,
                            "crawl-item-end",
                            status="error",
                            item_id=rec.item_id,
                            error=clean_text(detail.error, max_len=180),
                        )
                    else:
                        self._append_run_log(
                            run_log_path,
                            "crawl-item-end",
                            status="ok",
                            item_id=rec.item_id,
                        )
                    return rec, row, detail

            crawl_tasks = [asyncio.create_task(crawl_single_item(rec, row)) for rec, row in targets]
            llm_extract_executor = ThreadPoolExecutor(max_workers=max(1, int(llm_workers)))
            pending_llm_futures: list[tuple[tuple[UrlRecord, dict[str, str], ItemDetail], Any]] = []
            crawl_success = 0
            crawl_fail = 0

            for coro in asyncio.as_completed(crawl_tasks):
                rec, row, detail = await coro
                if detail.error:
                    crawl_fail += 1
                    self.task_tracker.add_failure(task_id, rec.item_id, detail.error)
                    finished += 1
                    self.task_tracker.update_progress(task_id, finished)
                    continue

                crawl_success += 1
                self._apply_crawl_detail(row, detail)
                future = llm_extract_executor.submit(self._extract_points_for_detail, detail)
                pending_llm_futures.append(((rec, row, detail), future))

            # Crawl stage ends when all crawl coroutines are done; LLM extraction futures
            # may still be running in the background.
            crawl_elapsed_sec = self._elapsed_seconds(crawl_started_at)
            self._append_run_log(
                run_log_path,
                "crawl-stage-end",
                status="ok",
                crawl_success=crawl_success,
                crawl_failed=crawl_fail,
                crawl_elapsed_sec=round(crawl_elapsed_sec, 3),
            )

            llm_extract_results: list[tuple[tuple[UrlRecord, dict[str, str], ItemDetail], Any]] = []
            for payload, future in pending_llm_futures:
                try:
                    llm_extract_results.append((payload, future.result()))
                except Exception as llm_error:
                    llm_extract_results.append((payload, llm_error))
            llm_extract_executor.shutdown(wait=True)

            llm_extract_success = 0
            llm_extract_fail = 0
            for payload, llm_result in llm_extract_results:
                rec, row, detail = payload
                if isinstance(llm_result, Exception):
                    llm_extract_fail += 1
                    self.task_tracker.add_failure(
                        task_id, rec.item_id, f"LLM extraction failed: {llm_result}"
                    )
                    row["selling_points_text"] = ""
                    row["selling_points_citation"] = ""
                    row["updated_at"] = now_iso()
                    finished += 1
                    self.task_tracker.update_progress(task_id, finished)
                    continue

                llm_extract_success += 1
                detail_summary, points = llm_result
                row["detail_summary"] = detail_summary or detail.detail_summary
                if points:
                    row["detail_summary"] = SellingPointExtractor.summarize_points(
                        points, fallback=row["detail_summary"]
                    )
                    row["selling_points_text"] = " | ".join(
                        [point["point"] for point in points]
                    )
                    row["selling_points_citation"] = " | ".join(
                        [point["citation"] for point in points]
                    )
                    row["process_status"] = "3"
                    selling_points[f"{workbook_id}:{rec.item_id}"] = {
                        "workbook_id": workbook_id,
                        "item_id": rec.item_id,
                        "points": points,
                        "detail_summary": row["detail_summary"],
                        "detail_blocks": detail.detail_blocks,
                        "updated_at": now_iso(),
                    }
                else:
                    row["selling_points_text"] = ""
                    row["selling_points_citation"] = ""
                row["updated_at"] = now_iso()
                finished += 1
                self.task_tracker.update_progress(task_id, finished)
            llm_extract_elapsed_sec = self._elapsed_seconds(llm_extract_started_at)
            self._append_run_log(
                run_log_path,
                "llm-extract-end",
                status="ok",
                llm_extract_success=llm_extract_success,
                llm_extract_failed=llm_extract_fail,
                llm_extract_elapsed_sec=round(llm_extract_elapsed_sec, 3),
            )

            workbook_rows = [r for r in products if r["workbook_id"] == workbook_id]
            workbook_points_map = self._build_workbook_points_map(
                workbook_id, selling_points
            )
            analyze_inputs: list[tuple[dict[str, str], list[dict[str, Any]]]] = []
            for row in workbook_rows:
                item_points = workbook_points_map.get(row["item_id"], [])
                if item_points:
                    analyze_inputs.append((row, item_points))

            llm_analyze_started_at = time.perf_counter()
            analyze_results = self._run_adaptive_stage(
                items=analyze_inputs,
                worker_fn=lambda payload: self.analyzer.analyze_item(
                    payload[0]["item_id"],
                    payload[1],
                    workbook_points_map,
                    item_title=payload[0].get("title", ""),
                ),
                max_workers=llm_workers,
                min_workers=llm_workers_min,
                stage_name="llm-analyze",
            )
            llm_analyze_fail = 0
            for payload, analyze_result in zip(analyze_inputs, analyze_results):
                row, _ = payload
                if isinstance(analyze_result, Exception):
                    llm_analyze_fail += 1
                    self.task_tracker.add_failure(
                        task_id,
                        row["item_id"],
                        f"LLM analyze failed: {analyze_result}",
                    )
                    continue
                row["competitor_analysis_text"] = analyze_result.get(
                    "competitor_analysis_text", ""
                )
                if analyze_result.get("market_tags"):
                    row["market_tags"] = " | ".join(analyze_result["market_tags"])
                row["process_status"] = "4"
                row["updated_at"] = now_iso()
            llm_analyze_elapsed_sec = self._elapsed_seconds(llm_analyze_started_at)
            self._append_run_log(
                run_log_path,
                "llm-analyze-end",
                status="ok",
                llm_analyze_inputs=len(analyze_inputs),
                llm_analyze_failed=llm_analyze_fail,
                llm_analyze_elapsed_sec=round(llm_analyze_elapsed_sec, 3),
            )

            export_started_at = time.perf_counter()
            (
                market_payload,
                batch_summary_text,
                market_summary_text,
                final_conclusion_text,
            ) = self._build_market_outputs(
                workbook_id=workbook_id,
                workbook_name=workbook["workbook_name"],
                workbook_rows=workbook_rows,
                workbook_points_map=workbook_points_map,
            )

            self.storage.save_products(products)
            self.storage.write_json(self.storage.selling_points_json, selling_points)
            self.storage.write_json(self.storage.market_report_json, market_payload)

            export_md_path, export_html_path = self.export_html_and_md(
                workbook_id=workbook_id,
                output_path=output_path,
                html_output_path=html_output_path,
                allow_incomplete=False,
                ignore_running_task_ids={task_id},
            )
            export_elapsed_sec = self._elapsed_seconds(export_started_at)
            total_elapsed_sec = self._elapsed_seconds(run_started_at)

            result = {
                "workbook_id": workbook_id,
                "workbook_name": workbook["workbook_name"],
                "task_id": task_id,
                "top_n": top_n,
                "items_found": len(search_records),
                "export_md": str(export_md_path),
                "export_csv": str(export_md_path),
                "export_html": str(export_html_path),
                "batch_competitor_summary": batch_summary_text,
                "market_summary": market_summary_text,
                "final_conclusion": final_conclusion_text,
                "source_mode": "input_urls" if normalized_inputs else "keyword_search",
                "login_recovery_events": self._collect_login_recovery_events(),
                "workers": {
                    "crawl_workers": effective_crawl_workers,
                    "llm_workers": max(1, int(llm_workers)),
                    "llm_workers_min": max(1, int(llm_workers_min)),
                },
                "timings_sec": {
                    "search": round(search_elapsed_sec, 3),
                    "crawl": round(crawl_elapsed_sec, 3),
                    "llm_extract": round(llm_extract_elapsed_sec, 3),
                    "llm_analyze": round(llm_analyze_elapsed_sec, 3),
                    "export": round(export_elapsed_sec, 3),
                    "total": round(total_elapsed_sec, 3),
                },
            }
            summary_payload, summary_path = self._write_run_summary(
                task_id=task_id,
                workbook_id=workbook_id,
                workbook_name=workbook["workbook_name"],
                keyword=keyword,
                status="completed",
                result_payload=result,
                run_log_path=run_log_path,
            )
            result["run_summary"] = summary_payload
            result["run_summary_path"] = str(summary_path)
            result["run_log_path"] = str(run_log_path)
            self.task_tracker.complete(task_id, "completed", result)
            self._append_run_log(
                run_log_path,
                "run-end",
                status="ok",
                success_rate=summary_payload.get("success_rate", 0),
                failed_items=summary_payload.get("failed_items", 0),
                total_runtime_sec=summary_payload.get("total_runtime_sec", 0),
                run_summary_path=str(summary_path),
            )
            return result

        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            fail_result = {
                "workbook_id": workbook_id,
                "workbook_name": workbook_name,
                "task_id": task_id,
                "timings_sec": {
                    "search": round(search_elapsed_sec, 3),
                    "crawl": round(crawl_elapsed_sec, 3),
                    "llm_extract": round(llm_extract_elapsed_sec, 3),
                    "llm_analyze": round(llm_analyze_elapsed_sec, 3),
                    "export": round(export_elapsed_sec, 3),
                    "total": round(self._elapsed_seconds(run_started_at), 3),
                },
            }
            summary_payload, summary_path = self._write_run_summary(
                task_id=task_id,
                workbook_id=workbook_id,
                workbook_name=workbook_name,
                keyword=keyword,
                status="failed",
                result_payload=fail_result,
                error_text=error_text,
                run_log_path=run_log_path,
            )
            failed_payload = {
                "error": error_text,
                "run_summary": summary_payload,
                "run_summary_path": str(summary_path),
                "run_log_path": str(run_log_path),
            }
            self.task_tracker.complete(task_id, "failed", failed_payload)
            self._append_run_log(
                run_log_path,
                "run-end",
                status="error",
                error=clean_text(error_text, max_len=300),
                run_summary_path=str(summary_path),
            )
            raise

    def run_keyword(
        self,
        keyword: str,
        top_n: int,
        workbook_name: str | None,
        search_url: str | None,
        search_sort: str,
        shop_filter_enabled: bool,
        output_path: str | None,
        html_output_path: str | None = None,
        input_urls: list[str] | None = None,
        crawl_workers: int = CRAWL_WORKERS,
        llm_workers: int = 24,
        llm_workers_min: int = 8,
    ) -> dict[str, Any]:
        run_started_at = time.perf_counter()
        keyword = keyword.strip()
        if not keyword:
            raise ValueError("keyword cannot be empty")
        recovered_tasks = self.task_tracker.recover_orphaned_running(
            stale_after_minutes=60
        )
        if recovered_tasks:
            LOG.warning(
                "Recovered orphaned running tasks before new run: %s",
                ",".join(recovered_tasks),
            )
        self._reset_login_recovery_events()

        workbook_name = (
            workbook_name
            or f"{keyword}-top{top_n}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        workbook = self.workbook_service.create(workbook_name)
        workbook_id = workbook["workbook_id"]

        normalized_inputs: list[UrlRecord] = []
        seen_input_ids: set[str] = set()
        for raw_url in input_urls or []:
            rec = normalize_url(raw_url)
            if not rec or rec.item_id in seen_input_ids:
                continue
            seen_input_ids.add(rec.item_id)
            normalized_inputs.append(rec)

        search_started_at = time.perf_counter()
        if normalized_inputs:
            search_records = normalized_inputs[:top_n]
            LOG.info(
                "Use %s direct input item URLs; skip search step", len(search_records)
            )
        else:
            search_records = self.search_client.search_top_items(
                keyword=keyword,
                top_n=top_n,
                search_url=search_url,
                search_sort=search_sort,
                official_only=shop_filter_enabled,
            )
        search_elapsed_sec = self._elapsed_seconds(search_started_at)
        if not search_records:
            raise RuntimeError(
                "no Taobao item URL found; provide --item-url/--item-urls-file or use a real signed-in browser session"
            )

        self.url_service.add_records(
            workbook_id=workbook_id,
            records=search_records[:top_n],
            group_code=DEFAULT_GROUP_CODE,
        )

        task_id = self.task_tracker.create(
            workbook_id=workbook_id,
            keyword=keyword,
            total_items=min(top_n, len(search_records)),
        )
        effective_crawl_workers = self._effective_crawl_workers(crawl_workers)
        finished = 0
        try:
            products = self.storage.list_products()
            product_map = {(r["workbook_id"], r["item_id"]): r for r in products}
            selling_points = self.storage.read_json(self.storage.selling_points_json)

            targets: list[tuple[UrlRecord, dict[str, str]]] = []
            for rec in search_records[:top_n]:
                row = product_map.get((workbook_id, rec.item_id))
                if not row:
                    continue
                self._apply_search_metadata(row, rec)
                targets.append((rec, row))

            # Crawl phase - Pipeline parallel: start LLM Extract as soon as each crawl completes
            crawl_started_at = time.perf_counter()
            llm_extract_started_at = time.perf_counter()  # Start LLM timer early for pipeline stats

            llm_inputs: list[tuple[UrlRecord, dict[str, str], ItemDetail]] = []
            pending_llm_futures: list[tuple[tuple, Any]] = []

            if targets:
                pool_size = max(1, min(effective_crawl_workers, len(targets)))
                llm_extract_executor = ThreadPoolExecutor(max_workers=llm_workers)

                with ThreadPoolExecutor(max_workers=pool_size) as executor:
                    future_to_target = {
                        executor.submit(
                            self.crawler.crawl,
                            workbook_id,
                            rec.item_id,
                            rec.normalized_url,
                        ): (rec, row)
                        for rec, row in targets
                    }
                    for future in as_completed(future_to_target):
                        rec, row = future_to_target[future]
                        try:
                            detail = future.result()
                        except Exception as exc:
                            detail = ItemDetail(
                                item_id=rec.item_id,
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
                                error=f"{type(exc).__name__}: {exc}",
                            )

                        if detail.error:
                            self.task_tracker.add_failure(task_id, rec.item_id, detail.error)
                            finished += 1
                            self.task_tracker.update_progress(task_id, finished)
                            continue

                        self._apply_crawl_detail(row, detail)
                        llm_inputs.append((rec, row, detail))

                        # Immediately submit to LLM Extract (pipeline parallel)
                        llm_future = llm_extract_executor.submit(
                            self._extract_points_for_detail, detail
                        )
                        pending_llm_futures.append(((rec, row, detail), llm_future))

                llm_extract_executor.shutdown(wait=True)

            crawl_elapsed_sec = self._elapsed_seconds(crawl_started_at)

            # Wait for all LLM Extract tasks to complete
            llm_extract_results: list[tuple[tuple, Any]] = []
            for payload, future in pending_llm_futures:
                try:
                    llm_result = future.result()
                    llm_extract_results.append((payload, llm_result))
                except Exception as llm_error:
                    llm_extract_results.append((payload, llm_error))

            # Process LLM Extract results
            for payload, llm_result in llm_extract_results:
                rec, row, detail = payload
                if isinstance(llm_result, Exception):
                    self.task_tracker.add_failure(
                        task_id, rec.item_id, f"LLM extraction failed: {llm_result}"
                    )
                    row["selling_points_text"] = ""
                    row["selling_points_citation"] = ""
                    row["updated_at"] = now_iso()
                    finished += 1
                    self.task_tracker.update_progress(task_id, finished)
                    continue

                detail_summary, points = llm_result
                row["detail_summary"] = detail_summary or detail.detail_summary
                if points:
                    row["detail_summary"] = SellingPointExtractor.summarize_points(
                        points, fallback=row["detail_summary"]
                    )
                    row["selling_points_text"] = " | ".join(
                        [point["point"] for point in points]
                    )
                    row["selling_points_citation"] = " | ".join(
                        [point["citation"] for point in points]
                    )
                    row["process_status"] = "3"
                    selling_points[f"{workbook_id}:{rec.item_id}"] = {
                        "workbook_id": workbook_id,
                        "item_id": rec.item_id,
                        "points": points,
                        "detail_summary": row["detail_summary"],
                        "detail_blocks": detail.detail_blocks,
                        "updated_at": now_iso(),
                    }
                else:
                    row["selling_points_text"] = ""
                    row["selling_points_citation"] = ""
                row["updated_at"] = now_iso()
                finished += 1
                self.task_tracker.update_progress(task_id, finished)
            llm_extract_elapsed_sec = self._elapsed_seconds(llm_extract_started_at)

            workbook_rows = [r for r in products if r["workbook_id"] == workbook_id]
            workbook_points_map = self._build_workbook_points_map(
                workbook_id, selling_points
            )
            analyze_inputs: list[tuple[dict[str, str], list[dict[str, Any]]]] = []
            for row in workbook_rows:
                item_points = workbook_points_map.get(row["item_id"], [])
                if item_points:
                    analyze_inputs.append((row, item_points))

            llm_analyze_started_at = time.perf_counter()
            analyze_results = self._run_adaptive_stage(
                items=analyze_inputs,
                worker_fn=lambda payload: self.analyzer.analyze_item(
                    payload[0]["item_id"],
                    payload[1],
                    workbook_points_map,
                    item_title=payload[0].get("title", ""),
                ),
                max_workers=llm_workers,
                min_workers=llm_workers_min,
                stage_name="llm-analyze",
            )
            for payload, analyze_result in zip(analyze_inputs, analyze_results):
                row, _ = payload
                if isinstance(analyze_result, Exception):
                    self.task_tracker.add_failure(
                        task_id,
                        row["item_id"],
                        f"LLM analyze failed: {analyze_result}",
                    )
                    continue
                row["competitor_analysis_text"] = analyze_result.get(
                    "competitor_analysis_text", ""
                )
                if analyze_result.get("market_tags"):
                    row["market_tags"] = " | ".join(analyze_result["market_tags"])
                row["process_status"] = "4"
                row["updated_at"] = now_iso()
            llm_analyze_elapsed_sec = self._elapsed_seconds(llm_analyze_started_at)

            export_started_at = time.perf_counter()
            (
                market_payload,
                batch_summary_text,
                market_summary_text,
                final_conclusion_text,
            ) = self._build_market_outputs(
                workbook_id=workbook_id,
                workbook_name=workbook["workbook_name"],
                workbook_rows=workbook_rows,
                workbook_points_map=workbook_points_map,
            )

            self.storage.save_products(products)
            self.storage.write_json(self.storage.selling_points_json, selling_points)
            self.storage.write_json(self.storage.market_report_json, market_payload)

            export_md_path, export_html_path = self.export_html_and_md(
                workbook_id=workbook_id,
                output_path=output_path,
                html_output_path=html_output_path,
                allow_incomplete=False,
                ignore_running_task_ids={task_id},
            )
            export_elapsed_sec = self._elapsed_seconds(export_started_at)
            total_elapsed_sec = self._elapsed_seconds(run_started_at)
            result = {
                "workbook_id": workbook_id,
                "workbook_name": workbook["workbook_name"],
                "task_id": task_id,
                "top_n": top_n,
                "items_found": len(search_records),
                "export_md": str(export_md_path),
                "export_csv": str(export_md_path),
                "export_html": str(export_html_path),
                "batch_competitor_summary": batch_summary_text,
                "market_summary": market_summary_text,
                "final_conclusion": final_conclusion_text,
                "source_mode": "input_urls" if normalized_inputs else "keyword_search",
                "login_recovery_events": self._collect_login_recovery_events(),
                "workers": {
                    "crawl_workers": effective_crawl_workers,
                    "llm_workers": max(1, int(llm_workers)),
                    "llm_workers_min": max(1, int(llm_workers_min)),
                },
                "timings_sec": {
                    "search": round(search_elapsed_sec, 3),
                    "crawl": round(crawl_elapsed_sec, 3),
                    "llm_extract": round(llm_extract_elapsed_sec, 3),
                    "llm_analyze": round(llm_analyze_elapsed_sec, 3),
                    "export": round(export_elapsed_sec, 3),
                    "total": round(total_elapsed_sec, 3),
                },
            }
            self.task_tracker.complete(task_id, "completed", result)
            return result
        except Exception as exc:
            self.task_tracker.complete(
                task_id, "failed", {"error": f"{type(exc).__name__}: {exc}"}
            )
            raise

    def run_existing_workbook(
        self,
        workbook_id: str,
        only_status: str = "1",
        output_path: str | None = None,
        html_output_path: str | None = None,
        crawl_workers: int = CRAWL_WORKERS,
        llm_workers: int = 24,
        llm_workers_min: int = 8,
    ) -> dict[str, Any]:
        run_started_at = time.perf_counter()
        self._reset_login_recovery_events()
        workbook = self.workbook_service.get(workbook_id)
        products = self.storage.list_products()
        targets = [
            row
            for row in products
            if row["workbook_id"] == workbook_id
            and row["process_status"] == str(only_status)
        ]
        if not targets:
            return {
                "workbook_id": workbook_id,
                "message": f"no items with process_status={only_status}",
            }

        task_id = self.task_tracker.create(
            workbook_id=workbook_id, keyword="", total_items=len(targets)
        )
        effective_crawl_workers = self._effective_crawl_workers(crawl_workers)
        finished = 0
        selling_points = self.storage.read_json(self.storage.selling_points_json)
        for row in targets:
            if not row.get("item_source_url"):
                row["item_source_url"] = row.get("normalized_url", "")
            if not row.get("source_type") and row.get("item_source_url", "").startswith(
                "http"
            ):
                row["source_type"] = "official"

        # Crawl phase - Pipeline parallel: start LLM Extract as soon as each crawl completes
        crawl_started_at = time.perf_counter()
        llm_extract_started_at = time.perf_counter()  # Start LLM timer early for pipeline stats

        llm_inputs: list[tuple[dict[str, str], ItemDetail]] = []
        pending_llm_futures: list[tuple[tuple, Any]] = []

        pool_size = max(1, min(effective_crawl_workers, len(targets)))
        llm_extract_executor = ThreadPoolExecutor(max_workers=llm_workers)

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            future_to_row = {
                executor.submit(
                    self.crawler.crawl,
                    workbook_id,
                    row["item_id"],
                    row["normalized_url"],
                ): row
                for row in targets
            }
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    detail = future.result()
                except Exception as exc:
                    detail = ItemDetail(
                        item_id=row["item_id"],
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
                        error=f"{type(exc).__name__}: {exc}",
                    )

                if detail.error:
                    self.task_tracker.add_failure(task_id, row["item_id"], detail.error)
                    finished += 1
                    self.task_tracker.update_progress(task_id, finished)
                    continue

                self._apply_crawl_detail(row, detail)
                llm_inputs.append((row, detail))

                # Immediately submit to LLM Extract (pipeline parallel)
                llm_future = llm_extract_executor.submit(
                    self._extract_points_for_detail, detail
                )
                pending_llm_futures.append(((row, detail), llm_future))

        llm_extract_executor.shutdown(wait=True)
        crawl_elapsed_sec = self._elapsed_seconds(crawl_started_at)

        # Wait for all LLM Extract tasks to complete
        llm_extract_results: list[tuple[tuple, Any]] = []
        for payload, future in pending_llm_futures:
            try:
                llm_result = future.result()
                llm_extract_results.append((payload, llm_result))
            except Exception as llm_error:
                llm_extract_results.append((payload, llm_error))

        # Process LLM Extract results
        for payload, llm_result in llm_extract_results:
            row, detail = payload
            if isinstance(llm_result, Exception):
                self.task_tracker.add_failure(
                    task_id, row["item_id"], f"LLM extraction failed: {llm_result}"
                )
                row["selling_points_text"] = ""
                row["selling_points_citation"] = ""
                row["updated_at"] = now_iso()
                finished += 1
                self.task_tracker.update_progress(task_id, finished)
                continue

            detail_summary, points = llm_result
            row["detail_summary"] = detail_summary or detail.detail_summary
            if points:
                row["detail_summary"] = SellingPointExtractor.summarize_points(
                    points, fallback=row["detail_summary"]
                )
                row["selling_points_text"] = " | ".join(
                    [point["point"] for point in points]
                )
                row["selling_points_citation"] = " | ".join(
                    [point["citation"] for point in points]
                )
                row["process_status"] = "3"
                selling_points[f"{workbook_id}:{row['item_id']}"] = {
                    "workbook_id": workbook_id,
                    "item_id": row["item_id"],
                    "points": points,
                    "detail_summary": row["detail_summary"],
                    "detail_blocks": detail.detail_blocks,
                    "updated_at": now_iso(),
                }
            row["updated_at"] = now_iso()
            finished += 1
            self.task_tracker.update_progress(task_id, finished)
        llm_extract_elapsed_sec = self._elapsed_seconds(llm_extract_started_at)

        workbook_rows = [r for r in products if r["workbook_id"] == workbook_id]
        workbook_points_map = self._build_workbook_points_map(
            workbook_id, selling_points
        )

        analyze_inputs: list[tuple[dict[str, str], list[dict[str, Any]]]] = []
        for row in workbook_rows:
            item_points = workbook_points_map.get(row["item_id"], [])
            if not item_points:
                row["competitor_analysis_text"] = "暂无有效卖点，跳过单品深度分析。"
                row["process_status"] = "4"
                row["updated_at"] = now_iso()
                continue
            analyze_inputs.append((row, item_points))

        llm_analyze_started_at = time.perf_counter()
        analyze_results = self._run_adaptive_stage(
            items=analyze_inputs,
            worker_fn=lambda payload: self.analyzer.analyze_item(
                payload[0]["item_id"],
                payload[1],
                workbook_points_map,
                item_title=payload[0].get("title", ""),
            ),
            max_workers=llm_workers,
            min_workers=llm_workers_min,
            stage_name="llm-analyze",
        )
        for payload, analyze_result in zip(analyze_inputs, analyze_results):
            row, _ = payload
            if isinstance(analyze_result, Exception):
                self.task_tracker.add_failure(
                    task_id,
                    row["item_id"],
                    f"LLM analyze failed: {analyze_result}",
                )
                continue
            row["competitor_analysis_text"] = analyze_result.get(
                "competitor_analysis_text", ""
            )
            if analyze_result.get("market_tags"):
                row["market_tags"] = " | ".join(analyze_result["market_tags"])
            row["process_status"] = "4"
            row["updated_at"] = now_iso()
        llm_analyze_elapsed_sec = self._elapsed_seconds(llm_analyze_started_at)

        export_started_at = time.perf_counter()
        (
            market_payload,
            batch_summary_text,
            market_summary_text,
            final_conclusion_text,
        ) = self._build_market_outputs(
            workbook_id=workbook_id,
            workbook_name=workbook["workbook_name"],
            workbook_rows=workbook_rows,
            workbook_points_map=workbook_points_map,
        )

        self.storage.save_products(products)
        self.storage.write_json(self.storage.selling_points_json, selling_points)
        self.storage.write_json(self.storage.market_report_json, market_payload)

        export_md_path, export_html_path = self.export_html_and_md(
            workbook_id=workbook_id,
            output_path=output_path,
            html_output_path=html_output_path,
            allow_incomplete=False,
            ignore_running_task_ids={task_id},
        )
        export_elapsed_sec = self._elapsed_seconds(export_started_at)
        total_elapsed_sec = self._elapsed_seconds(run_started_at)
        result = {
            "workbook_id": workbook_id,
            "task_id": task_id,
            "export_md": str(export_md_path),
            "export_csv": str(export_md_path),
            "export_html": str(export_html_path),
            "batch_competitor_summary": batch_summary_text,
            "market_summary": market_summary_text,
            "final_conclusion": final_conclusion_text,
            "login_recovery_events": self._collect_login_recovery_events(),
            "workers": {
                "crawl_workers": effective_crawl_workers,
                "llm_workers": max(1, int(llm_workers)),
                "llm_workers_min": max(1, int(llm_workers_min)),
            },
            "timings_sec": {
                "crawl": round(crawl_elapsed_sec, 3),
                "llm_extract": round(llm_extract_elapsed_sec, 3),
                "llm_analyze": round(llm_analyze_elapsed_sec, 3),
                "export": round(export_elapsed_sec, 3),
                "total": round(total_elapsed_sec, 3),
            },
        }
        self.task_tracker.complete(task_id, "completed", result)
        return result


def print_json(payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))


def _default_storage_state_file() -> str:
    env_value = (os.getenv("TAOBAO_STORAGE_STATE_FILE", "") or "").strip()
    if env_value:
        return env_value
    candidates = [
        Path("backend/data/taobao_storage_state.json"),
        Path("data/taobao_storage_state.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[1])


def build_parser() -> argparse.ArgumentParser:
    manual_wait_default = 0
    manual_login_timeout_default = 300
    gemini_timeout_default = 45
    gemini_pro_retries_default = 2
    shop_filter_default = (os.getenv("SHOP_FILTER", "on") or "on").strip().lower()
    crawl_workers_default = CRAWL_WORKERS
    llm_workers_default = LLM_WORKERS  # 64 for Flash/Vision models
    llm_workers_min_default = LLM_WORKERS_MIN  # 32
    try:
        manual_wait_default = max(0, int(os.getenv("MANUAL_WAIT_SECONDS", "0")))
    except ValueError:
        manual_wait_default = 0
    try:
        manual_login_timeout_default = max(
            30, int(os.getenv("TAOBAO_MANUAL_LOGIN_TIMEOUT_SEC", "300"))
        )
    except ValueError:
        manual_login_timeout_default = 300
    try:
        gemini_timeout_default = max(5, int(os.getenv("GEMINI_TIMEOUT_SEC", "45")))
    except ValueError:
        gemini_timeout_default = 45
    try:
        gemini_pro_retries_default = max(0, int(os.getenv("GEMINI_PRO_RETRIES", "2")))
    except ValueError:
        gemini_pro_retries_default = 2
    if shop_filter_default not in {"on", "off"}:
        shop_filter_default = "on"
    try:
        crawl_workers_default = max(
            1, int(os.getenv("CRAWL_WORKERS", str(CRAWL_WORKERS)))
        )
    except ValueError:
        crawl_workers_default = CRAWL_WORKERS
    try:
        llm_workers_default = max(1, int(os.getenv("LLM_WORKERS", str(LLM_WORKERS))))
    except ValueError:
        llm_workers_default = LLM_WORKERS
    try:
        llm_workers_min_default = max(1, int(os.getenv("LLM_WORKERS_MIN", str(LLM_WORKERS_MIN))))
    except ValueError:
        llm_workers_min_default = LLM_WORKERS_MIN
    if llm_workers_min_default > llm_workers_default:
        llm_workers_min_default = llm_workers_default

    parser = argparse.ArgumentParser(
        description="Taobao market-research pipeline: keyword/search-url -> top-n items -> CSV/HTML outputs."
    )
    parser.add_argument(
        "--data-dir", default=os.getenv("DATA_DIR", "data"), help="Data directory"
    )
    parser.add_argument(
        "--taobao-browser-mode",
        default=os.getenv("TAOBAO_BROWSER_MODE", "cdp"),
        choices=["cdp", "persistent"],
    )
    parser.add_argument(
        "--taobao-storage-state-file",
        default=_default_storage_state_file(),
        help="Playwright storage_state JSON path",
    )
    parser.add_argument(
        "--taobao-user-data-dir",
        default=_default_user_data_dir(),
        help="Persistent browser profile directory",
    )
    parser.add_argument(
        "--taobao-manual-login-timeout-sec",
        type=int,
        default=manual_login_timeout_default,
    )
    parser.add_argument(
        "--playwright-headless",
        default=os.getenv("PLAYWRIGHT_HEADLESS", "0"),
        choices=["0", "1"],
    )
    parser.add_argument(
        "--playwright-cdp-url",
        default=os.getenv(
            "PLAYWRIGHT_CDP_URL",
            os.getenv("TAOBAO_CDP_ENDPOINT", ""),
        ),
        help="Optional CDP endpoint of an existing browser session, e.g. http://127.0.0.1:9222",
    )
    parser.add_argument(
        "--manual-wait-seconds",
        type=int,
        default=manual_wait_default,
        help="Pause after page load to allow manual login/captcha solve before extraction",
    )
    parser.add_argument(
        "--gemini-flash-model",
        default=os.getenv("GEMINI_FLASH_MODEL", "gemini-flash-latest"),
        help="Gemini model for selling point extraction (image+text)",
    )
    parser.add_argument(
        "--gemini-pro-model",
        default=os.getenv("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview"),
        help="Gemini model for competitor analysis, market summary, and conclusions",
    )
    parser.add_argument("--gemini-proxy-url", default=os.getenv("GEMINI_PROXY_URL", ""))
    parser.add_argument(
        "--gemini-timeout-sec", type=int, default=gemini_timeout_default
    )
    parser.add_argument(
        "--gemini-pro-retries",
        type=int,
        default=gemini_pro_retries_default,
        help="Retries for preview/pro model before falling back to flash",
    )
    parser.add_argument(
        "--shop-filter",
        choices=["on", "off"],
        default=shop_filter_default,
        help="Filter search results by preferred shop names: official/flagship/Tmall+self-operated",
    )
    parser.add_argument(
        "--crawl-workers",
        type=int,
        default=crawl_workers_default,
        help="Concurrent workers for item crawling (default: 1; higher values may increase anti-bot risk)",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=llm_workers_default,
        help="Max concurrent workers for LLM extraction/analysis (default: 64 for Flash model)",
    )
    parser.add_argument(
        "--llm-workers-min",
        type=int,
        default=llm_workers_min_default,
        help="Min concurrent workers after transient errors (default: 32)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze-keyword", help="Create workbook and run full flow from one keyword."
    )
    analyze.add_argument("keyword")
    analyze.add_argument("--top-n", type=int, default=5)
    analyze.add_argument(
        "--search-url", default="", help="Optional Taobao search URL to open directly"
    )
    analyze.add_argument(
        "--item-url",
        action="append",
        default=[],
        help="Direct Taobao item URL; repeat for multiple",
    )
    analyze.add_argument(
        "--item-urls-file",
        default="",
        help="Text file containing one item URL per line",
    )
    analyze.add_argument(
        "--search-sort",
        choices=["page", "sales"],
        default="page",
        help="Search result selection order",
    )
    analyze.add_argument(
        "--official-only",
        action="store_true",
        help="Compatibility alias of --shop-filter on",
    )
    analyze.add_argument("--workbook-name", default="")
    analyze.add_argument("--output", default="")
    analyze.add_argument(
        "--html-output",
        default="",
        help="Optional HTML output path; default follows Markdown path",
    )

    final_csv = sub.add_parser(
        "final-csv", help="Alias of analyze-keyword for direct final HTML+Markdown output."
    )
    final_csv.add_argument("keyword")
    final_csv.add_argument("--top-n", type=int, default=5)
    final_csv.add_argument(
        "--search-url", default="", help="Optional Taobao search URL to open directly"
    )
    final_csv.add_argument(
        "--item-url",
        action="append",
        default=[],
        help="Direct Taobao item URL; repeat for multiple",
    )
    final_csv.add_argument(
        "--item-urls-file",
        default="",
        help="Text file containing one item URL per line",
    )
    final_csv.add_argument(
        "--search-sort",
        choices=["page", "sales"],
        default="page",
        help="Search result selection order",
    )
    final_csv.add_argument(
        "--official-only",
        action="store_true",
        help="Compatibility alias of --shop-filter on",
    )
    final_csv.add_argument("--workbook-name", default="")
    final_csv.add_argument("--output", default="")
    final_csv.add_argument(
        "--html-output",
        default="",
        help="Optional HTML output path; default follows Markdown path",
    )

    create_wb = sub.add_parser("create-workbook")
    create_wb.add_argument("workbook_name")
    rename_wb = sub.add_parser("rename-workbook")
    rename_wb.add_argument("workbook_id")
    rename_wb.add_argument("workbook_name")
    delete_wb = sub.add_parser("delete-workbook")
    delete_wb.add_argument("workbook_id")
    sub.add_parser("list-workbooks")

    create_group = sub.add_parser("create-group")
    create_group.add_argument("workbook_id")
    create_group.add_argument("group_name")
    create_group.add_argument("--group-code", default="")
    rename_group = sub.add_parser("rename-group")
    rename_group.add_argument("workbook_id")
    rename_group.add_argument("group_code")
    rename_group.add_argument("group_name")
    move_item = sub.add_parser("move-item-group")
    move_item.add_argument("workbook_id")
    move_item.add_argument("item_id")
    move_item.add_argument("group_code")

    add_urls = sub.add_parser("add-urls")
    add_urls.add_argument("workbook_id")
    add_urls.add_argument("urls", nargs="+")
    add_urls.add_argument("--group-code", default=DEFAULT_GROUP_CODE)
    del_item = sub.add_parser("delete-item")
    del_item.add_argument("workbook_id")
    del_item.add_argument("item_id")

    run_task = sub.add_parser("run-workbook-task")
    run_task.add_argument("workbook_id")
    run_task.add_argument("--only-status", default="1", choices=["1", "2", "3", "4"])
    run_task.add_argument("--output", default="")
    run_task.add_argument(
        "--html-output",
        default="",
        help="Optional HTML output path; default follows Markdown path",
    )

    export_csv = sub.add_parser(
        "export-csv",
        help="Compatibility alias: export Markdown report (with matching HTML).",
    )
    export_csv.add_argument("workbook_id")
    export_csv.add_argument("--output", default="")
    export_csv.add_argument(
        "--html-output",
        default="",
        help="Optional HTML output path; default follows Markdown path",
    )
    export_csv.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow exporting partial/intermediate workbook data.",
    )
    export_html = sub.add_parser("export-html")
    export_html.add_argument("workbook_id")
    export_html.add_argument("--output", default="")
    export_html.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow exporting partial/intermediate workbook data.",
    )
    show_task = sub.add_parser("show-task")
    show_task.add_argument("task_id")
    list_products = sub.add_parser("list-products")
    list_products.add_argument("workbook_id")
    market = sub.add_parser("show-market-report")
    market.add_argument("workbook_id")
    cdp_pages = sub.add_parser(
        "list-cdp-pages", help="List debuggable Chrome pages from CDP endpoint."
    )
    cdp_pages.add_argument(
        "--contains", default="", help="Optional substring filter for title/url"
    )
    return parser


def setup_services(
    args: argparse.Namespace,
) -> tuple[Pipeline, WorkbookService, GroupService, URLService, TaskTracker, Storage]:
    storage = Storage(data_dir=args.data_dir)
    workbook_service = WorkbookService(storage)
    group_service = GroupService(storage)
    url_service = URLService(storage, workbook_service, group_service)
    requested_headless = args.playwright_headless == "1"
    if requested_headless:
        LOG.warning(
            "Headless mode was requested but is forced off for Taobao to reduce anti-bot risk."
        )
    effective_headless = False
    browser_mode = str(args.taobao_browser_mode or "cdp").strip().lower()
    if browser_mode not in {"cdp", "persistent"}:
        browser_mode = "cdp"
    storage_state_file = Path(
        args.taobao_storage_state_file or _default_storage_state_file()
    )
    user_data_dir = Path(args.taobao_user_data_dir or _default_user_data_dir())
    search_client = SearchClient(
        headless=effective_headless,
        browser_mode=browser_mode,
        cdp_url=args.playwright_cdp_url or "",
        manual_wait_seconds=max(0, int(args.manual_wait_seconds)),
        storage_state_file=storage_state_file,
        user_data_dir=user_data_dir,
        manual_login_timeout_sec=max(30, int(args.taobao_manual_login_timeout_sec)),
    )
    crawler = Crawler(
        storage=storage,
        headless=effective_headless,
        browser_mode=browser_mode,
        cdp_url=args.playwright_cdp_url or "",
        manual_wait_seconds=max(0, int(args.manual_wait_seconds)),
        storage_state_file=storage_state_file,
        user_data_dir=user_data_dir,
        manual_login_timeout_sec=max(30, int(args.taobao_manual_login_timeout_sec)),
    )
    extractor = SellingPointExtractor(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_flash_model=args.gemini_flash_model,
        gemini_proxy_url=args.gemini_proxy_url or "",
        gemini_timeout_sec=max(5, int(args.gemini_timeout_sec)),
        gemini_raise_on_transient=True,
    )
    analyzer = Analyzer(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_flash_model=args.gemini_flash_model,
        gemini_pro_model=args.gemini_pro_model,
        gemini_proxy_url=args.gemini_proxy_url or "",
        gemini_timeout_sec=max(5, int(args.gemini_timeout_sec)),
        gemini_raise_on_transient=True,
        gemini_pro_retries=max(0, int(args.gemini_pro_retries)),
    )
    task_tracker = TaskTracker(storage)
    pipeline = Pipeline(
        storage=storage,
        workbook_service=workbook_service,
        group_service=group_service,
        url_service=url_service,
        search_client=search_client,
        crawler=crawler,
        extractor=extractor,
        analyzer=analyzer,
        task_tracker=task_tracker,
    )
    return pipeline, workbook_service, group_service, url_service, task_tracker, storage


def main(argv: list[str] | None = None) -> int:
    load_simple_dotenv(Path(".env"))
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    pipeline, workbook_service, group_service, url_service, task_tracker, storage = (
        setup_services(args)
    )
    try:
        if args.command in {"analyze-keyword", "final-csv"}:
            import asyncio
            from tools import cleanup_global_browser

            input_urls = list(args.item_url or [])
            input_urls.extend(load_url_lines(args.item_urls_file or None))
            shop_filter_enabled = str(args.shop_filter or "on").strip().lower() == "on"
            if bool(getattr(args, "official_only", False)):
                shop_filter_enabled = True

            # Use async version to maintain single event loop
            payload = asyncio.run(pipeline.run_keyword_async(
                keyword=args.keyword,
                top_n=max(1, min(args.top_n, MAX_TOP_N)),
                workbook_name=args.workbook_name or None,
                search_url=args.search_url or None,
                search_sort=args.search_sort,
                shop_filter_enabled=shop_filter_enabled,
                output_path=args.output or None,
                html_output_path=args.html_output or None,
                input_urls=input_urls,
                crawl_workers=max(1, int(args.crawl_workers)),
                llm_workers=max(1, int(args.llm_workers)),
                llm_workers_min=max(1, int(args.llm_workers_min)),
            ))

            # Cleanup browser to save cookies and session
            # This is important for persistent mode to save login state
            try:
                asyncio.run(cleanup_global_browser())
                LOG.info("Browser cleaned up, cookies saved")
            except Exception as cleanup_exc:
                LOG.warning(f"Browser cleanup failed (non-critical): {cleanup_exc}")

            print_json(payload)

            # Print timing summary
            if "timings_sec" in payload:
                t = payload["timings_sec"]
                LOG.info(
                    f"Timing summary: search={t.get('search', 0):.1f}s, "
                    f"crawl={t.get('crawl', 0):.1f}s, "
                    f"llm_extract={t.get('llm_extract', 0):.1f}s, "
                    f"llm_analyze={t.get('llm_analyze', 0):.1f}s, "
                    f"export={t.get('export', 0):.1f}s, "
                    f"total={t.get('total', 0):.1f}s"
                )

            return 0
        if args.command == "create-workbook":
            print_json(workbook_service.create(args.workbook_name))
            return 0
        if args.command == "rename-workbook":
            workbook_service.rename(args.workbook_id, args.workbook_name)
            print_json(
                {
                    "ok": True,
                    "workbook_id": args.workbook_id,
                    "workbook_name": args.workbook_name,
                }
            )
            return 0
        if args.command == "delete-workbook":
            workbook_service.delete(args.workbook_id)
            print_json({"ok": True, "workbook_id": args.workbook_id})
            return 0
        if args.command == "list-workbooks":
            print_json(workbook_service.list())
            return 0
        if args.command == "create-group":
            print_json(
                group_service.create(
                    args.workbook_id, args.group_name, args.group_code or None
                )
            )
            return 0
        if args.command == "rename-group":
            group_service.rename(args.workbook_id, args.group_code, args.group_name)
            print_json({"ok": True})
            return 0
        if args.command == "move-item-group":
            group_service.move_item(args.workbook_id, args.item_id, args.group_code)
            print_json({"ok": True})
            return 0
        if args.command == "add-urls":
            changed = url_service.add_urls(args.workbook_id, args.urls, args.group_code)
            print_json({"changed": len(changed), "items": changed})
            return 0
        if args.command == "delete-item":
            url_service.delete_item(args.workbook_id, args.item_id)
            print_json({"ok": True})
            return 0
        if args.command == "run-workbook-task":
            payload = pipeline.run_existing_workbook(
                args.workbook_id,
                only_status=args.only_status,
                output_path=args.output or None,
                html_output_path=args.html_output or None,
                crawl_workers=max(1, int(args.crawl_workers)),
                llm_workers=max(1, int(args.llm_workers)),
                llm_workers_min=max(1, int(args.llm_workers_min)),
            )
            print_json(payload)
            return 0
        if args.command == "export-csv":
            out_md, out_html = pipeline.export_html_and_md(
                args.workbook_id,
                output_path=args.output or None,
                html_output_path=args.html_output or None,
                allow_incomplete=bool(getattr(args, "allow_incomplete", False)),
            )
            print_json(
                {
                    "workbook_id": args.workbook_id,
                    "export_md": str(out_md),
                    "export_csv": str(out_md),
                    "export_html": str(out_html),
                }
            )
            return 0
        if args.command == "export-html":
            out = pipeline.export_html(
                args.workbook_id,
                args.output or None,
                allow_incomplete=bool(getattr(args, "allow_incomplete", False)),
            )
            print_json({"workbook_id": args.workbook_id, "export_html": str(out)})
            return 0
        if args.command == "show-task":
            print_json(task_tracker.get(args.task_id))
            return 0
        if args.command == "list-products":
            print_json(
                [
                    r
                    for r in storage.list_products()
                    if r["workbook_id"] == args.workbook_id
                ]
            )
            return 0
        if args.command == "show-market-report":
            report = storage.read_json(storage.market_report_json).get(
                args.workbook_id, {}
            )
            print_json(report)
            return 0
        if args.command == "list-cdp-pages":
            endpoint = (args.playwright_cdp_url or "").strip()
            if not endpoint:
                raise ValueError("--playwright-cdp-url is required for list-cdp-pages")
            pages = list_cdp_pages(endpoint)
            token = (args.contains or "").strip().lower()
            if token:
                pages = [
                    p
                    for p in pages
                    if token in p.get("url", "").lower()
                    or token in p.get("title", "").lower()
                ]
            print_json({"endpoint": endpoint, "pages": pages, "count": len(pages)})
            return 0
        parser.print_help()
        return 1
    except Exception as exc:
        LOG.error("%s: %s", type(exc).__name__, exc)
        LOG.debug("Traceback:\n%s", traceback.format_exc())
        print_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
