"""Export helpers for standalone review collection."""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any

from data import Storage, artifact_scope_dir, safe_path_slug
from review_models import REVIEW_CSV_COLUMNS, ReviewItemResult, ReviewRunSummary


class ReviewExporter:
    @staticmethod
    def default_output_dir(
        *,
        data_dir: str | Path,
        platform: str,
        target_name: str,
        months: int,
        days: int,
        limit: int,
        source_mode: str = "input_urls",
    ) -> Path:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        window = f"d{days}" if int(days) > 0 else f"m{months}"
        limit_token = "all" if int(limit) <= 0 else f"n{limit}"
        slug = safe_path_slug(target_name, default="reviews", lowercase=False)
        scope_dir = artifact_scope_dir(Path(data_dir) / "reviews_exports", platform, source_mode)
        return scope_dir / f"{slug}-{window}-{limit_token}-{stamp}"

    @staticmethod
    def export(
        *,
        output_dir: str | Path,
        item_results: list[ReviewItemResult],
        summary_payload: dict[str, Any],
    ) -> tuple[Path, Path, Path, Path]:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        jsonl_path = path / "reviews.jsonl"
        csv_path = path / "reviews.csv"
        summary_json_path = path / "run-summary.json"
        summary_md_path = path / "run-summary.md"

        rows = []
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for item_result in item_results:
                for review in item_result.reviews:
                    fh.write(json.dumps(review.to_json_dict(), ensure_ascii=False) + "\n")
                    rows.append(review.to_csv_row())

        Storage.write_csv(
            csv_path,
            REVIEW_CSV_COLUMNS,
            rows,
            excel_friendly=True,
        )
        Storage.write_json(summary_json_path, summary_payload)
        summary_md_path.write_text(
            ReviewExporter._render_summary_md(summary_payload),
            encoding="utf-8",
        )
        return jsonl_path, csv_path, summary_json_path, summary_md_path

    @staticmethod
    def _render_summary_md(summary_payload: dict[str, Any]) -> str:
        lines = [
            "# Review Collection Summary",
            "",
            f"- platform: {summary_payload.get('platform', '')}",
            f"- target_name: {summary_payload.get('target_name', '')}",
            f"- months: {summary_payload.get('months', 0)}",
            f"- days: {summary_payload.get('days', 0)}",
            f"- limit: {summary_payload.get('limit', 0)}",
            f"- item_count: {summary_payload.get('item_count', 0)}",
            f"- success_items: {summary_payload.get('success_items', 0)}",
            f"- failed_items: {summary_payload.get('failed_items', 0)}",
            f"- total_reviews: {summary_payload.get('total_reviews', 0)}",
            f"- login_recovery_event_count: {summary_payload.get('login_recovery_event_count', 0)}",
            f"- created_at: {summary_payload.get('created_at', '')}",
            "",
            "## Per Item Counts",
        ]
        per_item_counts = summary_payload.get("per_item_counts", [])
        if isinstance(per_item_counts, list) and per_item_counts:
            for item in per_item_counts:
                lines.append(
                    "- "
                    f"{item.get('platform', '')}:{item.get('item_id', '')} "
                    f"{item.get('count', 0)} "
                    f"stopped_reason={item.get('stopped_reason', '')}"
                )
        else:
            lines.append("- none")

        lines.extend(["", "## Top Errors"])
        top_errors = summary_payload.get("top_errors", [])
        if isinstance(top_errors, list) and top_errors:
            for item in top_errors:
                lines.append(f"- {item.get('reason', '')}: {item.get('count', 0)}")
        else:
            lines.append("- none")

        lines.extend(["", "## Stopped Reasons"])
        stopped_reasons = summary_payload.get("stopped_reasons", [])
        if isinstance(stopped_reasons, list) and stopped_reasons:
            for item in stopped_reasons:
                lines.append(f"- {item.get('reason', '')}: {item.get('count', 0)}")
        else:
            lines.append("- none")
        return "\n".join(lines).rstrip() + "\n"
