"""Data models for standalone marketplace review collection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from data import clean_text


REVIEW_CSV_COLUMNS = [
    "platform",
    "item_id",
    "item_url",
    "title",
    "shop_name",
    "brand",
    "comment_id",
    "comment_time",
    "comment_text",
    "rating",
    "sku_info",
    "user_name_masked",
    "user_level",
    "is_anonymous",
    "is_append",
    "append_time",
    "append_text",
    "like_count",
    "reply_count",
    "has_images",
    "has_video",
    "raw_tags",
    "raw_source",
    "collected_at",
]


@dataclass
class ReviewRecord:
    platform: str
    item_id: str
    item_url: str
    title: str = ""
    shop_name: str = ""
    brand: str = ""
    comment_id: str = ""
    comment_time: str = ""
    comment_text: str = ""
    rating: str = ""
    sku_info: str = ""
    user_name_masked: str = ""
    user_level: str = ""
    is_anonymous: bool = False
    is_append: bool = False
    append_time: str = ""
    append_text: str = ""
    like_count: int = 0
    reply_count: int = 0
    has_images: bool = False
    has_video: bool = False
    raw_tags: list[str] = field(default_factory=list)
    raw_source: Any = ""
    collected_at: str = ""

    def identity(self) -> str:
        comment_id = clean_text(self.comment_id, max_len=160)
        if comment_id:
            return f"id:{comment_id}"
        base = "|".join(
            [
                clean_text(self.comment_time, max_len=80),
                clean_text(self.comment_text, max_len=240),
                clean_text(self.sku_info, max_len=120),
            ]
        )
        return f"text:{base}"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_csv_row(self) -> dict[str, str]:
        raw_source = self.raw_source
        if isinstance(raw_source, (dict, list)):
            raw_source = json.dumps(raw_source, ensure_ascii=False, separators=(",", ":"))
        return {
            "platform": self.platform,
            "item_id": self.item_id,
            "item_url": self.item_url,
            "title": self.title,
            "shop_name": self.shop_name,
            "brand": self.brand,
            "comment_id": self.comment_id,
            "comment_time": self.comment_time,
            "comment_text": self.comment_text,
            "rating": self.rating,
            "sku_info": self.sku_info,
            "user_name_masked": self.user_name_masked,
            "user_level": self.user_level,
            "is_anonymous": "1" if self.is_anonymous else "0",
            "is_append": "1" if self.is_append else "0",
            "append_time": self.append_time,
            "append_text": self.append_text,
            "like_count": str(max(0, int(self.like_count or 0))),
            "reply_count": str(max(0, int(self.reply_count or 0))),
            "has_images": "1" if self.has_images else "0",
            "has_video": "1" if self.has_video else "0",
            "raw_tags": " | ".join(self.raw_tags),
            "raw_source": clean_text(str(raw_source or ""), max_len=800),
            "collected_at": self.collected_at,
        }


@dataclass
class ReviewItemResult:
    platform: str
    item_id: str
    item_url: str
    title: str = ""
    shop_name: str = ""
    brand: str = ""
    reviews: list[ReviewRecord] = field(default_factory=list)
    collected_count: int = 0
    cutoff_time: str = ""
    stopped_reason: str = ""
    login_recovery_events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def ok(self) -> bool:
        return not self.error


@dataclass
class ReviewRunSummary:
    platform: str
    target_name: str
    months: int
    days: int
    limit: int
    item_count: int
    success_items: int
    failed_items: int
    total_reviews: int
    output_dir: str
    jsonl_path: str
    csv_path: str
    run_summary_json_path: str
    run_summary_md_path: str
    per_item_counts: list[dict[str, Any]] = field(default_factory=list)
    stopped_reasons: list[dict[str, Any]] = field(default_factory=list)
    top_errors: list[dict[str, Any]] = field(default_factory=list)
    login_recovery_event_count: int = 0
    created_at: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
