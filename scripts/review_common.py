"""Shared helpers for standalone marketplace review collection."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any, Iterator

from data import clean_text
from review_models import ReviewRecord


def utc_now_local() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).astimezone()


def subtract_months(base: dt.datetime, months: int) -> dt.datetime:
    value = max(0, int(months))
    year = base.year
    month = base.month - value
    while month <= 0:
        month += 12
        year -= 1
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    month_start = base.replace(year=year, month=month, day=1)
    next_month_start = month_start.replace(year=next_year, month=next_month, day=1)
    last_day = (next_month_start - dt.timedelta(days=1)).day
    day = min(base.day, last_day)
    return month_start.replace(
        day=day,
        hour=base.hour,
        minute=base.minute,
        second=base.second,
        microsecond=base.microsecond,
    )


def subtract_days(base: dt.datetime, days: int) -> dt.datetime:
    value = max(0, int(days))
    return base - dt.timedelta(days=value)


def parse_review_datetime(value: Any, *, now: dt.datetime | None = None) -> dt.datetime | None:
    if value in (None, ""):
        return None
    base_now = now or utc_now_local()
    if isinstance(value, dt.datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=base_now.tzinfo)
    if isinstance(value, (int, float)):
        stamp = float(value)
        if stamp > 1e12:
            stamp /= 1000.0
        if stamp <= 0:
            return None
        return dt.datetime.fromtimestamp(stamp, tz=base_now.tzinfo)

    text = clean_text(str(value or ""), max_len=120)
    if not text:
        return None
    lowered = text.lower()
    if lowered.isdigit():
        return parse_review_datetime(int(lowered), now=base_now)

    relative_patterns = [
        (r"^今天\s*(\d{1,2}):(\d{2})(?::(\d{2}))?$", 0),
        (r"^昨日?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?$", 1),
        (r"^前天\s*(\d{1,2}):(\d{2})(?::(\d{2}))?$", 2),
    ]
    for pattern, days_back in relative_patterns:
        matched = re.match(pattern, text)
        if not matched:
            continue
        target = base_now - dt.timedelta(days=days_back)
        second = int(matched.group(3) or "0")
        return target.replace(
            hour=int(matched.group(1)),
            minute=int(matched.group(2)),
            second=second,
            microsecond=0,
        )

    days_ago_match = re.match(r"^(\d+)\s*天前$", text)
    if days_ago_match:
        return (base_now - dt.timedelta(days=int(days_ago_match.group(1)))).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    month_day_match = re.match(r"^(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$", text)
    if month_day_match:
        month = int(month_day_match.group(1))
        day = int(month_day_match.group(2))
        hour = int(month_day_match.group(3) or "0")
        minute = int(month_day_match.group(4) or "0")
        second = int(month_day_match.group(5) or "0")
        year = base_now.year
        try:
            parsed = dt.datetime(
                year,
                month,
                day,
                hour,
                minute,
                second,
                tzinfo=base_now.tzinfo,
            )
        except ValueError:
            parsed = None
        if parsed is not None and parsed > base_now + dt.timedelta(days=2):
            parsed = parsed.replace(year=year - 1)
        if parsed is not None:
            return parsed

    normalized = (
        text.replace("年", "-")
        .replace("月", "-")
        .replace("日", "")
        .replace("/", "-")
        .replace(".", "-")
        .replace("T", " ")
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", normalized)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    fmts = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y-%m",
        "%Y%m%d%H%M%S",
        "%Y%m%d",
    )
    for fmt in fmts:
        try:
            parsed = dt.datetime.strptime(normalized, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=base_now.tzinfo)
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=base_now.tzinfo)
    return parsed.astimezone(base_now.tzinfo)


def format_review_datetime(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone().isoformat(timespec="seconds")


def flatten_text(value: Any, *, max_len: int = 600) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [clean_text(chunk, max_len=max_len) for chunk in re.split(r"[\n\t]+", text)]
    merged = " ".join(part for part in parts if part)
    return clean_text(merged, max_len=max_len)


def flatten_tags(value: Any) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        parts = re.split(r"[|,，/；;]+", value)
        return [clean_text(part, max_len=40) for part in parts if clean_text(part, max_len=40)]
    if isinstance(value, (list, tuple, set)):
        tags: list[str] = []
        for item in value:
            tag = clean_text(str(item or ""), max_len=40)
            if tag:
                tags.append(tag)
        deduped: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(tag)
        return deduped
    return [clean_text(str(value), max_len=40)]


def pick_first_non_empty(container: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in container:
            continue
        value = container.get(key)
        if value in (None, "", [], {}):
            continue
        return value
    return None


def compose_text_parts(values: list[Any], *, max_len: int = 200) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = flatten_text(value, max_len=max_len)
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return clean_text(" | ".join(parts), max_len=max_len)


def mask_user_name(value: Any) -> str:
    text = clean_text(str(value or ""), max_len=40)
    if not text:
        return ""
    if len(text) <= 2:
        return text[0] + "*"
    if len(text) == 3:
        return text[0] + "*" + text[-1]
    return text[0] + "***" + text[-1]


def normalize_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text:
        return 0
    matched = re.search(r"-?\d+", text.replace(",", ""))
    if not matched:
        return 0
    try:
        return int(matched.group(0))
    except ValueError:
        return 0


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on", "匿名", "是", "有"}


def iter_nested_dicts(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            yield from iter_nested_dicts(value)
        return
    if isinstance(payload, list):
        for item in payload:
            yield from iter_nested_dicts(item)


def looks_like_review_dict(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict) or not row:
        return False
    keys = {str(key).lower() for key in row.keys()}
    text_keys = {"content", "commentdata", "commenttext", "ratecontent", "feedback", "text"}
    time_keys = {
        "creationtime",
        "commenttime",
        "commentdate",
        "ratedate",
        "feedbackdate",
        "datetime",
        "time",
        "date",
    }
    identity_keys = {"commentid", "guid", "comment_id", "id"}
    review_meta_keys = {
        "commentscore",
        "score",
        "star",
        "starlevel",
        "replycnt",
        "replycount",
        "praisecnt",
        "likecount",
        "wareattribute",
        "referencename",
        "productcolor",
        "productsize",
        "pictureinfolist",
        "imagelist",
        "images",
    }
    if keys & text_keys and keys & time_keys:
        return True
    if keys & text_keys and keys & identity_keys and (keys & time_keys or keys & review_meta_keys):
        return True
    if "appendcomment" in keys and keys & time_keys:
        return True
    return False


def dedupe_reviews(records: list[ReviewRecord]) -> list[ReviewRecord]:
    ordered: list[ReviewRecord] = []
    seen: set[str] = set()
    for record in records:
        key = record.identity()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(record)
    return ordered


def filter_and_limit_reviews(
    records: list[ReviewRecord],
    *,
    cutoff: dt.datetime,
    limit: int,
) -> list[ReviewRecord]:
    filtered: list[tuple[dt.datetime, ReviewRecord]] = []
    fallback_now = utc_now_local()
    for record in dedupe_reviews(records):
        parsed = parse_review_datetime(record.comment_time, now=fallback_now)
        if parsed is None or parsed < cutoff:
            continue
        filtered.append((parsed, record))
    filtered.sort(key=lambda item: item[0], reverse=True)
    effective_limit = int(limit)
    if effective_limit <= 0:
        return [item[1] for item in filtered]
    return [item[1] for item in filtered[:effective_limit]]


def summarize_reason_counts(values: list[str], *, limit: int = 5) -> list[dict[str, Any]]:
    counter: dict[str, int] = {}
    for value in values:
        key = clean_text(str(value or "Unknown"), max_len=120) or "Unknown"
        counter[key] = counter.get(key, 0) + 1
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [{"reason": key, "count": count} for key, count in ranked[: max(1, int(limit))]]
