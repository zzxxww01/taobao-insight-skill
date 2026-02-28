"""Data models, storage, and CRUD services for the Taobao market research pipeline."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from html import escape as html_escape, unescape as html_unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urlparse

from config import (
    ANY_URL_RE,
    BRAND_PRODUCT_KEYWORD_RE,
    BRAND_TOKEN_SPLIT_RE,
    DEFAULT_GROUP_CODE,
    DEFAULT_GROUP_NAME,
    DIGITS_ONLY_RE,
    GENERIC_BRAND_TOKENS_LOWER,
    ITEM_ID_IN_TEXT_RE,
    ITEM_ID_RE,
    ITEM_URL_RE,
    JSON_ITEM_ID_RE,
    LOGIN_COOKIE_NAMES,
    OFFICIAL_SHOP_MARKER_RE,
    PRICE_RE,
    PRODUCT_COLUMNS,
    PROMO_TITLE_PREFIX_RE,
    SALES_TEXT_RE,
    SEARCH_QUERY_PARAM_ALIASES,
    SHOP_SUFFIX_RE,
    SHOP_TRAILING_CATEGORY_RE,
    SKU_ID_RE,
    TAOBAO_DOMAINS,
    TITLE_SITE_SUFFIX_RE,
    TMALL_HINT_RE,
    WHITESPACE_RE,
    WORKBOOK_COLUMNS,
)


@dataclass
class UrlRecord:
    raw_url: str
    normalized_url: str
    item_id: str
    sku_id: str | None
    item_source_url: str = ""
    source_type: str = "official"
    search_rank: int | None = None
    title: str = ""
    shop_name: str = ""
    sales_text: str = ""
    is_official_store: bool = False


@dataclass
class ItemDetail:
    item_id: str
    title: str
    main_image_url: str
    shop_name: str
    brand: str
    prices: list[float]
    skus: list[dict[str, str]]
    detail_summary: str
    detail_text: str
    detail_blocks: list[dict[str, str]]
    citations: list[str]
    crawl_time: str
    error: str = ""


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def clean_text(value: str, max_len: int = 300) -> str:
    if not value:
        return ""
    value = WHITESPACE_RE.sub(" ", value).strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def preserve_multiline_text(value: Any, max_len: int = 4000) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def normalize_item_title(value: str, max_len: int = 160) -> str:
    text = clean_text(value, max_len=max_len * 2)
    if not text:
        return ""
    text = html_unescape(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = TITLE_SITE_SUFFIX_RE.sub("", text).strip(" -_|·•")
    text = WHITESPACE_RE.sub(" ", text).strip(" -_|·•")
    return clean_text(text, max_len=max_len)


def _clean_brand_candidate(value: str, max_len: int = 40) -> str:
    text = clean_text(value, max_len=max_len * 3)
    if not text:
        return ""
    text = html_unescape(text)
    text = WHITESPACE_RE.sub(" ", text).strip(" -_|·•\"'()[]{}【】（）")
    return clean_text(text, max_len=max_len)


def _looks_like_brand_candidate(value: str) -> bool:
    text = _clean_brand_candidate(value, max_len=40)
    if not text or len(text) < 2:
        return False
    normalized = WHITESPACE_RE.sub("", text).lower()
    if normalized in GENERIC_BRAND_TOKENS_LOWER:
        return False
    if len(text) > 36:
        return False
    if BRAND_PRODUCT_KEYWORD_RE.search(text) and len(text) >= 8:
        return False
    if " " in text and len(text) > 18:
        return False
    return True


def _extract_brand_token(value: str) -> str:
    text = normalize_item_title(value, max_len=140)
    if not text:
        return ""
    text = PROMO_TITLE_PREFIX_RE.sub("", text).strip()
    text = SHOP_SUFFIX_RE.sub("", text).strip(" -_|·•")
    text = SHOP_TRAILING_CATEGORY_RE.sub("", text).strip(" -_|·•")
    if not text:
        return ""

    latin_mix = re.match(
        r"^([A-Za-z][A-Za-z0-9]{1,19}(?:[·\-/]?[A-Za-z0-9]{1,10})?)([\u4e00-\u9fff]{1,6})?",
        text,
    )
    if latin_mix:
        combo = (latin_mix.group(1) + (latin_mix.group(2) or "")).strip()
        if _looks_like_brand_candidate(combo):
            return clean_text(combo, max_len=24)
        latin = latin_mix.group(1)
        if _looks_like_brand_candidate(latin):
            return clean_text(latin, max_len=24)

    head = BRAND_TOKEN_SPLIT_RE.split(text, maxsplit=1)[0]
    head = re.split(
        r"(粉饼|蜜粉|散粉|定妆|补妆|控油|遮瑕|持久|清透|雾面|哑光|柔焦|磨皮|防晒|防水|防汗|锁妆)",
        head,
        maxsplit=1,
    )[0]
    head = _clean_brand_candidate(head, max_len=24)
    if _looks_like_brand_candidate(head):
        return head
    return ""


def normalize_brand_name(brand: str, title: str = "", shop_name: str = "") -> str:
    ordered_candidates = [
        _extract_brand_token(shop_name),
        _extract_brand_token(brand),
        _extract_brand_token(title),
    ]
    shop = _clean_brand_candidate(shop_name, max_len=80)
    if shop:
        shop = SHOP_SUFFIX_RE.sub("", shop).strip(" -_|·•")
        shop = SHOP_TRAILING_CATEGORY_RE.sub("", shop).strip(" -_|·•")
        ordered_candidates.append(clean_text(shop, max_len=30))
    direct = _clean_brand_candidate(brand, max_len=40)
    if _looks_like_brand_candidate(direct):
        ordered_candidates.append(clean_text(direct, max_len=30))

    seen: set[str] = set()
    for candidate in ordered_candidates:
        value = _clean_brand_candidate(candidate, max_len=30)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        if _looks_like_brand_candidate(value):
            return value
    return ""


def parse_price_values(text: str) -> list[float]:
    prices: list[float] = []
    for token in PRICE_RE.findall(text):
        try:
            value = float(token)
        except ValueError:
            continue
        if 0 < value < 100000:
            prices.append(value)
    return sorted(prices)


def read_text_utf8_best(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_storage_state(storage_state_file: Path) -> dict[str, Any]:
    if not storage_state_file.exists():
        return {}
    try:
        payload = json.loads(read_text_utf8_best(storage_state_file))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def load_valid_cookies(
    storage_state_file: Path, min_ttl_sec: int = 120
) -> list[dict[str, Any]]:
    payload = _read_storage_state(storage_state_file)
    now_ts = int(time.time())
    valid: list[dict[str, Any]] = []
    for cookie in payload.get("cookies", []):
        if not isinstance(cookie, dict):
            continue
        domain = str(cookie.get("domain", "")).lower()
        if not any(d in domain for d in TAOBAO_DOMAINS):
            continue
        expires = cookie.get("expires")
        if expires in (None, "", -1):
            valid.append(cookie)
            continue
        try:
            exp_ts = int(float(expires))
        except Exception:
            continue
        if exp_ts > now_ts + min_ttl_sec:
            valid.append(cookie)
    return valid


def has_valid_login_cookie(storage_state_file: Path, min_ttl_sec: int = 120) -> bool:
    for cookie in load_valid_cookies(storage_state_file, min_ttl_sec=min_ttl_sec):
        name = str(cookie.get("name", "")).strip()
        if name in LOGIN_COOKIE_NAMES:
            return True
    return False


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def csv_cell(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return ""
    # Keep large ids and SKU values as plain text in Excel.
    if text.isdigit() and len(text) >= 12:
        return "'" + text
    # Prevent formula execution when CSV is opened by spreadsheet apps.
    if text[0] in {"=", "+", "-", "@"}:
        return "'" + text
    return text


def looks_like_tmall(url_text: str) -> bool:
    text = (url_text or "").strip()
    if not text:
        return False
    if TMALL_HINT_RE.search(text):
        return True
    parsed = urlparse(text)
    host = (parsed.netloc or "").lower()
    return "tmall.com" in host or "tmall.hk" in host


def _pick_query_value(
    query_pairs: list[tuple[str, str]], aliases: set[str]
) -> str | None:
    for key, value in query_pairs:
        if (key or "").strip().lower() in aliases and (value or "").strip():
            return (value or "").strip()
    return None


def _build_rich_item_url(
    parsed: Any, is_tmall: bool, item_id: str, sku_id: str | None
) -> str:
    scheme = (parsed.scheme or "https").lower()
    if scheme not in {"http", "https"}:
        scheme = "https"
    host = "detail.tmall.com" if is_tmall else "item.taobao.com"
    path = "/item.htm"

    # Keep existing query params to preserve detail-page options/context flags.
    original_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    merged_pairs: list[tuple[str, str]] = []
    id_written = False
    sku_written = False
    for key, value in original_pairs:
        key_clean = (key or "").strip()
        if not key_clean:
            continue
        key_lower = key_clean.lower()
        if key_lower == "id":
            if not id_written:
                merged_pairs.append(("id", item_id))
                id_written = True
            continue
        if key_lower in {"skuid", "sku_id"}:
            if sku_id and not sku_written:
                merged_pairs.append(("skuId", sku_id))
                sku_written = True
            continue
        merged_pairs.append((key_clean, (value or "").strip()))

    if not id_written:
        merged_pairs.insert(0, ("id", item_id))
    if sku_id and not sku_written:
        insert_index = 1 if merged_pairs and merged_pairs[0][0] == "id" else 0
        merged_pairs.insert(insert_index, ("skuId", sku_id))

    query = urlencode(merged_pairs, doseq=True)
    if query:
        return f"{scheme}://{host}{path}?{query}"
    return f"{scheme}://{host}{path}?id={item_id}"


def normalize_url(raw_url: str) -> UrlRecord | None:
    raw = (raw_url or "").strip()
    if not raw:
        return None
    if DIGITS_ONLY_RE.match(raw):
        return UrlRecord(
            raw_url=raw_url,
            normalized_url=f"https://item.taobao.com/item.htm?id={raw}",
            item_id=raw,
            sku_id=None,
            item_source_url=f"https://item.taobao.com/item.htm?id={raw}",
            source_type="official",
        )
    if raw.startswith("//"):
        raw = "https:" + raw
    elif not raw.lower().startswith("http"):
        host_hint = raw.lower()
        if "item.taobao.com" in host_hint or "detail.tmall.com" in host_hint:
            raw = "https://" + raw.lstrip("/")
        else:
            return None

    parsed = urlparse(raw)
    query = parse_qs(parsed.query)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    item_id = _pick_query_value(query_pairs, {"id"}) or (query.get("id") or [None])[0]
    sku_id = (
        _pick_query_value(query_pairs, {"skuid", "sku_id"})
        or (query.get("skuId") or query.get("skuid") or [None])[0]
    )
    is_tmall = looks_like_tmall(raw)

    if not is_tmall:
        for key in (
            "url",
            "target",
            "redirect",
            "redirect_url",
            "tu",
            "dest",
            "itemurl",
        ):
            for maybe_url in query.get(key, []):
                if looks_like_tmall(maybe_url):
                    is_tmall = True
                    break
            if is_tmall:
                break

    if not item_id:
        match = ITEM_ID_RE.search(raw)
        if match:
            item_id = match.group(1)
    if not sku_id:
        sku_match = SKU_ID_RE.search(raw)
        if sku_match:
            sku_id = sku_match.group(1)

    if not item_id:
        decoded = unquote(raw)
        if decoded != raw:
            return normalize_url(decoded)
        return None
    if not item_id.isdigit() or len(item_id) < 6:
        return None

    normalized = _build_rich_item_url(
        parsed=parsed, is_tmall=is_tmall, item_id=item_id, sku_id=sku_id
    )
    return UrlRecord(
        raw_url=raw_url,
        normalized_url=normalized,
        item_id=item_id,
        sku_id=sku_id,
        item_source_url=normalized,
        source_type="official",
    )


def parse_sales_to_int(text: str) -> int:
    source = (text or "").strip()
    if not source:
        return 0
    match = SALES_TEXT_RE.search(source)
    if not match:
        return 0
    try:
        number = float(match.group(1))
    except ValueError:
        return 0
    if match.group(2):
        number *= 10000.0
    return int(number)


def _normalize_search_keyword(value: str) -> str:
    text = WHITESPACE_RE.sub("", str(value or "")).strip().lower()
    return text


def _extract_search_keyword_from_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    except Exception:
        return ""
    for key, value in query_pairs:
        if (key or "").strip().lower() not in SEARCH_QUERY_PARAM_ALIASES:
            continue
        token = _normalize_search_keyword(value)
        if token:
            return token
    return ""


def _is_taobao_search_page(url: str) -> bool:
    return "s.taobao.com/search" in (url or "").lower()


def should_reuse_search_page(current_url: str, keyword: str) -> bool:
    if not _is_taobao_search_page(current_url):
        return False
    current_keyword = _extract_search_keyword_from_url(current_url)
    target_keyword = _normalize_search_keyword(keyword)
    if not current_keyword or not target_keyword:
        return False
    return current_keyword == target_keyword


def is_official_shop(shop_name: str, card_text: str = "") -> bool:
    shop = str(shop_name or "").strip()
    if shop and OFFICIAL_SHOP_MARKER_RE.search(shop):
        return True
    if not shop:
        fallback = str(card_text or "").strip()
        if fallback and OFFICIAL_SHOP_MARKER_RE.search(fallback):
            return True
    return False


def extract_candidate_item_urls(text: str, limit: int = 300) -> list[str]:
    if not text:
        return []

    candidates: list[str] = []
    seen_item_ids: set[str] = set()
    variants: list[str] = [text]
    try:
        decoded = unquote(text)
        if decoded != text:
            variants.append(decoded)
            decoded_twice = unquote(decoded)
            if decoded_twice != decoded:
                variants.append(decoded_twice)
    except Exception:
        pass

    def push_url(raw_url: str) -> None:
        if len(candidates) >= limit:
            return
        rec = normalize_url(raw_url)
        if not rec or rec.item_id in seen_item_ids:
            return
        seen_item_ids.add(rec.item_id)
        candidates.append(rec.normalized_url)

    def push_item_id(item_id: str) -> None:
        if len(candidates) >= limit:
            return
        if not item_id or item_id in seen_item_ids:
            return
        seen_item_ids.add(item_id)
        candidates.append(f"https://item.taobao.com/item.htm?id={item_id}")

    for body in variants:
        for raw_url in ITEM_URL_RE.findall(body):
            push_url(raw_url)
            if len(candidates) >= limit:
                return candidates

        for raw_url in ANY_URL_RE.findall(body):
            push_url(raw_url)
            if len(candidates) >= limit:
                return candidates

            try:
                parsed = urlparse(raw_url)
                query = parse_qs(parsed.query)
            except Exception:
                continue
            for key in (
                "url",
                "target",
                "redirect",
                "redirect_url",
                "tu",
                "dest",
                "itemurl",
            ):
                for maybe_url in query.get(key, []):
                    push_url(unquote(maybe_url))
                    if len(candidates) >= limit:
                        return candidates

        for item_id in ITEM_ID_IN_TEXT_RE.findall(body):
            push_item_id(item_id)
            if len(candidates) >= limit:
                return candidates

        for match in JSON_ITEM_ID_RE.finditer(body):
            push_item_id(match.group("id"))
            if len(candidates) >= limit:
                return candidates
    return candidates


def extract_json_object(raw: str) -> Any:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[idx:])
            return value
        except json.JSONDecodeError:
            continue
    return None


class Storage:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.products_csv = self.data_dir / "products.csv"
        self.workbooks_csv = self.data_dir / "workbooks.csv"
        self.selling_points_json = self.data_dir / "selling_points.json"
        self.market_report_json = self.data_dir / "market_report.json"
        self.groups_json = self.data_dir / "groups.json"
        self.tasks_json = self.data_dir / "tasks.json"
        self.images_dir = self.data_dir / "images"
        self.exports_dir = self.data_dir / "exports"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv(self.workbooks_csv, WORKBOOK_COLUMNS)
        self._ensure_csv(self.products_csv, PRODUCT_COLUMNS)
        self._ensure_json(self.selling_points_json, {})
        self._ensure_json(self.market_report_json, {})
        self._ensure_json(self.groups_json, {})
        self._ensure_json(self.tasks_json, {})

    def _ensure_csv(self, path: Path, columns: list[str]) -> None:
        if path.exists():
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

    def _ensure_json(self, path: Path, default_obj: Any) -> None:
        if path.exists():
            return
        self.write_json(path, default_obj)

    @staticmethod
    def read_json(path: Path) -> Any:
        if not path.exists():
            return {}
        text = read_text_utf8_best(path).strip()
        if not text:
            return {}
        return json.loads(text)

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)

    @staticmethod
    def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if not path.exists():
            return rows
        text = read_text_utf8_best(path)
        if not text.strip():
            return rows
        reader = csv.DictReader(io.StringIO(text))
        for raw in reader:
            normalized: dict[str, str] = {}
            for col in columns:
                value = raw.get(col, "") or ""
                # Backward-compatibility: recover ids that were exported in Excel-safe text format.
                if value.startswith("'") and value[1:].isdigit() and col in {"item_id"}:
                    value = value[1:]
                normalized[col] = value
            rows.append(normalized)
        return rows

    @staticmethod
    def write_csv(
        path: Path,
        columns: list[str],
        rows: list[dict[str, Any]],
        utf8_bom: bool = False,
        excel_friendly: bool = False,
    ) -> None:
        _ = utf8_bom
        encoding = "utf-8"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=columns,
                quoting=csv.QUOTE_ALL if excel_friendly else csv.QUOTE_MINIMAL,
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                if excel_friendly:
                    writer.writerow(
                        {col: csv_cell(row.get(col, "")) for col in columns}
                    )
                else:
                    writer.writerow(
                        {col: str(row.get(col, "") or "") for col in columns}
                    )
        tmp.replace(path)

    def list_workbooks(self) -> list[dict[str, str]]:
        return self.read_csv(self.workbooks_csv, WORKBOOK_COLUMNS)

    def save_workbooks(self, rows: list[dict[str, str]]) -> None:
        self.write_csv(self.workbooks_csv, WORKBOOK_COLUMNS, rows)

    def list_products(self) -> list[dict[str, str]]:
        return self.read_csv(self.products_csv, PRODUCT_COLUMNS)

    def save_products(self, rows: list[dict[str, str]]) -> None:
        self.write_csv(self.products_csv, PRODUCT_COLUMNS, rows)

    def list_groups(self) -> dict[str, list[dict[str, str]]]:
        payload = self.read_json(self.groups_json)
        return payload if isinstance(payload, dict) else {}

    def save_groups(self, payload: dict[str, list[dict[str, str]]]) -> None:
        self.write_json(self.groups_json, payload)

    def list_tasks(self) -> dict[str, dict[str, Any]]:
        payload = self.read_json(self.tasks_json)
        return payload if isinstance(payload, dict) else {}

    def save_tasks(self, payload: dict[str, dict[str, Any]]) -> None:
        self.write_json(self.tasks_json, payload)


class WorkbookService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def create(self, workbook_name: str) -> dict[str, str]:
        workbook_name = workbook_name.strip()
        if not workbook_name:
            raise ValueError("workbook_name cannot be empty")
        workbook_id = "wb_" + uuid.uuid4().hex[:12]
        ts = now_iso()
        row = {
            "workbook_id": workbook_id,
            "workbook_name": workbook_name,
            "created_at": ts,
            "updated_at": ts,
        }
        rows = self.storage.list_workbooks()
        rows.append(row)
        self.storage.save_workbooks(rows)

        groups = self.storage.list_groups()
        groups[workbook_id] = [
            {"group_code": DEFAULT_GROUP_CODE, "group_name": DEFAULT_GROUP_NAME}
        ]
        self.storage.save_groups(groups)
        return row

    def rename(self, workbook_id: str, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("new workbook_name cannot be empty")
        rows = self.storage.list_workbooks()
        found = False
        for row in rows:
            if row["workbook_id"] == workbook_id:
                row["workbook_name"] = new_name
                row["updated_at"] = now_iso()
                found = True
                break
        if not found:
            raise ValueError(f"workbook not found: {workbook_id}")
        self.storage.save_workbooks(rows)

        products = self.storage.list_products()
        for row in products:
            if row["workbook_id"] == workbook_id:
                row["workbook_name"] = new_name
                row["updated_at"] = now_iso()
        self.storage.save_products(products)

    def delete(self, workbook_id: str) -> None:
        rows = self.storage.list_workbooks()
        new_rows = [r for r in rows if r["workbook_id"] != workbook_id]
        if len(new_rows) == len(rows):
            raise ValueError(f"workbook not found: {workbook_id}")
        self.storage.save_workbooks(new_rows)

        products = [
            r for r in self.storage.list_products() if r["workbook_id"] != workbook_id
        ]
        self.storage.save_products(products)

        selling_points = self.storage.read_json(self.storage.selling_points_json)
        for key in list(selling_points.keys()):
            if key.startswith(f"{workbook_id}:"):
                del selling_points[key]
        self.storage.write_json(self.storage.selling_points_json, selling_points)

        market = self.storage.read_json(self.storage.market_report_json)
        market.pop(workbook_id, None)
        self.storage.write_json(self.storage.market_report_json, market)

        groups = self.storage.list_groups()
        groups.pop(workbook_id, None)
        self.storage.save_groups(groups)

        image_root = self.storage.images_dir / workbook_id
        if image_root.exists():
            shutil.rmtree(image_root)

    def get(self, workbook_id: str) -> dict[str, str]:
        for row in self.storage.list_workbooks():
            if row["workbook_id"] == workbook_id:
                return row
        raise ValueError(f"workbook not found: {workbook_id}")

    def list(self) -> list[dict[str, str]]:
        return self.storage.list_workbooks()


class GroupService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def list(self, workbook_id: str) -> list[dict[str, str]]:
        groups = self.storage.list_groups()
        if workbook_id not in groups:
            groups[workbook_id] = [
                {"group_code": DEFAULT_GROUP_CODE, "group_name": DEFAULT_GROUP_NAME}
            ]
            self.storage.save_groups(groups)
        return groups[workbook_id]

    def create(
        self, workbook_id: str, group_name: str, group_code: str | None = None
    ) -> dict[str, str]:
        group_name = group_name.strip()
        if not group_name:
            raise ValueError("group_name cannot be empty")
        groups_payload = self.storage.list_groups()
        groups = groups_payload.get(
            workbook_id,
            [{"group_code": DEFAULT_GROUP_CODE, "group_name": DEFAULT_GROUP_NAME}],
        )
        used_codes = {g["group_code"] for g in groups}
        if group_code:
            code = str(group_code)
            if code in used_codes:
                raise ValueError(f"group_code already exists: {code}")
        else:
            code = str(max((int(v) for v in used_codes if v.isdigit()), default=1) + 1)
        group = {"group_code": code, "group_name": group_name}
        groups.append(group)
        groups_payload[workbook_id] = groups
        self.storage.save_groups(groups_payload)
        return group

    def rename(self, workbook_id: str, group_code: str, group_name: str) -> None:
        group_name = group_name.strip()
        if not group_name:
            raise ValueError("group_name cannot be empty")
        groups_payload = self.storage.list_groups()
        groups = groups_payload.get(workbook_id, [])
        found = False
        for group in groups:
            if group["group_code"] == str(group_code):
                group["group_name"] = group_name
                found = True
                break
        if not found:
            raise ValueError(f"group not found: {group_code}")
        groups_payload[workbook_id] = groups
        self.storage.save_groups(groups_payload)

        products = self.storage.list_products()
        for row in products:
            if row["workbook_id"] == workbook_id and row["group_code"] == str(
                group_code
            ):
                row["group_name"] = group_name
                row["updated_at"] = now_iso()
        self.storage.save_products(products)

    def move_item(self, workbook_id: str, item_id: str, group_code: str) -> None:
        group_map = {g["group_code"]: g["group_name"] for g in self.list(workbook_id)}
        if str(group_code) not in group_map:
            raise ValueError(f"group not found: {group_code}")
        products = self.storage.list_products()
        found = False
        for row in products:
            if row["workbook_id"] == workbook_id and row["item_id"] == item_id:
                row["group_code"] = str(group_code)
                row["group_name"] = group_map[str(group_code)]
                row["updated_at"] = now_iso()
                found = True
                break
        if not found:
            raise ValueError(f"item not found in workbook: {item_id}")
        self.storage.save_products(products)


class URLService:
    def __init__(
        self,
        storage: Storage,
        workbook_service: WorkbookService,
        group_service: GroupService,
    ) -> None:
        self.storage = storage
        self.workbook_service = workbook_service
        self.group_service = group_service

    def add_urls(
        self,
        workbook_id: str,
        raw_urls: list[str],
        group_code: str = DEFAULT_GROUP_CODE,
    ) -> list[dict[str, str]]:
        workbook = self.workbook_service.get(workbook_id)
        group_map = {
            g["group_code"]: g["group_name"]
            for g in self.group_service.list(workbook_id)
        }
        if str(group_code) not in group_map:
            raise ValueError(f"group not found: {group_code}")
        products = self.storage.list_products()
        changed: list[dict[str, str]] = []

        for raw in raw_urls:
            rec = normalize_url(raw)
            if not rec:
                continue
            existing = next(
                (
                    r
                    for r in products
                    if r["workbook_id"] == workbook_id and r["item_id"] == rec.item_id
                ),
                None,
            )
            base = {
                "workbook_id": workbook_id,
                "workbook_name": workbook["workbook_name"],
                "group_code": str(group_code),
                "group_name": group_map[str(group_code)],
                "item_id": rec.item_id,
                "normalized_url": rec.normalized_url,
                "raw_url": rec.raw_url,
                "process_status": "1",
                "title": "",
                "main_image_url": "",
                "shop_name": "",
                "brand": "",
                "price_min": "",
                "price_max": "",
                "sku_count": "0",
                "sku_list": "",
                "detail_summary": "",
                "selling_points_text": "",
                "selling_points_citation": "",
                "competitor_analysis_text": "",
                "batch_competitor_summary_text": "",
                "market_summary_text": "",
                "final_conclusion_text": "",
                "market_tags": "",
                "crawl_time": "",
                "search_rank": "",
                "sales_text": "",
                "official_store": "",
                "item_source_url": rec.normalized_url,
                "source_type": "official",
                "updated_at": now_iso(),
            }
            if existing:
                for key, value in base.items():
                    if key in {
                        "title",
                        "main_image_url",
                        "shop_name",
                        "brand",
                        "price_min",
                        "price_max",
                        "sku_count",
                        "sku_list",
                    }:
                        continue
                    existing[key] = value
                changed.append(existing)
            else:
                products.append(base)
                changed.append(base)
        self.storage.save_products(products)
        return changed

    def add_records(
        self,
        workbook_id: str,
        records: list[UrlRecord],
        group_code: str = DEFAULT_GROUP_CODE,
    ) -> list[dict[str, str]]:
        workbook = self.workbook_service.get(workbook_id)
        group_map = {
            g["group_code"]: g["group_name"]
            for g in self.group_service.list(workbook_id)
        }
        if str(group_code) not in group_map:
            raise ValueError(f"group not found: {group_code}")
        products = self.storage.list_products()
        changed: list[dict[str, str]] = []

        for rec in records:
            existing = next(
                (
                    r
                    for r in products
                    if r["workbook_id"] == workbook_id and r["item_id"] == rec.item_id
                ),
                None,
            )
            base = {
                "workbook_id": workbook_id,
                "workbook_name": workbook["workbook_name"],
                "group_code": str(group_code),
                "group_name": group_map[str(group_code)],
                "item_id": rec.item_id,
                "normalized_url": rec.normalized_url,
                "raw_url": rec.raw_url or rec.normalized_url,
                "process_status": "1",
                "title": clean_text(rec.title, max_len=160),
                "main_image_url": "",
                "shop_name": clean_text(rec.shop_name, max_len=120),
                "brand": "",
                "price_min": "",
                "price_max": "",
                "sku_count": "0",
                "sku_list": "",
                "detail_summary": "",
                "selling_points_text": "",
                "selling_points_citation": "",
                "competitor_analysis_text": "",
                "batch_competitor_summary_text": "",
                "market_summary_text": "",
                "final_conclusion_text": "",
                "market_tags": "",
                "crawl_time": "",
                "search_rank": (
                    str(rec.search_rank) if rec.search_rank is not None else ""
                ),
                "sales_text": rec.sales_text,
                "official_store": "1" if rec.is_official_store else "",
                "item_source_url": rec.item_source_url
                or rec.normalized_url
                or rec.raw_url,
                "source_type": rec.source_type or "official",
                "updated_at": now_iso(),
            }
            if existing:
                for key, value in base.items():
                    if key in {
                        "title",
                        "main_image_url",
                        "shop_name",
                        "brand",
                        "price_min",
                        "price_max",
                        "sku_count",
                        "sku_list",
                        "detail_summary",
                    }:
                        continue
                    existing[key] = value
                changed.append(existing)
            else:
                products.append(base)
                changed.append(base)
        self.storage.save_products(products)
        return changed

    def delete_item(self, workbook_id: str, item_id: str) -> None:
        products = self.storage.list_products()
        new_rows = [
            r
            for r in products
            if not (r["workbook_id"] == workbook_id and r["item_id"] == item_id)
        ]
        if len(new_rows) == len(products):
            raise ValueError(f"item not found: {item_id}")
        self.storage.save_products(new_rows)

        selling_points = self.storage.read_json(self.storage.selling_points_json)
        selling_points.pop(f"{workbook_id}:{item_id}", None)
        self.storage.write_json(self.storage.selling_points_json, selling_points)

        image_dir = self.storage.images_dir / workbook_id / item_id
        if image_dir.exists():
            shutil.rmtree(image_dir)


class TaskTracker:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def create(self, workbook_id: str, keyword: str, total_items: int) -> str:
        tasks = self.storage.list_tasks()
        task_id = "task_" + uuid.uuid4().hex[:12]
        tasks[task_id] = {
            "task_id": task_id,
            "workbook_id": workbook_id,
            "keyword": keyword,
            "total_items": total_items,
            "finished_items": 0,
            "status": "running",
            "failures": [],
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        self.storage.save_tasks(tasks)
        return task_id

    def update_progress(self, task_id: str, finished: int) -> None:
        tasks = self.storage.list_tasks()
        if task_id not in tasks:
            return
        tasks[task_id]["finished_items"] = finished
        tasks[task_id]["updated_at"] = now_iso()
        self.storage.save_tasks(tasks)

    def add_failure(self, task_id: str, item_id: str, reason: str) -> None:
        tasks = self.storage.list_tasks()
        if task_id not in tasks:
            return
        tasks[task_id]["failures"].append(
            {"item_id": item_id, "reason": clean_text(reason, max_len=400)}
        )
        tasks[task_id]["updated_at"] = now_iso()
        self.storage.save_tasks(tasks)

    def complete(self, task_id: str, status: str, result: dict[str, Any]) -> None:
        tasks = self.storage.list_tasks()
        if task_id not in tasks:
            return
        tasks[task_id]["status"] = status
        tasks[task_id]["result"] = result
        tasks[task_id]["updated_at"] = now_iso()
        self.storage.save_tasks(tasks)

    def get(self, task_id: str) -> dict[str, Any]:
        tasks = self.storage.list_tasks()
        if task_id not in tasks:
            raise ValueError(f"task not found: {task_id}")
        return tasks[task_id]
