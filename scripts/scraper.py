"""Web scraping and data extraction for Taobao/Tmall."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.request
from html import escape as html_escape, unescape as html_unescape
from pathlib import Path
from typing import Any
from urllib.parse import (
    parse_qs,
    parse_qsl,
    quote_plus,
    unquote,
    urlencode,
    urljoin,
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

try:
    from tools import (
        GlobalBrowserManager,
        LoginHandleResult,
        TaobaoLogin,
        get_global_browser_manager,
        cleanup_global_browser,
    )
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    LoginHandleResult = Any  # type: ignore
    # Log the error for debugging
    logging.warning(f"Failed to import tools: {e}")

from analysis import SellingPointExtractor
from config import (
    ANTI_BOT_MARKERS,
    ANY_URL_RE,
    BANNED_SHOP_MARKER_RE,
    BRAND_RE,
    ITEM_ID_IN_TEXT_RE,
    ITEM_ID_RE,
    ITEM_URL_RE,
    JSON_ITEM_ID_RE,
    LINE_BREAK_RE,
    NON_WORD_RE,
    SEARCH_BLOCK_HINT,
    SHOP_NAME_RE,
    SKU_ID_RE,
    SKU_MAP_RE,
    UA,
)
from data import (
    ItemDetail,
    Storage,
    UrlRecord,
    clean_text,
    extract_candidate_item_urls,
    extract_json_object,
    has_valid_login_cookie,
    is_official_shop,
    load_valid_cookies,
    looks_like_tmall,
    normalize_brand_name,
    normalize_item_title,
    normalize_url,
    now_iso,
    parse_price_values,
    parse_sales_to_int,
    read_text_utf8_best,
    should_reuse_search_page,
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


def _looks_like_search_content(body_text: str) -> bool:
    body_lower = (body_text or "").lower()
    search_markers = (
        "人付款",
        "已售",
        "综合排序",
        "销量",
        "筛选",
        "收货地",
        "店铺",
        "价格",
        "taobao",
        "tmall",
    )
    hits = sum(1 for token in search_markers if token in body_text or token in body_lower)
    return hits >= 2


def detect_non_product_page(current_url: str, title: str, body_text: str) -> str:
    url_lower = (current_url or "").lower()
    title_lower = (title or "").lower()
    body_lower = (body_text or "").lower()

    if "login.taobao.com" in url_lower or "member/login" in url_lower:
        return "redirected to Taobao login page"
    if any(
        token in url_lower for token in ("captcha", "punish", "x5sec", "_____tmd_____")
    ):
        return "redirected to anti-bot verification page"
    if "rgv587" in body_lower or "fail_sys_user_validate" in body_lower:
        return "anti-bot validation detected in response"

    if is_search_result_url(url_lower):
        has_item_link = (
            "item.taobao.com/item.htm?id=" in body_lower
            or "detail.tmall.com/item.htm?id=" in body_lower
        )
        if has_item_link or _looks_like_search_content(body_text):
            return ""

    login_markers = (
        "扫码登录",
        "请扫码登录",
        "扫一扫登录",
        "账号密码登录",
        "密码登录",
        "短信登录",
        "验证码登录",
        "忘记密码",
        "免费注册",
        "请登录",
        "登录淘宝",
        "登录天猫",
        "taobao login",
        "tmall login",
    )
    login_hits = sum(
        1 for marker in login_markers if marker in body_text or marker.lower() in body_lower
    )
    if login_hits >= 2:
        return "login page content detected"
    if ("登录" in title and ("淘宝" in title or "天猫" in title)) or (
        "login" in title_lower and ("taobao" in title_lower or "tmall" in title_lower)
    ):
        return "login page title detected"
    if "captcha" in title_lower:
        return "captcha title detected"
    return ""


def is_search_result_url(url: str) -> bool:
    lower = (url or "").lower()
    if "s.taobao.com/search" in lower:
        return True
    if "s.taobao.com" in lower and ("q=" in lower or "search" in lower or "sort=" in lower):
        return True
    if "list.tmall.com/search_product.htm" in lower:
        return True
    if "list.tmall.com" in lower and "q=" in lower:
        return True
    return False


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


def fetch_text_http(
    url: str, timeout_sec: int = 30, max_retries: int = 3
) -> tuple[str, str]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                payload = response.read()
                final_url = response.geturl() or url
                charset = ""
                try:
                    content_type = str(response.headers.get("content-type", "") or "")
                    match = re.search(
                        r"charset=([A-Za-z0-9._-]+)", content_type, flags=re.IGNORECASE
                    )
                    if match:
                        charset = match.group(1).strip()
                except Exception:
                    charset = ""
            if charset:
                try:
                    return payload.decode(charset, errors="replace"), final_url
                except Exception:
                    pass
            for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
                try:
                    return payload.decode(encoding), final_url
                except Exception:
                    continue
            return payload.decode("utf-8", errors="replace"), final_url
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(1.5 * attempt, 4.5))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to fetch url: {url}")


def decode_numeric_char_html(raw_text: str) -> str:
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    if len(lines) < 60:
        return ""
    numeric_lines = [line for line in lines if line.isdigit()]
    if len(numeric_lines) < 60 or len(numeric_lines) < int(len(lines) * 0.75):
        return ""
    chars: list[str] = []
    for token in numeric_lines[:50000]:
        try:
            code = int(token)
        except ValueError:
            continue
        if 0 <= code <= 0x10FFFF:
            chars.append(chr(code))
    decoded = "".join(chars)
    lowered = decoded.lower()
    if (
        "<html" in lowered
        or "real_jump_address" in lowered
        or "s.click.taobao.com" in lowered
    ):
        return decoded
    return ""


def extract_real_jump_address(page_text: str) -> str:
    source = page_text or ""
    match = re.search(r"real_jump_address\s*=\s*'([^']+)'", source, flags=re.IGNORECASE)
    if not match:
        return ""
    return (
        html_unescape(match.group(1)).strip().replace("&amp;", "&").replace("\\/", "/")
    )


def resolve_official_from_click_chain(click_url: str, max_hops: int = 5) -> str:
    current = html_unescape((click_url or "").strip()).replace("&amp;", "&")
    seen: set[str] = set()
    for _ in range(max_hops):
        if not current or current in seen:
            break
        seen.add(current)
        direct = normalize_url(current)
        if direct:
            return direct.normalized_url
        try:
            text, final_url = fetch_text_http(current, timeout_sec=12, max_retries=2)
        except Exception:
            break
        final_rec = normalize_url(final_url)
        if final_rec:
            return final_rec.normalized_url
        decoded = decode_numeric_char_html(text)
        for blob in (text, decoded):
            for candidate in extract_candidate_item_urls(blob, limit=12):
                rec = normalize_url(candidate)
                if rec:
                    return rec.normalized_url
        next_url = extract_real_jump_address(decoded or text)
        if not next_url or next_url == current:
            break
        current = next_url
    return ""


def resolve_official_item_url(item_page_url: str, page_html: str) -> str:
    for source in (item_page_url, page_html, decode_numeric_char_html(page_html)):
        for candidate in extract_candidate_item_urls(source, limit=20):
            rec = normalize_url(candidate)
            if rec:
                return rec.normalized_url

    redirect_urls = [
        html_unescape(url).replace("&amp;", "&")
        for url in re.findall(
            r"https?://(?:s\.click\.taobao\.com|[^\"'<> ]*go\.php\?u=)[^\"'<> ]+",
            page_html or "",
            flags=re.IGNORECASE,
        )
    ]
    for link in list(dict.fromkeys(redirect_urls))[:10]:
        if "s.click.taobao.com/" in link:
            resolved = resolve_official_from_click_chain(link, max_hops=5)
            if resolved:
                return resolved
            continue
        try:
            go_html, _ = fetch_text_http(link, timeout_sec=10, max_retries=1)
        except Exception:
            continue
        decoded_go = decode_numeric_char_html(go_html)
        for blob in (go_html, decoded_go):
            for candidate in extract_candidate_item_urls(blob, limit=20):
                rec = normalize_url(candidate)
                if rec:
                    return rec.normalized_url
            click_urls = [
                html_unescape(url).replace("&amp;", "&")
                for url in re.findall(
                    r"https?://s\.click\.taobao\.com/[^\s\"'<>]+",
                    blob or "",
                    flags=re.IGNORECASE,
                )
            ]
            jump = extract_real_jump_address(blob)
            if jump:
                click_urls.append(jump)
            for click_url in list(dict.fromkeys(click_urls))[:5]:
                resolved = resolve_official_from_click_chain(click_url, max_hops=5)
                if resolved:
                    return resolved
    return ""


def parse_mirror_list_items(list_url: str, top_n: int) -> list[dict[str, str]]:
    try:
        html, final_url = fetch_text_http(list_url, timeout_sec=20, max_retries=2)
    except Exception:
        return []
    base = final_url or list_url
    entries: list[dict[str, str]] = []
    seen: set[str] = set()

    # Prefer entry-title links from WordPress mirror pages.
    for match in re.finditer(
        r"<a[^>]+href=[\"'](?P<href>[^\"']+/item/[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        href = html_unescape(match.group("href")).strip()
        if href.startswith("/"):
            href = urljoin(base, href)
        if href.startswith("//"):
            href = "https:" + href
        if not href.startswith("http"):
            continue
        if href in seen:
            continue
        seen.add(href)
        raw_title = re.sub(r"<[^>]+>", " ", match.group("title") or "")
        title = normalize_item_title(html_unescape(raw_title), max_len=220)
        entries.append({"url": href, "title": title})
        if len(entries) >= top_n:
            return entries
    return entries


def derive_mirror_item_id(item_url: str, index: int) -> str:
    text = (item_url or "").strip()
    match = re.search(r"/item/([^/?#]+)", text, flags=re.IGNORECASE)
    if match:
        token = re.sub(r"[^A-Za-z0-9_-]+", "", match.group(1))
        if token:
            return f"mirror_{token[:32]}"
    token = re.sub(r"[^A-Za-z0-9]+", "", text)
    if token:
        return f"mirror_{token[-24:]}"
    return f"mirror_item_{index:03d}"


def resolve_browser_use_cmd(cmd: str) -> str:
    value = (cmd or "").strip() or "browser-use"
    as_path = Path(value)
    if as_path.exists():
        return str(as_path)

    resolved = shutil.which(value)
    if resolved:
        return resolved

    appdata = os.getenv("APPDATA", "")
    if appdata:
        scripts_dir = (
            Path(appdata)
            / "Python"
            / f"Python{sys.version_info.major}{sys.version_info.minor}"
            / "Scripts"
        )
        candidates = [scripts_dir / value]
        if not value.lower().endswith(".exe"):
            candidates.append(scripts_dir / f"{value}.exe")
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return value


def build_cdp_endpoint_candidates(cdp_url: str) -> list[str]:
    """Generate a list of CDP endpoint URLs to try for connection."""
    raw = (cdp_url or "").strip()
    if not raw:
        return []
    candidates: list[str] = [raw]
    # If the URL doesn't have an explicit path, also try /json/version style
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https"):
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in candidates:
            candidates.append(base)
    elif parsed.scheme in ("ws", "wss"):
        http_scheme = "http" if parsed.scheme == "ws" else "https"
        http_base = f"{http_scheme}://{parsed.netloc}"
        if http_base not in candidates:
            candidates.append(http_base)
    return list(dict.fromkeys(candidates))


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


def pick_cdp_page(
    cdp_url: str,
    url_contains: str = "",
    preferred_domains: tuple[str, ...] = (),
) -> dict[str, str] | None:
    pages = list_cdp_pages(cdp_url)
    if not pages:
        return None
    token = (url_contains or "").strip().lower()
    if token:
        for page in pages:
            if (
                token in page.get("url", "").lower()
                or token in page.get("title", "").lower()
            ):
                return page
    for domain in preferred_domains:
        d = domain.lower()
        for page in pages:
            if d in page.get("url", "").lower():
                return page
    return pages[0]


class AsyncCdpPage:
    def __init__(self, websocket_url: str, timeout_sec: float = 30.0) -> None:
        self.websocket_url = websocket_url
        self.timeout_sec = max(5.0, float(timeout_sec))
        self._ws = None
        self._seq = 0

    async def __aenter__(self) -> "AsyncCdpPage":
        import websockets

        self._ws = await websockets.connect(
            self.websocket_url, open_timeout=self.timeout_sec, ping_interval=20
        )
        await self.call("Runtime.enable")
        await self.call("Page.enable")
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def call(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        import asyncio

        if self._ws is None:
            raise RuntimeError("CDP websocket not connected")
        self._seq += 1
        msg_id = self._seq
        payload = {"id": msg_id, "method": method, "params": params or {}}
        await self._ws.send(json.dumps(payload, ensure_ascii=False))
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_sec)
            data = json.loads(raw)
            if data.get("id") != msg_id:
                continue
            if "error" in data:
                raise RuntimeError(f"CDP error on {method}: {data['error']}")
            return data.get("result", {})

    async def evaluate(self, expression: str, return_by_value: bool = True) -> Any:
        result = await self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": return_by_value,
                "awaitPromise": True,
            },
        )
        eval_result = result.get("result", {})
        if isinstance(result.get("exceptionDetails"), dict):
            details = result["exceptionDetails"]
            raise RuntimeError(f"CDP evaluate exception: {details}")
        if return_by_value:
            return eval_result.get("value")
        return eval_result

    async def navigate(self, url: str) -> None:
        await self.call("Page.navigate", {"url": url})

    async def wait_ready(self, timeout_sec: float = 20.0) -> None:
        import asyncio
        import time

        deadline = time.time() + max(1.0, float(timeout_sec))
        while time.time() < deadline:
            try:
                state = await self.evaluate("document.readyState")
            except Exception:
                state = ""
            if state in {"complete", "interactive"}:
                return
            await asyncio.sleep(0.35)
        return


class SearchClient:
    def __init__(
        self,
        browser_use_cmd: str = "browser-use",
        headless: bool = False,
        browser_mode: str = "cdp",
        cdp_url: str = "",
        manual_wait_seconds: int = 0,
        cdp_context_index: int = 0,
        cdp_page_url_contains: str = "",
        cdp_connect_timeout_ms: int = 60_000,
        storage_state_file: str | Path | None = None,
        user_data_dir: str | Path | None = None,
        manual_login_on_demand: bool = False,
        manual_login_timeout_sec: int = 300,
        prefer_cdp: bool = True,
        allow_http_fallback: bool = True,
        http_fallback_list_url: str = "",
        allow_mirror_source: bool = False,
        # New parameter for using global browser manager
        use_global_browser: bool = False,
    ) -> None:
        self.browser_use_cmd = browser_use_cmd
        self.headless = headless
        self.browser_mode = (browser_mode or "cdp").strip().lower()
        if self.browser_mode not in {"cdp", "persistent"}:
            self.browser_mode = "cdp"
        self.cdp_url = cdp_url.strip()
        self.manual_wait_seconds = max(0, int(manual_wait_seconds))
        self.cdp_context_index = max(0, int(cdp_context_index))
        self.cdp_page_url_contains = cdp_page_url_contains.strip()
        self.cdp_connect_timeout_ms = max(10_000, int(cdp_connect_timeout_ms))
        self.storage_state_file = (
            Path(storage_state_file).resolve()
            if storage_state_file
            else Path("data/taobao_storage_state.json").resolve()
        )
        self.user_data_dir = (
            Path(user_data_dir).resolve()
            if user_data_dir
            else Path("data/taobao_profile").resolve()
        )
        self.manual_login_on_demand = bool(manual_login_on_demand)
        self.manual_login_timeout_sec = max(30, int(manual_login_timeout_sec))
        self.prefer_cdp = bool(prefer_cdp)
        self.allow_http_fallback = bool(allow_http_fallback)
        self.http_fallback_list_url = (http_fallback_list_url or "").strip()
        self.allow_mirror_source = bool(allow_mirror_source)
        self.use_global_browser = bool(use_global_browser) and TOOLS_AVAILABLE

        # Global browser manager reference (if using)
        self._global_browser_manager: Any = None
        # Structured login recovery events for diagnostics/output
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

    def _guess_http_fallback_list_url(
        self, keyword: str, search_url: str | None
    ) -> str:
        _ = keyword
        if self.http_fallback_list_url:
            return self.http_fallback_list_url
        if not search_url:
            return ""
        parsed = urlparse(search_url)
        host = (parsed.netloc or "").lower()
        if "taobao.com" in host or "tmall.com" in host:
            return search_url
        if self.allow_mirror_source and "tmall.lc" in host:
            return search_url
        return ""

    def _search_with_http_fallback(
        self,
        keyword: str,
        top_n: int,
        search_url: str | None = None,
    ) -> list[UrlRecord]:
        list_url = self._guess_http_fallback_list_url(keyword, search_url)
        if not list_url:
            return []
        parsed = urlparse(list_url)
        host = (parsed.netloc or "").lower()
        if (
            "taobao.com" not in host
            and "tmall.com" not in host
            and not self.allow_mirror_source
        ):
            LOG.warning("Skip non-official HTTP fallback source: %s", list_url)
            return []
        entries = parse_mirror_list_items(list_url, top_n=max(top_n * 2, top_n))
        if not entries:
            return []

        rows: list[UrlRecord] = []
        seen_ids: set[str] = set()
        for idx, entry in enumerate(entries, start=1):
            item_page_url = str(entry.get("url", "") or "").strip()
            title = normalize_item_title(str(entry.get("title", "") or ""), max_len=160)
            if not item_page_url:
                continue
            try:
                item_html, _ = fetch_text_http(
                    item_page_url, timeout_sec=20, max_retries=2
                )
            except Exception:
                item_html = ""

            official_url = (
                resolve_official_item_url(item_page_url, item_html) if item_html else ""
            )
            official_rec = normalize_url(official_url) if official_url else None
            if official_rec:
                if official_rec.item_id in seen_ids:
                    continue
                seen_ids.add(official_rec.item_id)
                rows.append(
                    UrlRecord(
                        raw_url=item_page_url,
                        normalized_url=official_rec.normalized_url,
                        item_id=official_rec.item_id,
                        sku_id=official_rec.sku_id,
                        item_source_url=official_rec.normalized_url,
                        source_type="mixed",
                        title=title,
                    )
                )
            else:
                if not self.allow_mirror_source:
                    continue
                mirror_item_id = derive_mirror_item_id(item_page_url, idx)
                if mirror_item_id in seen_ids:
                    continue
                seen_ids.add(mirror_item_id)
                rows.append(
                    UrlRecord(
                        raw_url=item_page_url,
                        normalized_url=item_page_url,
                        item_id=mirror_item_id,
                        sku_id=None,
                        item_source_url=item_page_url,
                        source_type="mirror",
                        title=title,
                    )
                )
            if len(rows) >= top_n:
                break
        return rows

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

    async def _inject_saved_cookies_async(self, context: Any) -> None:
        cookies = load_valid_cookies(self.storage_state_file)
        if not cookies:
            return
        try:
            await context.add_cookies(cookies)
        except Exception:
            return

    async def _persist_storage_state_async(self, context: Any) -> None:
        self.storage_state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            await context.storage_state(path=str(self.storage_state_file))
        except Exception:
            return

    async def _open_context_async(
        self, playwright: Any, headless: bool, use_cdp: bool
    ) -> tuple[Any, Any | None, bool, bool]:
        """Connect to existing CDP browser or launch persistent context."""
        # Use global browser manager if enabled
        if self.use_global_browser and TOOLS_AVAILABLE:
            if self._global_browser_manager is None:
                self._global_browser_manager = await get_global_browser_manager()

            # Initialize global browser if needed
            if not self._global_browser_manager.is_initialized:
                browser_context = await self._global_browser_manager.initialize(
                    browser_mode=self.browser_mode,
                    headless=self.headless,
                    cdp_url=self.cdp_url,
                    user_data_dir=str(self.user_data_dir),
                )
            else:
                browser_context = self._global_browser_manager.browser_context

            # Return context, browser (None), close flags
            return browser_context, None, False, False

        # Original logic for backwards compatibility
        if use_cdp:
            if not self.cdp_url:
                raise RuntimeError("CDP URL is required; set --playwright-cdp-url")
            last_error = ""
            for candidate in build_cdp_endpoint_candidates(self.cdp_url):
                try:
                    browser = await playwright.chromium.connect_over_cdp(
                        candidate, timeout=self.cdp_connect_timeout_ms
                    )
                    contexts = browser.contexts
                    if contexts:
                        index = min(max(0, int(self.cdp_context_index)), len(contexts) - 1)
                        context = contexts[index]
                    else:
                        context = await browser.new_context()
                    await self._inject_saved_cookies_async(context)
                    return context, browser, False, False
                except Exception as exc:
                    last_error = f"{candidate}: {exc}"
            raise RuntimeError(f"CDP connect failed ({self.cdp_url}): {last_error}")
        else:
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ]
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                headless=headless,
                args=args,
                viewport={"width": 1440, "height": 900},
                channel="chrome",
            )
            await self._inject_saved_cookies_async(context)
            return context, None, True, False

    @staticmethod
    async def _close_context_async(
        context: Any, browser: Any | None, close_context: bool, close_browser: bool
    ) -> None:
        if close_context:
            try:
                await context.close()
            except Exception:
                pass
        if close_browser and browser is not None:
            try:
                await browser.close()
            except Exception:
                pass

    async def _select_page_from_context(self, context: Any) -> tuple[Any, bool]:
        page = None
        created_new_page = False
        if self.cdp_page_url_contains:
            token = self.cdp_page_url_contains.lower()
            for existing in context.pages:
                existing_url = (existing.url or "").lower()
                if token in existing_url:
                    page = existing
                    break
        if page is None:
            for existing in context.pages:
                existing_url = (existing.url or "").lower()
                if "taobao.com" in existing_url or "tmall.com" in existing_url:
                    page = existing
                    break
        if page is None:
            page = await context.new_page()
            created_new_page = True
        return page, created_new_page

    async def _wait_manual_login_async(
        self, page: Any, timeout_sec: int, stage: str
    ) -> None:
        deadline = time.time() + max(30, int(timeout_sec))
        try:
            await page.bring_to_front()
        except Exception:
            pass
        while time.time() < deadline:
            current_url = page.url
            try:
                title = await page.title()
            except Exception:
                title = ""
            try:
                body_text = await page.inner_text("body")
            except Exception:
                body_text = ""
            reason = detect_non_product_page(current_url, title, body_text)
            if not reason:
                return
            await page.wait_for_timeout(2000)
        raise RuntimeError(
            f"manual login timeout in {stage}, still blocked by login/anti-bot"
        )

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
        if self.use_global_browser and TOOLS_AVAILABLE:
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
        else:
            await self._wait_manual_login_async(
                page, timeout_sec=self.manual_login_timeout_sec, stage=stage
            )
            self.login_recovery_events.append(
                {
                    "source": "search",
                    "stage": stage,
                    "ok": True,
                    "blocked_reason": reason,
                    "final_state": "SUCCESS",
                    "reason": "manual_login_wait_success",
                    "elapsed_sec": 0.0,
                    "url": current_url,
                    "decision_trace": [],
                    "updated_at": now_iso(),
                }
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

    def _search_with_browser_use(
        self,
        keyword: str,
        top_n: int,
        search_url: str | None = None,
        search_sort: str = "page",
        official_only: bool = False,
    ) -> list[UrlRecord]:
        sort_hint = (
            "by visible page order" if search_sort == "page" else "by sales descending"
        )
        official_hint = "yes" if official_only else "no"
        if search_url:
            prompt = textwrap.dedent(
                f"""
                Open this Taobao search URL directly:
                {search_url}
                Return only JSON:
                {{"items":[{{"url":"https://detail.tmall.com/item.htm?id=123&skuId=456&ns=1&abbucket=8", "title":"...", "shop_name":"...", "sales_text":"..."}}]}}
                Rules:
                - Keep only Taobao/Tmall product links containing /item.htm?id=
                - Keep the original href with full query params; do not trim URL to only id
                - Return top {top_n} items {sort_hint}
                - Filter to official shops only: {official_hint}
                - Prefer the current signed-in real browser session if available
                - No markdown
                """
            ).strip()
        else:
            prompt = textwrap.dedent(
                f"""
                Open Taobao and search keyword: {keyword}
                Return only JSON:
                {{"items":[{{"url":"https://detail.tmall.com/item.htm?id=123&skuId=456&ns=1&abbucket=8", "title":"...", "shop_name":"...", "sales_text":"..."}}]}}
                Rules:
                - Keep only Taobao/Tmall product links containing /item.htm?id=
                - Keep the original href with full query params; do not trim URL to only id
                - Return top {top_n} items {sort_hint}
                - Filter to official shops only: {official_hint}
                - Prefer the current signed-in real browser session if available
                - No markdown
                """
            ).strip()
        commands = [
            [self.browser_use_cmd, "--browser", "real", "run", prompt],
            [self.browser_use_cmd, "--browser", "real", "--task", prompt],
        ]
        LOG.info("Run browser-use search with real browser")
        proc = None
        for cmd in commands:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=600,
                check=False,
            )
            merged_try = (proc.stdout or "") + "\n" + (proc.stderr or "")
            # New browser-use CLI uses `run`; older releases use `--task`.
            if (
                "invalid choice" in merged_try.lower()
                or "unrecognized arguments" in merged_try.lower()
            ):
                continue
            break
        if proc is None:
            return []
        merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
        candidate_urls: list[str] = []

        payload = extract_json_object(merged)
        parsed_cards: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            items = payload.get("items", [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        parsed_cards.append(item)
                        if item.get("url"):
                            candidate_urls.append(str(item["url"]))
        if parsed_cards:
            records = self._records_from_cards(
                parsed_cards,
                top_n=top_n,
                search_sort=search_sort,
                official_only=official_only,
            )
            if records:
                return records
        candidate_urls.extend(extract_candidate_item_urls(merged, limit=top_n * 20))
        return self._records_from_candidate_urls(candidate_urls, top_n)

    @staticmethod
    async def _search_with_cdp_socket_async(
        keyword: str,
        top_n: int,
        cdp_url: str,
        manual_wait_seconds: int = 0,
        search_url: str | None = None,
        search_sort: str = "page",
        official_only: bool = False,
        cdp_page_url_contains: str = "",
    ) -> tuple[list[UrlRecord], str]:
        import asyncio

        target_url = (
            search_url or f"https://s.taobao.com/search?q={quote_plus(keyword)}"
        )
        selected = pick_cdp_page(
            cdp_url=cdp_url,
            url_contains=cdp_page_url_contains or "s.taobao.com/search",
            preferred_domains=("s.taobao.com/search", "taobao.com/search"),
        )
        if not selected or not selected.get("webSocketDebuggerUrl"):
            raise RuntimeError("no debuggable page found from CDP /json/list")

        card_records: list[dict[str, Any]] = []
        hrefs: list[str] = []
        page_text = ""
        page_html = ""
        async with AsyncCdpPage(
            selected["webSocketDebuggerUrl"], timeout_sec=35.0
        ) as page:
            current_url = str(await page.evaluate("location.href || ''"))
            should_navigate = True
            if not search_url and should_reuse_search_page(current_url, keyword):
                should_navigate = False
            if should_navigate:
                await page.navigate(target_url)
                await page.wait_ready(timeout_sec=35.0)

            await asyncio.sleep(max(4.5, float(manual_wait_seconds)))
            scroll_rounds = max(4, 4 + (manual_wait_seconds // 3))
            for _ in range(scroll_rounds):
                await page.evaluate("window.scrollBy(0, 2400); true")
                await asyncio.sleep(1.1)

            card_records_raw = await page.evaluate(
                """
                (() => {
                  const nodes = Array.from(document.querySelectorAll('a[id^="item_id_"], a[href*="item.htm?id="]'));
                  const out = [];
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
                    const titleNode = card ? card.querySelector('[class*="title"], [class*="Title"], h3, h4') : null;
                    const title = ((titleNode && titleNode.textContent) || node.textContent || '').replace(/\\s+/g, ' ').trim();
                    const shopNode = card ? card.querySelector('[class*="shopName"], [class*="shop"], [class*="store"]') : null;
                    const shopName = ((shopNode && shopNode.textContent) || '').replace(/\\s+/g, ' ').trim();
                    const salesMatch = cardText.match(/\\d+(?:\\.\\d+)?\\s*万?\\+?\\s*(?:人付款|已售)/);
                    out.push({
                      href,
                      title,
                      shop_name: shopName,
                      sales_text: salesMatch ? salesMatch[0] : '',
                      card_text: cardText
                    });
                    if (out.length >= 260) break;
                  }
                  return out;
                })()
                """
            )
            if isinstance(card_records_raw, list):
                card_records = [
                    row for row in card_records_raw if isinstance(row, dict)
                ]

            hrefs_raw = await page.evaluate(
                "Array.from(document.querySelectorAll('a[href]')).map(n => n.href).filter(Boolean).slice(0, 5000)"
            )
            if isinstance(hrefs_raw, list):
                hrefs = [str(v) for v in hrefs_raw if v]

            page_html = str(
                await page.evaluate(
                    "document.documentElement ? document.documentElement.outerHTML : ''"
                )
                or ""
            )
            page_text = str(
                await page.evaluate("document.body ? document.body.innerText : ''")
                or ""
            )

        candidate_urls: list[str] = []
        candidate_urls.extend(hrefs)
        candidate_urls.extend(extract_candidate_item_urls(page_html, limit=top_n * 60))
        candidate_urls.extend(extract_candidate_item_urls(page_text, limit=top_n * 60))
        records = SearchClient._records_from_cards(
            card_records,
            top_n=top_n,
            search_sort=search_sort,
            official_only=official_only,
        )
        if not records:
            records = SearchClient._records_from_candidate_urls(candidate_urls, top_n)
        anti_bot_signal = detect_antibot_signal(page_text, page_html)
        return records, anti_bot_signal

    async def _search_with_playwright_async(
        self,
        keyword: str,
        top_n: int,
        search_url: str | None = None,
        search_sort: str = "page",
        official_only: bool = False,
        use_cdp: bool = True,
    ) -> tuple[list[UrlRecord], str]:
        import asyncio

        from playwright.async_api import async_playwright

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

        # When using global browser, don't create a separate Playwright instance
        if self.use_global_browser and TOOLS_AVAILABLE:
            return await self._search_with_global_browser_async(
                keyword=keyword,
                top_n=top_n,
                search_url=search_url,
                search_sort=search_sort,
                official_only=official_only,
            )

        # Original code path with local Playwright instance
        async with async_playwright() as p:
            browser = None
            context = None
            page = None
            created_new_page = False
            close_context = False
            close_browser = False
            try:
                context, browser, close_context, close_browser = (
                    await self._open_context_async(
                        playwright=p,
                        headless=self.headless,
                        use_cdp=bool(use_cdp and self.browser_mode == "cdp"),
                    )
                )
                page, created_new_page = await self._select_page_from_context(context)

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
                should_navigate = True
                current_url = page.url or ""
                if (
                    not search_url
                    and not created_new_page
                    and should_reuse_search_page(current_url, keyword)
                ):
                    should_navigate = False
                if should_navigate:
                    await page.goto(
                        target_url, wait_until="domcontentloaded", timeout=120_000
                    )
                await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))

                await self._ensure_search_surface_async(
                    page=page,
                    context=context,
                    target_url=target_url,
                    stage="search-initial",
                )

                scroll_rounds = max(4, 4 + (self.manual_wait_seconds // 3))
                for _ in range(scroll_rounds):
                    await page.mouse.wheel(0, 2400)
                    await page.wait_for_timeout(1200)

                await self._ensure_search_surface_async(
                    page=page,
                    context=context,
                    target_url=target_url,
                    stage="search-post-scroll",
                )

                try:
                    card_records = await page.evaluate(
                        """
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
                            const salesMatch = cardText.match(/\\d+(?:\\.\\d+)?\\s*万?\\+?\\s*(?:人付款|已售)/);
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
                        """
                    )
                except Exception:
                    card_records = []

                hrefs = await page.eval_on_selector_all(
                    "a[href]", "nodes => nodes.map(n => n.href)"
                )
                candidate_urls.extend([str(v) for v in hrefs if v])

                try:
                    resource_urls_raw = await page.evaluate(
                        "() => performance.getEntriesByType('resource').map(e => e.name)"
                    )
                    resource_urls = [
                        str(v) for v in resource_urls_raw if isinstance(v, str)
                    ]
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
                if page and created_new_page and not page.is_closed():
                    try:
                        await page.close()
                    except Exception:
                        pass
                if context is not None:
                    await self._persist_storage_state_async(context)
                if context is not None:
                    await self._close_context_async(
                        context, browser, close_context, close_browser
                    )

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

    async def _search_with_global_browser_async(
        self,
        keyword: str,
        top_n: int,
        search_url: str | None = None,
        search_sort: str = "page",
        official_only: bool = False,
    ) -> tuple[list[UrlRecord], str]:
        """Search using GlobalBrowserManager - doesn't create separate Playwright instance"""
        import asyncio

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
                        const salesMatch = cardText.match(/\\d+(?:\\.\\d+)?\\s*万?\\+?\\s*(?:人付款|已售)/);
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
        backend: str = "browser-use",
        search_url: str | None = None,
        search_sort: str = "page",
        official_only: bool = False,
    ) -> list[UrlRecord]:
        """Single-path search: Playwright via CDP only, supports one-time manual login, error on failure."""
        import asyncio

        if not self.cdp_url:
            raise RuntimeError("CDP URL is required; set --playwright-cdp-url")

        try:
            records, anti_bot_signal = asyncio.run(
                self._search_with_playwright_async(
                    keyword=keyword,
                    top_n=top_n,
                    search_url=search_url,
                    search_sort=search_sort,
                    official_only=official_only,
                    use_cdp=True,
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
        cdp_context_index: int = 0,
        cdp_page_url_contains: str = "",
        cdp_connect_timeout_ms: int = 60_000,
        storage_state_file: str | Path | None = None,
        user_data_dir: str | Path | None = None,
        manual_login_on_demand: bool = False,
        manual_login_timeout_sec: int = 300,
        prefer_cdp: bool = True,
        allow_http_fallback: bool = True,
        use_global_browser: bool = False,  # New parameter for global browser manager
    ) -> None:
        self.storage = storage
        self.headless = headless
        self.browser_mode = (browser_mode or "cdp").strip().lower()
        if self.browser_mode not in {"cdp", "persistent"}:
            self.browser_mode = "cdp"
        self.cdp_url = cdp_url.strip()
        self.manual_wait_seconds = max(0, int(manual_wait_seconds))
        self.cdp_context_index = max(0, int(cdp_context_index))
        self.cdp_page_url_contains = cdp_page_url_contains.strip()
        self.cdp_connect_timeout_ms = max(10_000, int(cdp_connect_timeout_ms))
        self.storage_state_file = (
            Path(storage_state_file).resolve()
            if storage_state_file
            else Path("data/taobao_storage_state.json").resolve()
        )
        self.user_data_dir = (
            Path(user_data_dir).resolve()
            if user_data_dir
            else Path("data/taobao_profile").resolve()
        )
        self.manual_login_on_demand = bool(manual_login_on_demand)
        self.manual_login_timeout_sec = max(30, int(manual_login_timeout_sec))
        self.prefer_cdp = bool(prefer_cdp)
        self.allow_http_fallback = bool(allow_http_fallback)

        # Global browser manager support
        self.use_global_browser = bool(use_global_browser) and TOOLS_AVAILABLE
        self._global_browser_manager: Any = None
        # Structured login recovery events for diagnostics/output
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
            "领券",
            "立减",
            "优惠",
            "权益",
            "联系客服",
            "赠",
            "入会",
            "咨询",
            "活动",
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
            if "旗舰店" in snippet or "官方店" in snippet or "专卖店" in snippet:
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
            "淘宝",
            "天猫",
            "首页",
            "登录",
            "购物车",
            "客服",
            "举报",
            "收藏",
            "分享",
            "店铺",
            "旺旺",
        )
        seen_text: set[str] = set()
        chunks = [clean_text(raw, max_len=180) for raw in LINE_BREAK_RE.split(text)]
        chunks = [chunk for chunk in chunks if chunk]
        if len(chunks) <= 2:
            for fragment in re.split(r"[。；;!！?？|]", text):
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
        detail_summary = "；".join(detail_summary_parts)

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

    @staticmethod
    def _html_to_text(html: str) -> str:
        source = str(html or "")
        if not source:
            return ""
        source = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", source)
        source = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", source)
        source = re.sub(
            r"(?i)</?(br|p|div|li|h[1-6]|tr|td|section|article|span)[^>]*>",
            "\n",
            source,
        )
        source = re.sub(r"(?is)<[^>]+>", " ", source)
        source = html_unescape(source)
        lines = [clean_text(line, max_len=400) for line in source.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def _crawl_with_http_fallback(
        self, url: str, item_id: str, image_dir: Path
    ) -> ItemDetail:
        image_dir.mkdir(parents=True, exist_ok=True)
        crawl_time = now_iso()
        html, final_url = fetch_text_http(url, timeout_sec=30, max_retries=2)
        title_match = re.search(
            r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL
        )
        title = clean_text(title_match.group(1) if title_match else "", max_len=120)
        body_text = self._html_to_text(html)
        blocked_reason = detect_non_product_page(final_url, title, body_text)
        if blocked_reason:
            raise RuntimeError(blocked_reason)
        image_urls = re.findall(
            r"https?://[^\"'\s>]+(?:jpg|jpeg|png|webp|avif|gif)(?:[^\"'\s>]*)",
            html,
            flags=re.IGNORECASE,
        )
        image_meta = [
            {"src": src, "alt": "", "class_name": "", "width": 0, "height": 0}
            for src in image_urls[:120]
        ]
        return self._build_item_detail_from_raw(
            item_id=item_id,
            crawl_time=crawl_time,
            title=title,
            html=html,
            body_text=body_text,
            image_meta=image_meta,
            image_urls=image_urls,
        )

    async def _inject_saved_cookies_async(self, context: Any) -> None:
        cookies = load_valid_cookies(self.storage_state_file)
        if not cookies:
            return
        try:
            await context.add_cookies(cookies)
        except Exception:
            return

    async def _persist_storage_state_async(self, context: Any) -> None:
        self.storage_state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            await context.storage_state(path=str(self.storage_state_file))
        except Exception:
            return

    async def _open_context_async(
        self, playwright: Any, use_cdp: bool
    ) -> tuple[Any, Any | None, bool, bool]:
        """Connect to existing CDP browser or launch persistent context."""
        # Use global browser manager if enabled
        if self.use_global_browser and TOOLS_AVAILABLE:
            if self._global_browser_manager is None:
                self._global_browser_manager = await get_global_browser_manager()

            # Check if the browser is still valid
            context = self._global_browser_manager.browser_context
            needs_own_browser = False
            if context is None:
                needs_own_browser = True
            elif context._impl_obj is None:
                needs_own_browser = True
            elif not self._global_browser_manager.is_initialized:
                needs_own_browser = True

            if needs_own_browser:
                # Global browser connection is broken, create own browser
                LOG.warning("[Crawler] Global browser connection broken, creating own browser instance")
                # Create own Playwright instance
                from playwright.async_api import async_playwright
                if playwright is None:
                    playwright = await async_playwright().start()
                    self._own_playwright = playwright  # Store for cleanup
                # Fall through to original logic
                pass
            else:
                # Return context, browser (None), close flags
                return context, None, False, False

        # Original logic for backwards compatibility
        if use_cdp:
            if not self.cdp_url:
                raise RuntimeError("CDP URL is required; set --playwright-cdp-url")
            last_error = ""
            for candidate in build_cdp_endpoint_candidates(self.cdp_url):
                try:
                    browser = await playwright.chromium.connect_over_cdp(
                        candidate, timeout=self.cdp_connect_timeout_ms
                    )
                    contexts = browser.contexts
                    if contexts:
                        index = min(max(0, int(self.cdp_context_index)), len(contexts) - 1)
                        context = contexts[index]
                    else:
                        context = await browser.new_context(
                            viewport={"width": 1440, "height": 2200}
                        )
                    return context, browser, False, False
                except Exception as exc:
                    last_error = f"{candidate}: {exc}"
            raise RuntimeError(f"CDP connect failed ({self.cdp_url}): {last_error}")
        else:
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ]
            context = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.user_data_dir),
                headless=self.headless,
                args=args,
                viewport={"width": 1440, "height": 2200},
                channel="chrome",
            )
            return context, None, True, False

    @staticmethod
    async def _close_context_async(
        context: Any, browser: Any | None, close_context: bool, close_browser: bool
    ) -> None:
        if close_context:
            try:
                await context.close()
            except Exception:
                pass
        if close_browser and browser is not None:
            try:
                await browser.close()
            except Exception:
                pass

    async def _select_page_from_context(self, context: Any) -> tuple[Any, bool]:
        # Crawl tasks may run concurrently; always isolate each task in a new tab.
        # Use GlobalBrowserManager's new_page method if available to prevent race conditions
        if self.use_global_browser and self._global_browser_manager:
            page = await self._global_browser_manager.new_page()
            return page, True
        else:
            page = await context.new_page()
            return page, True

    async def _wait_manual_login_async(self, page: Any, stage: str) -> None:
        deadline = time.time() + max(30, int(self.manual_login_timeout_sec))
        try:
            await page.bring_to_front()
        except Exception:
            pass
        while time.time() < deadline:
            current_url = page.url
            try:
                title = await page.title()
            except Exception:
                title = ""
            try:
                body_text = await page.inner_text("body")
            except Exception:
                body_text = ""
            blocked_reason = detect_non_product_page(current_url, title, body_text)
            if not blocked_reason:
                return
            await page.wait_for_timeout(2000)
        raise RuntimeError(
            f"manual login timeout in {stage}, still blocked by login/anti-bot"
        )

    async def _try_open_detail_tab(self, page: Any) -> None:
        selectors = [
            "text=图文详情",
            "a:has-text('图文详情')",
            "li:has-text('图文详情')",
            "[role='tab']:has-text('图文详情')",
            "text=商品详情",
            "a:has-text('商品详情')",
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

    async def _crawl_async(
        self, url: str, item_id: str, image_dir: Path, allow_cdp: bool = True
    ) -> ItemDetail:
        from playwright.async_api import async_playwright

        image_dir.mkdir(parents=True, exist_ok=True)
        crawl_time = now_iso()
        html = ""
        body_text = ""
        title = ""
        image_urls: list[str] = []
        image_meta: list[dict[str, Any]] = []
        current_url = ""
        detail_blocks: list[dict[str, str]] = []
        async with async_playwright() as p:
            browser = None
            context = None
            page = None
            created_new_page = False
            close_context = False
            close_browser = False
            try:
                context, browser, close_context, close_browser = (
                    await self._open_context_async(
                        playwright=p,
                        use_cdp=bool(allow_cdp and self.browser_mode == "cdp"),
                    )
                )
                page, created_new_page = await self._select_page_from_context(context)

                await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))
                initial_reason = detect_non_product_page(
                    page.url, await page.title(), await page.inner_text("body")
                )
                if initial_reason:
                    if self.manual_login_on_demand:
                        await self._wait_manual_login_async(
                            page, stage=f"crawl:{item_id}"
                        )
                    elif not has_valid_login_cookie(self.storage_state_file):
                        raise RuntimeError(
                            "taobao item page requires login and local cookies are missing/expired. "
                            "Please log in manually when prompted."
                        )
                    else:
                        raise RuntimeError(
                            f"taobao item page blocked: {initial_reason}"
                        )

                scroll_rounds = max(4, 4 + (self.manual_wait_seconds // 4))
                for _ in range(scroll_rounds):
                    await page.mouse.wheel(0, 1800)
                    await page.wait_for_timeout(1200)
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
                detail_blocks = await self._collect_detail_blocks_from_page(
                    page, item_id=item_id, image_dir=image_dir
                )
                current_url = page.url
                html = await page.content()
                body_text = await page.inner_text("body")
                title = await page.title()
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
                blocked_reason = detect_non_product_page(current_url, title, body_text)
                if blocked_reason:
                    raise RuntimeError(blocked_reason)
                await self._persist_storage_state_async(context)
            finally:
                if page and created_new_page and not page.is_closed():
                    try:
                        await page.close()
                    except Exception:
                        pass
                if context is not None:
                    await self._persist_storage_state_async(context)
                    await self._close_context_async(
                        context, browser, close_context, close_browser
                    )

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

    async def _crawl_with_cdp_socket_async(
        self, url: str, item_id: str, image_dir: Path
    ) -> ItemDetail:
        import asyncio

        image_dir.mkdir(parents=True, exist_ok=True)
        crawl_time = now_iso()
        selected = pick_cdp_page(
            cdp_url=self.cdp_url,
            url_contains=self.cdp_page_url_contains or "detail.tmall.com/item.htm",
            preferred_domains=(
                "detail.tmall.com/item.htm",
                "item.taobao.com/item.htm",
                "taobao.com",
            ),
        )
        if not selected or not selected.get("webSocketDebuggerUrl"):
            raise RuntimeError("no debuggable page found from CDP /json/list")

        html = ""
        body_text = ""
        title = ""
        current_url = ""
        image_meta: list[dict[str, Any]] = []
        image_urls: list[str] = []

        async with AsyncCdpPage(
            selected["webSocketDebuggerUrl"], timeout_sec=35.0
        ) as page:
            await page.navigate(url)
            await page.wait_ready(timeout_sec=35.0)
            await asyncio.sleep(max(4.5, float(self.manual_wait_seconds)))
            scroll_rounds = max(4, 4 + (self.manual_wait_seconds // 4))
            for _ in range(scroll_rounds):
                await page.evaluate("window.scrollBy(0, 1800); true")
                await asyncio.sleep(1.0)

            title = str(await page.evaluate("document.title || ''") or "")
            current_url = str(await page.evaluate("location.href || ''") or "")
            html = str(
                await page.evaluate(
                    "document.documentElement ? document.documentElement.outerHTML : ''"
                )
                or ""
            )
            body_text = str(
                await page.evaluate("document.body ? document.body.innerText : ''")
                or ""
            )
            image_meta_raw = await page.evaluate(
                """
                (() => {
                  return Array.from(document.querySelectorAll('img')).map((n) => {
                    const rect = n.getBoundingClientRect();
                    return {
                      src: (n.currentSrc || n.src || '').trim(),
                      alt: (n.alt || '').trim(),
                      class_name: String(n.className || ''),
                      width: Number(n.naturalWidth || n.width || rect.width || 0),
                      height: Number(n.naturalHeight || n.height || rect.height || 0)
                    };
                  }).filter(v => v.src);
                })()
                """
            )
            if isinstance(image_meta_raw, list):
                image_meta = [row for row in image_meta_raw if isinstance(row, dict)]
                image_urls = [str(v.get("src", "")) for v in image_meta]

        blocked_reason = detect_non_product_page(current_url, title, body_text)
        if blocked_reason:
            raise RuntimeError(blocked_reason)
        return self._build_item_detail_from_raw(
            item_id=item_id,
            crawl_time=crawl_time,
            title=title,
            html=html,
            body_text=body_text,
            image_meta=image_meta,
            image_urls=image_urls,
        )

    def crawl(self, workbook_id: str, item_id: str, url: str) -> ItemDetail:
        """Single-path crawl: CDP only, one login attempt, error on failure."""
        import asyncio

        # Use global lock to serialize Playwright access across threads
        # This prevents connection corruption when multiple crawlers run concurrently
        with PLAYWRIGHT_LOCK:
            image_dir = self.storage.images_dir / workbook_id / item_id
            try:
                return asyncio.run(
                    self._crawl_async(url, item_id, image_dir, allow_cdp=True)
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
        """Async crawl using global browser - for single event loop architecture.

        This method is designed to be called within a single event loop, avoiding
        the asyncio.run() wrapper that causes Playwright connection issues.

        IMPORTANT: Login should be handled during the search phase. This method
        assumes the user is already logged in. If a login page is detected,
        it will fail immediately rather than blocking.
        """
        from tools import get_global_browser_manager

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

        try:
            # Get the global browser manager (should already be initialized)
            global_manager = await get_global_browser_manager()

            # Get or create a page for crawling
            page = await global_manager.get_page()

            # Navigate to the item page
            await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))

            # Check for non-product page (login required, blocked, etc.)
            initial_reason = detect_non_product_page(
                page.url, await page.title(), await page.inner_text("body")
            )
            if initial_reason:
                recovered = False
                if TOOLS_AVAILABLE:
                    LOG.warning(
                        "item page blocked for item %s, trying login recovery once: %s",
                        item_id,
                        initial_reason,
                    )
                    login_handler = TaobaoLogin(
                        browser_context=global_manager.browser_context,
                        context_page=page,
                        login_timeout_sec=self.manual_login_timeout_sec,
                    )
                    login_result: LoginHandleResult = await login_handler.check_and_handle_login()
                    self.login_recovery_events.append(
                        {
                            "source": "crawl",
                            "stage": "item-initial",
                            "item_id": item_id,
                            "ok": bool(login_result.ok),
                            "blocked_reason": initial_reason,
                            "final_state": login_result.final_state,
                            "reason": login_result.reason,
                            "elapsed_sec": round(float(login_result.elapsed_sec), 3),
                            "url": page.url,
                            "decision_trace": login_result.decision_trace,
                            "updated_at": now_iso(),
                        }
                    )
                    if login_result.ok:
                        await self._persist_storage_state_async(global_manager.browser_context)
                        await page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                        await page.wait_for_timeout(max(4500, self.manual_wait_seconds * 1000))
                        retry_reason = detect_non_product_page(
                            page.url, await page.title(), await page.inner_text("body")
                        )
                        if not retry_reason:
                            recovered = True
                        else:
                            initial_reason = retry_reason
                if not recovered:
                    raise RuntimeError(
                        f"taobao item page blocked: {initial_reason}. "
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
            await self._persist_storage_state_async(global_manager.browser_context)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
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
