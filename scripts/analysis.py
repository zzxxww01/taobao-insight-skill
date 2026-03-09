"""AI-driven analysis (Gemini), selling point extraction, and competitor comparisons."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import statistics
import textwrap
import time
import threading
import uuid
from contextlib import contextmanager
from html import escape as html_escape, unescape as html_unescape
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from config import (
    TITLE_POINT_RULES,
    UA,
)
from data import (
    ItemDetail,
    Storage,
    clean_text,
    normalize_brand_name,
    now_iso,
    preserve_multiline_text,
    read_text_utf8_best,
    _extract_brand_token,
)

LOG = logging.getLogger("taobao_insight")


class TransientGeminiError(RuntimeError):
    """Raised for temporary Gemini API errors that should be retried."""


class _GeminiKeyPool:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_index = 0
        self._cooldowns: dict[str, float] = {}

    def ordered_keys(self, keys: list[str]) -> list[tuple[str, str]]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in keys:
            key = str(raw or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        if not normalized:
            return []
        with self._lock:
            start = self._next_index % len(normalized)
            self._next_index = (self._next_index + 1) % len(normalized)
            ordered = normalized[start:] + normalized[:start]
            now = time.time()
            ready = [key for key in ordered if self._cooldowns.get(key, 0.0) <= now]
            cooling = [key for key in ordered if self._cooldowns.get(key, 0.0) > now]
        final_order = ready + cooling
        return [
            (key, "primary" if idx == 0 else f"alternate_{idx}")
            for idx, key in enumerate(final_order)
        ]

    def mark_transient(self, key: str, cooldown_sec: int = 90) -> None:
        if not key:
            return
        with self._lock:
            self._cooldowns[key] = time.time() + max(30, int(cooldown_sec))

    def mark_success(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            self._cooldowns.pop(key, None)


_GEMINI_KEY_POOL = _GeminiKeyPool()


def _split_env_key_list(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,;]+", str(raw_value or "").strip()):
        value = token.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _collect_gemini_keys(primary_key: str) -> list[str]:
    keys: list[str] = []
    for key in [primary_key]:
        value = str(key or "").strip()
        if value:
            keys.append(value)
    for env_name in ("GEMINI_API_KEYS", "GEMINI_API_KEY_FALLBACK", "GEMINI_API_KEY_ALT"):
        keys.extend(_split_env_key_list(os.getenv(env_name, "")))
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _mask_api_key(api_key: str) -> str:
    value = str(api_key or "").strip()
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _gemini_debug_dir() -> Path | None:
    raw = os.getenv("TAOBAO_INSIGHT_GEMINI_DEBUG_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _serialize_gemini_contents_for_debug(contents: Any) -> list[dict[str, Any]]:
    items = contents if isinstance(contents, list) else [contents]
    serialized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            data = item.get("data")
            mime_type = clean_text(str(item.get("mime_type", "")), max_len=80)
            if isinstance(data, (bytes, bytearray)):
                payload = bytes(data)
                serialized.append(
                    {
                        "kind": "inline_data",
                        "mime_type": mime_type or "application/octet-stream",
                        "byte_length": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                continue
        if isinstance(item, (bytes, bytearray)):
            payload = bytes(item)
            serialized.append(
                {
                    "kind": "inline_data",
                    "mime_type": "application/octet-stream",
                    "byte_length": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            continue
        serialized.append({"kind": "text", "text": str(item or "")})
    return serialized


def _dump_gemini_debug_record(
    *,
    model: str,
    contents: Any,
    backend: str,
    response_text: str = "",
    error: str = "",
    debug_meta: dict[str, Any] | None = None,
    api_key_alias: str = "",
    api_key_mask: str = "",
) -> None:
    target_dir = _gemini_debug_dir()
    if target_dir is None:
        return
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        serialized_contents = _serialize_gemini_contents_for_debug(contents)
        text_parts = [
            str(part.get("text", ""))
            for part in serialized_contents
            if part.get("kind") == "text"
        ]
        payload = {
            "ts": now_iso(),
            "backend": backend,
            "model": model,
            "api_key_alias": api_key_alias or "",
            "api_key_mask": api_key_mask or "",
            "debug_meta": debug_meta or {},
            "has_binary_parts": any(
                part.get("kind") == "inline_data" for part in serialized_contents
            ),
            "text_part_count": len(text_parts),
            "text_parts": text_parts,
            "contents": serialized_contents,
            "response_text": response_text or "",
            "error": error or "",
        }
        filename = (
            f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:10]}-"
            f"{re.sub(r'[^a-zA-Z0-9._-]+', '-', model or 'gemini').strip('-') or 'gemini'}.json"
        )
        (target_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        LOG.warning("Failed to dump Gemini debug record: %s", exc)


def _is_transient_gemini_exception(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = (
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
    return any(token in text for token in tokens)


@contextmanager
def gemini_proxy_env(proxy_url: str) -> Any:
    keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    old_values = {k: os.environ.get(k) for k in keys}
    try:
        if proxy_url:
            for key in keys:
                os.environ[key] = proxy_url
        yield
    finally:
        for key in keys:
            previous = old_values.get(key)
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _extract_text_from_genai_response(response: Any) -> str:
    text_value = getattr(response, "text", "")
    if isinstance(text_value, str) and text_value.strip():
        return text_value.strip()
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not isinstance(parts, list):
                continue
            fragments: list[str] = []
            for part in parts:
                part_text = getattr(part, "text", "")
                if isinstance(part_text, str) and part_text.strip():
                    fragments.append(part_text.strip())
            if fragments:
                return "\n".join(fragments).strip()
    return ""


def _build_genai_contents(contents: Any, genai_types: Any) -> Any:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        built_parts: list[Any] = []
        for item in contents:
            if isinstance(item, str):
                built_parts.append(genai_types.Part.from_text(text=item))
                continue
            if isinstance(item, dict):
                data = item.get("data")
                mime_type = str(item.get("mime_type", "") or "image/jpeg")
                if isinstance(data, (bytes, bytearray)):
                    built_parts.append(
                        genai_types.Part.from_bytes(
                            data=bytes(data), mime_type=mime_type
                        )
                    )
                    continue
            built_parts.append(genai_types.Part.from_text(text=str(item)))
        return built_parts
    return str(contents or "")


def gemini_generate_json_text(
    api_key: str,
    model: str,
    contents: Any,
    proxy_url: str = "",
    timeout_sec: int = 45,
    raise_on_transient: bool = False,
    debug_meta: dict[str, Any] | None = None,
) -> str:
    keys = _collect_gemini_keys(api_key)
    if not keys:
        LOG.error(f"Gemini API key is missing. Skipping API call to {model}.")
        return ""

    with gemini_proxy_env(proxy_url):
        try:
            from google import genai as google_genai
            from google.genai import types as genai_types
        except Exception:
            google_genai = None
            genai_types = None

        last_error: Exception | None = None
        transient_error: Exception | None = None
        ordered_keys = _GEMINI_KEY_POOL.ordered_keys(keys)
        for index, (key, key_alias) in enumerate(ordered_keys, start=1):
            key_mask = _mask_api_key(key)
            key_errors: list[Exception] = []

            if google_genai is not None and genai_types is not None:
                try:
                    client_kwargs: dict[str, Any] = {"api_key": key}
                    try:
                        timeout_ms = max(5_000, int(timeout_sec * 1000))
                        client_kwargs["http_options"] = genai_types.HttpOptions(
                            timeout=timeout_ms
                        )
                    except Exception:
                        pass

                    client = google_genai.Client(**client_kwargs)
                    genai_contents = _build_genai_contents(contents, genai_types)
                    config = genai_types.GenerateContentConfig(
                        responseMimeType="application/json"
                    )
                    response = client.models.generate_content(
                        model=model,
                        contents=genai_contents,
                        config=config,
                    )
                    text = _extract_text_from_genai_response(response)
                    _dump_gemini_debug_record(
                        model=model,
                        contents=contents,
                        backend="google.genai",
                        response_text=text or "",
                        debug_meta=debug_meta,
                        api_key_alias=key_alias,
                        api_key_mask=key_mask,
                    )
                    _GEMINI_KEY_POOL.mark_success(key)
                    return text or ""
                except Exception as exc:
                    key_errors.append(exc)
                    _dump_gemini_debug_record(
                        model=model,
                        contents=contents,
                        backend="google.genai",
                        error=str(exc),
                        debug_meta=debug_meta,
                        api_key_alias=key_alias,
                        api_key_mask=key_mask,
                    )

            try:
                import google.generativeai as legacy_genai

                legacy_genai.configure(api_key=key)
                legacy_model = legacy_genai.GenerativeModel(model)
                response = legacy_model.generate_content(
                    contents,
                    generation_config={"response_mime_type": "application/json"},
                    request_options={"timeout": max(5, int(timeout_sec))},
                )
                text = (response.text or "").strip()
                _dump_gemini_debug_record(
                    model=model,
                    contents=contents,
                    backend="google.generativeai",
                    response_text=text,
                    debug_meta=debug_meta,
                    api_key_alias=key_alias,
                    api_key_mask=key_mask,
                )
                _GEMINI_KEY_POOL.mark_success(key)
                return text
            except Exception as exc:
                key_errors.append(exc)
                _dump_gemini_debug_record(
                    model=model,
                    contents=contents,
                    backend="google.generativeai",
                    error=str(exc),
                    debug_meta=debug_meta,
                    api_key_alias=key_alias,
                    api_key_mask=key_mask,
                )

            for exc in key_errors:
                last_error = exc
                if _is_transient_gemini_exception(exc):
                    transient_error = exc
            if transient_error and any(
                _is_transient_gemini_exception(exc) for exc in key_errors
            ):
                _GEMINI_KEY_POOL.mark_transient(key)
                if len(ordered_keys) > 1 and index < len(ordered_keys):
                    LOG.warning(
                        "Gemini key %s hit transient limit; switching key for model %s",
                        key_alias,
                        model,
                    )
                    continue
            else:
                _GEMINI_KEY_POOL.mark_success(key)
            if last_error is not None:
                LOG.error(
                    "Failed to generate content with Gemini (%s via %s): %s",
                    model,
                    key_alias,
                    last_error,
                    exc_info=True,
                )
                break

        if raise_on_transient and transient_error is not None:
            raise TransientGeminiError(str(transient_error)) from transient_error
        return ""


class SellingPointExtractor:
    def __init__(
        self,
        gemini_api_key: str = "",
        gemini_proxy_url: str = "",
        gemini_flash_model: str = "gemini-flash-latest",
        gemini_pro_model: str = "gemini-3.1-pro-preview",
        gemini_timeout_sec: int = 45,
        gemini_raise_on_transient: bool = False,
    ) -> None:
        self.gemini_api_key = (gemini_api_key or "").strip()
        self.gemini_flash_model = gemini_flash_model
        self.gemini_pro_model = gemini_pro_model
        self.gemini_proxy_url = gemini_proxy_url.strip()
        self.gemini_timeout_sec = max(5, int(gemini_timeout_sec))
        self.gemini_raise_on_transient = bool(gemini_raise_on_transient)
        try:
            self.gemini_image_max_candidates = max(
                6, int(os.getenv("GEMINI_IMAGE_MAX_CANDIDATES", "24"))
            )
        except ValueError:
            self.gemini_image_max_candidates = 24
        try:
            self.gemini_image_batch_max_images = max(
                1, int(os.getenv("GEMINI_IMAGE_BATCH_MAX_IMAGES", "8"))
            )
        except ValueError:
            self.gemini_image_batch_max_images = 8
        try:
            self.gemini_image_batch_max_bytes = max(
                1_000_000, int(os.getenv("GEMINI_IMAGE_BATCH_MAX_BYTES", str(18 * 1024 * 1024)))
            )
        except ValueError:
            self.gemini_image_batch_max_bytes = 18 * 1024 * 1024

    @staticmethod
    def _platform_key(item_context: dict[str, str]) -> str:
        return clean_text(str(item_context.get("platform", "")), max_len=20).lower()

    @staticmethod
    def _platform_name(item_context: dict[str, str]) -> str:
        platform = SellingPointExtractor._platform_key(item_context)
        if platform == "jd":
            return "京东"
        if platform == "taobao":
            return "淘宝"
        return "电商"

    @staticmethod
    def _detail_term(item_context: dict[str, str]) -> str:
        platform = SellingPointExtractor._platform_key(item_context)
        if platform == "taobao":
            return "图文详情"
        return "商品详情"

    @staticmethod
    def _detail_reference_label(item_context: dict[str, str]) -> str:
        if SellingPointExtractor._platform_key(item_context) == "taobao":
            return "图文详情参考图："
        return "商品详情参考图："

    @staticmethod
    def _detail_citation_label(item_platform: str) -> str:
        platform = clean_text(str(item_platform or ""), max_len=20).lower()
        if platform == "taobao":
            return "图文详情页引用："
        return "商品详情引用："

    @staticmethod
    def _is_noise_point_text(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        noise_tokens = (
            "公益宝贝",
            "每笔成交将",
            "累计捐赠",
            "捐赠",
            "运费险",
            "店铺首页",
            "进店逛逛",
            "客服",
            "收藏店铺",
            "领券",
            "入会",
            "优惠",
            "活动规则",
            "满减",
            "88vip",
            "抢购",
            "立减",
            "优惠券",
            "券后",
            "到手价",
            "赠品",
        )
        lowered = value.lower()
        extra_non_product_tokens = (
            "???",
            "???",
            "?????????",
            "????",
            "???",
            "???",
            "???",
            "??????",
            "??????",
            "??????",
            "??????",
            "??????",
            "???",
            "????",
            "???",
            "???",
            "????",
            "??????",
            "????",
            "???",
        )
        if any(token in lowered for token in extra_non_product_tokens):
            return True
        if any(token in value for token in noise_tokens):
            return True
        if any(
            token in lowered
            for token in ("tmall.com", "taobao.com", "http://", "https://")
        ):
            return True
        return False

    @staticmethod
    def _is_promotional_point_text(text: str) -> bool:
        value = str(text or "").strip().lower()
        if not value:
            return False
        promo_tokens = (
            "88vip",
            "满减",
            "领券",
            "抢购",
            "立减",
            "优惠",
            "券后",
            "到手价",
            "赠品",
            "活动",
        )
        extra_service_tokens = (
            "顺丰",
            "速达",
            "退货",
            "无理由",
            "顾问",
            "咨询",
            "物流",
            "服务",
            "礼券",
            "礼遇",
            "官方直供",
            "原装正品",
            "线上专属",
            "伪素颜",
            "氛围感",
            "送女友",
            "女神礼物",
            "生日礼物",
            "38节",
            "销售额第一",
            "销量第一",
            "排名第一",
        )
        if any(token in value for token in extra_service_tokens):
            return True
        return any(token in value for token in promo_tokens)

    @staticmethod
    def summarize_detail(
        item_context: dict[str, str], detail_blocks: list[dict[str, str]]
    ) -> str:
        _ = item_context
        if not detail_blocks:
            return ""
        rows: list[str] = []
        seen: set[str] = set()
        image_refs: list[str] = []
        for block in detail_blocks:
            ref = clean_text(str(block.get("source_ref", "")), max_len=24)
            if not ref:
                continue
            source_type = clean_text(str(block.get("source_type", "")), max_len=12)
            content = clean_text(str(block.get("content", "")), max_len=120)
            if not content:
                continue
            if SellingPointExtractor._is_noise_point_text(content):
                continue
            if source_type == "image":
                image_refs.append(ref)
            else:
                key = f"{content}|{ref}"
                if key in seen:
                    continue
                seen.add(key)
                rows.append(f"{content}[{ref}]")
            if len(rows) >= 6:
                break
        if image_refs:
            unique_refs: list[str] = []
            for ref in image_refs:
                if ref and ref not in unique_refs:
                    unique_refs.append(ref)
                if len(unique_refs) >= 6:
                    break
            if unique_refs:
                rows.append(
                    SellingPointExtractor._detail_reference_label(item_context)
                    + "、".join(unique_refs)
                )
        return "；".join(rows)

    @staticmethod
    def summarize_points(
        points: list[dict[str, Any]], fallback: str = "", max_items: int = 8
    ) -> str:
        rows: list[str] = []
        seen: set[str] = set()
        for point in points:
            if not isinstance(point, dict):
                continue
            point_text = clean_text(str(point.get("point", "")), max_len=70)
            citation = clean_text(str(point.get("citation", "")), max_len=40)
            if not point_text:
                continue
            if SellingPointExtractor._is_noise_point_text(point_text):
                continue
            row = f"{point_text}[{citation}]" if citation else point_text
            if row in seen:
                continue
            seen.add(row)
            rows.append(row)
            if len(rows) >= max_items:
                break
        if rows:
            return "；".join(rows)
        return clean_text(fallback, max_len=2000)

    @staticmethod
    def _extract_json(text: str) -> Any:
        if not text:
            return []
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"(\[[\s\S]*\])", text)
        if match:
            snippet = match.group(1)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def _build_prompt(
        item_context: dict[str, str], detail_blocks: list[dict[str, str]]
    ) -> str:
        detail_term = SellingPointExtractor._detail_term(item_context)
        platform_name = SellingPointExtractor._platform_name(item_context)
        price_text = clean_text(str(item_context.get("price_text", "")), max_len=120)
        sku_summary = clean_text(str(item_context.get("sku_summary", "")), max_len=240)
        lines: list[str] = []
        for block in detail_blocks:
            source_type = clean_text(str(block.get("source_type", "")), max_len=20).lower()
            if source_type != "text":
                continue
            raw_content = str(block.get("content", "") or "")
            content = clean_text(raw_content, max_len=220)
            if any(
                token in raw_content
                for token in (
                    "我的淘宝",
                    "购物车",
                    "收藏夹",
                    "免费开店",
                    "千牛卖家中心",
                    "帮助中心",
                    "无障碍",
                )
            ):
                continue
            if not content or SellingPointExtractor._is_noise_point_text(content):
                continue
            lines.append(f"- {block.get('source_ref', '')} | text | {content}")
            if len(lines) >= 40:
                break
        payload = "\n".join(lines) or "\u65e0\u53ef\u9760\u6587\u672c\u8bc1\u636e"
        return textwrap.dedent(
            f"""
            \u4f60\u662f\u8d44\u6df1{platform_name}\u5546\u54c1\u7814\u7a76\u5206\u6790\u5e08\u3002\u8bf7\u53ea\u4f9d\u636e\u7ed9\u5b9a{detail_term}\u6587\u672c\u8bc1\u636e\uff0c\u63d0\u70bc\u6d88\u8d39\u8005\u53ef\u76f4\u63a5\u611f\u77e5\u7684\u5546\u54c1\u5356\u70b9\u3002
            \u5546\u54c1\u4e0a\u4e0b\u6587\uff1a
            - item_id: {item_context.get("item_id", "")}
            - platform: {platform_name}
            - title: {item_context.get("title", "")}
            - brand: {item_context.get("brand", "")}
            - shop_name: {item_context.get("shop_name", "")}
            - price: {price_text}
            - sku_summary: {sku_summary}

            {detail_term}\u6587\u672c\u8bc1\u636e\uff08source_ref | source_type | content\uff09\uff1a
            {payload}

            \u7ea6\u675f\uff1a
            1) \u53ea\u80fd\u5f15\u7528\u7ed9\u5b9a\u8bc1\u636e\uff0c\u4e0d\u5f97\u8865\u5145\u5916\u90e8\u5e38\u8bc6\u3002
            2) \u5982\u679c\u8bc1\u636e\u4e0d\u8db3\uff0c\u5b81\u53ef\u5c11\u5199\uff1b\u5b8c\u5168\u4e0d\u8db3\u65f6\u8fd4\u56de []\u3002
            3) \u5356\u70b9\u5fc5\u987b\u662f\u6d88\u8d39\u8005\u53ef\u76f4\u63a5\u611f\u77e5\u7684\u8868\u8fbe\uff0c\u4f18\u5148\u5986\u6548\u3001\u80a4\u611f\u3001\u6750\u8d28\u3001\u89c4\u683c\u7ec4\u5408\u3001\u793c\u8d60\u4fe1\u606f\u3002
            4) \u5ffd\u7565\u5e73\u53f0\u5bfc\u822a\u3001\u5e97\u94fa\u5165\u53e3\u3001\u7269\u6d41\u552e\u540e\u3001\u4f1a\u5458\u6743\u76ca\u3001\u7eaf\u4fc3\u9500\u8bcd\u3001\u9875\u9762\u58f3\u4fe1\u606f\u3002
            5) \u6bcf\u6761\u5fc5\u987b\u7ed1\u5b9a citation\uff0c\u4e14 citation \u5fc5\u987b\u6765\u81ea\u4e0a\u8ff0 source_ref\u3002
            6) \u4e0d\u5141\u8bb8\u628a\u540c\u4e00\u53e5\u8bdd\u62c6\u6210\u591a\u4e2a\u8fd1\u4e49\u91cd\u590d\u5356\u70b9\uff1b\u53bb\u91cd\u540e\u63a7\u5236\u5728 2-6 \u6761\u3002
            7) \u56fe\u7247\u91cc\u624d\u770b\u5f97\u5230\u7684\u4fe1\u606f\u4e0d\u8981\u5728\u8fd9\u91cc\u8f93\u51fa\u3002
            \u4ec5\u8f93\u51fa JSON \u6570\u7ec4\uff1a
            [{{"point": "\u4e1d\u7ed2\u96fe\u9762\u5986\u6548", "citation": "text_1"}}]
            """
        ).strip()

    @staticmethod
    def _guess_mime_type(url_or_path: str) -> str:
        value = str(url_or_path or "").strip().lower()
        if value.endswith(".png"):
            return "image/png"
        if value.endswith(".webp"):
            return "image/webp"
        if value.endswith(".gif"):
            return "image/gif"
        return "image/jpeg"

    @staticmethod
    def _load_image_binary(content: str) -> tuple[bytes, str]:
        source = (content or "").strip()
        if not source:
            return b"", ""
        as_path = Path(source)
        if as_path.exists() and as_path.is_file():
            try:
                data = as_path.read_bytes()
                return data, SellingPointExtractor._guess_mime_type(source)
            except Exception:
                return b"", ""
        if not source.lower().startswith("http"):
            return b"", ""
        lower = source.lower()
        referer = ""
        if any(token in lower for token in ("jd.com", "360buyimg", "jdimg", "3.cn")):
            referer = "https://item.jd.com/"
        elif any(token in lower for token in ("tmall", "taobao", "alicdn", "tbcdn")):
            referer = "https://detail.tmall.com/"
        headers = {"User-Agent": UA}
        if referer:
            headers["Referer"] = referer
        try:
            request = urllib.request.Request(
                source,
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                data = response.read()
                content_type = (
                    str(response.headers.get("content-type", "") or "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                mime = (
                    content_type
                    if content_type.startswith("image/")
                    else SellingPointExtractor._guess_mime_type(source)
                )
                return data, mime
        except Exception:
            return b"", ""

    @staticmethod
    def _ref_sort_key(ref: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", str(ref or ""))
        index = int(match.group(1)) if match else 10_000
        return index, str(ref or "")

    def _build_image_candidates(
        self, detail_blocks: list[dict[str, str]]
    ) -> list[tuple[str, str, bytes]]:
        image_blocks = [
            b
            for b in detail_blocks
            if str(b.get("source_type", "")).strip().lower() == "image"
        ]
        if not image_blocks:
            return []
        ranked: list[tuple[int, tuple[str, str, bytes]]] = []
        seen_refs: set[str] = set()
        seen_sources: set[str] = set()
        for block in image_blocks[: max(len(image_blocks), self.gemini_image_max_candidates * 2)]:
            ref = clean_text(str(block.get("source_ref", "")), max_len=24)
            source = str(block.get("content", "") or "").strip().lower()
            if not ref or ref in seen_refs:
                continue
            source_key = source
            if source_key and source_key in seen_sources:
                continue
            if any(
                token in source
                for token in ("/shaidan/", "default.image", "i.imageupload", "avatar", "comment", "review")
            ):
                continue
            seen_refs.add(ref)
            if source_key:
                seen_sources.add(source_key)
            data, mime = self._load_image_binary(str(block.get("content", "")))
            if not data:
                continue
            if len(data) > 8 * 1024 * 1024:
                continue
            if len(data) < 2 * 1024:
                continue
            mime_norm = (mime or "image/jpeg").strip().lower()
            if mime_norm == "image/gif" and len(data) < 24 * 1024:
                continue
            score = 0
            if 24 * 1024 <= len(data) <= 4 * 1024 * 1024:
                score += 3
            elif len(data) >= 8 * 1024:
                score += 1
            if mime_norm in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
                score += 2
            if ref == "image_1":
                score += 1
            ranked.append((score, (ref, mime_norm or "image/jpeg", data)))
        ranked.sort(key=lambda item: (-item[0], self._ref_sort_key(item[1][0])))
        return [row for _, row in ranked[: self.gemini_image_max_candidates]]

    def _split_image_candidate_batches(
        self, candidates: list[tuple[str, str, bytes]]
    ) -> list[list[tuple[str, str, bytes]]]:
        if not candidates:
            return []
        batches: list[list[tuple[str, str, bytes]]] = []
        current: list[tuple[str, str, bytes]] = []
        current_bytes = 0
        for row in candidates:
            row_size = len(row[2])
            over_image_cap = len(current) >= self.gemini_image_batch_max_images
            over_byte_cap = current and (current_bytes + row_size > self.gemini_image_batch_max_bytes)
            if current and (over_image_cap or over_byte_cap):
                batches.append(current)
                current = []
                current_bytes = 0
            current.append(row)
            current_bytes += row_size
        if current:
            batches.append(current)
        return batches

    def _generate_image_points_with_gemini(
        self,
        item_context: dict[str, str],
        candidates: list[tuple[str, str, bytes]],
        max_points: int = 8,
        debug_meta: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        if not self.gemini_api_key:
            return []
        if not candidates:
            return []

        refs = [ref for ref, _, _ in candidates]
        parts: list[Any] = []
        for ref, mime, data in candidates:
            parts.append(f"{ref}:")
            parts.append({"mime_type": mime or "image/jpeg", "data": data})

        detail_term = self._detail_term(item_context)
        platform_name = self._platform_name(item_context)
        price_text = clean_text(str(item_context.get("price_text", "")), max_len=120)
        sku_summary = clean_text(str(item_context.get("sku_summary", "")), max_len=240)
        prompt = textwrap.dedent(
            f"""
            \u4f60\u662f{platform_name}{detail_term}\u56fe\u7247\u5206\u6790\u52a9\u624b\u3002\u8bf7\u53ea\u6839\u636e\u8f93\u5165\u56fe\u7247\uff0c\u63d0\u70bc\u5546\u54c1\u5356\u70b9\u3002
            \u5546\u54c1ID: {item_context.get("item_id", "")}
            \u5546\u54c1\u6807\u9898: {item_context.get("title", "")}
            \u54c1\u724c: {item_context.get("brand", "")}
            \u5e97\u94fa: {item_context.get("shop_name", "")}
            \u4ef7\u683c: {price_text}
            SKU \u6458\u8981: {sku_summary}

            \u7ea6\u675f\uff1a
            1) \u53ea\u80fd\u4f9d\u636e\u56fe\u7247\u91cc\u76f4\u63a5\u53ef\u89c1\u7684\u6587\u5b57\u3001\u5305\u88c5\u3001\u989c\u8272\u3001\u7ed3\u6784\u3001\u89c4\u683c\u3001\u793c\u76d2\u4e0e\u8d60\u54c1\u4fe1\u606f\u3002
            2) \u5ffd\u7565\u5e73\u53f0\u5bfc\u822a\u3001\u5e97\u94fa\u5165\u53e3\u3001\u5ba2\u670d\u7269\u6d41\u552e\u540e\u3001\u4f1a\u5458\u6743\u76ca\u3001\u7eaf\u4fc3\u9500\u6a2a\u5e45\u3001\u4e0e\u5546\u54c1\u65e0\u5173\u7684\u9875\u9762\u88c5\u9970\u3002
            3) \u8bc1\u636e\u4e0d\u8db3\u65f6\u8fd4\u56de []\uff0c\u4e0d\u8981\u731c\u6d4b\u3002
            4) \u6bcf\u6761\u5356\u70b9\u5fc5\u987b\u5305\u542b point \u4e0e citation\u3002
            5) citation \u53ea\u80fd\u662f\u4ee5\u4e0b\u7f16\u53f7\u4e4b\u4e00: {", ".join(sorted(refs, key=self._ref_sort_key))}.
            6) \u4e0d\u8981\u8f93\u51fa\u8fd1\u4e49\u91cd\u590d\u70b9\uff1b\u53ea\u63d0\u70bc\u6700\u6709\u4fe1\u606f\u91cf\u7684 1-{max(2, max_points)} points.
            \u4ec5\u8f93\u51fa JSON \u6570\u7ec4\uff1a
            [{{"point": "Gift box with limited packaging", "citation": "image_1"}}]
            """
        ).strip()
        raw_text = gemini_generate_json_text(
            api_key=self.gemini_api_key,
            model=self.gemini_flash_model,
            contents=[prompt] + parts,
            proxy_url=self.gemini_proxy_url,
            timeout_sec=self.gemini_timeout_sec,
            raise_on_transient=self.gemini_raise_on_transient,
            debug_meta=debug_meta,
        )
        if not raw_text:
            return []
        payload = self._extract_json(raw_text)
        if not isinstance(payload, list):
            return []
        valid_rows = self._validate(payload, set(refs))
        return valid_rows[:max_points]

    def _extract_image_points_with_gemini(
        self,
        item_context: dict[str, str],
        detail_blocks: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        candidates = self._build_image_candidates(detail_blocks)
        if not candidates:
            return []

        batches = self._split_image_candidate_batches(candidates)
        merged: list[dict[str, str]] = []
        seen: set[str] = set()
        valid_refs = {
            b.get("source_ref", "") for b in detail_blocks if b.get("source_ref")
        }
        base_meta = {
            "stage": "llm_extract_images",
            "item_id": str(item_context.get("item_id", "")),
            "workbook_id": str(item_context.get("workbook_id", "")),
            "task_id": str(item_context.get("task_id", "")),
            "platform": str(item_context.get("platform", "")),
        }
        for batch_index, batch in enumerate(batches, start=1):
            batch_points = self._generate_image_points_with_gemini(
                item_context=item_context,
                candidates=batch,
                max_points=min(8, max(3, len(batch) + 1)),
                debug_meta={
                    **base_meta,
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "candidate_refs": [ref for ref, _, _ in batch],
                },
            )
            for point in batch_points:
                key = f"{point.get('point', '')}|{point.get('citation', '')}"
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(point)
        if merged:
            return self._validate(merged, valid_refs)

        for row in candidates:
            single = self._generate_image_points_with_gemini(
                item_context=item_context,
                candidates=[row],
                max_points=2,
                debug_meta={
                    **base_meta,
                    "stage": "llm_extract_images_single_fallback",
                    "candidate_refs": [row[0]],
                },
            )
            for point in single:
                key = f"{point.get('point', '')}|{point.get('citation', '')}"
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(point)
                if len(merged) >= 12:
                    break
            if len(merged) >= 12:
                break
        if merged:
            return self._validate(merged, valid_refs)
        return []

    @staticmethod
    def _validate(
        points: list[dict[str, Any]], valid_refs: set[str], limit: int = 12
    ) -> list[dict[str, str]]:
        clean_rows: list[dict[str, str]] = []
        seen: set[str] = set()
        citation_counts: dict[str, int] = {}
        for point_obj in points:
            if not isinstance(point_obj, dict):
                continue
            point = clean_text(str(point_obj.get("point", "")), max_len=140)
            citation = clean_text(str(point_obj.get("citation", "")), max_len=80)
            if not point or not citation:
                continue
            if SellingPointExtractor._is_noise_point_text(point):
                continue
            if citation not in valid_refs:
                continue
            if citation_counts.get(citation, 0) >= 3:
                continue
            key = f"{point}|{citation}"
            if key in seen:
                continue
            seen.add(key)
            citation_counts[citation] = citation_counts.get(citation, 0) + 1
            clean_rows.append({"point": point, "citation": citation})
            if len(clean_rows) >= max(1, int(limit)):
                break
        return clean_rows

    def _extract_with_gemini(
        self, item_context: dict[str, str], detail_blocks: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        if not self.gemini_api_key:
            return []

        prompt = self._build_prompt(item_context, detail_blocks)
        raw = gemini_generate_json_text(
            api_key=self.gemini_api_key,
            model=self.gemini_flash_model,
            contents=prompt,
            proxy_url=self.gemini_proxy_url,
            timeout_sec=self.gemini_timeout_sec,
            raise_on_transient=self.gemini_raise_on_transient,
            debug_meta={
                "stage": "llm_extract_text",
                "item_id": str(item_context.get("item_id", "")),
                "workbook_id": str(item_context.get("workbook_id", "")),
                "task_id": str(item_context.get("task_id", "")),
                "platform": str(item_context.get("platform", "")),
            },
        )
        if not raw:
            return []
        payload = self._extract_json(raw)
        if not isinstance(payload, list):
            return []
        valid_refs = {
            b.get("source_ref", "") for b in detail_blocks if b.get("source_ref")
        }
        return self._validate(payload, valid_refs)

    def extract(
        self,
        item_context: dict[str, str],
        detail_blocks: list[dict[str, str]],
        detail_summary: str = "",
    ) -> list[dict[str, str]]:
        _ = detail_summary
        if not self.gemini_api_key:
            return []
        valid_refs = {
            b.get("source_ref", "") for b in detail_blocks if b.get("source_ref")
        }
        points = self._extract_image_points_with_gemini(item_context, detail_blocks)
        if not points:
            points = self._extract_with_gemini(item_context, detail_blocks)
        validated = self._validate(points, valid_refs)
        if validated:
            validated = [
                p
                for p in validated
                if not self._is_promotional_point_text(p.get("point", ""))
            ]
        return validated


class Analyzer:
    def __init__(
        self,
        gemini_api_key: str | None,
        gemini_flash_model: str = "gemini-flash-latest",
        gemini_pro_model: str = "gemini-3.1-pro-preview",
        gemini_proxy_url: str = "",
        gemini_timeout_sec: int = 45,
        gemini_raise_on_transient: bool = False,
        gemini_pro_retries: int = 2,
    ) -> None:
        self.gemini_api_key = (gemini_api_key or "").strip()
        self.gemini_flash_model = gemini_flash_model
        self.gemini_pro_model = gemini_pro_model
        self.gemini_proxy_url = gemini_proxy_url.strip()
        self.gemini_timeout_sec = max(5, int(gemini_timeout_sec))
        self.gemini_raise_on_transient = bool(gemini_raise_on_transient)
        self.gemini_pro_retries = max(0, int(gemini_pro_retries))

    @staticmethod
    def _extract_json(text: str) -> Any:
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            snippet = match.group(1)
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _to_string_list(value: Any, max_len: int = 120) -> list[str]:
        if not isinstance(value, list):
            return []
        clean_rows: list[str] = []
        for item in value:
            text = clean_text(str(item), max_len=max_len)
            if text:
                clean_rows.append(text)
        return clean_rows

    @staticmethod
    def _theme_of_point(point: str) -> str:
        value = point or ""
        if any(
            token in value
            for token in ("控油", "持妆", "定妆", "防汗", "防水", "不脱妆")
        ):
            return "控油持妆"
        if any(
            token in value for token in ("遮瑕", "柔焦", "磨皮", "毛孔", "提亮", "细腻")
        ):
            return "底妆修饰"
        if any(
            token in value for token in ("轻薄", "服帖", "不卡粉", "不暗沉", "妆感")
        ):
            return "肤感体验"
        if any(
            token in value
            for token in ("油皮", "干皮", "混油", "敏感", "补妆", "通勤", "学生")
        ):
            return "场景人群"
        return "其他表达"

    @staticmethod
    def _price_quantiles(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        ordered = sorted(values)
        q1 = ordered[max(0, int((len(ordered) - 1) * 0.33))]
        q2 = ordered[max(0, int((len(ordered) - 1) * 0.66))]
        return float(q1), float(q2)

    @staticmethod
    def _price_tier(price: float | None, q1: float, q2: float) -> str:
        if price is None:
            return "未知"
        if price <= q1:
            return "入门"
        if price <= q2:
            return "中端"
        return "高端"

    @staticmethod
    def _normalize_note_text(note: str, max_len: int = 180) -> str:
        text = clean_text(str(note or ""), max_len=max_len)
        if not text:
            return ""
        if "..." in text:
            text = text.split("...", 1)[0].rstrip("，、；:：- ")
        return clean_text(text, max_len=max_len)

    @staticmethod
    def _extract_refs_from_text(text: str, max_refs: int = 10) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()
        for token in re.findall(
            r"(?:image|text)_\d+", str(text or ""), flags=re.IGNORECASE
        ):
            ref = token.lower()
            if ref in seen:
                continue
            seen.add(ref)
            refs.append(ref)
            if len(refs) >= max_refs:
                break
        return refs

    @staticmethod
    def _count_brands(rows: list[dict[str, str]]) -> tuple[dict[str, int], int]:
        brand_count: dict[str, int] = {}
        unknown_brand_count = 0
        for row in rows:
            # For overview statistics, prefer explicit brand/shop fields.
            # Title-derived tokens are only used when short and stable enough.
            brand = normalize_brand_name(
                row.get("brand", ""),
                title="",
                shop_name=row.get("shop_name", ""),
            )
            if not brand:
                title_token = _extract_brand_token(row.get("title", ""))
                if (
                    title_token
                    and len(title_token) <= 12
                    and not any(ch.isdigit() for ch in title_token)
                ):
                    brand = title_token
            if not brand:
                unknown_brand_count += 1
                continue
            brand_count[brand] = brand_count.get(brand, 0) + 1
        return brand_count, unknown_brand_count

    @staticmethod
    def _build_brand_context_line(
        rows: list[dict[str, str]],
        tier_text: str,
        theme_text: str = "",
    ) -> str:
        tier_display = tier_text
        if tier_text and "价格信息缺失" not in tier_text:
            tier_display = f"{tier_text}（按样本分位）"
        brand_count, unknown_brand_count = Analyzer._count_brands(rows)
        sorted_brands = sorted(
            brand_count.items(), key=lambda item: (-item[1], item[0])
        )
        unique_item_ids = len(
            {
                str(row.get("item_id", "")).strip("' ").strip()
                for row in rows
                if str(row.get("item_id", "")).strip()
            }
        )
        if unique_item_ids <= 0:
            unique_item_ids = len(rows)

        if sorted_brands:
            brand_text = " | ".join(
                [f"{name}({count})" for name, count in sorted_brands]
            )
            context_line = (
                f"样本去重：按item_id去重后{unique_item_ids}款；"
                f"品牌概览：已识别{len(brand_count)}个品牌，品牌分布为{brand_text}；"
                f"价格层级：{tier_display}。"
            )
        else:
            context_line = (
                f"样本去重：按item_id去重后{unique_item_ids}款；"
                f"品牌概览：品牌信息暂未解析；价格层级：{tier_display}。"
            )

        if unknown_brand_count:
            context_line += f" 另有{unknown_brand_count}款品牌字段为空。"
        if theme_text:
            context_line += f" 主导主题：{theme_text}。"
        return context_line

    @classmethod
    def _format_batch_summary_text(
        cls,
        sample_size: int,
        core_points: list[str],
        differentiated_points: list[str],
        homogenized_points: list[str],
        citation_notes: list[str],
        context_line: str = "",
        include_evidence: bool = False,
    ) -> str:
        core = [
            clean_text(v, max_len=120)
            for v in core_points
            if clean_text(v, max_len=120)
        ]
        diff = [
            clean_text(v, max_len=160)
            for v in differentiated_points
            if clean_text(v, max_len=160)
        ]
        homo = [
            clean_text(v, max_len=120)
            for v in homogenized_points
            if clean_text(v, max_len=120)
        ]

        sections: list[str] = []
        scope = f"样本范围：本批共纳入{sample_size}款商品。"
        if context_line:
            scope += " " + clean_text(context_line, max_len=2000)
        sections.append(scope)

        if core:
            sections.append(
                "市场共性：\n"
                + "\n".join(
                    [f"{idx}. {point}" for idx, point in enumerate(core[:5], start=1)]
                )
            )
        if diff:
            sections.append(
                "主要差异：\n"
                + "\n".join(
                    [f"{idx}. {point}" for idx, point in enumerate(diff[:5], start=1)]
                )
            )
        if homo:
            sections.append(
                "竞争焦点：\n"
                + "\n".join(
                    [f"{idx}. {point}" for idx, point in enumerate(homo[:5], start=1)]
                )
            )
        return "\n\n".join(sections)

    def _generate_json_text(
        self,
        prompt: str,
        model: str,
        raise_on_transient: bool,
        debug_meta: dict[str, Any] | None = None,
    ) -> str:
        return gemini_generate_json_text(
            api_key=self.gemini_api_key,
            model=model,
            contents=prompt,
            proxy_url=self.gemini_proxy_url,
            timeout_sec=self.gemini_timeout_sec,
            raise_on_transient=raise_on_transient,
            debug_meta=debug_meta,
        )

    def _generate_json(
        self, prompt: str, debug_meta: dict[str, Any] | None = None
    ) -> Any:
        if not self.gemini_api_key:
            return {}

        max_attempts = 1 + self.gemini_pro_retries
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                text = self._generate_json_text(
                    prompt=prompt,
                    model=self.gemini_pro_model,
                    raise_on_transient=True,
                    debug_meta=debug_meta,
                )
            except Exception as exc:
                last_error = exc
                LOG.warning(
                    "Gemini primary model failed (%s, attempt %s/%s): %s",
                    self.gemini_pro_model,
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts and _is_transient_gemini_exception(exc):
                    continue
                break
            if text:
                return self._extract_json(text)
            LOG.warning(
                "Gemini primary model returned empty payload (%s, attempt %s/%s)",
                self.gemini_pro_model,
                attempt,
                max_attempts,
            )

        flash_model = (self.gemini_flash_model or "").strip()
        if flash_model and flash_model != self.gemini_pro_model:
            try:
                flash_text = self._generate_json_text(
                    prompt=prompt,
                    model=flash_model,
                    raise_on_transient=False,
                    debug_meta={
                        **(debug_meta or {}),
                        "fallback_model": flash_model,
                        "fallback_from": self.gemini_pro_model,
                    },
                )
                if flash_text:
                    LOG.warning(
                        "Gemini fallback activated: primary=%s fallback=%s",
                        self.gemini_pro_model,
                        flash_model,
                    )
                    return self._extract_json(flash_text)
                LOG.warning(
                    "Gemini fallback model returned empty payload (%s)",
                    flash_model,
                )
            except Exception as exc:
                LOG.warning("Gemini fallback model failed (%s): %s", flash_model, exc)

        if last_error and self.gemini_raise_on_transient and _is_transient_gemini_exception(last_error):
            raise TransientGeminiError(str(last_error)) from last_error
        return {}

    @staticmethod
    def _single_sample_points_from_row(row: dict[str, str], limit: int = 4) -> list[str]:
        values = [clean_text(p, max_len=80) for p in str(row.get("selling_points_text", "")).split("|") if clean_text(p, max_len=80)]
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _single_sample_tags(row: dict[str, str], limit: int = 4) -> list[str]:
        tags = [clean_text(p, max_len=24) for p in str(row.get("market_tags", "")).split("|") if clean_text(p, max_len=24)]
        deduped: list[str] = []
        for tag in tags:
            if tag not in deduped:
                deduped.append(tag)
            if len(deduped) >= limit:
                break
        return deduped or Analyzer._single_sample_points_from_row(row, limit=limit)

    @staticmethod
    def _single_sample_item_analysis(item_id: str, item_title: str, item_platform: str, item_points: list[dict[str, Any]]) -> dict[str, Any]:
        point_rows = [clean_text(str(p.get("point", "")), max_len=80) for p in item_points if clean_text(str(p.get("point", "")), max_len=80)]
        citations = [clean_text(str(p.get("citation", "")), max_len=80) for p in item_points if clean_text(str(p.get("citation", "")), max_len=80)]
        point_rows = list(dict.fromkeys(point_rows))[:4]
        item_name = clean_text(item_title, max_len=64) or clean_text(item_id, max_len=48) or "This item"
        if not point_rows:
            return {"competitor_analysis_text": "", "market_tags": []}
        lines = [f"\u5546\u54c1\uff1a{item_name}", "\u8bf4\u660e\uff1a\u5f53\u524d\u4ec5\u6709\u5355\u5546\u54c1\u6837\u672c\uff0c\u4ee5\u4e0b\u4e3a\u5546\u54c1\u81ea\u8eab\u5356\u70b9\u89c2\u5bdf\uff0c\u4e0d\u6784\u6210\u7ade\u54c1\u5bf9\u6bd4\u7ed3\u8bba\u3002", "\u6838\u5fc3\u5356\u70b9\uff1a" + ", ".join(point_rows[:3])]
        if len(point_rows) >= 4:
            lines.append("\u8865\u5145\u89c2\u5bdf\uff1a" + point_rows[3])
        if citations:
            lines.append(SellingPointExtractor._detail_citation_label(item_platform) + ", ".join(list(dict.fromkeys(citations))[:6]))
        return {"competitor_analysis_text": "\n".join(lines), "market_tags": point_rows[:4]}

    @staticmethod
    def _single_sample_batch_summary(row: dict[str, str]) -> dict[str, Any]:
        points = Analyzer._single_sample_points_from_row(row, limit=4)
        if not points:
            return {"batch_competitor_summary_text": "", "batch_tags": []}
        item_name = clean_text(row.get("title", ""), max_len=64) or clean_text(row.get("item_id", ""), max_len=48) or "该商品"
        text = "\n\n".join([
            "\u6837\u672c\u8303\u56f4\uff1a\u5f53\u524d\u4ec5\u7eb3\u5165 1 \u6b3e\u5546\u54c1\uff0c\u4ee5\u4e0b\u5185\u5bb9\u4ec5\u63cf\u8ff0\u8be5\u6837\u672c\u81ea\u8eab\u8868\u8fbe\uff0c\u4e0d\u6784\u6210\u6279\u91cf\u7ade\u54c1\u7efc\u8ff0\u3002",
            f"\u5546\u54c1\uff1a{item_name}",
            "\u5355\u6837\u672c\u6279\u91cf\u6982\u89c8\uff1a\n" + "\n".join([f"{idx}. {point}" for idx, point in enumerate(points, start=1)]),
        ])
        return {"batch_competitor_summary_text": text, "batch_tags": Analyzer._single_sample_tags(row)}

    @staticmethod
    def _single_sample_market_summary(row: dict[str, str]) -> dict[str, Any]:
        points = Analyzer._single_sample_points_from_row(row, limit=4)
        item_name = clean_text(row.get("title", ""), max_len=64) or clean_text(row.get("item_id", ""), max_len=48) or "该商品"
        price_text = clean_text(row.get("price_min", "") or row.get("price_max", ""), max_len=32)
        common = f"\u5f53\u524d\u4ec5\u57fa\u4e8e 1 \u6b3e\u6837\u672c\uff0c\u65e0\u6cd5\u53ef\u9760\u5224\u65ad\u5e02\u573a\u5171\u6027\u3002\u53ef\u786e\u8ba4 {item_name} \u7684\u4e3b\u8981\u5356\u70b9\u96c6\u4e2d\u5728\uff1a{'\u3001'.join(points[:3]) or '\u8bc1\u636e\u4e0d\u8db3'}\u3002"
        opportunities = f"\u5f53\u524d\u6837\u672c\u4e2d\u53ef\u76f4\u63a5\u89c2\u5bdf\u5230\u7684\u7279\u8272\u8868\u8fbe\u4e3a\uff1a{'\u3001'.join(points[:2]) or '\u6682\u65e0'}"
        risks = "\u5f53\u524d\u4ec5\u6709 1 \u6b3e\u6837\u672c\uff0c\u65e0\u6cd5\u53ef\u9760\u5224\u65ad\u7ade\u4e89\u5f3a\u5ea6\u3001\u540c\u8d28\u5316\u7a0b\u5ea6\u6216\u5e02\u573a\u673a\u4f1a\u5206\u5e03\u3002"
        if price_text:
            common += f"\u89c2\u5bdf\u5230\u7684\u4ef7\u683c\u4e3a\uff1a{price_text}\u3002"
        return {"summary_text": f"\u5e02\u573a\u5171\u6027\uff1a{common}\n\u5dee\u5f02\u5316\u5206\u5e03\uff1a{opportunities}\n\u7ade\u4e89\u72b6\u6001\uff1a{risks}", "common_points": [common], "opportunities": [opportunities], "risks": [risks], "market_tags": Analyzer._single_sample_tags(row), "updated_at": now_iso(), "sample_size": 1, "point_count": len(points)}

    @staticmethod
    def _single_sample_final_conclusion(row: dict[str, str]) -> str:
        points = Analyzer._single_sample_points_from_row(row, limit=4)
        item_name = clean_text(row.get("title", ""), max_len=64) or clean_text(row.get("item_id", ""), max_len=48) or "该商品"
        if not points:
            return "\u5f53\u524d\u4ec5\u6709 1 \u6b3e\u6837\u672c\uff0c\u4e14\u8bc1\u636e\u4e0d\u8db3\uff0c\u4e0d\u5e94\u5916\u63a8\u4e3a\u5e02\u573a\u7ed3\u8bba\u3002"
        lead = f"\u5f53\u524d\u4ec5\u6709 1 \u6b3e\u6837\u672c\uff0c\u4e0d\u5e94\u5916\u63a8\u4e3a\u5e02\u573a\u7ed3\u8bba\u3002\u53ef\u786e\u8ba4 {item_name} \u7684\u4e3b\u8981\u5356\u70b9\u4e3a\uff1a{'\u3001'.join(points[:3])}\u3002"
        if len(points) > 3:
            lead += f"\u8865\u5145\u8868\u8fbe\u5305\u62ec\uff1a{points[3]}\u3002"
        return lead

    def analyze_item(self, item_id: str, item_points: list[dict[str, Any]], workbook_points_map: dict[str, list[dict[str, Any]]], item_title: str = "", item_platform: str = "", workbook_id: str = "", task_id: str = "") -> dict[str, Any]:
        if not self.gemini_api_key:
            return {"competitor_analysis_text": "", "market_tags": []}
        current_points = [clean_text(str(p.get("point", "")), max_len=80) for p in item_points if str(p.get("point", "")).strip()]
        citations = [clean_text(str(p.get("citation", "")), max_len=80) for p in item_points if str(p.get("citation", "")).strip()]
        peer_points: list[str] = []
        for peer_item_id, points in workbook_points_map.items():
            if peer_item_id == item_id:
                continue
            for point_obj in points:
                value = clean_text(str(point_obj.get("point", "")), max_len=80)
                if value:
                    peer_points.append(value)
        if not peer_points:
            return self._single_sample_item_analysis(item_id, item_title, item_platform, item_points)
        prompt = textwrap.dedent(f"""
            \u4f60\u662f\u5e02\u573a\u54a8\u8be2\u516c\u53f8\u8d44\u6df1\u7ade\u54c1\u987e\u95ee\u3002\u8bf7\u9488\u5bf9\u5355\u4e00\u5546\u54c1\u8f93\u51fa\u6709\u8bc1\u636e\u7ea6\u675f\u7684\u7ade\u54c1\u89c2\u5bdf\u3002
            \u5546\u54c1ID: {item_id}
            \u5f53\u524d\u5546\u54c1\u5356\u70b9\uff1a{json.dumps(current_points[:12], ensure_ascii=False)}
            \u540c\u7c7b\u5176\u4ed6\u5546\u54c1\u5356\u70b9\uff1a{json.dumps(peer_points[:120], ensure_ascii=False)}
            \u5f53\u524d\u5f15\u7528\uff1a{json.dumps(citations[:12], ensure_ascii=False)}
            \u4ec5\u8f93\u51fa JSON\uff0c\u5305\u542b core_points\u3001differentiated_points\u3001homogenized_points\u3001citation_notes\u3001market_tags\u3002
        """).strip()
        payload = self._generate_json(prompt, debug_meta={"stage": "llm_analyze_item", "item_id": item_id, "workbook_id": workbook_id, "task_id": task_id, "platform": item_platform})
        if not isinstance(payload, dict):
            return {"competitor_analysis_text": "", "market_tags": []}
        core = self._to_string_list(payload.get("core_points"))[:3]
        diff = self._to_string_list(payload.get("differentiated_points"))[:3]
        homo = self._to_string_list(payload.get("homogenized_points"))[:3]
        notes = self._to_string_list(payload.get("citation_notes"), max_len=80)[:6]
        tags = self._to_string_list(payload.get("market_tags"), max_len=40)[:6]
        if current_points:
            def _aligned(point_text: str) -> bool:
                value = clean_text(point_text, max_len=80)
                return any(value in current or current in value for current in current_points if current)
            core = [point for point in core if _aligned(point)]
            diff = [point for point in diff if _aligned(point)]
            homo = [point for point in homo if _aligned(point)]
            if not core:
                core = current_points[:3]
            if not diff:
                diff = [point for point in current_points if point not in peer_points][:3]
            if not homo:
                homo = [point for point in current_points if point in peer_points][:3]
        if not core and not diff and not homo:
            return {"competitor_analysis_text": "", "market_tags": []}
        item_name = clean_text(item_title, max_len=64) or clean_text(item_id, max_len=48) or "This item"
        conclusion = f"{item_name} \u5728\u6837\u672c\u4e2d\u5177\u5907\u4e00\u5b9a\u533a\u5206\u5ea6\uff0c\u91cd\u70b9\u4f53\u73b0\u5728\uff1a{', '.join(diff)}." if diff else f"{item_name} \u4ee5\u4e3b\u6d41\u5356\u70b9\u4e3a\u4e3b\uff0c\u5dee\u5f02\u5316\u7a7a\u95f4\u6709\u9650\u3002"
        lines = [f"\u5546\u54c1\uff1a{item_name}"]
        if core:
            lines.append("\u6838\u5fc3\u5356\u70b9\uff1a" + ", ".join(core))
        if diff:
            lines.append("\u5dee\u5f02\u5316\u5356\u70b9\uff1a" + ", ".join(diff))
        if homo:
            lines.append("\u540c\u8d28\u5316\u5356\u70b9\uff1a" + ", ".join(homo))
        lines.append("\u7ed3\u8bba\uff1a" + conclusion)
        refs = notes or citations[:6]
        if refs:
            lines.append(SellingPointExtractor._detail_citation_label(item_platform) + ", ".join(refs))
        text = "\n".join(lines)
        if not tags:
            tags = (core[:2] + diff[:2])[:6]
        return {"competitor_analysis_text": text, "market_tags": tags}

    def generate_batch_competitor_summary(self, rows: list[dict[str, str]], workbook_points_map: dict[str, list[dict[str, Any]]], *, workbook_id: str = "", task_id: str = "") -> dict[str, Any]:
        if not self.gemini_api_key:
            return {"batch_competitor_summary_text": "", "batch_tags": []}
        if len(rows) <= 1:
            return self._single_sample_batch_summary(rows[0]) if rows else {"batch_competitor_summary_text": "", "batch_tags": []}
        payload_rows = []
        for row in rows[:120]:
            item_id = row.get("item_id", "")
            points = workbook_points_map.get(item_id, [])
            payload_rows.append({"item_id": item_id, "title": clean_text(row.get("title", ""), max_len=120), "brand": clean_text(row.get("brand", ""), max_len=80), "selling_points": [{"point": clean_text(str(point.get("point", "")), max_len=80), "citation": clean_text(str(point.get("citation", "")), max_len=80)} for point in points[:8] if str(point.get("point", "")).strip()]})
        prompt = textwrap.dedent(f"""
            \u4f60\u662f\u5e02\u573a\u54a8\u8be2\u516c\u53f8\u9879\u76ee\u7ecf\u7406\uff0c\u8bf7\u57fa\u4e8e\u6837\u672c\u8f93\u51fa\u6279\u91cf\u7ade\u54c1\u7efc\u8ff0\u3002
            \u6837\u672c\uff1a{json.dumps(payload_rows, ensure_ascii=False)}
            Output JSON only with core_points, differentiated_points, homogenized_points, citation_notes, batch_tags.
        """).strip()
        payload = self._generate_json(prompt, debug_meta={"stage": "llm_analyze_batch_summary", "workbook_id": workbook_id, "task_id": task_id})
        if not isinstance(payload, dict):
            return {"batch_competitor_summary_text": "", "batch_tags": []}
        core = self._to_string_list(payload.get("core_points"), max_len=120)[:5]
        diff = self._to_string_list(payload.get("differentiated_points"), max_len=180)[:5]
        homo = self._to_string_list(payload.get("homogenized_points"), max_len=120)[:5]
        notes = self._to_string_list(payload.get("citation_notes"), max_len=180)[:8]
        tags = self._to_string_list(payload.get("batch_tags"), max_len=30)[:10]
        if not core and not diff and not homo:
            return {"batch_competitor_summary_text": "", "batch_tags": []}
        if not notes:
            for row in rows:
                for point in workbook_points_map.get(row.get("item_id", ""), []):
                    citation = clean_text(str(point.get("citation", "")), max_len=80)
                    if citation:
                        notes.append(citation)
                    if len(notes) >= 8:
                        break
                if len(notes) >= 8:
                    break
        prices = []
        tiers = {"??": 0, "??": 0, "??": 0, "??": 0}
        for row in rows:
            price = None
            for key in ("price_min", "price_max"):
                raw = (row.get(key, "") or "").strip()
                if not raw:
                    continue
                try:
                    price = float(raw)
                    break
                except ValueError:
                    continue
            if price is not None:
                prices.append(price)
        q1, q2 = Analyzer._price_quantiles(prices)
        for row in rows:
            price = None
            for key in ("price_min", "price_max"):
                raw = (row.get(key, "") or "").strip()
                if not raw:
                    continue
                try:
                    price = float(raw)
                    break
                except ValueError:
                    continue
            tier = Analyzer._price_tier(price, q1, q2)
            tiers[tier] = tiers.get(tier, 0) + 1
        tier_text = f"Entry {tiers.get('??',0)} | Mid {tiers.get('??',0)} | High {tiers.get('??',0)}" if prices else f"No price info (unknown {tiers.get('??',0)})"
        context_line = Analyzer._build_brand_context_line(rows=rows, tier_text=tier_text, theme_text="")
        summary_text = Analyzer._format_batch_summary_text(sample_size=len(rows), core_points=core, differentiated_points=diff, homogenized_points=homo, citation_notes=list(dict.fromkeys(notes))[:8], context_line=context_line)
        if not tags:
            tags = (core[:3] + homo[:2])[:10]
        return {"batch_competitor_summary_text": summary_text, "batch_tags": tags}

    @staticmethod
    def build_row_market_mapping(row: dict[str, str], point_freq: dict[str, int], q1: float, q2: float) -> dict[str, str]:
        points = [clean_text(p, max_len=80) for p in str(row.get("selling_points_text", "")).split("|") if clean_text(p, max_len=80)]
        if not points:
            return {"batch": "", "market": "", "final": ""}
        item_name = clean_text(row.get("title", ""), max_len=48) or clean_text(row.get("item_id", ""), max_len=48) or "This item"
        core = points[:3]
        unique = [p for p in points if point_freq.get(p, 0) == 1][:2]
        homo = [p for p in points if point_freq.get(p, 0) >= 2][:2]
        price = None
        for field in ("price_min", "price_max"):
            raw = (row.get(field, "") or "").strip()
            if not raw:
                continue
            try:
                price = float(raw)
                break
            except ValueError:
                continue
        tier = Analyzer._price_tier(price, q1, q2)
        theme = Analyzer._theme_of_point(points[0]) if points else "other"
        core_text = ", ".join(core)
        diff_text = ", ".join(unique)
        same_text = ", ".join(homo)
        overlap_ratio = len(homo) / len(points) if points else 0.0
        overlap_level = "high" if overlap_ratio >= 0.66 else "medium" if overlap_ratio >= 0.34 else "low"
        price_text = f"{price:.2f}" if price is not None else "n/a"
        batch_parts = [f"{item_name} core points: {core_text}"]
        if diff_text:
            batch_parts.append(f"differentiated: {diff_text}")
        if same_text:
            batch_parts.append(f"shared: {same_text}")
        batch_text = "; ".join(batch_parts)
        market_parts = [f"{item_name} price tier: {tier} ({price_text})", f"theme: {theme}", f"overlap: {overlap_level}"]
        if diff_text:
            market_parts.append(f"distinctive points: {diff_text}")
        market_text = "; ".join(market_parts)
        final_text = f"{item_name} main points: {core_text}; differentiated points: {diff_text or 'not obvious'}."
        return {"batch": batch_text, "market": market_text, "final": final_text}

    def generate_final_conclusion(self, rows: list[dict[str, str]], batch_competitor_summary_text: str, market_summary_text: str, *, workbook_id: str = "", task_id: str = "") -> str:
        if not self.gemini_api_key:
            return ""
        if len(rows) <= 1:
            return self._single_sample_final_conclusion(rows[0]) if rows else ""
        sample_rows = []
        for row in rows[:80]:
            sample_rows.append({"item_id": row.get("item_id", ""), "title": clean_text(row.get("title", ""), max_len=120), "brand": clean_text(row.get("brand", ""), max_len=80), "price_min": row.get("price_min", ""), "price_max": row.get("price_max", ""), "selling_points_text": clean_text(row.get("selling_points_text", ""), max_len=300)})
        prompt = textwrap.dedent(f"""
            \u4f60\u662f\u5e02\u573a\u54a8\u8be2\u516c\u53f8\u603b\u76d1\uff0c\u8bf7\u57fa\u4e8e\u8f93\u5165\u5f62\u6210\u6700\u7ec8\u5e02\u573a\u7ed3\u8bba\u3002
            \u6279\u91cf\u7ade\u54c1\u603b\u7ed3\uff1a{batch_competitor_summary_text}
            \u5e02\u573a\u603b\u7ed3\uff1a{market_summary_text}
            \u6837\u672c\uff1a{json.dumps(sample_rows, ensure_ascii=False)}
            Output JSON only: {{"final_conclusion": "3-6 sentences"}}
        """).strip()
        payload = self._generate_json(prompt, debug_meta={"stage": "llm_analyze_final_conclusion", "workbook_id": workbook_id, "task_id": task_id})
        if not isinstance(payload, dict):
            return ""
        final_conclusion = clean_text(str(payload.get("final_conclusion", "")), max_len=1500)
        if not final_conclusion:
            return ""
        point_freq = {}
        prices = []
        for row in rows:
            for point in str(row.get("selling_points_text", "")).split("|"):
                text = clean_text(point, max_len=80)
                if text:
                    point_freq[text] = point_freq.get(text, 0) + 1
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
        sections = []
        for row in rows:
            mapping = Analyzer.build_row_market_mapping(row, point_freq, q1, q2)
            if mapping["final"]:
                sections.append(mapping["final"])
        if not sections:
            return final_conclusion
        return "\n".join([final_conclusion, "", "\u6309\u4ea7\u54c1\u7ed3\u8bba\uff1a", "\n\n".join(sections)])

    def generate_market_summary(self, rows: list[dict[str, str]], workbook_points_map: dict[str, list[dict[str, Any]]], *, workbook_id: str = "", task_id: str = "") -> dict[str, Any]:
        if not self.gemini_api_key:
            return {"summary_text": "", "common_points": [], "opportunities": [], "risks": [], "market_tags": [], "updated_at": now_iso(), "sample_size": len(rows), "point_count": 0}
        if len(rows) <= 1:
            return self._single_sample_market_summary(rows[0]) if rows else {"summary_text": "", "common_points": [], "opportunities": [], "risks": [], "market_tags": [], "updated_at": now_iso(), "sample_size": 0, "point_count": 0}
        payload_rows = []
        for row in rows[:120]:
            item_id = row.get("item_id", "")
            points = workbook_points_map.get(item_id, [])
            payload_rows.append({"item_id": item_id, "title": row.get("title", ""), "brand": row.get("brand", ""), "shop_name": row.get("shop_name", ""), "selling_points": [{"point": clean_text(str(point.get("point", "")), max_len=80), "citation": clean_text(str(point.get("citation", "")), max_len=80)} for point in points[:8] if str(point.get("point", "")).strip()]})
        prompt = textwrap.dedent(f"""
            \u4f60\u662f\u5e02\u573a\u54a8\u8be2\u516c\u53f8\u9ad8\u7ea7\u7814\u7a76\u5458\u3002\u8bf7\u57fa\u4e8e\u6837\u672c\u505a\u5e02\u573a\u8c03\u7814\u603b\u7ed3\u3002
            \u6837\u672c\uff1a{json.dumps(payload_rows, ensure_ascii=False)}
            \u4ec5\u8f93\u51fa JSON\uff0c\u5305\u542b common_insights\u3001opportunities\u3001risks\u3001market_tags\u3002
        """).strip()
        payload = self._generate_json(prompt, debug_meta={"stage": "llm_analyze_market_summary", "workbook_id": workbook_id, "task_id": task_id})
        if not isinstance(payload, dict):
            return {"summary_text": "", "common_points": [], "opportunities": [], "risks": [], "market_tags": [], "updated_at": now_iso(), "sample_size": len(rows), "point_count": 0}
        common = clean_text(str(payload.get("common_insights", "")), max_len=1200)
        opportunities = clean_text(str(payload.get("opportunities", "")), max_len=1200)
        risks = clean_text(str(payload.get("risks", "")), max_len=1200)
        tags = self._to_string_list(payload.get("market_tags"), max_len=30)[:10]
        if not common or not opportunities or not risks:
            return {"summary_text": "", "common_points": [], "opportunities": [], "risks": [], "market_tags": [], "updated_at": now_iso(), "sample_size": len(rows), "point_count": 0}
        summary_text = f"\u5e02\u573a\u5171\u6027\uff1a{common}\n\u5dee\u5f02\u5316\u5206\u5e03\uff1a{opportunities}\n\u7ade\u4e89\u72b6\u6001\uff1a{risks}"
        return {"summary_text": summary_text, "common_points": [common], "opportunities": [opportunities], "risks": [risks], "market_tags": tags, "updated_at": now_iso(), "sample_size": len(rows), "point_count": sum(len(workbook_points_map.get(r.get('item_id', ''), [])) for r in rows)}
