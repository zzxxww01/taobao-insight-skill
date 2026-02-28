"""AI-driven analysis (Gemini), selling point extraction, and competitor comparisons."""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import textwrap
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
) -> str:
    key = (api_key or "").strip()
    if not key:
        LOG.error(f"Gemini API key is missing. Skipping API call to {model}.")
        return ""

    with gemini_proxy_env(proxy_url):
        try:
            from google import genai as google_genai
            from google.genai import types as genai_types
        except Exception:
            google_genai = None
            genai_types = None

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
                return text or ""
            except Exception as e:
                if raise_on_transient and _is_transient_gemini_exception(e):
                    raise TransientGeminiError(str(e)) from e
                LOG.error(
                    "Failed to generate content with google.genai: %s", e, exc_info=True
                )
                return ""

        try:
            import google.generativeai as legacy_genai

            legacy_genai.configure(api_key=key)
            legacy_model = legacy_genai.GenerativeModel(model)
            response = legacy_model.generate_content(
                contents,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": max(5, int(timeout_sec))},
            )
            return (response.text or "").strip()
        except Exception as e:
            if raise_on_transient and _is_transient_gemini_exception(e):
                raise TransientGeminiError(str(e)) from e
            LOG.error(
                "Failed to generate content with legacy google.generativeai: %s",
                e,
                exc_info=True,
            )
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
                rows.append("图文详情参考图：" + "、".join(unique_refs))
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
        lines: list[str] = []
        for block in detail_blocks[:80]:
            lines.append(
                f"- {block.get('source_ref', '')} | {block.get('source_type', '')} | {clean_text(block.get('content', ''), max_len=220)}"
            )
        payload = "\n".join(lines)
        return textwrap.dedent(
            f"""
            你是资深美妆电商策略分析师，请从图文详情中抽取“可被消费者感知的产品卖点”，并严格引用证据。

            商品上下文:
            - title: {item_context.get("title", "")}
            - brand: {item_context.get("brand", "")}
            - shop_name: {item_context.get("shop_name", "")}

            详情证据块（source_ref | source_type | content）:
            {payload}

            抽取原则:
            1) 只依据给定证据，不得补充外部常识，不得推断未出现的信息。
            2) 卖点必须是“消费者价值表达”，优先功效/妆效/肤感/场景/人群适配。
            3) 禁止空泛措辞（如“质量好”“口碑好”）与纯营销话术。
            4) 每条必须绑定 citation，且 citation 必须来自上述 source_ref。
            5) 去重并控制在 4-8 条，中文精炼表达，每条 <= 28 字。

            仅输出 JSON 数组，不要输出任何其他文字。元素结构:
            [
              {{
                "point": "卖点文本",
                "citation": "text_1",
                "consumer_value": "解决了什么问题",
                "scenario": "适用场景或人群"
              }}
            ]
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
        try:
            request = urllib.request.Request(
                source,
                headers={
                    "User-Agent": UA,
                    "Referer": "https://detail.tmall.com/",
                },
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
        for block in image_blocks[:20]:
            ref = clean_text(str(block.get("source_ref", "")), max_len=24)
            if not ref or ref in seen_refs:
                continue
            seen_refs.add(ref)
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
        return [row for _, row in ranked[:10]]

    def _generate_image_points_with_gemini(
        self,
        item_context: dict[str, str],
        candidates: list[tuple[str, str, bytes]],
        max_points: int = 8,
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

        prompt = textwrap.dedent(
            f"""
            你是电商图文详情分析助手。请基于输入图片提炼商品卖点。
            商品标题: {item_context.get("title", "")}
            品牌: {item_context.get("brand", "")}
            店铺: {item_context.get("shop_name", "")}

            规则:
            1) 只能依据图片信息，不得编造。
            2) 每条卖点必须包含 point 与 citation。
            3) citation 只能是以下编号之一: {", ".join(sorted(refs, key=self._ref_sort_key))}。
            4) 输出 2-{max(2, max_points)} 条，中文简洁表达。

            仅输出 JSON 数组:
            [{{"point":"控油持妆","citation":"image_1"}}]
            """
        ).strip()
        raw_text = gemini_generate_json_text(
            api_key=self.gemini_api_key,
            model=self.gemini_flash_model,
            contents=[prompt] + parts,
            proxy_url=self.gemini_proxy_url,
            timeout_sec=self.gemini_timeout_sec,
            raise_on_transient=self.gemini_raise_on_transient,
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

        primary = self._generate_image_points_with_gemini(
            item_context=item_context, candidates=candidates[:6], max_points=8
        )
        if primary:
            return primary

        without_first = [row for row in candidates if row[0] != "image_1"] or candidates
        retry = self._generate_image_points_with_gemini(
            item_context=item_context, candidates=without_first[:4], max_points=6
        )
        if retry:
            return retry

        merged: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in without_first[:4]:
            single = self._generate_image_points_with_gemini(
                item_context=item_context, candidates=[row], max_points=2
            )
            for point in single:
                key = f"{point.get('point', '')}|{point.get('citation', '')}"
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(point)
                if len(merged) >= 6:
                    break
            if len(merged) >= 6:
                break
        if merged:
            valid_refs = {
                b.get("source_ref", "") for b in detail_blocks if b.get("source_ref")
            }
            return self._validate(merged, valid_refs)
        return []

    @staticmethod
    def _validate(
        points: list[dict[str, Any]], valid_refs: set[str]
    ) -> list[dict[str, str]]:
        clean_rows: list[dict[str, str]] = []
        seen: set[str] = set()
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
            key = f"{point}|{citation}"
            if key in seen:
                continue
            seen.add(key)
            clean_rows.append({"point": point, "citation": citation})
            if len(clean_rows) >= 8:
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

    def _generate_json_text(self, prompt: str, model: str, raise_on_transient: bool) -> str:
        return gemini_generate_json_text(
            api_key=self.gemini_api_key,
            model=model,
            contents=prompt,
            proxy_url=self.gemini_proxy_url,
            timeout_sec=self.gemini_timeout_sec,
            raise_on_transient=raise_on_transient,
        )

    def _generate_json(self, prompt: str) -> Any:
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

    def analyze_item(
        self,
        item_id: str,
        item_points: list[dict[str, Any]],
        workbook_points_map: dict[str, list[dict[str, Any]]],
        item_title: str = "",
    ) -> dict[str, Any]:
        if not self.gemini_api_key:
            return {"competitor_analysis_text": "", "market_tags": []}

        current_points = [
            clean_text(str(p.get("point", "")), max_len=80)
            for p in item_points
            if str(p.get("point", "")).strip()
        ]
        citations = [
            clean_text(str(p.get("citation", "")), max_len=80)
            for p in item_points
            if str(p.get("citation", "")).strip()
        ]
        peer_points: list[str] = []
        for peer_item_id, points in workbook_points_map.items():
            if peer_item_id == item_id:
                continue
            for point_obj in points:
                value = clean_text(str(point_obj.get("point", "")), max_len=80)
                if value:
                    peer_points.append(value)

        prompt = textwrap.dedent(
            f"""
            你是市场咨询公司资深竞品顾问。请针对单一商品输出“对业务有指导意义”的竞品洞察。
            输入数据全部来自卖点与引用，不得编造事实。

            商品ID: {item_id}
            当前商品卖点: {json.dumps(current_points[:12], ensure_ascii=False)}
            同类其他商品卖点: {json.dumps(peer_points[:120], ensure_ascii=False)}
            当前商品引用: {json.dumps(citations[:12], ensure_ascii=False)}

            分析要求:
            1) core_points: 当前商品最核心、最有价值感知的卖点（最多3条）。
            2) differentiated_points: 相比同类更有差异化的表达（最多3条）。
            3) homogenized_points: 与同类高度重叠、易被替代的表达（最多3条）。
            4) citation_notes: 对应引用编号与简要说明，格式建议“text_3: xxx”（最多6条）。
            5) market_tags: 可用于看板过滤的关键词标签（2-6条，短词）。

            仅输出 JSON:
            {{
              "core_points": ["..."],
              "differentiated_points": ["..."],
              "homogenized_points": ["..."],
              "citation_notes": ["..."],
              "market_tags": ["..."]
            }}
            """
        ).strip()
        payload = self._generate_json(prompt)
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
                return any(
                    value in current or current in value
                    for current in current_points
                    if current
                )

            core = [point for point in core if _aligned(point)]
            diff = [point for point in diff if _aligned(point)]
            homo = [point for point in homo if _aligned(point)]
            if not core:
                core = current_points[:3]
            if not diff:
                diff = [point for point in current_points if point not in peer_points][
                    :3
                ]
            if not homo:
                homo = [point for point in current_points if point in peer_points][:3]
        if not core and not diff and not homo:
            return {"competitor_analysis_text": "", "market_tags": []}
        item_name = (
            clean_text(item_title, max_len=64)
            or clean_text(item_id, max_len=48)
            or "该商品"
        )
        conclusion = (
            f"{item_name}在样本中有一定区分度，重点体现在{'、'.join(diff)}。"
            if diff
            else f"{item_name}以主流卖点为主，差异化空间有限。"
        )
        lines: list[str] = [f"商品：{item_name}"]
        if core:
            lines.append("核心卖点：" + "、".join(core))
        if diff:
            lines.append("差异化卖点：" + "、".join(diff))
        if homo:
            lines.append("同质化卖点：" + "、".join(homo))
        lines.append(f"结论：{conclusion}")
        refs = notes or citations[:6]
        if refs:
            lines.append("图文详情页引用：" + "、".join(refs))
        text = "\n".join(lines)
        if not tags:
            tags = (core[:2] + diff[:2])[:6]
        return {"competitor_analysis_text": text, "market_tags": tags}

    def generate_batch_competitor_summary(
        self,
        rows: list[dict[str, str]],
        workbook_points_map: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if not self.gemini_api_key:
            return {"batch_competitor_summary_text": "", "batch_tags": []}

        payload_rows: list[dict[str, Any]] = []
        for row in rows[:120]:
            item_id = row.get("item_id", "")
            points = workbook_points_map.get(item_id, [])
            payload_rows.append(
                {
                    "item_id": item_id,
                    "title": clean_text(row.get("title", ""), max_len=120),
                    "brand": clean_text(row.get("brand", ""), max_len=80),
                    "selling_points": [
                        {
                            "point": clean_text(
                                str(point.get("point", "")), max_len=80
                            ),
                            "citation": clean_text(
                                str(point.get("citation", "")), max_len=80
                            ),
                        }
                        for point in points[:8]
                        if str(point.get("point", "")).strip()
                    ],
                }
            )

        prompt = textwrap.dedent(
            f"""
            你是市场咨询公司项目经理，请基于样本输出“批量竞品综述”。
            所有观点必须可追溯到样本卖点和引用，不得虚构。
            样本:
            {json.dumps(payload_rows, ensure_ascii=False)}

            分析要求:
            1) core_points: 赛道高频核心诉求（最多5条）。
            2) differentiated_points: 能形成区隔的表达（最多5条，可带商品标题）。
            3) homogenized_points: 同质化最严重的表达（最多5条）。
            4) citation_notes: 关键证据说明，优先引用编号（最多8条）。
            5) batch_tags: 适合聚合分析的标签词（最多10条）。

            仅输出 JSON:
            {{
              "core_points": ["..."],
              "differentiated_points": ["..."],
              "homogenized_points": ["..."],
              "citation_notes": ["..."],
              "batch_tags": ["..."]
            }}
            """
        ).strip()
        payload = self._generate_json(prompt)
        if not isinstance(payload, dict):
            return {"batch_competitor_summary_text": "", "batch_tags": []}

        core = self._to_string_list(payload.get("core_points"), max_len=120)[:5]
        diff = self._to_string_list(payload.get("differentiated_points"), max_len=180)[
            :5
        ]
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

        prices: list[float] = []
        tiers = {"入门": 0, "中端": 0, "高端": 0, "未知": 0}
        for row in rows:
            price: float | None = None
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
            price: float | None = None
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
        tier_text = (
            f"入门{tiers.get('入门',0)} | 中端{tiers.get('中端',0)} | 高端{tiers.get('高端',0)}"
            if prices
            else f"价格信息缺失(未知{tiers.get('未知', 0)})"
        )
        context_line = Analyzer._build_brand_context_line(
            rows=rows,
            tier_text=tier_text,
            theme_text="",
        )

        summary_text = Analyzer._format_batch_summary_text(
            sample_size=len(rows),
            core_points=core,
            differentiated_points=diff,
            homogenized_points=homo,
            citation_notes=list(dict.fromkeys(notes))[:8],
            context_line=context_line,
        )
        if not tags:
            tags = (core[:3] + homo[:2])[:10]
        return {
            "batch_competitor_summary_text": summary_text,
            "batch_tags": tags,
        }

    @staticmethod
    def build_row_market_mapping(
        row: dict[str, str],
        point_freq: dict[str, int],
        q1: float,
        q2: float,
    ) -> dict[str, str]:
        points = [
            clean_text(p, max_len=80)
            for p in str(row.get("selling_points_text", "")).split("|")
            if clean_text(p, max_len=80)
        ]
        if not points:
            return {"batch": "", "market": "", "final": ""}
        item_name = (
            clean_text(row.get("title", ""), max_len=48)
            or clean_text(row.get("item_id", ""), max_len=48)
            or "该商品"
        )
        core = points[:3]
        unique = [p for p in points if point_freq.get(p, 0) == 1][:2]
        homo = [p for p in points if point_freq.get(p, 0) >= 2][:2]

        price: float | None = None
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
        theme = Analyzer._theme_of_point(points[0]) if points else "其他表达"
        core_text = "、".join(core)
        diff_text = "、".join(unique)
        same_text = "、".join(homo)

        overlap_ratio = len(homo) / len(points) if points else 0.0
        if overlap_ratio >= 0.66:
            overlap_level = "较高"
        elif overlap_ratio >= 0.34:
            overlap_level = "中等"
        else:
            overlap_level = "较低"

        price_text = f"{price:.2f}元" if price is not None else "价格信息缺失"
        batch_parts: list[str] = [f"{item_name}核心卖点为{core_text}"]
        if diff_text:
            batch_parts.append(f"可区分卖点是{diff_text}")
        if same_text:
            batch_parts.append(f"与同类重合点有{same_text}")
        batch_text = "；".join(batch_parts) + "。"

        market_parts: list[str] = [
            f"{item_name}位于{tier}价格档（{price_text}）",
            f"主打{theme}方向",
            f"同质化程度{overlap_level}",
        ]
        if diff_text:
            market_parts.append(f"可识别区分点为{diff_text}")
        market_text = "；".join(market_parts) + "。"

        final_text = (
            f"{item_name}主要卖点：{core_text}；"
            f"差异化卖点：{diff_text or '不明显'}。"
        )
        return {"batch": batch_text, "market": market_text, "final": final_text}

    def generate_final_conclusion(
        self,
        rows: list[dict[str, str]],
        batch_competitor_summary_text: str,
        market_summary_text: str,
    ) -> str:
        if not self.gemini_api_key:
            return ""

        sample_rows: list[dict[str, Any]] = []
        for row in rows[:80]:
            sample_rows.append(
                {
                    "item_id": row.get("item_id", ""),
                    "title": clean_text(row.get("title", ""), max_len=120),
                    "brand": clean_text(row.get("brand", ""), max_len=80),
                    "price_min": row.get("price_min", ""),
                    "price_max": row.get("price_max", ""),
                    "selling_points_text": clean_text(
                        row.get("selling_points_text", ""), max_len=300
                    ),
                }
            )

        prompt = textwrap.dedent(
            f"""
            你是市场咨询公司总监，请基于输入形成“最终市场结论”。
            输出应清晰可读，禁止编造，禁止输出建议动作。
            批量竞品总结:
            {batch_competitor_summary_text}
            市场总结:
            {market_summary_text}
            样本:
            {json.dumps(sample_rows, ensure_ascii=False)}

            输出 JSON:
            {{
              "final_conclusion": "3-6句，包含赛道判断、竞争格局、差异化空间"
            }}
            """
        ).strip()
        payload = self._generate_json(prompt)
        if not isinstance(payload, dict):
            return ""

        final_conclusion = clean_text(
            str(payload.get("final_conclusion", "")), max_len=1500
        )
        if not final_conclusion:
            return ""
        final_conclusion = final_conclusion.replace("。 按产品结论", "。\n\n按产品结论")
        final_conclusion = final_conclusion.replace("。 商品：", "。\n\n商品：")

        point_freq: dict[str, int] = {}
        prices: list[float] = []
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
        sections: list[str] = []
        for row in rows:
            mapping = Analyzer.build_row_market_mapping(row, point_freq, q1, q2)
            if mapping["final"]:
                sections.append(mapping["final"])
        if not sections:
            return final_conclusion
        return "\n".join([final_conclusion, "", "按产品结论：", "\n\n".join(sections)])

    def generate_market_summary(
        self,
        rows: list[dict[str, str]],
        workbook_points_map: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if not self.gemini_api_key:
            return {
                "summary_text": "",
                "common_points": [],
                "opportunities": [],
                "risks": [],
                "market_tags": [],
                "updated_at": now_iso(),
                "sample_size": len(rows),
                "point_count": 0,
            }

        payload_rows: list[dict[str, Any]] = []
        for row in rows[:120]:
            item_id = row.get("item_id", "")
            points = workbook_points_map.get(item_id, [])
            payload_rows.append(
                {
                    "item_id": item_id,
                    "title": row.get("title", ""),
                    "brand": row.get("brand", ""),
                    "shop_name": row.get("shop_name", ""),
                    "selling_points": [
                        {
                            "point": clean_text(
                                str(point.get("point", "")), max_len=80
                            ),
                            "citation": clean_text(
                                str(point.get("citation", "")), max_len=80
                            ),
                        }
                        for point in points[:8]
                        if str(point.get("point", "")).strip()
                    ],
                }
            )

        prompt = textwrap.dedent(
            f"""
            你是市场咨询公司高级研究员。请基于样本做“市场调研总结”。
            只可使用输入样本，不得虚构，不输出建议动作。
            样本:
            {json.dumps(payload_rows, ensure_ascii=False)}

            仅输出 JSON:
            {{
              "common_insights":"2-4句，描述主流卖点共性与市场主轴",
              "opportunities":"2-4句，描述样本中的差异化表达现状",
              "risks":"2-4句，描述竞争状态与同质化程度",
              "market_tags":["标签1","标签2"]
            }}
            """
        ).strip()
        payload = self._generate_json(prompt)
        if not isinstance(payload, dict):
            return {
                "summary_text": "",
                "common_points": [],
                "opportunities": [],
                "risks": [],
                "market_tags": [],
                "updated_at": now_iso(),
                "sample_size": len(rows),
                "point_count": 0,
            }

        common = clean_text(str(payload.get("common_insights", "")), max_len=1200)
        opportunities = clean_text(str(payload.get("opportunities", "")), max_len=1200)
        risks = clean_text(str(payload.get("risks", "")), max_len=1200)
        tags = self._to_string_list(payload.get("market_tags"), max_len=30)[:10]
        if not common or not opportunities or not risks:
            return {
                "summary_text": "",
                "common_points": [],
                "opportunities": [],
                "risks": [],
                "market_tags": [],
                "updated_at": now_iso(),
                "sample_size": len(rows),
                "point_count": 0,
            }
        summary_text = (
            f"市场共性：{common}\n差异化分布：{opportunities}\n竞争状态：{risks}"
        )
        return {
            "summary_text": summary_text,
            "common_points": [common],
            "opportunities": [opportunities],
            "risks": [risks],
            "market_tags": tags,
            "updated_at": now_iso(),
            "sample_size": len(rows),
            "point_count": sum(
                len(workbook_points_map.get(r.get("item_id", ""), [])) for r in rows
            ),
        }
