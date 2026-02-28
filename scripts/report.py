"""CSV and HTML generation and export logic."""

from __future__ import annotations

import csv
import logging
import re
from html import escape as html_escape
from pathlib import Path
from typing import Any

try:
    from pypinyin import pinyin, Style
except ImportError:
    pinyin = None

from config import (
    EXTENDED_PRODUCT_COLUMNS,
    PRODUCT_COLUMNS,
    NON_WORD_RE,
)
from data import (
    Storage,
    csv_cell,
    now_iso,
    preserve_multiline_text,
    read_text_utf8_best,
    clean_text,
    normalize_brand_name,
)
from analysis import Analyzer

LOG = logging.getLogger("taobao_insight")


class ReportGenerator:
    def __init__(self, storage: Storage, workbook_service: Any, analyzer: Any = None):
        self.storage = storage
        self.workbook_service = workbook_service
        self.analyzer = analyzer

    def _get_export_filename(self, workbook: dict[str, Any], ext: str) -> str:
        name = workbook.get("workbook_name", "")
        if not name:
            name = workbook.get("workbook_id", "")

        if pinyin:
            try:
                parts = pinyin(name, style=Style.NORMAL)
                name = "".join([p[0] for p in parts if p])
            except Exception:
                pass

        name = re.sub(r"[^a-zA-Z0-9_\-]+", "", name)
        if not name:
            name = workbook.get("workbook_id", "export")
        return f"{name}.{ext}"

    def export_csv(self, workbook_id: str, output_path: str | None = None) -> Path:
        workbook = self.workbook_service.get(workbook_id)
        rows = [
            dict(r)
            for r in self.storage.list_products()
            if r["workbook_id"] == workbook_id
        ]
        for row in rows:
            row["brand"] = normalize_brand_name(
                row.get("brand", ""),
                title=row.get("title", ""),
                shop_name=row.get("shop_name", ""),
            )
        target = (
            Path(output_path)
            if output_path
            else (self.storage.exports_dir / self._get_export_filename(workbook, "csv"))
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        self.storage.write_csv(
            target, PRODUCT_COLUMNS, rows, utf8_bom=True, excel_friendly=True
        )
        return target

    def export_html(
        self,
        workbook_id: str,
        output_path: str | None = None,
        md_output_path: str | None = None,
        write_md_copy: bool = False,
    ) -> Path:
        workbook = self.workbook_service.get(workbook_id)
        rows = [
            r for r in self.storage.list_products() if r["workbook_id"] == workbook_id
        ]
        target = (
            Path(output_path)
            if output_path
            else (
                self.storage.exports_dir / self._get_export_filename(workbook, "html")
            )
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        market_report = self.storage.read_json(self.storage.market_report_json).get(
            workbook_id, {}
        )

        def preserve_report_text(value: Any, max_len: int = 4000) -> str:
            text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if len(text) <= max_len:
                return text
            return text[: max_len - 3].rstrip() + "..."

        batch_summary_text = preserve_report_text(
            market_report.get("batch_competitor_summary_text", ""), max_len=4000
        )
        market_summary_text = preserve_report_text(
            market_report.get("summary_text", ""), max_len=4000
        )
        final_conclusion_text = preserve_report_text(
            market_report.get("final_conclusion_text", ""), max_len=4000
        )

        market_tags_value = market_report.get("market_tags", [])
        market_tags_text = ""
        if isinstance(market_tags_value, list):
            market_tags_text = " | ".join(
                [
                    clean_text(str(v), max_len=30)
                    for v in market_tags_value
                    if str(v).strip()
                ]
            )
        elif market_tags_value:
            market_tags_text = clean_text(str(market_tags_value), max_len=300)
        if not market_tags_text:
            market_tags_text = next(
                (r.get("market_tags", "") for r in rows if r.get("market_tags")), ""
            )

        sample_count = len(rows)
        selling_count = sum(
            1 for r in rows if str(r.get("selling_points_text", "")).strip()
        )
        price_count = sum(
            1
            for r in rows
            if str(r.get("price_min", "")).strip()
            or str(r.get("price_max", "")).strip()
        )
        analyzed_count = sum(
            1 for r in rows if str(r.get("competitor_analysis_text", "")).strip()
        )

        point_freq: dict[str, int] = {}
        prices: list[float] = []
        for row in rows:
            for point in str(row.get("selling_points_text", "")).split("|"):
                value = clean_text(point, max_len=80)
                if value:
                    point_freq[value] = point_freq.get(value, 0) + 1
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
        product_titles = [
            clean_text(row.get("title", ""), max_len=120)
            for row in rows
            if clean_text(row.get("title", ""), max_len=120)
        ]
        title_highlights = sorted(
            list(dict.fromkeys(product_titles)), key=len, reverse=True
        )

        def render_text(value: str) -> str:
            return html_escape(str(value or "")).replace("\n", "<br>")

        def render_paragraphs(value: str, highlight_titles: bool = False) -> str:
            text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                return ""
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if not paragraphs:
                paragraphs = [text]
            rendered_parts: list[str] = []
            for paragraph in paragraphs:
                html_line = render_text(paragraph)
                if highlight_titles:
                    for title_name in title_highlights:
                        escaped_title = html_escape(title_name)
                        html_line = html_line.replace(
                            escaped_title, f"<strong>{escaped_title}</strong>"
                        )
                rendered_parts.append(f"<p>{html_line}</p>")
            return "".join(rendered_parts)

        def strip_item_section(value: str) -> str:
            text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                return ""
            for marker in (
                "按产品结论：",
                "按产品结论",
                "单品简明结论：",
                "单品结论：",
            ):
                index = text.find(marker)
                if index >= 0:
                    return text[:index].strip()
            return text

        def extract_named_block(source_text: str, header: str) -> str:
            text = str(source_text or "").replace("\r\n", "\n").replace("\r", "\n")
            if not text:
                return ""
            pattern = re.compile(re.escape(header) + r"\n(?P<body>[\s\S]*?)(?:\n\n|$)")
            match = pattern.search(text)
            if not match:
                return ""
            return match.group("body").strip()

        def first_paragraph(value: str) -> str:
            text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                return ""
            blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
            if not blocks:
                return ""
            return blocks[0]

        def resolve_title_alias(name: str) -> str:
            alias = clean_text(name, max_len=120)
            if not alias:
                return ""
            alias_lower = alias.lower()
            for title_name in title_highlights:
                if alias_lower == title_name.lower():
                    return title_name
            for title_name in title_highlights:
                if alias_lower in title_name.lower():
                    return title_name
            for title_name in title_highlights:
                if title_name.lower() in alias_lower:
                    return title_name
            return alias

        def normalize_major_diff_block(value: str) -> str:
            raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not raw:
                return ""
            line_re = re.compile(
                r"^(?P<prefix>\d+[\.、\)]\s*)?(?P<diff>.+?)\s*[（(](?P<name>[^()（）]{2,120})[）)]\s*$"
            )
            lines: list[str] = []
            for line in raw.split("\n"):
                text = line.strip()
                if not text:
                    lines.append("")
                    continue
                match = line_re.match(text)
                if not match:
                    lines.append(text)
                    continue
                diff_text = clean_text(match.group("diff"), max_len=220)
                product_name = resolve_title_alias(match.group("name"))
                if not (diff_text and product_name):
                    lines.append(text)
                    continue
                prefix = match.group("prefix") or ""
                lines.append(f"{prefix}{product_name}：{diff_text}")
            return "\n".join(lines).strip()

        final_overall_text = strip_item_section(final_conclusion_text)
        market_overall_text = strip_item_section(market_summary_text)
        batch_core_block = extract_named_block(
            batch_summary_text, "市场共性："
        ) or extract_named_block(batch_summary_text, "高频卖点（共性）：")
        batch_diff_block = extract_named_block(
            batch_summary_text, "主要差异："
        ) or extract_named_block(batch_summary_text, "差异化表达（主要区分点）：")
        batch_homo_block = extract_named_block(
            batch_summary_text, "竞争焦点："
        ) or extract_named_block(batch_summary_text, "同质化表达（竞争焦点）：")

        overall_sections: list[str] = []
        section_keys: set[str] = set()

        def add_overall_section(label: str, content: str) -> None:
            body = str(content or "").strip()
            if not body:
                return
            if label == "主要差异":
                body = normalize_major_diff_block(body)
            key = NON_WORD_RE.sub("", body.lower())
            if not key or key in section_keys:
                return
            section_keys.add(key)
            overall_sections.append(f"{label}:\n{body}")

        add_overall_section("样本概览", first_paragraph(batch_summary_text))
        add_overall_section("市场共性", batch_core_block)
        add_overall_section("主要差异", batch_diff_block)
        add_overall_section("竞争焦点", batch_homo_block)
        if not (batch_core_block or batch_diff_block or batch_homo_block):
            add_overall_section(
                "整体结论",
                first_paragraph(final_overall_text)
                or first_paragraph(market_overall_text),
            )
        if not overall_sections:
            add_overall_section(
                "整体结论",
                final_overall_text or market_overall_text or batch_summary_text,
            )
        overall_summary_text = "\n\n".join(overall_sections)

        def split_points_text(value: str, max_items: int = 8) -> list[str]:
            if not value:
                return []
            chunks = re.split(r"[|锛?]", value)
            out: list[str] = []
            seen: set[str] = set()
            for chunk in chunks:
                text = clean_text(chunk, max_len=80)
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append(text)
                if len(out) >= max_items:
                    break
            return out

        def split_sku_lines(value: str, max_items: int = 40) -> list[str]:
            chunks = re.split(r"[;\r\n]+", str(value or ""))
            out: list[str] = []
            seen: set[str] = set()
            for chunk in chunks:
                text = clean_text(chunk, max_len=200)
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append(text)
                if len(out) >= max_items:
                    break
            return out

        def one_sentence(value: str, max_len: int = 220) -> str:
            text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not text:
                return ""
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if not lines:
                return ""
            first = lines[0]
            if first.startswith("商品：") and len(lines) > 1:
                first = lines[1]
            if first.startswith(("结论：", "简要结论：")) and "：" in first:
                first = first.split("：", 1)[1].strip()
            return clean_text(first, max_len=max_len)

        def to_main_diff_sentence(
            value: str, title_text: str = "", max_len: int = 260
        ) -> str:
            raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not raw:
                return ""
            merged = "；".join([line.strip() for line in raw.split("\n") if line.strip()])
            if not merged:
                return ""
            if title_text and merged.startswith(title_text):
                merged = merged[len(title_text) :].lstrip("：:，,。;； ")
            core_text = ""
            diff_text = ""

            core_match = re.search(
                r"(?:主要卖点|核心卖点)[:：]\s*(.*?)(?:[；;。]|$)", merged
            )
            if core_match:
                core_text = clean_text(core_match.group(1), max_len=max_len)
            if not core_text:
                legacy_core = re.search(r"以(.+?)作为主要沟通卖点", merged)
                if legacy_core:
                    core_text = clean_text(legacy_core.group(1), max_len=max_len)

            diff_match = re.search(
                r"差异化卖点(?:是|为)?[:：]?\s*(.*?)(?:[；;。]|$)", merged
            )
            if diff_match:
                diff_text = clean_text(diff_match.group(1), max_len=max_len)
            if not diff_text and "差异化卖点不明显" in merged:
                diff_text = "不明显"

            if core_text:
                return f"主要卖点：{core_text}；差异化卖点：{diff_text or '不明显'}。"
            return one_sentence(raw, max_len=max_len)

        def competitor_sentence(text: str, fallback: str = "") -> str:
            raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if not raw:
                return one_sentence(fallback)
            lines = [line.strip() for line in raw.split("\n") if line.strip()]
            for line in lines:
                if line.startswith(("结论：", "简要结论：")) and "：" in line:
                    return clean_text(line.split("：", 1)[1].strip(), max_len=220)
            for line in lines:
                if line.startswith(
                    (
                        "商品：",
                        "核心卖点：",
                        "差异化卖点：",
                        "同质化卖点：",
                        "图文详情页引用：",
                    )
                ):
                    continue
                return clean_text(line, max_len=220)
            return one_sentence(fallback)

        def strip_title_prefix(sentence: str, title_text: str) -> str:
            value = str(sentence or "").strip()
            title_value = str(title_text or "").strip()
            if not value:
                return ""
            if title_value and value.startswith(title_value):
                value = value[len(title_value) :].lstrip("锛?锛?銆?")
            return value

        def md_normalize_text(value: str) -> str:
            return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

        def md_escape_cell(value: str) -> str:
            text = md_normalize_text(value)
            if not text:
                return ""
            return text.replace("|", r"\|").replace("\n", "<br>")

        def md_table(headers: list[str], table_rows: list[list[str]]) -> str:
            if not headers:
                return ""
            lines: list[str] = []
            lines.append("| " + " | ".join([md_escape_cell(h) for h in headers]) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row_values in table_rows:
                row = list(row_values)
                if len(row) < len(headers):
                    row.extend([""] * (len(headers) - len(row)))
                elif len(row) > len(headers):
                    row = row[: len(headers)]
                lines.append("| " + " | ".join([md_escape_cell(v) for v in row]) + " |")
            return "\n".join(lines)

        concise_headers = [
            "序号",
            "商品",
            "价格",
            "核心卖点",
            "图文卖点摘要",
            "单品竞品结论",
            "市场观察",
            "最终结论",
            "商品链接",
        ]
        concise_col_classes = [
            "concise-col-index",
            "concise-col-title",
            "concise-col-price",
            "concise-col-points",
            "concise-col-detail",
            "concise-col-competitor",
            "concise-col-market",
            "concise-col-final",
            "concise-col-link",
        ]
        concise_table_head = "".join(
            [
                f'<th class="{concise_col_classes[idx]}">{html_escape(col)}</th>'
                for idx, col in enumerate(concise_headers)
            ]
        )
        concise_table_rows: list[str] = []
        quick_conclusion_items: list[str] = []
        concise_table_rows_md: list[list[str]] = []
        quick_conclusion_items_md: list[str] = []

        def raw_col_class(col: str) -> str:
            token = NON_WORD_RE.sub("-", str(col or "").strip().lower()).strip("-")
            token = token.replace("_", "-")
            return f"raw-col-{token or 'unknown'}"

        raw_wrap_cols = {
            "sku_list",
            "detail_summary",
            "selling_points_text",
            "selling_points_citation",
            "competitor_analysis_text",
            "batch_competitor_summary_text",
            "market_summary_text",
            "final_conclusion_text",
        }
        raw_table_head = "".join(
            [
                f'<th class="{raw_col_class(col)}">{html_escape(col)}</th>'
                for col in PRODUCT_COLUMNS
            ]
        )
        raw_table_rows: list[str] = []
        for row in rows:
            cells: list[str] = []
            for col in PRODUCT_COLUMNS:
                value = str(row.get(col, "") or "")
                col_class = raw_col_class(col)
                if col in {
                    "normalized_url",
                    "raw_url",
                    "main_image_url",
                    "item_source_url",
                } and value.startswith("http"):
                    href = html_escape(value, quote=True)
                    label = html_escape(clean_text(value, max_len=70))
                    rendered = f'<a href="{href}" target="_blank" rel="noopener noreferrer">{label}</a>'
                elif col == "sku_list":
                    rendered = "<br>".join(
                        [
                            html_escape(line)
                            for line in split_sku_lines(value, max_items=80)
                        ]
                    )
                else:
                    rendered = render_text(value)
                if col in raw_wrap_cols:
                    rendered = f'<div class="raw-clamp" title="{html_escape(value[:1000], quote=True)}">{rendered}</div>'
                cells.append(f'<td class="{col_class}">{rendered}</td>')
            raw_table_rows.append("<tr>" + "".join(cells) + "</tr>")
        if not raw_table_rows:
            raw_table_rows.append(
                f'<tr><td colspan="{len(PRODUCT_COLUMNS)}"></td></tr>'
            )

        item_cards: list[str] = []
        item_cards_md: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            mapping = self.analyzer.build_row_market_mapping(row, point_freq, q1, q2)
            title_plain = (
                clean_text(row.get("title", ""), max_len=120)
                or clean_text(row.get("item_id", ""), max_len=50)
                or f"鍟嗗搧{idx}"
            )
            title = html_escape(title_plain)
            brand_plain = normalize_brand_name(
                row.get("brand", ""),
                title=title_plain,
                shop_name=row.get("shop_name", ""),
            )
            brand = html_escape(brand_plain)
            shop_plain = clean_text(row.get("shop_name", ""), max_len=50)
            shop = html_escape(shop_plain)
            price_min = clean_text(row.get("price_min", ""), max_len=20)
            price_max = clean_text(row.get("price_max", ""), max_len=20)
            if price_min and price_max and price_min != price_max:
                price_text = f"{price_min}-{price_max}元"
            else:
                price_text = (price_min or price_max) + (
                    "元" if (price_min or price_max) else ""
                )
            source_url = str(
                row.get("item_source_url", "") or row.get("normalized_url", "") or ""
            )
            source_link = ""
            if source_url.startswith("http"):
                href = html_escape(source_url, quote=True)
                source_link = f'<a href="{href}" target="_blank" rel="noopener noreferrer">商品链接</a>'

            points = split_points_text(
                str(row.get("selling_points_text", "")), max_items=8
            )
            point_chips = "".join(
                [
                    f'<span class="point-chip">{html_escape(point)}</span>'
                    for point in points
                ]
            )
            if not point_chips:
                point_chips = '<span class="point-chip muted"></span>'
            point_text = " | ".join(points[:4])

            detail_sentence = one_sentence(
                str(row.get("detail_summary", "")), max_len=240
            )
            market_sentence_text = one_sentence(
                str(row.get("market_summary_text", "")) or mapping.get("market", ""),
                max_len=240,
            )
            final_sentence = to_main_diff_sentence(
                str(row.get("final_conclusion_text", "")) or mapping.get("final", ""),
                title_text=title_plain,
                max_len=300,
            )
            comp_sentence = to_main_diff_sentence(
                str(row.get("competitor_analysis_text", "")),
                title_text=title_plain,
                max_len=300,
            ) or competitor_sentence(
                str(row.get("competitor_analysis_text", "")),
                fallback=mapping.get("batch", ""),
            )
            market_sentence_short = strip_title_prefix(
                market_sentence_text, title_plain
            )
            final_sentence_short = strip_title_prefix(final_sentence, title_plain)
            comp_sentence_short = strip_title_prefix(comp_sentence, title_plain)
            summary_text = (
                final_sentence_short or comp_sentence_short or "暂缺相关分析数据"
            )
            quick_conclusion_items.append(
                f"<li><strong>{html_escape(title_plain)}</strong>：{render_text(summary_text)}</li>"
            )
            quick_conclusion_items_md.append(f"**{title_plain}**：{summary_text}")

            concise_cells: list[str] = [
                str(idx),
                title_plain,
                price_text,
                point_text,
                detail_sentence,
                comp_sentence_short,
                market_sentence_short,
                final_sentence_short,
                source_url,
            ]
            concise_row_cells: list[str] = []
            for col_idx, value in enumerate(concise_cells):
                col_class = concise_col_classes[col_idx]
                if col_idx == len(concise_cells) - 1 and source_link:
                    concise_row_cells.append(f'<td class="{col_class}">{source_link}</td>')
                else:
                    concise_row_cells.append(f'<td class="{col_class}">{render_text(value)}</td>')
            concise_table_rows.append("<tr>" + "".join(concise_row_cells) + "</tr>")
            concise_table_rows_md.append(
                concise_cells[:-1]
                + (
                    [f"[商品链接]({source_url})"]
                    if source_url.startswith("http")
                    else [""]
                )
            )

            sku_lines = split_sku_lines(str(row.get("sku_list", "")), max_items=40)
            sku_html = (
                '<ul class="sku-list">'
                + "".join([f"<li>{html_escape(line)}</li>" for line in sku_lines])
                + "</ul>"
                if sku_lines
                else ""
            )
            main_image_url = str(row.get("main_image_url", "") or "")
            main_image_block = ""
            main_image_src = ""
            if main_image_url.startswith("http"):
                main_image_src = main_image_url
            else:
                try:
                    image_path = Path(main_image_url)
                    if image_path.exists() and image_path.is_file():
                        main_image_src = image_path.resolve().as_uri()
                except Exception:
                    main_image_src = ""
            if main_image_src:
                href = html_escape(main_image_src, quote=True)
                main_image_block = (
                    '<div class="main-image">'
                    f'<a href="{href}" target="_blank" rel="noopener noreferrer">'
                    f'<img src="{href}" alt="main-image" loading="lazy"></a></div>'
                )
            item_cards_md.append(
                {
                    "idx": idx,
                    "title": title_plain,
                    "summary": summary_text,
                    "brand": brand_plain,
                    "shop": shop_plain,
                    "price_text": price_text,
                    "source_url": source_url if source_url.startswith("http") else "",
                    "points": points,
                    "detail_sentence": detail_sentence,
                    "comp_sentence_short": comp_sentence_short,
                    "market_sentence_short": market_sentence_short,
                    "main_image_src": main_image_src,
                    "sku_lines": sku_lines,
                    "detail_summary_raw": str(row.get("detail_summary", "")),
                    "competitor_analysis_raw": str(
                        row.get("competitor_analysis_text", "")
                    ),
                }
            )

            item_cards.append(
                f"""
                <details class="item-card"{" open" if idx == 1 else ""}>
                  <summary>
                    <div class="item-head">
                      <span class="item-index">{idx}</span>
                      <span class="item-title">{title}</span>
                    </div>
                    <div class="item-summary">{render_text(summary_text)}</div>
                  </summary>
                  <div class="item-body">
                    <div class="item-meta">品牌：{brand} | 店铺：{shop} | 价格：{html_escape(price_text)} {source_link}</div>
                    <div class="item-block"><div class="item-label">核心卖点</div><div class="point-list">{point_chips}</div></div>
                    <div class="item-block"><div class="item-label">图文卖点摘要</div><div class="item-text">{render_text(detail_sentence)}</div></div>
                    <div class="item-block"><div class="item-label">单品竞品结论</div><div class="item-text">{render_text(comp_sentence_short)}</div></div>
                    <div class="item-block"><div class="item-label">市场观察</div><div class="item-text">{render_text(market_sentence_short)}</div></div>
                    {main_image_block}
                    <details class="sub-detail">
                      <summary>查看 SKU 与完整分析字段</summary>
                      <div class="item-block"><div class="item-label">SKU 列表</div><div class="item-text">{sku_html}</div></div>
                      <div class="item-block"><div class="item-label">detail_summary</div><div class="item-text">{render_text(str(row.get("detail_summary", "")))}</div></div>
                      <div class="item-block"><div class="item-label">competitor_analysis_text</div><div class="item-text">{render_text(str(row.get("competitor_analysis_text", "")))}</div></div>
                    </details>
                  </div>
                </details>
                """
            )
        if not concise_table_rows:
            concise_table_rows.append(
                f'<tr><td colspan="{len(concise_headers)}"></td></tr>'
            )
        generated_at = now_iso()

        html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>淘宝市场调研报告</title>
  <style>
    :root {{
      --bg: #f3f6fa;
      --card: #ffffff;
      --text: #111827;
      --muted: #475569;
      --line: #dbe2ea;
      --head: #eef2f7;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      max-width: 100%;
      overflow-x: hidden;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
    }}
    .layout {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      gap: 12px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      overflow-x: hidden;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }}
    .stat {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .stat-k {{
      font-size: 12px;
      color: #64748b;
      margin-bottom: 4px;
    }}
    .stat-v {{
      font-size: 20px;
      font-weight: 700;
      color: #0f172a;
    }}
    .section-text p {{
      margin: 0 0 10px;
      line-height: 1.75;
      font-size: 14px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .quick-list {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.75;
      font-size: 14px;
      color: #1f2937;
    }}
    .quick-list li {{
      margin-bottom: 8px;
      word-break: break-word;
    }}
    .item-grid {{
      display: grid;
      gap: 10px;
    }}
    .item-card {{
      border: 1px solid #dbe2ea;
      border-radius: 10px;
      background: #fff;
    }}
    .item-card > summary {{
      list-style: none;
      cursor: pointer;
      padding: 12px;
      display: grid;
      gap: 6px;
    }}
    .item-card > summary::-webkit-details-marker {{ display: none; }}
    .item-card[open] > summary {{
      border-bottom: 1px solid #e2e8f0;
    }}
    .item-head {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .item-index {{
      display: inline-block;
      min-width: 24px;
      height: 24px;
      border-radius: 999px;
      line-height: 24px;
      text-align: center;
      background: #e6f7f3;
      color: #0f766e;
      font-size: 13px;
      font-weight: 700;
    }}
    .item-title {{
      font-size: 15px;
      font-weight: 700;
      word-break: break-word;
    }}
    .item-summary {{
      font-size: 13px;
      line-height: 1.6;
      color: #1f2937;
      word-break: break-word;
    }}
    .item-body {{
      padding: 12px;
    }}
    .item-meta {{
      font-size: 12px;
      color: #475569;
      line-height: 1.6;
      word-break: break-word;
    }}
    .item-block {{
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px dashed #e2e8f0;
    }}
    .item-label {{
      font-size: 13px;
      font-weight: 700;
      color: #0f766e;
      margin-bottom: 4px;
    }}
    .item-text {{
      font-size: 13px;
      line-height: 1.7;
      color: #1f2937;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .point-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .point-chip {{
      background: #f1f5f9;
      border: 1px solid #dbe2ea;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      color: #334155;
    }}
    .point-chip.muted {{
      background: #f8fafc;
      color: #64748b;
    }}
    .main-image img {{
      margin-top: 8px;
      max-width: 280px;
      width: 100%;
      max-height: 280px;
      object-fit: contain;
      border: 1px solid #dbe2ea;
      border-radius: 10px;
      background: #fff;
    }}
    .sub-detail {{
      margin-top: 10px;
      border: 1px solid #dbe2ea;
      border-radius: 10px;
      background: #fff;
    }}
    .sub-detail > summary {{
      cursor: pointer;
      list-style: none;
      padding: 8px 10px;
      font-size: 13px;
      font-weight: 600;
      color: #1f2937;
    }}
    .sub-detail > summary::-webkit-details-marker {{ display: none; }}
    .sku-list {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.7;
    }}
    .sku-list li {{
      margin-bottom: 4px;
      word-break: break-word;
    }}
    .table-wrap {{
      width: 100%;
      max-width: 100%;
      display: block;
      overflow: auto;
      border: 1px solid #dbe2ea;
      border-radius: 10px;
    }}
    .raw-table-wrap {{
      width: 100%;
      overflow-x: auto;
      overflow-y: auto;
      border: 1px solid #dbe2ea;
      border-radius: 10px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: fixed;
      font-size: 12px;
    }}
    th, td {{
      border: 1px solid #dbe2ea;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    th {{
      background: var(--head);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .section-detail + .section-detail {{
      margin-top: 10px;
    }}
    .section-detail > summary {{
      cursor: pointer;
      list-style: none;
      font-size: 14px;
      font-weight: 700;
      color: #0f766e;
      margin: 0;
    }}
    .section-detail > summary::-webkit-details-marker {{ display: none; }}
    .section-detail > div {{
      margin-top: 8px;
    }}
    .concise-table-wrap {{
      overflow-x: auto;
      overflow-y: auto;
    }}
    .concise-table {{
      min-width: 1380px;
      width: max-content;
      table-layout: fixed;
    }}
    .concise-table .concise-col-index {{ width: 56px; text-align: center; }}
    .concise-table .concise-col-title {{ width: 180px; }}
    .concise-table .concise-col-price {{ width: 96px; }}
    .concise-table .concise-col-points {{ width: 190px; }}
    .concise-table .concise-col-detail {{ width: 180px; }}
    .concise-table .concise-col-competitor {{ width: 220px; }}
    .concise-table .concise-col-market {{ width: 200px; }}
    .concise-table .concise-col-final {{ width: 220px; }}
    .concise-table .concise-col-link {{ width: 90px; text-align: center; }}
    .concise-table td {{
      line-height: 1.45;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .raw-table {{
      min-width: 2200px;
      width: max-content;
      table-layout: fixed;
    }}
    .raw-table th,
    .raw-table td {{
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .raw-clamp {{
      display: -webkit-box;
      -webkit-line-clamp: 5;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .raw-table .raw-col-workbook-id,
    .raw-table .raw-col-group-code,
    .raw-table .raw-col-group-name,
    .raw-table .raw-col-item-id,
    .raw-table .raw-col-process-status,
    .raw-table .raw-col-price-min,
    .raw-table .raw-col-price-max,
    .raw-table .raw-col-sku-count,
    .raw-table .raw-col-search-rank,
    .raw-table .raw-col-official-store,
    .raw-table .raw-col-source-type {{
      width: 90px;
    }}
    .raw-table .raw-col-workbook-name,
    .raw-table .raw-col-shop-name,
    .raw-table .raw-col-brand,
    .raw-table .raw-col-sales-text,
    .raw-table .raw-col-market-tags {{
      width: 130px;
    }}
    .raw-table .raw-col-normalized-url,
    .raw-table .raw-col-raw-url,
    .raw-table .raw-col-main-image-url,
    .raw-table .raw-col-item-source-url {{
      width: 170px;
    }}
    .raw-table .raw-col-title {{
      width: 220px;
    }}
    .raw-table .raw-col-sku-list,
    .raw-table .raw-col-detail-summary,
    .raw-table .raw-col-selling-points-text,
    .raw-table .raw-col-selling-points-citation {{
      width: 220px;
    }}
    .raw-table .raw-col-competitor-analysis-text,
    .raw-table .raw-col-batch-competitor-summary-text,
    .raw-table .raw-col-market-summary-text,
    .raw-table .raw-col-final-conclusion-text {{
      width: 280px;
    }}
    .raw-table .raw-col-crawl-time,
    .raw-table .raw-col-updated-at {{
      width: 150px;
    }}
    a {{
      color: #0b7285;
      text-decoration: none;
      word-break: break-all;
    }}
    @media (max-width: 900px) {{
      .stats {{
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }}
      .layout {{
        padding: 12px;
      }}
    }}
  </style>
</head>
<body>
  <main class="layout">
    <section class="card">
      <h1>淘宝市场调研报告</h1>
      <div class="meta">工作簿：{html_escape(workbook.get("workbook_name", workbook_id))} | 样本量：{sample_count} | 更新时间：{html_escape(generated_at)}</div>
      <div class="meta">市场标签：{html_escape(market_tags_text or "")}</div>
    </section>
    <section class="stats">
      <div class="stat"><div class="stat-k">样本数量</div><div class="stat-v">{sample_count}</div></div>
      <div class="stat"><div class="stat-k">已提取卖点</div><div class="stat-v">{selling_count}</div></div>
      <div class="stat"><div class="stat-k">有价格信息</div><div class="stat-v">{price_count}</div></div>
      <div class="stat"><div class="stat-k">已完成单品分析</div><div class="stat-v">{analyzed_count}</div></div>
    </section>
    <section class="card">
      <h2>整体聚合结论</h2>
      <div class="section-text">{render_paragraphs(overall_summary_text or "", highlight_titles=True)}</div>
    </section>
    <section class="card">
      <h2>单品结论（按商品）</h2>
      <ol class="quick-list">{"".join(quick_conclusion_items)}</ol>
    </section>
    <section class="card">
      <h2>商品分析明细（可展开）</h2>
      <div class="item-grid">
        {"".join(item_cards)}
      </div>
    </section>
    <section class="card">
      <details class="section-detail" open>
        <summary><strong>简表（单句描述）</strong></summary>
        <div>
          <div class="table-wrap concise-table-wrap">
            <table class="concise-table">
              <thead><tr>{concise_table_head}</tr></thead>
              <tbody>
                {"".join(concise_table_rows)}
              </tbody>
            </table>
          </div>
        </div>
      </details>
      <details class="section-detail">
        <summary><strong>原始明细（全部字段）</strong></summary>
        <div>
          <div class="table-wrap raw-table-wrap">
            <table class="raw-table">
              <thead><tr>{raw_table_head}</tr></thead>
              <tbody>
                {"".join(raw_table_rows)}
              </tbody>
            </table>
          </div>
        </div>
      </details>
    </section>
  </main>
</body>
</html>
"""
        markdown_lines: list[str] = [
            "# 淘宝市场调研报告",
            "",
            f"- 工作簿：{workbook.get('workbook_name', workbook_id)}",
            f"- 样本量：{sample_count}",
            f"- 更新时间：{generated_at}",
        ]
        if market_tags_text:
            markdown_lines.append(f"- 市场标签：{market_tags_text}")
        markdown_lines.extend(
            [
                "",
                "## 核心指标",
                "",
                md_table(
                    ["指标", "数值"],
                    [
                        ["样本数量", str(sample_count)],
                        ["已提取卖点", str(selling_count)],
                        ["有价格信息", str(price_count)],
                        ["已完成单品分析", str(analyzed_count)],
                    ],
                ),
                "",
                "## 整体聚合结论",
                "",
            ]
        )
        summary_sections = [
            block.strip()
            for block in re.split(r"\n\s*\n", overall_summary_text or "")
            if block.strip()
        ]
        if summary_sections:
            for block in summary_sections:
                markdown_lines.append(block)
                markdown_lines.append("")
        else:
            markdown_lines.extend(["暂无整体结论。", ""])

        markdown_lines.extend(["## 单品结论（按商品）", ""])
        if quick_conclusion_items_md:
            for conclusion in quick_conclusion_items_md:
                markdown_lines.append(f"- {conclusion}")
        else:
            markdown_lines.append("- 暂无单品结论。")
        markdown_lines.append("")

        markdown_lines.extend(["## 商品分析明细（可展开）", ""])
        for item in item_cards_md:
            markdown_lines.extend(
                [
                    f"### {item['idx']}. {item['title']}",
                    "",
                    f"- 摘要：{item['summary']}",
                    f"- 品牌：{item['brand']} | 店铺：{item['shop']} | 价格：{item['price_text']}",
                ]
            )
            if item["source_url"]:
                markdown_lines.append(f"- 商品链接：[商品链接]({item['source_url']})")
            if item["points"]:
                markdown_lines.append(f"- 核心卖点：{' | '.join(item['points'])}")
            else:
                markdown_lines.append("- 核心卖点：")
            markdown_lines.extend(
                [
                    f"- 图文卖点摘要：{item['detail_sentence']}",
                    f"- 单品竞品结论：{item['comp_sentence_short']}",
                    f"- 市场观察：{item['market_sentence_short']}",
                ]
            )
            if item["main_image_src"]:
                markdown_lines.append(f"- 主图：[查看主图]({item['main_image_src']})")
            markdown_lines.append("- 查看 SKU 与完整分析字段：")
            if item["sku_lines"]:
                markdown_lines.append("  - SKU 列表：")
                for sku in item["sku_lines"]:
                    markdown_lines.append(f"    - {sku}")
            else:
                markdown_lines.append("  - SKU 列表：")
            markdown_lines.append(
                f"  - detail_summary：{md_normalize_text(item['detail_summary_raw'])}"
            )
            markdown_lines.append(
                "  - competitor_analysis_text："
                + md_normalize_text(item["competitor_analysis_raw"])
            )
            markdown_lines.append("")

        markdown_lines.extend(
            [
                "## 简表（单句描述）",
                "",
                md_table(concise_headers, concise_table_rows_md),
                "",
                "## 原始明细（全部字段）",
                "",
                md_table(
                    PRODUCT_COLUMNS,
                    [
                        [str(row.get(col, "") or "") for col in PRODUCT_COLUMNS]
                        for row in rows
                    ],
                ),
                "",
            ]
        )
        markdown_doc = "\n".join(markdown_lines)

        target.write_text(html_doc, encoding="utf-8")
        if write_md_copy:
            md_target = (
                Path(md_output_path) if md_output_path else target.with_suffix(".md")
            )
            md_target.parent.mkdir(parents=True, exist_ok=True)
            md_target.write_text(markdown_doc, encoding="utf-8")
        return target

