from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import urllib.request
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

from config import UA


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_task(data_dir: Path, task_id: str) -> dict[str, Any]:
    tasks = _read_json(data_dir / "tasks.json")
    payload = tasks.get(task_id, {})
    if not isinstance(payload, dict):
        return {}
    return payload


def _load_products(data_dir: Path, workbook_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (data_dir / "products.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("workbook_id") == workbook_id:
                rows.append({key: str(value or "") for key, value in row.items()})
    rows.sort(
        key=lambda row: (
            int(row.get("search_rank", "99999") or 99999)
            if str(row.get("search_rank", "")).strip().isdigit()
            else 99999,
            row.get("item_id", ""),
        )
    )
    return rows


def _load_workbook_selling_points(
    data_dir: Path, workbook_id: str
) -> dict[str, dict[str, Any]]:
    payload = _read_json(data_dir / "selling_points.json")
    out: dict[str, dict[str, Any]] = {}
    prefix = f"{workbook_id}:"
    for key, value in payload.items():
        if not isinstance(key, str) or not key.startswith(prefix) or not isinstance(value, dict):
            continue
        out[key.split(":", 1)[1]] = value
    return out


def _load_debug_records(debug_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    if not debug_dir.exists():
        return records
    for path in sorted(debug_dir.glob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            records.append((path, payload))
    return records


def _guess_referer(url: str) -> str:
    lower = str(url or "").lower()
    if any(token in lower for token in ("jd.com", "360buyimg", "jdimg", "3.cn")):
        return "https://item.jd.com/"
    if any(token in lower for token in ("taobao", "tmall", "alicdn", "tbcdn")):
        return "https://detail.tmall.com/"
    return ""


def _copy_or_download_image(source: str, target: Path) -> None:
    source_path = Path(str(source or ""))
    if source_path.exists() and source_path.is_file():
        shutil.copyfile(source_path, target)
        return

    headers = {"User-Agent": UA}
    referer = _guess_referer(source)
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(str(source), headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _local_asset_filename(ref: str, source: str) -> str:
    suffix = Path(urlparse(source).path).suffix or Path(source).suffix or ".img"
    if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".img"}:
        suffix = ".img"
    return f"{ref}{suffix}"


def _ensure_item_assets(
    bundle_dir: Path,
    item_id: str,
    image_blocks: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    asset_dir = bundle_dir / "assets" / item_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_map: dict[str, dict[str, str]] = {}
    for block in image_blocks:
        ref = str(block.get("source_ref", "") or "").strip()
        source = str(block.get("content", "") or "").strip()
        if not ref or not source:
            continue
        filename = _local_asset_filename(ref, source)
        target = asset_dir / filename
        if not target.exists() or target.stat().st_size == 0:
            try:
                _copy_or_download_image(source, target)
            except Exception:
                pass
        asset_map[ref] = {
            "filename": filename,
            "relative_path": f"assets/{item_id}/{filename}",
            "absolute_path": str(target.resolve()),
            "remote_url": source,
        }
    return asset_map


def _extract_prompt_text(payload: dict[str, Any]) -> str:
    parts = payload.get("text_parts", [])
    if not isinstance(parts, list):
        return ""
    return "\n\n".join(str(part or "") for part in parts)


def _extract_image_refs(payload: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    contents = payload.get("contents", [])
    if not isinstance(contents, list):
        return refs
    for part in contents:
        if not isinstance(part, dict):
            continue
        if part.get("kind") != "text":
            continue
        text = str(part.get("text", "") or "")
        if text.startswith("image_") and text.endswith(":"):
            refs.append(text[:-1])
    return refs


def _record_sort_key(record: tuple[Path, dict[str, Any]]) -> tuple[str, str]:
    path, payload = record
    ts = str(payload.get("ts", "") or path.name)
    return ts, path.name


def _stage_label(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if not normalized:
        return "Unknown Stage"
    mapping = [
        ("llm_extract_images", "Image Extraction"),
        ("llm_extract_text", "Text Extraction"),
        ("llm_extract", "Selling Point Extraction"),
        ("competitor_analysis", "Competitor Analysis"),
        ("batch_competitor_summary", "Batch Summary"),
        ("market_summary", "Market Summary"),
        ("final_conclusion", "Final Conclusion"),
    ]
    for token, label in mapping:
        if token in normalized:
            return label
    return stage


def _short_url_label(url: str, max_len: int = 96) -> str:
    value = str(url or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _default_search_url(platform: str, keyword: str) -> str:
    normalized_platform = str(platform or "taobao").strip().lower()
    if normalized_platform == "jd":
        return f"https://search.jd.com/Search?keyword={quote_plus(keyword)}"
    return f"https://s.taobao.com/search?q={quote_plus(keyword)}"


def _extract_tps_dimensions(url: str) -> tuple[int, int]:
    match = re.search(r"tps-(?P<w>\d+)-(?P<h>\d+)", str(url or ""), flags=re.IGNORECASE)
    if not match:
        return 0, 0
    try:
        return int(match.group("w")), int(match.group("h"))
    except Exception:
        return 0, 0


def _is_debug_noise_image(platform: str, url: str) -> bool:
    normalized_platform = str(platform or "").strip().lower()
    lower = str(url or "").strip().lower()
    if not lower:
        return False
    if normalized_platform == "taobao":
        if any(
            token in lower
            for token in (
                "service",
                "refund",
                "vip",
                "wuying",
                "barrier",
                "accessible",
                "green",
                "carbon",
            )
        ):
            return True
        width, height = _extract_tps_dimensions(lower)
        if width and height:
            ratio = width / max(height, 1)
            if min(width, height) < 180:
                return True
            if height <= 320 and width >= 240 and ratio >= 2.0:
                return True
    return False


def _infer_task_inputs(
    task_payload: dict[str, Any],
    task_result: dict[str, Any],
    product_rows: list[dict[str, str]],
) -> dict[str, Any]:
    source_mode = str(task_result.get("source_mode", "") or "").strip() or "keyword_search"
    keyword = str(
        task_result.get("run_summary", {}).get("keyword", "")
        or task_payload.get("keyword", "")
        or ""
    ).strip()
    platform = str(
        task_result.get("platform", "")
        or (product_rows[0].get("platform", "") if product_rows else "")
        or "taobao"
    ).strip()
    raw_input_urls = task_result.get("input_urls_used", [])
    input_urls = [str(url or "").strip() for url in raw_input_urls if str(url or "").strip()] if isinstance(raw_input_urls, list) else []
    if source_mode == "input_urls" and not input_urls:
        seen: set[str] = set()
        for row in product_rows:
            for key in ("item_source_url", "normalized_url", "raw_url"):
                candidate = str(row.get(key, "") or "").strip()
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                input_urls.append(candidate)
    search_target_url = str(task_result.get("search_target_url", "") or "").strip()
    if source_mode == "keyword_search" and not search_target_url and keyword:
        search_target_url = _default_search_url(platform, keyword)
    return {
        "source_mode": source_mode,
        "keyword": keyword,
        "platform": platform,
        "input_urls": input_urls,
        "search_target_url": search_target_url,
    }


def _guess_export_html_path(
    data_dir: Path,
    task_result: dict[str, Any],
    task_payload: dict[str, Any],
    product_rows: list[dict[str, str]],
) -> Path | None:
    direct = str(task_result.get("export_html", "") or "").strip()
    if direct:
        candidate = Path(direct)
        if not candidate.is_absolute():
            candidate = (data_dir / candidate).resolve()
        if candidate.exists():
            return candidate
    task_inputs = _infer_task_inputs(task_payload, task_result, product_rows)
    workbook_name = str(task_result.get("workbook_name", "") or "").strip()
    platform = str(task_inputs.get("platform", "") or "taobao").strip().lower()
    source_mode = str(task_inputs.get("source_mode", "") or "keyword_search").strip()
    mode_dir = "direct" if source_mode == "input_urls" else "search"
    if workbook_name:
        guessed = data_dir / "exports" / platform / mode_dir / f"{workbook_name}.html"
        if guessed.exists():
            return guessed.resolve()
    return None


def _render_item_section(
    *,
    row: dict[str, str],
    selling_points: dict[str, Any],
    asset_map: dict[str, dict[str, str]],
    debug_records: list[tuple[Path, dict[str, Any]]],
) -> str:
    item_id = row.get("item_id", "")
    points = selling_points.get("points", [])
    if not isinstance(points, list):
        points = []
    detail_blocks = selling_points.get("detail_blocks", [])
    if not isinstance(detail_blocks, list):
        detail_blocks = []
    visit_chain_raw = selling_points.get("visit_chain", [])
    visit_chain = [entry for entry in visit_chain_raw if isinstance(entry, dict)]
    image_blocks = [
        block for block in detail_blocks if str(block.get("source_type", "")).lower() == "image"
    ]
    product_image_blocks = [
        block
        for block in image_blocks
        if not _is_debug_noise_image(row.get("platform", ""), str(block.get("content", "") or ""))
    ]
    noise_image_blocks = [
        block
        for block in image_blocks
        if _is_debug_noise_image(row.get("platform", ""), str(block.get("content", "") or ""))
    ]
    text_blocks = [
        block for block in detail_blocks if str(block.get("source_type", "")).lower() == "text"
    ]
    citation_counts = Counter(
        str(point.get("citation", "") or "") for point in points if isinstance(point, dict)
    )
    sent_counts: Counter[str] = Counter()
    for _, payload in debug_records:
        meta = payload.get("debug_meta", {})
        if isinstance(meta, dict) and str(meta.get("item_id", "")) not in {"", item_id}:
            continue
        for ref in _extract_image_refs(payload):
            sent_counts[ref] += 1

    used_image_cards: list[str] = []
    unused_image_cards: list[str] = []
    noise_image_cards: list[str] = []

    def _build_image_card(block: dict[str, Any], badge_label: str, badge_class: str) -> str:
        ref = str(block.get("source_ref", "") or "")
        asset = asset_map.get(ref, {})
        local_src = escape(asset.get("relative_path", ""))
        remote_src = escape(asset.get("remote_url", str(block.get("content", "") or "")))
        cited_label = f"Cited {citation_counts.get(ref, 0)}x"
        return f"""
            <article class="card">
              <div class="card-head">
                <strong>{escape(ref)}</strong>
                <span class="badge {badge_class}">{escape(badge_label)}</span>
              </div>
              <a href="{remote_src}" target="_blank" rel="noreferrer">
                <img src="{local_src}" alt="{escape(ref)}" loading="lazy" onerror="this.onerror=null;this.src='{remote_src}';">
              </a>
              <div class="meta">
                <div><strong>{escape(cited_label)}</strong></div>
                <div>Local: <code>{escape(asset.get("absolute_path", ""))}</code></div>
                <div>Remote: <a href="{remote_src}" target="_blank" rel="noreferrer">{escape(_short_url_label(asset.get("remote_url", str(block.get("content", "") or ""))))}</a></div>
              </div>
            </article>
            """

    for block in product_image_blocks:
        ref = str(block.get("source_ref", "") or "")
        was_sent = sent_counts.get(ref, 0) > 0
        card_html = _build_image_card(
            block,
            badge_label="Sent to Gemini" if was_sent else "Captured Only",
            badge_class="used" if was_sent else "unused",
        )
        if was_sent:
            used_image_cards.append(card_html)
        else:
            unused_image_cards.append(card_html)

    for block in noise_image_blocks:
        ref = str(block.get("source_ref", "") or "")
        was_sent = sent_counts.get(ref, 0) > 0
        noise_image_cards.append(
            _build_image_card(
                block,
                badge_label="Legacy Noise (Sent)" if was_sent else "Filtered Noise",
                badge_class="unused",
            )
        )

    point_rows = "".join(
        f"<tr><td>{escape(str(point.get('point', '') or ''))}</td>"
        f"<td><code>{escape(str(point.get('citation', '') or ''))}</code></td></tr>"
        for point in points
        if isinstance(point, dict)
    ) or '<tr><td colspan="2">No structured selling points were extracted.</td></tr>'
    text_rows = "".join(
        f"<tr><td><code>{escape(str(block.get('source_ref', '') or ''))}</code></td>"
        f"<td>{escape(str(block.get('content', '') or ''))}</td></tr>"
        for block in text_blocks
    ) or '<tr><td colspan="2">No text evidence captured.</td></tr>'

    seen_urls: set[str] = set()
    url_rows: list[str] = []
    for label, value in (
        ("raw_url", row.get("raw_url", "")),
        ("normalized_url", row.get("normalized_url", "")),
        ("item_source_url", row.get("item_source_url", "")),
        ("main_image_url", row.get("main_image_url", "")),
    ):
        url_value = str(value or "").strip()
        if not url_value or url_value in seen_urls:
            continue
        seen_urls.add(url_value)
        safe_url = escape(url_value)
        url_rows.append(
            f"<tr><td>{escape(label)}</td><td><a href=\"{safe_url}\" target=\"_blank\" rel=\"noreferrer\">{escape(_short_url_label(url_value))}</a></td></tr>"
        )
    url_rows_html = "".join(url_rows) or '<tr><td colspan="2">No item-level URLs recorded.</td></tr>'

    visit_rows = "".join(
        f"<tr><td>{escape(str(entry.get('stage', '') or ''))}</td>"
        f"<td><a href=\"{escape(str(entry.get('url', '') or ''))}\" target=\"_blank\" rel=\"noreferrer\">{escape(_short_url_label(str(entry.get('url', '') or '')))}</a></td>"
        f"<td>{escape(str(entry.get('title', '') or ''))}</td>"
        f"<td>{escape(str(entry.get('note', '') or ''))}</td></tr>"
        for entry in visit_chain
        if str(entry.get("url", "") or "").strip()
    ) or '<tr><td colspan="4">No crawler visit chain recorded for this item.</td></tr>'

    main_image_asset = asset_map.get("image_1", {})
    main_src = escape(main_image_asset.get("relative_path", ""))
    main_remote = escape(main_image_asset.get("remote_url", row.get("main_image_url", "")))
    item_chips = "".join(
        [
            f'<span class="chip">points: {len(points)}</span>',
            f'<span class="chip">sent images: {sum(1 for count in sent_counts.values() if count > 0)}</span>',
            f'<span class="chip">captured images: {len(product_image_blocks)}</span>',
            f'<span class="chip">filtered noise: {len(noise_image_blocks)}</span>',
            f'<span class="chip">text blocks: {len(text_blocks)}</span>',
        ]
    )
    unused_section = ""
    if unused_image_cards:
        unused_section = f"""
        <details>
          <summary>Other Captured Images ({len(unused_image_cards)})</summary>
          <div class="gallery">{''.join(unused_image_cards)}</div>
        </details>
        """
    noise_section = ""
    if noise_image_cards:
        noise_section = f"""
        <details>
          <summary>Filtered Noise Images ({len(noise_image_cards)})</summary>
          <div class="empty">These images do not look like product detail evidence. If marked as "Legacy Noise (Sent)", they were part of an older run and should be excluded by the current filter.</div>
          <div class="gallery">{''.join(noise_image_cards)}</div>
        </details>
        """
    return f"""
    <section class="panel item-section" id="item-{escape(item_id)}">
      <div class="hero">
        <div>
          <div class="section-kicker">Item Overview</div>
          <h2>{escape(row.get("title", "") or item_id)}</h2>
          <div class="meta-grid">
            <div><strong>item_id</strong>: <code>{escape(item_id)}</code></div>
            <div><strong>platform</strong>: {escape(row.get("platform", ""))}</div>
            <div><strong>shop_name</strong>: {escape(row.get("shop_name", ""))}</div>
            <div><strong>brand</strong>: {escape(row.get("brand", ""))}</div>
            <div><strong>price_min</strong>: {escape(row.get("price_min", ""))}</div>
            <div><strong>price_max</strong>: {escape(row.get("price_max", ""))}</div>
            <div><strong>sku_count</strong>: {escape(row.get("sku_count", ""))}</div>
            <div><strong>detail_summary</strong>: {escape(str(selling_points.get("detail_summary", "") or ""))}</div>
          </div>
          <div class="chips">{item_chips}</div>
        </div>
        <div>
          <a href="{main_remote}" target="_blank" rel="noreferrer">
            <img src="{main_src}" alt="main image" onerror="this.onerror=null;this.src='{main_remote}';">
          </a>
        </div>
      </div>

      <div class="subgrid">
        <section class="subpanel">
          <h3>Workflow URLs</h3>
          <table>
            <thead><tr><th>type</th><th>url</th></tr></thead>
            <tbody>{url_rows_html}</tbody>
          </table>
        </section>
        <section class="subpanel">
          <h3>Visited URLs</h3>
          <table>
            <thead><tr><th>stage</th><th>url</th><th>title</th><th>note</th></tr></thead>
            <tbody>{visit_rows}</tbody>
          </table>
        </section>
      </div>

      <h3>Selling Points</h3>
      <table>
        <thead><tr><th>point</th><th>citation</th></tr></thead>
        <tbody>{point_rows}</tbody>
      </table>

      <details open>
        <summary>Text Evidence ({len(text_blocks)})</summary>
        <table>
          <thead><tr><th>source_ref</th><th>content</th></tr></thead>
          <tbody>{text_rows}</tbody>
        </table>
      </details>

      <h3>Images Sent To Gemini</h3>
      <div class="gallery">{''.join(used_image_cards) or '<div class="empty">No image evidence was sent to Gemini.</div>'}</div>
      {unused_section}
      {noise_section}
    </section>
    """


def _render_debug_calls(debug_records: list[tuple[Path, dict[str, Any]]]) -> str:
    sections: list[str] = []
    for index, (path, payload) in enumerate(sorted(debug_records, key=_record_sort_key), start=1):
        meta = payload.get("debug_meta", {})
        if not isinstance(meta, dict):
            meta = {}
        stage = str(meta.get("stage", "") or payload.get("model", "") or path.name)
        prompt_text = _extract_prompt_text(payload)
        response_text = str(payload.get("response_text", "") or "")
        refs = _extract_image_refs(payload)
        ref_chips = "".join(f'<span class="chip chip-soft">{escape(ref)}</span>' for ref in refs)
        meta_rows = [
            f"<div><strong>ts</strong>: {escape(str(payload.get('ts', '') or ''))}</div>",
            f"<div><strong>model</strong>: {escape(str(payload.get('model', '') or ''))}</div>",
            f"<div><strong>backend</strong>: {escape(str(payload.get('backend', '') or ''))}</div>",
            f"<div><strong>stage</strong>: {escape(stage)}</div>",
            f"<div><strong>item_id</strong>: {escape(str(meta.get('item_id', '') or ''))}</div>",
            f"<div><strong>workbook_id</strong>: <code>{escape(str(meta.get('workbook_id', '') or ''))}</code></div>",
            f"<div><strong>task_id</strong>: <code>{escape(str(meta.get('task_id', '') or ''))}</code></div>",
            f"<div><strong>api_key</strong>: {escape(str(payload.get('api_key_alias', '') or ''))} / {escape(str(payload.get('api_key_mask', '') or ''))}</div>",
            f"<div><strong>has_binary_parts</strong>: {escape(str(payload.get('has_binary_parts', False)))}</div>",
            f"<div><strong>source_file</strong>: <code>{escape(str(path.name))}</code></div>",
        ]
        sections.append(
            f"""
            <section class="panel record" id="call-{index}">
              <div class="record-head">
                <div>
                  <div class="section-kicker">Step {index}</div>
                  <h3>{escape(_stage_label(stage))}</h3>
                </div>
                <div class="chips">{ref_chips}</div>
              </div>
              <div class="meta-grid compact">{''.join(meta_rows)}</div>
              <div class="io-grid">
                <section class="subpanel">
                  <h4>Input Prompt</h4>
                  <pre>{escape(prompt_text)}</pre>
                </section>
                <section class="subpanel">
                  <h4>Output</h4>
                  <pre>{escape(response_text)}</pre>
                </section>
              </div>
            </section>
            """
        )
    return "".join(sections) or '<div class="empty">No Gemini debug records found.</div>'


def _render_html(
    *,
    bundle_dir: Path,
    html_path: Path,
    analysis_json_path: Path,
    report_copy_path: Path,
    task_payload: dict[str, Any],
    task_result: dict[str, Any],
    product_rows: list[dict[str, str]],
    item_payloads: list[dict[str, Any]],
    debug_records: list[tuple[Path, dict[str, Any]]],
) -> None:
    timings = task_result.get("run_summary", {}).get("timings_sec", {})
    call_counts = Counter(str(payload.get("model", "") or "") for _, payload in debug_records)
    task_inputs = _infer_task_inputs(task_payload, task_result, product_rows)
    item_sections = [
        _render_item_section(
            row=item_payload["row"],
            selling_points=item_payload["selling_points"],
            asset_map=item_payload["asset_map"],
            debug_records=debug_records,
        )
        for item_payload in item_payloads
    ]
    chips = "".join(
        f'<span class="chip">{escape(model)}: {count}</span>'
        for model, count in sorted(call_counts.items())
    )
    nav_links = "".join(
        f'<a class="nav-chip" href="#item-{escape(str(item_payload["row"].get("item_id", "") or ""))}">{escape(str(item_payload["row"].get("item_id", "") or ""))}</a>'
        for item_payload in item_payloads
        if str(item_payload["row"].get("item_id", "") or "").strip()
    )
    nav_links += '<a class="nav-chip" href="#gemini-calls">Gemini Calls</a>'
    input_url_rows = "".join(
        f"<tr><td>input_url</td><td><a href=\"{escape(url)}\" target=\"_blank\" rel=\"noreferrer\">{escape(_short_url_label(url))}</a></td></tr>"
        for url in task_inputs["input_urls"]
    )
    if task_inputs["search_target_url"]:
        input_url_rows += (
            f"<tr><td>search_target_url</td><td><a href=\"{escape(task_inputs['search_target_url'])}\" target=\"_blank\" rel=\"noreferrer\">"
            f"{escape(_short_url_label(task_inputs['search_target_url']))}</a></td></tr>"
        )
    if not input_url_rows:
        input_url_rows = '<tr><td colspan="2">No workflow URLs recorded at task level.</td></tr>'
    login_events = task_result.get("login_recovery_events", [])
    if not isinstance(login_events, list):
        login_events = []
    login_rows = "".join(
        f"<tr><td>{escape(str(event.get('source', '') or ''))}</td>"
        f"<td>{escape(str(event.get('stage', '') or ''))}</td>"
        f"<td><a href=\"{escape(str(event.get('current_url', '') or ''))}\" target=\"_blank\" rel=\"noreferrer\">{escape(_short_url_label(str(event.get('current_url', '') or '')))}</a></td>"
        f"<td>{escape(str(event.get('blocked_reason', '') or ''))}</td></tr>"
        for event in login_events
        if isinstance(event, dict)
    ) or '<tr><td colspan="4">No login recovery events.</td></tr>'
    path_rows = "".join(
        [
            f"<tr><td>bundle_dir</td><td><code>{escape(str(bundle_dir.resolve()))}</code></td></tr>",
            f"<tr><td>index_html</td><td><code>{escape(str(html_path.resolve()))}</code></td></tr>",
            f"<tr><td>analysis_json</td><td><code>{escape(str(analysis_json_path.resolve()))}</code></td></tr>",
            f"<tr><td>report_html</td><td><code>{escape(str(report_copy_path.resolve()))}</code></td></tr>",
            f"<tr><td>run_log_path</td><td><code>{escape(str(task_result.get('run_log_path', '') or ''))}</code></td></tr>",
        ]
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(str(task_result.get("platform", "") or "market"))} Gemini Debug</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf3;
      --ink: #2e241c;
      --muted: #6e5a49;
      --line: #d9c8b0;
      --accent: #a6313d;
      --accent-soft: #f2d8dc;
      --used: #2f6f4f;
      --unused: #8a6d3b;
      --code: #f3e6d7;
      --shadow: 0 10px 30px rgba(63, 38, 15, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Microsoft YaHei UI", "PingFang SC", sans-serif; color: var(--ink); background: linear-gradient(180deg, #f9f3ea, #efe5d6); }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3, h4 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 24px; margin-top: 10px; }}
    h3 {{ font-size: 18px; margin-top: 18px; }}
    h4 {{ font-size: 15px; color: var(--muted); }}
    a {{ color: var(--accent); }}
    .panel, .subpanel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); }}
    .panel {{ padding: 20px 22px; margin-bottom: 18px; }}
    .subpanel {{ padding: 16px 18px; }}
    .section-kicker {{ display: inline-flex; padding: 4px 10px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-size: 12px; font-weight: 700; margin-bottom: 10px; }}
    .hero {{ display: grid; grid-template-columns: 1.25fr 0.9fr; gap: 18px; align-items: start; }}
    .hero img {{ width: 100%; border-radius: 16px; border: 1px solid var(--line); background: #fff; min-height: 240px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 20px; }}
    .meta-grid.compact {{ grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 12px; }}
    .meta-grid code {{ background: var(--code); padding: 2px 6px; border-radius: 6px; }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
    .chip, .nav-chip {{ border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 6px 10px; font-size: 13px; text-decoration: none; color: var(--ink); }}
    .chip-soft {{ background: var(--accent-soft); border-color: transparent; color: var(--accent); }}
    .subgrid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin: 18px 0; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }}
    .card img {{ display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: cover; background: #f3ede4; }}
    .card-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 12px 14px; border-bottom: 1px solid var(--line); }}
    .meta {{ padding: 12px 14px; font-size: 13px; color: var(--muted); word-break: break-all; }}
    .badge {{ display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .badge.used {{ background: rgba(47,111,79,.14); color: var(--used); }}
    .badge.unused {{ background: rgba(138,109,59,.14); color: var(--unused); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 10px 12px; text-align: left; vertical-align: top; }}
    th {{ background: #f3e6d7; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: var(--code); padding: 14px; border-radius: 12px; border: 1px solid var(--line); overflow-x: auto; min-height: 220px; }}
    details summary {{ cursor: pointer; font-weight: 700; margin-bottom: 10px; }}
    .record-head {{ display: flex; justify-content: space-between; align-items: start; gap: 16px; }}
    .io-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 14px; }}
    .sticky-nav {{ position: static; z-index: auto; backdrop-filter: none; }}
    .empty {{ padding: 16px; border: 1px dashed var(--line); border-radius: 14px; color: var(--muted); background: rgba(255,255,255,.6); }}
    @media (max-width: 1080px) {{
      .hero, .subgrid, .io-grid {{ grid-template-columns: 1fr; }}
      .meta-grid, .meta-grid.compact {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel sticky-nav">
      <div class="section-kicker">Debug Overview</div>
      <h1>Gemini Debug Workflow</h1>
      <div class="meta-grid">
        <div><strong>task_id</strong>: <code>{escape(str(task_payload.get("task_id", "") or task_result.get("task_id", "")))}</code></div>
        <div><strong>workbook_id</strong>: <code>{escape(str(task_payload.get("workbook_id", "") or task_result.get("workbook_id", "")))}</code></div>
        <div><strong>platform</strong>: {escape(str(task_inputs.get("platform", "") or ""))}</div>
        <div><strong>source_mode</strong>: {escape(str(task_inputs.get("source_mode", "") or ""))}</div>
        <div><strong>keyword</strong>: {escape(str(task_inputs.get("keyword", "") or ""))}</div>
        <div><strong>items</strong>: {escape(str(len(product_rows)))}</div>
        <div><strong>gemini_calls</strong>: {escape(str(len(debug_records)))}</div>
        <div><strong>timings</strong>: search {escape(str(timings.get("search", "")))}s / crawl {escape(str(timings.get("crawl", "")))}s / llm_extract {escape(str(timings.get("llm_extract", "")))}s / llm_analyze {escape(str(timings.get("llm_analyze", "")))}s / export {escape(str(timings.get("export", "")))}s</div>
      </div>
      <div class="chips">{chips}{nav_links}</div>
    </section>

    <section class="panel">
      <div class="section-kicker">Workflow</div>
      <h2>Task Inputs And Access Chain</h2>
      <div class="subgrid">
        <section class="subpanel">
          <h3>Task URLs</h3>
          <table>
            <thead><tr><th>type</th><th>url</th></tr></thead>
            <tbody>{input_url_rows}</tbody>
          </table>
        </section>
        <section class="subpanel">
          <h3>Login Recovery Events</h3>
          <table>
            <thead><tr><th>source</th><th>stage</th><th>url</th><th>reason</th></tr></thead>
            <tbody>{login_rows}</tbody>
          </table>
        </section>
      </div>
    </section>

    {''.join(item_sections)}

    <section class="panel" id="gemini-calls">
      <div class="section-kicker">LLM Workflow</div>
      <h2>Gemini Calls</h2>
      {_render_debug_calls(debug_records)}
    </section>

    <section class="panel">
      <div class="section-kicker">Paths</div>
      <h2>Artifact Paths</h2>
      <table>
        <thead><tr><th>type</th><th>path</th></tr></thead>
        <tbody>{path_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def build_debug_bundle(
    *,
    data_dir: str | Path,
    debug_dir: str | Path,
    workbook_id: str,
    task_id: str,
    bundle_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Path]:
    data_path = Path(data_dir).resolve()
    debug_path = Path(debug_dir).resolve()
    bundle_path = Path(bundle_dir).resolve()
    bundle_path.mkdir(parents=True, exist_ok=True)

    task_payload = _load_task(data_path, task_id)
    task_result = task_payload.get("result", {}) if isinstance(task_payload, dict) else {}
    if not isinstance(task_result, dict):
        task_result = {}
    product_rows = _load_products(data_path, workbook_id)
    selling_points_map = _load_workbook_selling_points(data_path, workbook_id)
    debug_records = _load_debug_records(debug_path)

    item_payloads: list[dict[str, Any]] = []
    for row in product_rows:
        item_id = row.get("item_id", "")
        selling_points = selling_points_map.get(item_id, {})
        if not isinstance(selling_points, dict):
            selling_points = {}
        detail_blocks = selling_points.get("detail_blocks", [])
        if not isinstance(detail_blocks, list):
            detail_blocks = []
        image_blocks = [
            block for block in detail_blocks if str(block.get("source_type", "")).lower() == "image"
        ]
        asset_map = _ensure_item_assets(bundle_path, item_id, image_blocks)
        item_payloads.append(
            {
                "row": row,
                "selling_points": selling_points,
                "asset_map": asset_map,
            }
        )

    report_html_path = _guess_export_html_path(data_path, task_result, task_payload, product_rows)
    report_copy_path = bundle_path / "report.html"
    if report_html_path and report_html_path.exists():
        shutil.copyfile(report_html_path, report_copy_path)
    else:
        report_copy_path.write_text("", encoding="utf-8")

    analysis_json_path = bundle_path / "analysis_result.json"
    analysis_payload = {
        "task": task_payload,
        "task_result": task_result,
        "products": product_rows,
        "items": [
            {
                "row": item_payload["row"],
                "selling_points": item_payload["selling_points"],
                "asset_map": item_payload["asset_map"],
            }
            for item_payload in item_payloads
        ],
        "debug_records": [
            {"filename": path.name, "payload": payload}
            for path, payload in debug_records
        ],
        "bundle": {
            "bundle_dir": str(bundle_path.resolve()),
            "report_html": str(report_copy_path.resolve()),
            "assets_dir": str((bundle_path / "assets").resolve()),
        },
    }
    analysis_json_path.write_text(
        json.dumps(analysis_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    html_output = Path(output_path).resolve() if output_path else bundle_path / "index.html"
    _render_html(
        bundle_dir=bundle_path,
        html_path=html_output,
        analysis_json_path=analysis_json_path,
        report_copy_path=report_copy_path,
        task_payload=task_payload,
        task_result=task_result,
        product_rows=product_rows,
        item_payloads=item_payloads,
        debug_records=debug_records,
    )
    return {
        "bundle_dir": bundle_path,
        "html": html_output,
        "analysis_json": analysis_json_path,
        "report_html": report_copy_path,
    }


def rebuild_debug_bundle_index(bundle_dir: str | Path) -> Path:
    bundle_path = Path(bundle_dir).resolve()
    analysis_json_path = bundle_path / "analysis_result.json"
    payload = _read_json(analysis_json_path)
    data_path = bundle_path.parent.parent

    task_payload = payload.get("task", {})
    if not isinstance(task_payload, dict):
        task_payload = {}
    task_result = payload.get("task_result", {})
    if not isinstance(task_result, dict):
        task_result = {}

    if "product" in payload and "selling_points" in payload:
        product = payload.get("product", {})
        if not isinstance(product, dict):
            product = {}
        selling_points = payload.get("selling_points", {})
        if not isinstance(selling_points, dict):
            selling_points = {}
        flash_inputs_raw = payload.get("flash_inputs", [])
        flash_inputs = flash_inputs_raw if isinstance(flash_inputs_raw, list) else []
        asset_map: dict[str, dict[str, str]] = {}
        for row in flash_inputs:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref", "") or "")
            if not ref:
                continue
            local_relative = str(row.get("local_relative", "") or "")
            local_absolute = str(row.get("local_absolute", "") or "")
            remote_url = str(row.get("remote_url", "") or "")
            filename = Path(local_relative).name or f"{ref}.img"
            asset_map[ref] = {
                "filename": filename,
                "relative_path": local_relative,
                "absolute_path": local_absolute,
                "remote_url": remote_url,
            }
        product_rows = [{key: str(value or "") for key, value in product.items()}] if product else []
        item_payloads = []
        if product_rows:
            item_payloads.append(
                {
                    "row": product_rows[0],
                    "selling_points": selling_points,
                    "asset_map": asset_map,
                }
            )
        item_id = str(product.get("item_id", "") or "")
        workbook_id = str(task_result.get("workbook_id", "") or product.get("workbook_id", ""))
        task_id = str(task_result.get("task_id", "") or "")
        pseudo_record = {
            "ts": str(task_result.get("run_summary", {}).get("updated_at", "") or ""),
            "backend": "bundle.legacy",
            "model": "gemini-flash-latest",
            "api_key_alias": "legacy",
            "api_key_mask": "",
            "debug_meta": {
                "stage": "llm_extract_images",
                "item_id": item_id,
                "workbook_id": workbook_id,
                "task_id": task_id,
            },
            "has_binary_parts": bool(flash_inputs),
            "text_parts": [
                "Legacy debug bundle: full Gemini prompt/response was not preserved in analysis_result.json. "
                "This section only records which image assets were sent to Gemini."
            ],
            "contents": [
                {"kind": "text", "text": f"{str(row.get('ref', '') or '')}:"}
                for row in flash_inputs
                if isinstance(row, dict) and str(row.get("ref", "") or "")
            ],
            "response_text": json.dumps(
                [
                    {
                        "ref": str(row.get("ref", "") or ""),
                        "byte_length": int(row.get("byte_length", 0) or 0),
                        "remote_url": str(row.get("remote_url", "") or ""),
                        "used_count": int(row.get("used_count", 0) or 0),
                    }
                    for row in flash_inputs
                    if isinstance(row, dict)
                ],
                ensure_ascii=False,
                indent=2,
            ),
        }
        html_path = bundle_path / "index.html"
        _render_html(
            bundle_dir=bundle_path,
            html_path=html_path,
            analysis_json_path=analysis_json_path,
            report_copy_path=bundle_path / "report.html",
            task_payload=task_payload,
            task_result=task_result,
            product_rows=product_rows,
            item_payloads=item_payloads,
            debug_records=[(bundle_path / "legacy_flash_inputs.json", pseudo_record)],
        )
        return html_path

    product_rows_raw = payload.get("products", [])
    item_payloads_raw = payload.get("items", [])
    debug_records_raw = payload.get("debug_records", [])

    product_rows = [
        {key: str(value or "") for key, value in row.items()}
        for row in product_rows_raw
        if isinstance(row, dict)
    ]
    item_payloads = [
        {
            "row": item.get("row", {}) if isinstance(item.get("row", {}), dict) else {},
            "selling_points": item.get("selling_points", {}) if isinstance(item.get("selling_points", {}), dict) else {},
            "asset_map": item.get("asset_map", {}) if isinstance(item.get("asset_map", {}), dict) else {},
        }
        for item in item_payloads_raw
        if isinstance(item, dict)
    ]
    debug_records: list[tuple[Path, dict[str, Any]]] = []
    if isinstance(debug_records_raw, list):
        for record in debug_records_raw:
            if not isinstance(record, dict):
                continue
            filename = str(record.get("filename", "") or "")
            inner_payload = record.get("payload", {})
            if not filename or not isinstance(inner_payload, dict):
                continue
            debug_records.append((bundle_path / filename, inner_payload))

    workbook_id = str(task_payload.get("workbook_id", "") or task_result.get("workbook_id", ""))
    if not product_rows and workbook_id:
        product_rows = _load_products(data_path, workbook_id)

    if not item_payloads and workbook_id:
        selling_points_map = _load_workbook_selling_points(data_path, workbook_id)
        for row in product_rows:
            item_id = str(row.get("item_id", "") or "")
            selling_points = selling_points_map.get(item_id, {})
            if not isinstance(selling_points, dict):
                selling_points = {}
            detail_blocks = selling_points.get("detail_blocks", [])
            if not isinstance(detail_blocks, list):
                detail_blocks = []
            image_blocks = [
                block for block in detail_blocks if str(block.get("source_type", "")).lower() == "image"
            ]
            item_payloads.append(
                {
                    "row": row,
                    "selling_points": selling_points,
                    "asset_map": _ensure_item_assets(bundle_path, item_id, image_blocks),
                }
            )

    html_path = bundle_path / "index.html"
    _render_html(
        bundle_dir=bundle_path,
        html_path=html_path,
        analysis_json_path=analysis_json_path,
        report_copy_path=bundle_path / "report.html",
        task_payload=task_payload,
        task_result=task_result,
        product_rows=product_rows,
        item_payloads=item_payloads,
        debug_records=debug_records,
    )
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a UTF-8 Gemini debug bundle.")
    parser.add_argument("--debug-dir", required=True)
    parser.add_argument("--workbook-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    bundle = build_debug_bundle(
        data_dir=args.data_dir,
        debug_dir=args.debug_dir,
        workbook_id=args.workbook_id,
        task_id=args.task_id,
        bundle_dir=args.bundle_dir,
        output_path=args.output or None,
    )
    print(str(bundle["html"]))


if __name__ == "__main__":
    main()
