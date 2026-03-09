import asyncio
import csv
import datetime as dt
import json
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analysis import Analyzer, SellingPointExtractor
from generate_gemini_debug_html import build_debug_bundle, rebuild_debug_bundle_index
from pipeline import (
    Pipeline,
    _default_debug_bundle_dir,
    _default_run_basename,
    _safe_debug_slug,
)
from scraper import Crawler
from data import normalize_url
from tools.login_rules import detect_non_product_page
from tools.taobao_login import LoginDecision, TaobaoLogin


class _FixedDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 3, 9, 12, 0, 0)
        if tz is not None:
            return value.replace(tzinfo=tz)
        return value


def _workspace_tmpdir(prefix: str) -> Path:
    path = ROOT / f"{prefix}{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TaobaoImageFilterTests(unittest.TestCase):
    def test_rejects_promo_or_service_detail_images(self) -> None:
        self.assertFalse(
            Crawler._is_relevant_detail_dom_image_row(
                {
                    "src": "https://img.alicdn.com/imgextra/test-1.jpg",
                    "alt": "",
                    "class_name": "detail-image",
                    "width": 790,
                    "height": 890,
                    "context_text": "88VIP 会员礼券礼遇 官方旗舰店",
                }
            )
        )
        self.assertFalse(
            Crawler._is_relevant_detail_dom_image_row(
                {
                    "src": "https://img.alicdn.com/imgextra/test-2.jpg",
                    "alt": "",
                    "class_name": "detail-image",
                    "width": 790,
                    "height": 890,
                    "context_text": "顺丰速达 7天无理由退货 服务保障",
                }
            )
        )
        self.assertTrue(
            Crawler._is_relevant_detail_dom_image_row(
                {
                    "src": "https://img.alicdn.com/imgextra/test-3.jpg",
                    "alt": "产品详情",
                    "class_name": "detail-image",
                    "width": 790,
                    "height": 1139,
                    "context_text": "151红茶烟棕色 细腻缎光质地",
                }
            )
        )

    def test_trims_tail_service_banners(self) -> None:
        rows = [
            {
                "src": f"https://img.alicdn.com/imgextra/product-{idx}.jpg",
                "width": 790,
                "height": 1139,
                "alt": "",
                "class_name": "detail-image",
                "context_text": "",
            }
            for idx in range(1, 6)
        ]
        rows.extend(
            [
                {
                    "src": "https://img.alicdn.com/imgextra/service-card.jpg",
                    "width": 790,
                    "height": 890,
                    "alt": "",
                    "class_name": "detail-image",
                    "context_text": "",
                },
                {
                    "src": "https://img.alicdn.com/imgextra/banner-1.jpg",
                    "width": 2480,
                    "height": 624,
                    "alt": "",
                    "class_name": "detail-image",
                    "context_text": "",
                },
                {
                    "src": "https://img.alicdn.com/imgextra/banner-2.jpg",
                    "width": 790,
                    "height": 226,
                    "alt": "",
                    "class_name": "detail-image",
                    "context_text": "",
                },
                {
                    "src": "https://img.alicdn.com/imgextra/banner-3.jpg",
                    "width": 790,
                    "height": 226,
                    "alt": "",
                    "class_name": "detail-image",
                    "context_text": "",
                },
            ]
        )
        trimmed = Crawler._trim_non_product_tail_image_rows(rows)
        self.assertEqual(len(trimmed), 5)
        self.assertEqual(trimmed[-1]["src"], "https://img.alicdn.com/imgextra/product-5.jpg")

    def test_trims_two_tail_noise_images(self) -> None:
        rows = [
            {
                "src": f"https://img.alicdn.com/imgextra/product-{idx}.jpg",
                "width": 790,
                "height": 1139,
                "alt": "",
                "class_name": "detail-image",
                "context_text": "",
            }
            for idx in range(1, 5)
        ]
        rows.extend(
            [
                {
                    "src": "https://gw.alicdn.com/imgextra/i4/O1CN012YkS1S20pKuSLCT05_!!6000000006898-0-tps-720-280.jpg",
                    "width": 720,
                    "height": 280,
                    "alt": "",
                    "class_name": "",
                    "context_text": "",
                },
                {
                    "src": "https://gw.alicdn.com/imgextra/i1/O1CN01Hs1MVu1IDvM07YrGh_!!6000000000860-2-tps-293-143.png",
                    "width": 293,
                    "height": 143,
                    "alt": "",
                    "class_name": "",
                    "context_text": "",
                },
            ]
        )
        trimmed = Crawler._trim_non_product_tail_image_rows(rows)
        self.assertEqual(len(trimmed), 4)
        self.assertEqual(trimmed[-1]["src"], "https://img.alicdn.com/imgextra/product-4.jpg")


class GeminiBatchingTests(unittest.TestCase):
    def test_image_batches_cover_all_candidates(self) -> None:
        extractor = SellingPointExtractor(gemini_api_key="")
        extractor.gemini_image_batch_max_images = 8
        extractor.gemini_image_batch_max_bytes = 300_000
        candidates = [
            (f"image_{index}", "image/jpeg", b"x" * 40_000)
            for index in range(1, 18)
        ]
        batches = extractor._split_image_candidate_batches(candidates)
        refs = [ref for batch in batches for ref, _, _ in batch]
        self.assertEqual(refs, [row[0] for row in candidates])
        self.assertEqual([len(batch) for batch in batches], [7, 7, 3])

    def test_build_image_candidates_skips_tiny_or_review_images(self) -> None:
        extractor = SellingPointExtractor(gemini_api_key="")
        detail_blocks = [
            {
                "source_ref": "image_1",
                "source_type": "image",
                "content": "https://img.alicdn.com/imgextra/test-main.jpg",
            },
            {
                "source_ref": "image_2",
                "source_type": "image",
                "content": "https://img.alicdn.com/imgextra/test-review.jpg?review=true",
            },
            {
                "source_ref": "image_3",
                "source_type": "image",
                "content": "https://g.alicdn.com/s.gif",
            },
        ]
        def _fake_load(content: str) -> tuple[bytes, str]:
            if "s.gif" in content:
                return b"z" * 43, "image/gif"
            return b"x" * 50_000, "image/jpeg"

        with patch.object(
            SellingPointExtractor,
            "_load_image_binary",
            side_effect=_fake_load,
        ):
            candidates = extractor._build_image_candidates(detail_blocks)
        self.assertEqual([ref for ref, _, _ in candidates], ["image_1"])

    def test_service_points_are_filtered_as_promotional(self) -> None:
        self.assertTrue(SellingPointExtractor._is_promotional_point_text("顺丰速达配送服务"))
        self.assertTrue(SellingPointExtractor._is_promotional_point_text("支持7天无理由退货"))
        self.assertTrue(SellingPointExtractor._is_promotional_point_text("官方直供原装正品"))
        self.assertTrue(SellingPointExtractor._is_promotional_point_text("适合打造清透伪素颜妆容"))
        self.assertTrue(SellingPointExtractor._is_promotional_point_text("营造剔透氛围感妆效"))
        self.assertTrue(SellingPointExtractor._is_promotional_point_text("中国高端唇釉销售额第一"))
        self.assertFalse(SellingPointExtractor._is_promotional_point_text("摩登极细管设计"))


class UrlNormalizationTests(unittest.TestCase):
    def test_taobao_search_ad_url_normalizes_to_clean_item_url(self) -> None:
        rec = normalize_url(
            "https://click.simba.taobao.com/cc_im?id=564093547806&skuId=5579065060987&spm=a21n57.1.item.1&p=abc"
        )
        self.assertIsNotNone(rec)
        self.assertEqual(
            rec.normalized_url,
            "https://item.taobao.com/item.htm?id=564093547806&skuId=5579065060987",
        )

    def test_tmall_direct_detail_url_preserves_context_params(self) -> None:
        rec = normalize_url(
            "https://detail.tmall.com/item.htm?abbucket=5&id=671553774918&pisk=test&rn=abc&spm=foo&utm_source=bar"
        )
        self.assertIsNotNone(rec)
        self.assertEqual(
            rec.normalized_url,
            "https://detail.tmall.com/item.htm?abbucket=5&id=671553774918&pisk=test&rn=abc&spm=foo",
        )


class TaobaoLoginRegressionTests(unittest.TestCase):
    def test_tmall_item_title_is_not_detected_as_login_page(self) -> None:
        reason = detect_non_product_page(
            "https://detail.tmall.com/item.htm?id=671553774918",
            "YSL圣罗兰小金条口红 1936 - 天猫Tmall.com",
            "YSL圣罗兰官方旗舰店 商品详情 参数 评价",
        )
        self.assertEqual(reason, "")

    def test_cookie_change_on_login_page_keeps_waiting_until_page_recovers(self) -> None:
        class DummyContext:
            async def cookies(self, *_args, **_kwargs):
                return []

        page = AsyncMock()
        login = TaobaoLogin(DummyContext(), page, login_timeout_sec=30)
        login._evaluate_login_decision = AsyncMock(
            side_effect=[
                LoginDecision(
                    is_login_page=True,
                    has_login_cookie=False,
                    has_search_dom=False,
                    reason="title_login",
                    url="https://login.taobao.com/",
                    timestamp=time.time(),
                ),
                LoginDecision(
                    is_login_page=True,
                    has_login_cookie=True,
                    has_search_dom=False,
                    reason="title_login",
                    url="https://login.taobao.com/",
                    timestamp=time.time(),
                ),
                LoginDecision(
                    is_login_page=False,
                    has_login_cookie=True,
                    has_search_dom=False,
                    reason="non_login_default",
                    url="https://detail.tmall.com/item.htm?id=671553774918",
                    timestamp=time.time(),
                ),
            ]
        )
        login._login_cookie_fingerprint = AsyncMock(
            side_effect=["cookie-before", "cookie-before", "cookie-after", "cookie-after"]
        )

        with patch("tools.taobao_login.asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(
                login._wait_for_login("https://detail.tmall.com/item.htm?id=671553774918")
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.cookie_changed)
        self.assertEqual(login._evaluate_login_decision.await_count, 3)


class DebugBundleTests(unittest.TestCase):
    def test_safe_debug_slug_is_ascii_stable(self) -> None:
        self.assertEqual(_safe_debug_slug("口红 top1"), "kouhong-top1")
        self.assertEqual(_safe_debug_slug("jd/debug"), "jd-debug")

    def test_default_run_basename_uses_short_platform_suffix(self) -> None:
        with patch("pipeline.dt.datetime", _FixedDateTime):
            self.assertEqual(
                _default_run_basename("口红", 1, "taobao", "keyword_search"),
                "kouhong_260309_top1_tb",
            )
            self.assertEqual(
                _default_run_basename("direct-items", 1, "jd", "input_urls"),
                "direct_260309_jd",
            )

    def test_next_default_workbook_name_appends_version_suffix(self) -> None:
        tmp = _workspace_tmpdir("tmp_debug_workbook_")
        try:
            pipeline = object.__new__(Pipeline)
            pipeline.platform = "taobao"
            pipeline.storage = type(
                "StorageStub",
                (),
                {"exports_dir": Path(tmp) / "data" / "exports"},
            )()
            scope_dir = pipeline.storage.exports_dir / "taobao" / "search"
            scope_dir.mkdir(parents=True, exist_ok=True)
            (scope_dir / "kouhong_260309_top1_tb.md").write_text("", encoding="utf-8")
            (scope_dir / "kouhong_260309_top1_tb.html").write_text("", encoding="utf-8")

            with patch("pipeline.dt.datetime", _FixedDateTime):
                candidate = pipeline._next_default_workbook_name(
                    keyword="口红",
                    top_n=1,
                    source_mode="keyword_search",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(candidate, "kouhong_260309_top1_tb_1")

    def test_default_debug_bundle_dir_groups_by_platform_and_mode(self) -> None:
        target = _default_debug_bundle_dir(
            Path("data"),
            platform="jd",
            keyword="jd-debug",
            top_n=1,
            source_mode="input_urls",
        )
        self.assertEqual(target.parent.name, "direct")
        self.assertEqual(target.parent.parent.name, "jd")
        self.assertTrue(target.name.startswith("jd-debug-top1-"))

    def test_build_debug_bundle_outputs_expected_files(self) -> None:
        tmp = _workspace_tmpdir("tmp_debug_bundle_")
        try:
            root = Path(tmp)
            data_dir = root / "data"
            debug_dir = root / "debug_raw"
            bundle_dir = root / "bundle"
            exports_dir = data_dir / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            debug_dir.mkdir(parents=True, exist_ok=True)

            (exports_dir / "report.html").write_text("<html><body>report</body></html>", encoding="utf-8")
            image_path = root / "image.jpg"
            image_path.write_bytes(b"\xff\xd8\xff\xd9")

            with (data_dir / "products.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "workbook_id",
                        "workbook_name",
                        "group_code",
                        "group_name",
                        "item_id",
                        "normalized_url",
                        "raw_url",
                        "process_status",
                        "title",
                        "main_image_url",
                        "shop_name",
                        "brand",
                        "price_min",
                        "price_max",
                        "sku_count",
                        "sku_list",
                        "detail_summary",
                        "selling_points_text",
                        "selling_points_citation",
                        "competitor_analysis_text",
                        "batch_competitor_summary_text",
                        "market_summary_text",
                        "final_conclusion_text",
                        "market_tags",
                        "crawl_time",
                        "updated_at",
                        "platform",
                        "search_rank",
                        "sales_text",
                        "official_store",
                        "item_source_url",
                        "source_type",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "workbook_id": "wb_test",
                        "workbook_name": "测试",
                        "group_code": "1",
                        "group_name": "default-group",
                        "item_id": "123",
                        "normalized_url": "https://detail.tmall.com/item.htm?id=123",
                        "raw_url": "https://detail.tmall.com/item.htm?id=123",
                        "process_status": "4",
                        "title": "测试商品",
                        "main_image_url": str(image_path),
                        "shop_name": "测试店铺",
                        "brand": "测试品牌",
                        "price_min": "99.00",
                        "price_max": "99.00",
                        "sku_count": "1",
                        "sku_list": "123001|default-sku|99.00",
                        "detail_summary": "测试摘要",
                        "selling_points_text": "测试卖点",
                        "selling_points_citation": "image_1",
                        "competitor_analysis_text": "",
                        "batch_competitor_summary_text": "",
                        "market_summary_text": "",
                        "final_conclusion_text": "",
                        "market_tags": "",
                        "crawl_time": "",
                        "updated_at": "",
                        "platform": "taobao",
                        "search_rank": "1",
                        "sales_text": "",
                        "official_store": "1",
                        "item_source_url": "https://detail.tmall.com/item.htm?id=123",
                        "source_type": "official",
                    }
                )

            (data_dir / "selling_points.json").write_text(
                json.dumps(
                    {
                        "wb_test:123": {
                            "workbook_id": "wb_test",
                            "item_id": "123",
                            "points": [{"point": "测试卖点", "citation": "image_1"}],
                            "detail_summary": "测试摘要",
                            "detail_blocks": [
                                {
                                    "source_type": "image",
                                    "source_ref": "image_1",
                                    "content": str(image_path),
                                },
                                {
                                    "source_type": "text",
                                    "source_ref": "text_1",
                                    "content": "测试文本",
                                },
                            ],
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            selling_points_payload = json.loads(
                (data_dir / "selling_points.json").read_text(encoding="utf-8")
            )
            selling_points_payload["wb_test:123"]["visit_chain"] = [
                {
                    "stage": "item_initial",
                    "url": "https://detail.tmall.com/item.htm?id=123",
                    "title": "test item",
                    "note": "initial goto",
                },
                {
                    "stage": "item_ready",
                    "url": "https://detail.tmall.com/item.htm?id=123",
                    "title": "test item",
                    "note": "detail page ready for extraction",
                },
            ]
            (data_dir / "selling_points.json").write_text(
                json.dumps(selling_points_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (data_dir / "tasks.json").write_text(
                json.dumps(
                    {
                        "task_test": {
                            "task_id": "task_test",
                            "workbook_id": "wb_test",
                            "result": {
                                "task_id": "task_test",
                                "workbook_id": "wb_test",
                                "platform": "taobao",
                                "export_html": str(exports_dir / "report.html"),
                                "run_summary": {
                                    "keyword": "口红",
                                    "timings_sec": {"search": 1, "crawl": 2, "llm_extract": 3, "llm_analyze": 4, "export": 5},
                                },
                                "final_conclusion": "测试结论",
                            },
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            task_payload = json.loads((data_dir / "tasks.json").read_text(encoding="utf-8"))
            task_payload["task_test"]["result"]["source_mode"] = "input_urls"
            task_payload["task_test"]["result"]["input_urls_used"] = [
                "https://detail.tmall.com/item.htm?id=123"
            ]
            (data_dir / "tasks.json").write_text(
                json.dumps(task_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (data_dir / "market_report.json").write_text("{}", encoding="utf-8")
            record = {
                "ts": "2026-03-08T00:00:00+08:00",
                "backend": "google.genai",
                "model": "gemini-flash-latest",
                "api_key_alias": "primary",
                "api_key_mask": "AIza...test",
                "debug_meta": {
                    "stage": "llm_extract_images",
                    "item_id": "123",
                    "workbook_id": "wb_test",
                    "task_id": "task_test",
                },
                "has_binary_parts": True,
                "text_part_count": 2,
                "text_parts": ["测试 prompt", "image_1:"],
                "contents": [
                    {"kind": "text", "text": "测试 prompt"},
                    {"kind": "text", "text": "image_1:"},
                    {"kind": "inline_data", "mime_type": "image/jpeg", "byte_length": 4, "sha256": "abcd"},
                ],
                "response_text": '[{"point":"测试卖点","citation":"image_1"}]',
                "error": "",
            }
            (debug_dir / "1.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            bundle = build_debug_bundle(
                data_dir=data_dir,
                debug_dir=debug_dir,
                workbook_id="wb_test",
                task_id="task_test",
                bundle_dir=bundle_dir,
            )
            self.assertTrue(bundle["html"].exists())
            self.assertTrue(bundle["analysis_json"].exists())
            self.assertTrue(bundle["report_html"].exists())
            self.assertTrue((bundle_dir / "assets" / "123" / "image_1.jpg").exists())
            html_text = bundle["html"].read_text(encoding="utf-8")
            self.assertIn("Workflow URLs", html_text)
            self.assertIn("Visited URLs", html_text)
            self.assertIn("https://detail.tmall.com/item.htm?id=123", html_text)
            self.assertIn("Gemini Calls", html_text)
            rebuilt_index = rebuild_debug_bundle_index(bundle_dir)
            self.assertEqual(rebuilt_index, bundle_dir / "index.html")
            self.assertTrue(rebuilt_index.exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class PromptAndSingleSampleTests(unittest.TestCase):
    def test_rejects_taobao_nav_text_noise(self) -> None:
        self.assertTrue(Crawler._is_noise_detail_text("我的淘宝 购物车 收藏夹 免费开店 千牛卖家中心"))
        self.assertTrue(Crawler._is_noise_detail_text("Language 无障碍 帮助中心"))
        self.assertFalse(Crawler._is_noise_detail_text("丝绒雾面妆效 持久显色"))

    def test_text_prompt_uses_only_text_blocks(self) -> None:
        prompt = SellingPointExtractor._build_prompt(
            {
                "platform": "taobao",
                "title": "测试口红",
                "price_text": "299.00",
                "sku_summary": "3个SKU；示例：1966暖棕红:299.00",
            },
            [
                {"source_ref": "image_1", "source_type": "image", "content": "https://img.example/a.jpg"},
                {"source_ref": "text_1", "source_type": "text", "content": "丝绒雾面妆效"},
                {"source_ref": "text_2", "source_type": "text", "content": "我的淘宝 购物车 收藏夹"},
            ],
        )
        self.assertIn("text_1", prompt)
        self.assertNotIn("image_1", prompt)
        self.assertNotIn("consumer_value", prompt)
        self.assertNotIn("scenario", prompt)
        self.assertNotIn("购物车", prompt)

    def test_validate_caps_points_per_citation(self) -> None:
        rows = SellingPointExtractor._validate(
            [
                {"point": "卖点1", "citation": "image_1"},
                {"point": "卖点2", "citation": "image_1"},
                {"point": "卖点3", "citation": "image_1"},
                {"point": "卖点4", "citation": "image_1"},
            ],
            {"image_1"},
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["point"] for row in rows], ["卖点1", "卖点2", "卖点3"])

    def test_single_sample_batch_summary_is_deterministic(self) -> None:
        analyzer = Analyzer(gemini_api_key="test-key")
        result = analyzer.generate_batch_competitor_summary(
            [{
                "item_id": "1",
                "title": "测试口红",
                "selling_points_text": "丝绒雾面妆效 | 限定礼盒包装 | 节日礼赠",
                "market_tags": "丝绒 | 礼盒",
            }],
            {"1": [{"point": "丝绒雾面妆效", "citation": "image_1"}]},
        )
        self.assertIn("单样本批量概览", result["batch_competitor_summary_text"])
        self.assertIn("测试口红", result["batch_competitor_summary_text"])
        self.assertGreaterEqual(len(result["batch_tags"]), 1)

    def test_single_sample_market_summary_is_deterministic(self) -> None:
        analyzer = Analyzer(gemini_api_key="test-key")
        result = analyzer.generate_market_summary(
            [{
                "item_id": "1",
                "title": "测试口红",
                "selling_points_text": "丝绒雾面妆效 | 限定礼盒包装",
                "price_min": "299.00",
                "market_tags": "丝绒 | 礼盒",
            }],
            {"1": [{"point": "丝绒雾面妆效", "citation": "image_1"}]},
        )
        self.assertIn("当前仅基于 1 款样本", result["summary_text"])
        self.assertIn("测试口红", result["summary_text"])
        self.assertIn("299.00", result["summary_text"])

    def test_single_sample_final_conclusion_is_deterministic(self) -> None:
        analyzer = Analyzer(gemini_api_key="test-key")
        result = analyzer.generate_final_conclusion(
            [{
                "item_id": "1",
                "title": "测试口红",
                "selling_points_text": "丝绒雾面妆效 | 限定礼盒包装 | 节日礼赠",
            }],
            "",
            "",
        )
        self.assertIn("当前仅有 1 款样本", result)
        self.assertIn("测试口红", result)


if __name__ == "__main__":
    unittest.main()
