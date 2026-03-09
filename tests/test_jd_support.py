import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data import normalize_url
from analysis import SellingPointExtractor
from jd_scraper import JDCrawler, parse_jd_search_cards_from_html
from report import ReportGenerator
from tools.jd_login import JDLogin
from tools.jd_login_rules import detect_non_product_page
from tools.taobao_login import LoginDecision

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class JDSupportTests(unittest.TestCase):
    def test_refine_price_values_drops_unit_price_outlier(self) -> None:
        prices = JDCrawler._refine_price_values([0.08, 13.65, 19.9, 59.0])
        self.assertEqual(prices, [13.65, 19.9, 59.0])

    def test_normalize_jd_item_url(self) -> None:
        rec = normalize_url(
            "https://item.jd.com/8142476.html",
            default_platform="jd",
            allowed_platforms={"jd"},
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec.platform, "jd")
        self.assertEqual(rec.item_id, "8142476")
        self.assertEqual(rec.normalized_url, "https://item.jd.com/8142476.html")

    def test_normalize_digits_for_jd(self) -> None:
        rec = normalize_url(
            "8142476",
            default_platform="jd",
            allowed_platforms={"jd"},
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec.platform, "jd")
        self.assertEqual(rec.normalized_url, "https://item.jd.com/8142476.html")

    def test_reject_jd_url_for_taobao_pipeline(self) -> None:
        rec = normalize_url(
            "https://item.jd.com/8142476.html",
            default_platform="taobao",
            allowed_platforms={"taobao"},
        )
        self.assertIsNone(rec)

    def test_parse_jd_search_cards_from_fixture(self) -> None:
        html = (FIXTURES / "jd_search_results_fixture.html").read_text(encoding="utf-8")
        cards = parse_jd_search_cards_from_html(html)
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["href"], "//item.jd.com/1001.html")
        self.assertEqual(cards[0]["shop_name"], "品牌官方旗舰店")
        self.assertIn("2万+评价", cards[0]["sales_text"])

    def test_parse_jd_search_cards_from_react_card(self) -> None:
        html = """
<div class="_wrapper_1m8y1_3 plugin_goodsCardWrapper _row_6_1m8y1_13" data-point-id="card-1" data-sku="100047236536">
  <div class="_wrapper_8g5fc_1" title="YSL圣罗兰口红香水礼盒小金条1966+自由之水50"></div>
  <div class="_goods_title_container_1g56m_1 _clip2_1g56m_14">
    <span title="YSL圣罗兰口红香水礼盒小金条1966+自由之水50" class="_text_1g56m_31">YSL圣罗兰口红香水礼盒小金条1966+自由之水50</span>
  </div>
  <div class="_goods_volume_container_1xkku_1">
    <span class="_goods_volume_1xkku_1">
      <span title="已售1万+">已售1万+</span>
      <span class="_tml_1xkku_12" title="30天加购飙升5倍">30天加购飙升5倍</span>
    </span>
  </div>
  <div class="_shopFloor_b6zo3_1">
    <span class="_name_b6zo3_45"><span>YSL圣罗兰京东自营旗舰店</span></span>
  </div>
</div>
"""
        cards = parse_jd_search_cards_from_html(html)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["href"], "https://item.jd.com/100047236536.html")
        self.assertEqual(cards[0]["title"], "YSL圣罗兰口红香水礼盒小金条1966+自由之水50")
        self.assertEqual(cards[0]["shop_name"], "YSL圣罗兰京东自营旗舰店")
        self.assertIn("已售1万+", cards[0]["sales_text"])

    def test_report_title_uses_jd_platform(self) -> None:
        title = ReportGenerator._report_title_for_platforms(
            [{"platform": "jd"}],
            {"platform": "jd"},
        )
        self.assertEqual(title, "京东市场调研报告")

    def test_report_detail_label_uses_jd_product_detail(self) -> None:
        label = ReportGenerator._detail_summary_label_for_platforms(
            [{"platform": "jd"}],
            {"platform": "jd"},
        )
        self.assertEqual(label, "商品详情卖点摘要")

    def test_detect_jd_risk_page(self) -> None:
        html = (FIXTURES / "jd_risk_page_fixture.html").read_text(encoding="utf-8")
        reason = detect_non_product_page(
            "https://search.jd.com/Search?keyword=口红",
            "京东验证",
            html,
        )
        self.assertIn("anti-bot", reason.lower())

    def test_extract_jd_color_size_payload(self) -> None:
        html = (FIXTURES / "jd_item_fixture.html").read_text(encoding="utf-8")
        payload = JDCrawler._extract_color_size_payload(html)
        self.assertEqual(len(payload), 2)
        self.assertEqual(str(payload[0]["skuId"]), "8142476")
        self.assertEqual(payload[1]["规格"], "豆沙红")

    def test_extract_jd_shop_name(self) -> None:
        html = (FIXTURES / "jd_item_fixture.html").read_text(encoding="utf-8")
        shop_name = JDCrawler._extract_shop_name(html, html)
        self.assertEqual(shop_name, "品牌京东自营旗舰店")

    def test_extract_jd_shop_name_from_script_field(self) -> None:
        html = '<script>var pageConfig = {"shopName":"YSL圣罗兰京东自营旗舰店"};</script>'
        shop_name = JDCrawler._extract_shop_name(html, "")
        self.assertEqual(shop_name, "YSL圣罗兰京东自营旗舰店")

    def test_resolve_jd_shop_name_rejects_polluted_mixed_text(self) -> None:
        shop_name = JDCrawler._resolve_shop_name(
            [
                "【YSL圣罗兰YSL小金条】YSL圣罗兰全新小金条口红1936 YSL圣罗兰京东自营旗舰店 ￥410.00 累计评价 100万+"
            ],
            [],
        )
        self.assertEqual(shop_name, "YSL圣罗兰京东自营旗舰店")

    def test_extract_jd_brand_prefers_html_brand_field(self) -> None:
        html = '<script>var pageConfig = {"brandName":"YSL圣罗兰","shopName":"YSL圣罗兰京东自营旗舰店"};</script>'
        brand = JDCrawler._extract_brand(
            "【YSL圣罗兰YSL小金条】YSL圣罗兰全新小金条口红1936",
            html,
        )
        self.assertEqual(brand, "YSL圣罗兰")

    def test_merge_jd_page_texts_keeps_unique_lines(self) -> None:
        merged = JDCrawler._merge_page_texts(
            "标题\n价格信息：￥410.00\nYSL圣罗兰京东自营旗舰店",
            "标题\n商品详情\nYSL圣罗兰京东自营旗舰店",
        )
        self.assertIn("标题", merged)
        self.assertIn("价格信息：￥410.00", merged)
        self.assertIn("商品详情", merged)
        self.assertEqual(merged.count("YSL圣罗兰京东自营旗舰店"), 1)

    def test_extract_jd_gallery_image_urls(self) -> None:
        html = """
<div id="preview">
  <img id="spec-img" data-origin="//img11.360buyimg.com/n1/s720x720_jfs/t1/test/cover.png" />
</div>
<img src="//img13.360buyimg.com/imagetools/jfs/t1/test/icon.png" />
<script src="//storage.360buyimg.com/jsresource/ws_js/jdwebm.js?v=ProDetail"></script>
<script>
  var pageConfig = {
    imageList: ["jfs/t1/test/detail-1.jpg", "jfs/t1/test/detail-2.jpg"],
    cat: [1, 2, 3]
  };
</script>
"""
        urls = JDCrawler._extract_gallery_image_urls(html)
        self.assertGreaterEqual(len(urls), 2)
        self.assertTrue(any("360buyimg.com" in url for url in urls))
        self.assertFalse(any(url.endswith(".js?v=ProDetail") for url in urls))
        self.assertFalse(any("icon.png" in url for url in urls))

    def test_pick_main_image_skips_video_candidate(self) -> None:
        image = JDCrawler._pick_main_image(
            [
                {
                    "src": "https://img10.360buyimg.com/n1/video-poster.jpg",
                    "class_name": "video poster",
                    "width": 800,
                    "height": 800,
                    "is_video": True,
                },
                {
                    "src": "https://img10.360buyimg.com/n1/jfs/t1/test/real-main.jpg",
                    "class_name": "spec-img preview",
                    "width": 800,
                    "height": 800,
                    "is_video": False,
                },
            ]
        )
        self.assertEqual(
            image,
            "https://img10.360buyimg.com/n1/jfs/t1/test/real-main.jpg",
        )

    def test_reject_jd_review_or_placeholder_image_urls(self) -> None:
        self.assertFalse(
            JDCrawler._looks_like_product_image_url(
                "https://img30.360buyimg.com/shaidan/s300x300_jfs/t1/test/review.jpg.dpg"
            )
        )
        self.assertFalse(
            JDCrawler._looks_like_product_image_url(
                "https://storage.360buyimg.com/default.image/test_sma.jpg"
            )
        )
        self.assertFalse(
            JDCrawler._looks_like_product_image_url(
                "https://storage.360buyimg.com/i.imageUpload/test_sma.jpg"
            )
        )
        self.assertFalse(
            JDCrawler._looks_like_product_image_url(
                "https://img13.360buyimg.com/imagetools/jfs/t1/test/icon.png"
            )
        )

    def test_reject_jd_ui_or_review_dom_image_rows(self) -> None:
        self.assertFalse(
            JDCrawler._is_relevant_dom_image_row(
                {
                    "src": "https://img12.360buyimg.com/imagetools/jfs/t1/test/play.png",
                    "class_name": "video-play-icon",
                    "width": 204,
                    "height": 204,
                    "is_main": False,
                    "is_video": False,
                }
            )
        )
        self.assertFalse(
            JDCrawler._is_relevant_dom_image_row(
                {
                    "src": "https://img30.360buyimg.com/shaidan/s300x300_jfs/t1/test/review.jpg.dpg",
                    "class_name": "comment-item",
                    "width": 225,
                    "height": 300,
                    "is_main": False,
                    "is_video": False,
                }
            )
        )
        self.assertTrue(
            JDCrawler._is_relevant_dom_image_row(
                {
                    "src": "https://img10.360buyimg.com/pcpubliccms/s1440x1440_jfs/t1/test/main.jpg.avif",
                    "class_name": "spec-img preview",
                    "width": 800,
                    "height": 800,
                    "is_main": True,
                    "is_video": False,
                }
            )
        )

    def test_filter_jd_video_noise_text(self) -> None:
        self.assertTrue(JDCrawler._is_noise_detail_text("0:00 / 0:16"))
        self.assertTrue(JDCrawler._is_noise_detail_text("edv7kd44c82z..."))
        self.assertTrue(JDCrawler._is_noise_detail_text("4736"))
        self.assertTrue(JDCrawler._is_noise_detail_text("超98%买家赞不绝口"))
        self.assertTrue(JDCrawler._is_noise_detail_text("兰蔻烤栗棕和圣罗兰1936哪个好看？"))
        self.assertTrue(
            JDCrawler._is_noise_detail_text(
                "被种草很久的口红，入手后真的太喜欢了！膏体细腻滋润，爱不释手！"
            )
        )
        self.assertTrue(JDCrawler._is_noise_detail_text("1936 素颜都🉑"))
        self.assertTrue(JDCrawler._is_noise_detail_text("LF182500"))
        self.assertTrue(JDCrawler._is_noise_detail_text("3年"))
        self.assertTrue(JDCrawler._is_noise_detail_text("产品（注册/备案）名称"))
        self.assertTrue(JDCrawler._is_noise_detail_text("满500减50"))
        self.assertTrue(JDCrawler._is_noise_detail_text("最高返41京豆"))
        self.assertTrue(JDCrawler._is_noise_detail_text("一键找同款，比价更省心"))
        self.assertTrue(JDCrawler._is_noise_detail_text("口红热卖榜·第1名"))
        self.assertTrue(JDCrawler._is_noise_detail_text("海外美国 ALABAMA Abbeville"))
        self.assertTrue(JDCrawler._is_noise_detail_text("查看更多4个回答"))
        self.assertFalse(JDCrawler._is_noise_detail_text("YSL圣罗兰京东自营旗舰店"))

    def test_normalize_jd_price_text_candidates(self) -> None:
        values = JDCrawler._normalize_price_text_candidates(
            [
                "￥410.00",
                "410.00",
                "￥410.00 累计评价 100万+ 降价通知",
            ],
            html='{"price":"410.00"}',
        )
        self.assertEqual(values, ["价格信息：￥410.00"])

    def test_append_jd_image_candidate_dedupes_same_asset(self) -> None:
        out: list[str] = []
        seen: set[str] = set()
        JDCrawler._append_image_candidate(
            out,
            seen,
            "https://img10.360buyimg.com/pcpubliccms/s1440x1440_jfs/t1/test/real-main.jpg.avif",
        )
        JDCrawler._append_image_candidate(
            out,
            seen,
            "https://img10.360buyimg.com/n1/jfs/t1/test/real-main.jpg.avif",
        )
        self.assertEqual(out, ["https://img10.360buyimg.com/pcpubliccms/s1440x1440_jfs/t1/test/real-main.jpg.avif"])

    def test_extract_jd_detail_api_url(self) -> None:
        html = """
<script>
  var pageConfig = {
    desc: "//api.m.jd.com/description/channel?appid=item-v3&functionId=pc_description_channel&skuId=8142476"
  };
</script>
"""
        desc_url = JDCrawler._extract_jd_detail_api_url(html)
        self.assertIn("pc_description_channel", desc_url)
        self.assertIn("skuId=8142476", desc_url)

    def test_extract_jd_desc_payload_texts_and_images(self) -> None:
        payload = (
            '{"content":"<div><p>丝绒雾面妆效</p>'
            '<img src=\\"//img10.360buyimg.com/test.jpg\\" /></div>"}'
        )
        texts, images = JDCrawler._extract_texts_and_images_from_desc_payload(
            payload,
            "https://api.m.jd.com/description/channel",
        )
        self.assertIn("丝绒雾面妆效", texts)
        self.assertEqual(images[0], "https://img10.360buyimg.com/test.jpg")

    def test_jd_prompt_uses_product_detail_context(self) -> None:
        extractor = SellingPointExtractor(gemini_api_key="")
        prompt = extractor._build_prompt(
            item_context={
                "platform": "jd",
                "title": "YSL 小金条口红",
                "brand": "YSL",
                "shop_name": "YSL京东自营旗舰店",
                "price_text": "299.00",
                "sku_summary": "1966暖棕红:299.00 | 21复古正红:299.00",
            },
            detail_blocks=[
                {
                    "source_ref": "text_1",
                    "source_type": "text",
                    "content": "丝绒雾面妆效，显色浓郁。",
                }
            ],
        )
        self.assertIn("商品详情文本证据", prompt)
        self.assertIn("price: 299.00", prompt)
        self.assertIn("sku_summary: 1966暖棕红:299.00 | 21复古正红:299.00", prompt)
        self.assertNotIn("图文详情证据", prompt)

    def test_analysis_skips_review_images_for_gemini_candidates(self) -> None:
        extractor = SellingPointExtractor(gemini_api_key="")
        detail_blocks = [
            {
                "source_ref": "image_1",
                "source_type": "image",
                "content": "https://img10.360buyimg.com/pcpubliccms/s1440x1440_jfs/t1/test/main.jpg.avif",
            },
            {
                "source_ref": "image_17",
                "source_type": "image",
                "content": "https://img30.360buyimg.com/shaidan/s300x300_jfs/t1/test/review.jpg.dpg",
            },
        ]
        with patch.object(
            SellingPointExtractor,
            "_load_image_binary",
            side_effect=[
                (b"x" * 30000, "image/jpeg"),
                (b"y" * 30000, "image/jpeg"),
            ],
        ):
            candidates = extractor._build_image_candidates(detail_blocks)
        self.assertEqual([ref for ref, _, _ in candidates], ["image_1"])


class _FakeJDContext:
    async def cookies(self, urls=None):
        return []


class _FakeJDPage:
    url = "https://passport.jd.com/new/login.aspx"

    async def bring_to_front(self) -> None:
        return None


class _SequencedJDLogin(JDLogin):
    def __init__(self, decisions, fingerprints):
        super().__init__(_FakeJDContext(), _FakeJDPage(), login_timeout_sec=30)
        self._decisions = list(decisions)
        self._fingerprints = list(fingerprints)

    async def _evaluate_login_decision(self) -> LoginDecision:
        if self._decisions:
            return self._decisions.pop(0)
        return LoginDecision(
            is_login_page=False,
            has_login_cookie=True,
            has_search_dom=True,
            reason="done",
            url="https://search.jd.com/Search?keyword=test",
            timestamp=0.0,
        )

    async def _login_cookie_fingerprint(self) -> str:
        if self._fingerprints:
            return self._fingerprints.pop(0)
        return "thor=ready"


class JDLoginBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def test_jd_login_waits_until_page_leaves_blocked_state(self) -> None:
        decisions = [
            LoginDecision(
                is_login_page=True,
                has_login_cookie=False,
                has_search_dom=False,
                reason="url_login",
                url="https://passport.jd.com/new/login.aspx",
                timestamp=1.0,
            ),
            LoginDecision(
                is_login_page=True,
                has_login_cookie=True,
                has_search_dom=False,
                reason="url_login",
                url="https://passport.jd.com/new/login.aspx",
                timestamp=2.0,
            ),
            LoginDecision(
                is_login_page=False,
                has_login_cookie=True,
                has_search_dom=True,
                reason="search_surface_ready",
                url="https://search.jd.com/Search?keyword=%E5%8F%A3%E7%BA%A2",
                timestamp=3.0,
            ),
        ]
        fingerprints = ["", "", "thor=ok", "thor=ok"]
        login = _SequencedJDLogin(decisions, fingerprints)
        login._started_at = 1.0

        with patch("tools.jd_login.asyncio.sleep", new=AsyncMock()) as mocked_sleep:
            result = await login._wait_for_login("https://passport.jd.com/new/login.aspx")

        self.assertTrue(result.ok)
        self.assertEqual(result.final_state, "SUCCESS")
        self.assertEqual(result.reason, "cookie_and_page_confirmed")
        states = [row.get("state") for row in result.decision_trace]
        self.assertIn("COOKIE_UPDATED", states)
        self.assertEqual(states.count("WAIT_VERIFICATION"), 3)
        self.assertGreaterEqual(mocked_sleep.await_count, 1)


if __name__ == "__main__":
    unittest.main()
