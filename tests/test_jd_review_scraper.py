import asyncio
import shutil
import unittest
from pathlib import Path
import sys
import uuid
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data import Storage
from jd_review_scraper import JDReviewCrawler
from review_common import looks_like_review_dict


class JDReviewScraperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = ROOT / f"review_test_jd_{uuid.uuid4().hex[:8]}"
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.storage = Storage(data_dir=self.tmpdir)
        self.crawler = JDReviewCrawler(storage=self.storage, headless=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_from_api_dict_maps_jd_review_fields(self) -> None:
        row = {
            "commentId": "c1",
            "creationTime": "2026-03-01 10:20:30",
            "content": "质地很好，颜色也正",
            "score": 5,
            "referenceName": "小金条",
            "productColor": "1966暖棕红",
            "nickname": "YSL粉丝",
            "userLevelName": "PLUS",
            "anonymousFlag": 1,
            "images": [{"imgUrl": "https://img30.360buyimg.com/test.jpg"}],
            "appendComment": {"content": "追评也很满意", "creationTime": "2026-03-03 10:20:30"},
            "tagList": ["显色", "丝滑"],
        }
        record = self.crawler._record_from_api_dict(
            row,
            item_id="8142476",
            item_url="https://item.jd.com/8142476.html",
            title="YSL 小金条",
            shop_name="YSL京东自营旗舰店",
            brand="YSL",
        )
        assert record is not None
        self.assertEqual(record.platform, "jd")
        self.assertEqual(record.comment_id, "c1")
        self.assertIn("质地很好", record.comment_text)
        self.assertTrue(record.is_append)
        self.assertTrue(record.is_anonymous)
        self.assertTrue(record.has_images)
        self.assertEqual(record.raw_tags, ["显色", "丝滑"])

    def test_extract_reviews_from_payloads_walks_nested_json(self) -> None:
        payload = """
        {
          "comments": {
            "list": [
              {
                "commentId": "c2",
                "creationTime": "2026-03-02 08:00:00",
                "content": "上嘴顺滑",
                "nickname": "tester"
              }
            ]
          }
        }
        """
        records = self.crawler._extract_reviews_from_payloads(
            [payload],
            item_id="8142476",
            item_url="https://item.jd.com/8142476.html",
            title="YSL 小金条",
            shop_name="YSL京东自营旗舰店",
            brand="YSL",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].comment_id, "c2")
        self.assertIn("上嘴顺滑", records[0].comment_text)


    def test_ensure_item_detail_ready_reloads_after_login_recovery(self) -> None:
        class FakePage:
            def __init__(self) -> None:
                self.url = "https://passport.jd.com/new/login.aspx"
                self.goto_calls: list[str] = []

            async def goto(self, url: str, **kwargs) -> None:
                _ = kwargs
                self.goto_calls.append(url)
                self.url = url

            async def wait_for_timeout(self, ms: int) -> None:
                _ = ms

            async def title(self) -> str:
                return "商品详情"

            async def inner_text(self, selector: str) -> str:
                _ = selector
                return "商品详情页"

        async def fake_handle_login(self, page, context, stage, item_id):
            _ = page, context, stage, item_id
            return True

        page = FakePage()
        self.crawler._handle_login_if_needed_async = MethodType(fake_handle_login, self.crawler)
        handled = asyncio.run(
            self.crawler._ensure_item_detail_ready_async(
                page=page,
                context=object(),
                url="https://item.jd.com/100259348596.html",
                item_id="100259348596",
                stage="review-item-initial",
            )
        )
        self.assertTrue(handled)
        self.assertEqual(page.goto_calls, ["https://item.jd.com/100259348596.html"])

    def test_ensure_item_detail_ready_raises_when_page_is_still_blocked(self) -> None:
        class FakePage:
            def __init__(self) -> None:
                self.url = "https://passport.jd.com/new/login.aspx"

            async def goto(self, url: str, **kwargs) -> None:
                _ = url, kwargs

            async def wait_for_timeout(self, ms: int) -> None:
                _ = ms

            async def title(self) -> str:
                return "京东-欢迎登录"

            async def inner_text(self, selector: str) -> str:
                _ = selector
                return "请登录 扫码登录"

        async def fake_handle_login(self, page, context, stage, item_id):
            _ = page, context, stage, item_id
            return False

        page = FakePage()
        self.crawler._handle_login_if_needed_async = MethodType(fake_handle_login, self.crawler)
        with self.assertRaises(RuntimeError):
            asyncio.run(
                self.crawler._ensure_item_detail_ready_async(
                    page=page,
                    context=object(),
                    url="https://item.jd.com/100259348596.html",
                    item_id="100259348596",
                    stage="review-item-initial",
                )
            )

    def test_looks_like_review_dict_ignores_question_answers(self) -> None:
        answer_row = {
            "systemId": "20",
            "nickName": "tester",
            "content": "我觉得这款更好看",
        }
        review_row = {
            "commentId": "103752680164929211",
            "commentDate": "2026-03-03 22:40:20",
            "commentData": "这款口红真的绝了",
            "commentScore": "5",
        }
        self.assertFalse(looks_like_review_dict(answer_row))
        self.assertTrue(looks_like_review_dict(review_row))

    def test_extract_reviews_from_comment_list_payload_maps_commentlist_list(self) -> None:
        payload = {
            "code": "0",
            "result": {
                "pageInfo": {"data": {"maxPage": "3"}},
                "floors": [
                    {
                        "mId": "commentlist-list",
                        "data": [
                            {
                                "commentInfo": {
                                    "commentId": "api-page-1",
                                    "commentDate": "2026-03-08 23:22:24",
                                    "commentData": "latest review",
                                    "commentScore": "5",
                                    "userNickName": "tester",
                                    "wareAttribute": [{"颜色": "1936"}],
                                    "pictureInfoList": [],
                                }
                            }
                        ],
                    }
                ],
            },
        }
        records = self.crawler._extract_reviews_from_comment_list_payload(
            payload,
            item_id="100259348596",
            item_url="https://item.jd.com/100259348596.html",
            title="YSL Lipstick",
            shop_name="JD Self Operated",
            brand="YSL",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].comment_id, "api-page-1")
        self.assertIn("latest review", records[0].comment_text)
        self.assertEqual(self.crawler._extract_comment_list_max_page(payload), 3)

if __name__ == "__main__":
    unittest.main()
