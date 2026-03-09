import asyncio
import datetime as dt
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data import Storage
from review_common import utc_now_local
from taobao_review_scraper import TaobaoReviewCrawler


class TaobaoReviewScraperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = ROOT / f"review_test_taobao_{uuid.uuid4().hex[:8]}"
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.storage = Storage(data_dir=self.tmpdir)
        self.crawler = TaobaoReviewCrawler(storage=self.storage, headless=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_from_api_dict_maps_rate_api_fields(self) -> None:
        row = {
            "id": "t1",
            "feedbackDate": "2026年3月4日",
            "feedback": "Great color and comfortable texture.",
            "skuValueStr": "01 Rose",
            "skuMap": {"Color": "01 Rose"},
            "reduceUserNick": "w**2",
            "creditLevel": "8",
            "feedPicList": [{"thumbnail": "//img.alicdn.com/test-rate.jpg"}],
            "rateTagList": [{"title": "smooth"}],
            "interactInfo": {"likeCount": "12", "commentCount": "3"},
            "append": {"feedbackDate": "2026年3月6日", "feedback": "Still looks good."},
        }
        record = self.crawler._record_from_api_dict(
            row,
            item_id="564093547806",
            item_url="https://detail.tmall.com/item.htm?id=564093547806",
            title="Lipstick",
            shop_name="Official Store",
            brand="YSL",
        )
        assert record is not None
        self.assertEqual(record.platform, "taobao")
        self.assertEqual(record.comment_id, "t1")
        self.assertIn("Great color", record.comment_text)
        self.assertEqual(record.user_name_masked, "w**2")
        self.assertEqual(record.reply_count, 3)
        self.assertEqual(record.like_count, 12)
        self.assertTrue(record.has_images)
        self.assertTrue(record.is_append)
        self.assertEqual(record.raw_tags, ["smooth"])
        self.assertIn("2026-03-04T23:59:59", record.comment_time)

    def test_parse_taobao_review_datetime_uses_end_of_day_for_date_only(self) -> None:
        parsed = self.crawler._parse_taobao_review_datetime("2026年3月4日")
        assert parsed is not None
        self.assertEqual(parsed.year, 2026)
        self.assertEqual(parsed.month, 3)
        self.assertEqual(parsed.day, 4)
        self.assertEqual(parsed.hour, 23)
        self.assertEqual(parsed.minute, 59)
        self.assertEqual(parsed.second, 59)

    def test_extract_reviews_from_payloads_walks_nested_json(self) -> None:
        payload = """
        {
          "data": {
            "rateList": [
              {
                "id": "t2",
                "feedbackDate": "2026年3月2日",
                "feedback": "Looks natural.",
                "reduceUserNick": "tester"
              }
            ]
          }
        }
        """
        records = self.crawler._extract_reviews_from_payloads(
            [payload],
            item_id="564093547806",
            item_url="https://detail.tmall.com/item.htm?id=564093547806",
            title="Lipstick",
            shop_name="Official Store",
            brand="YSL",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].comment_id, "t2")
        self.assertIn("Looks natural", records[0].comment_text)

    def test_collect_reviews_via_rate_api_stops_at_cutoff(self) -> None:
        page = object()
        timezone = utc_now_local().tzinfo
        cutoff = dt.datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone)
        payloads = [
            {
                "ret": ["SUCCESS::调用成功"],
                "data": {
                    "hasNext": "true",
                    "rateList": [
                        {
                            "id": "n1",
                            "feedbackDate": "2026年3月9日",
                            "feedback": "newest",
                            "reduceUserNick": "a**1",
                        },
                        {
                            "id": "n2",
                            "feedbackDate": "2026年3月8日",
                            "feedback": "recent",
                            "reduceUserNick": "b**2",
                        },
                    ],
                },
            },
            {
                "ret": ["SUCCESS::调用成功"],
                "data": {
                    "hasNext": "true",
                    "rateList": [
                        {
                            "id": "o1",
                            "feedbackDate": "2026年3月4日",
                            "feedback": "older",
                            "reduceUserNick": "c**3",
                        }
                    ],
                },
            },
        ]

        async def fake_fetch(self, page_obj, *, referer_url, request_data):
            _ = self, page_obj, referer_url, request_data
            return "mtopjsonp1(" + json.dumps(payloads.pop(0), ensure_ascii=False) + ")"

        self.crawler._fetch_rate_payload_text_async = MethodType(fake_fetch, self.crawler)
        reviews, stop_reason = asyncio.run(
            self.crawler._collect_reviews_via_rate_api_async(
                page,
                item_id="564093547806",
                item_url="https://detail.tmall.com/item.htm?id=564093547806",
                title="Lipstick",
                shop_name="Official Store",
                brand="YSL",
                cutoff=cutoff,
                limit=0,
            )
        )
        self.assertEqual(stop_reason, "reached_cutoff")
        self.assertEqual([row.comment_id for row in reviews], ["n1", "n2"])


if __name__ == "__main__":
    unittest.main()
