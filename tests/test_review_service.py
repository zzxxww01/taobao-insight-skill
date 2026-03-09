import asyncio
import shutil
import unittest
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data import Storage, normalize_url
from review_common import filter_and_limit_reviews, utc_now_local
from review_models import ReviewItemResult, ReviewRecord
from review_service import ReviewCollectionService


class _FakeCrawler:
    def __init__(self, result: ReviewItemResult) -> None:
        self._result = result

    async def collect_reviews_async(self, **kwargs):
        _ = kwargs
        return self._result


class ReviewServiceTests(unittest.TestCase):
    def test_filter_and_limit_reviews_keeps_recent_sorted(self) -> None:
        now = utc_now_local()
        cutoff = now.replace(month=max(1, now.month), day=1, hour=0, minute=0, second=0, microsecond=0)
        records = [
            ReviewRecord(
                platform="jd",
                item_id="1",
                item_url="https://item.jd.com/1.html",
                comment_id="old",
                comment_time=(cutoff.replace(year=cutoff.year - 1)).isoformat(timespec="seconds"),
                comment_text="old review",
            ),
            ReviewRecord(
                platform="jd",
                item_id="1",
                item_url="https://item.jd.com/1.html",
                comment_id="new-1",
                comment_time=now.isoformat(timespec="seconds"),
                comment_text="new review 1",
            ),
            ReviewRecord(
                platform="jd",
                item_id="1",
                item_url="https://item.jd.com/1.html",
                comment_id="new-2",
                comment_time=(now.replace(hour=max(0, now.hour - 1))).isoformat(timespec="seconds"),
                comment_text="new review 2",
            ),
        ]
        filtered = filter_and_limit_reviews(records, cutoff=cutoff, limit=1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].comment_id, "new-1")

    def test_review_service_exports_jsonl_csv_and_summary(self) -> None:
        tmpdir = ROOT / f"review_test_service_{uuid.uuid4().hex[:8]}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            storage = Storage(data_dir=tmpdir)
            record = normalize_url(
                "https://item.jd.com/8142476.html",
                default_platform="jd",
                allowed_platforms={"jd"},
            )
            assert record is not None
            result = ReviewItemResult(
                platform="jd",
                item_id="8142476",
                item_url="https://item.jd.com/8142476.html",
                title="YSL Lipstick",
                shop_name="JD Self Operated",
                brand="YSL",
                reviews=[
                    ReviewRecord(
                        platform="jd",
                        item_id="8142476",
                        item_url="https://item.jd.com/8142476.html",
                        title="YSL Lipstick",
                        shop_name="JD Self Operated",
                        brand="YSL",
                        comment_id="r1",
                        comment_time=utc_now_local().isoformat(timespec="seconds"),
                        comment_text="nice",
                        collected_at=utc_now_local().isoformat(timespec="seconds"),
                    )
                ],
                collected_count=1,
                cutoff_time=utc_now_local().isoformat(timespec="seconds"),
                stopped_reason="limit_reached",
            )
            crawler = _FakeCrawler(result)
            service = ReviewCollectionService(crawler, platform="jd", data_dir=storage.data_dir)
            payload = asyncio.run(
                service.run_async(
                    target_name="lipstick",
                    targets=[record],
                    months=2,
                    days=0,
                    limit=100,
                )
            )
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["total_reviews"], 1)
            self.assertTrue(Path(payload["jsonl_path"]).exists())
            self.assertTrue(Path(payload["csv_path"]).exists())
            self.assertTrue(Path(payload["run_summary_json_path"]).exists())
            self.assertTrue(Path(payload["run_summary_md_path"]).exists())
            output_dir = Path(payload["output_dir"])
            self.assertEqual(output_dir.parent.name, "direct")
            self.assertEqual(output_dir.parent.parent.name, "jd")
            self.assertTrue(output_dir.name.startswith("8142476-m2-n100-"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
