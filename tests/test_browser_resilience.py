import asyncio
import base64
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    platform_default_profile_name,
    platform_default_storage_state_name,
    platform_legacy_profile_names,
    platform_legacy_storage_state_names,
)
from data import Storage
from jd_scraper import JDCrawler, JDSearchClient
from scraper import SearchClient
from tools.cdp_browser import RawCdpPage, _RawCdpLocator
from tools.jd_login import JDLogin
from tools.taobao_login import LoginDecision


class _FakeContext:
    def __init__(self) -> None:
        self.calls = 0
        self.cookies = []

    async def add_cookies(self, cookies):
        self.calls += 1
        self.cookies.append(list(cookies))


class _FakeConn:
    def __init__(self) -> None:
        self.handlers = {}
        self.send_calls = []

    def on(self, method, handler) -> None:
        self.handlers.setdefault(method, set()).add(handler)

    def off(self, method, handler) -> None:
        handlers = self.handlers.get(method, set())
        handlers.discard(handler)

    async def send(self, method, params=None, *, session_id="", timeout_sec=15.0):
        _ = timeout_sec
        self.send_calls.append((method, params, session_id))
        if method == "Network.getResponseBody":
            return {
                "body": base64.b64encode(b'{"ok": true}').decode("ascii"),
                "base64Encoded": True,
            }
        return {}


class _FakeEvalPage:
    def __init__(self) -> None:
        self.expressions = []

    async def evaluate(self, expression, return_by_value=True):
        _ = return_by_value
        self.expressions.append(expression)
        if "getAttribute" in expression:
            return {"found": True, "value": "pager next"}
        return {"found": True, "value": True}


class _FakeFetchPage:
    def __init__(self) -> None:
        self.url = "https://item.jd.com/1001.html"
        self.expressions = []

    async def evaluate(self, expression, return_by_value=True):
        _ = return_by_value
        self.expressions.append(expression)
        if "fetch(" in expression:
            return '{"content":"browser-session"}'
        if "navigator.userAgent" in expression:
            return "Mozilla/5.0 Test"
        return ""


class _FakeJDContext:
    async def cookies(self, urls=None):
        _ = urls
        return []


class _FakeJDPage:
    url = "https://passport.jd.com/new/login.aspx"

    async def bring_to_front(self) -> None:
        return None


class _SequencedFingerprintJDLogin(JDLogin):
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
        return "thor=ok"


class BrowserResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_storage_state_restore_is_per_context(self) -> None:
        tmpdir = ROOT / f"tmp_storage_restore_{uuid.uuid4().hex[:8]}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        storage_state = tmpdir / "state.json"
        storage_state.write_text(
            json.dumps({"cookies": [{"name": "sid", "value": "1", "domain": ".jd.com"}]}),
            encoding="utf-8",
        )
        client = JDSearchClient(
            storage_state_file=storage_state,
            user_data_dir=tmpdir / "profile",
        )
        try:
            first = _FakeContext()
            second = _FakeContext()
            await client._restore_storage_state_async(first)
            await client._restore_storage_state_async(first)
            await client._restore_storage_state_async(second)
            self.assertEqual(first.calls, 1)
            self.assertEqual(second.calls, 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def test_raw_cdp_response_event_exposes_body_text(self) -> None:
        conn = _FakeConn()
        manager = SimpleNamespace(connection=conn, _stealth_script_source="")
        page = RawCdpPage(manager=manager, target_id="target-1", session_id="session-1")
        seen = []

        def _handler(response) -> None:
            seen.append(response)

        page.on("response", _handler)
        page._dispatch_network_response(
            {
                "sessionId": "session-1",
                "params": {
                    "requestId": "request-1",
                    "response": {
                        "url": "https://api.example.com/data",
                        "status": 200,
                    },
                },
            }
        )
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].url, "https://api.example.com/data")
        self.assertEqual(await seen[0].text(), '{"ok": true}')

    async def test_raw_cdp_locator_supports_text_and_attributes(self) -> None:
        fake_page = _FakeEvalPage()
        locator = _RawCdpLocator(fake_page, text="\u4e0b\u4e00\u9875", exact=False)
        await locator.scroll_into_view_if_needed()
        await locator.click()
        classes = await locator.get_attribute("class")
        self.assertEqual(classes, "pager next")
        self.assertTrue(any("\u4e0b\u4e00\u9875" in expression for expression in fake_page.expressions))

    async def test_jd_desc_fetch_prefers_browser_session(self) -> None:
        tmpdir = ROOT / f"tmp_jd_fetch_{uuid.uuid4().hex[:8]}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        storage = Storage(data_dir=tmpdir)
        crawler = JDCrawler(storage=storage, headless=False)
        try:
            target, text = await crawler._fetch_jd_desc_payload_text_async(
                _FakeFetchPage(),
                "/description/channel?skuId=1001",
            )
            self.assertIn("description/channel", target)
            self.assertEqual(text, '{"content":"browser-session"}')
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def test_jd_login_result_includes_cookie_fingerprint_changes(self) -> None:
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
                is_login_page=False,
                has_login_cookie=True,
                has_search_dom=True,
                reason="search_surface_ready",
                url="https://search.jd.com/Search?keyword=test",
                timestamp=2.0,
            ),
        ]
        login = _SequencedFingerprintJDLogin(decisions, ["", "", "thor=ok"])
        result = await login._wait_for_login("https://passport.jd.com/new/login.aspx")
        self.assertTrue(result.ok)
        self.assertEqual(result.cookie_fingerprint_before, "")
        self.assertEqual(result.cookie_fingerprint_after, "thor=ok")
        self.assertTrue(result.cookie_changed)

    def test_jd_classes_use_platform_specific_login_config(self) -> None:
        self.assertEqual(JDSearchClient.PLATFORM_LABEL, "JD")
        self.assertEqual(JDCrawler.PLATFORM_LABEL, "JD")
        self.assertTrue(callable(JDSearchClient.DETECT_NON_PRODUCT_PAGE_FN))
        self.assertTrue(callable(JDCrawler.DETECT_NON_PRODUCT_PAGE_FN))


class PlatformDefaultsTests(unittest.TestCase):
    def test_platform_defaults_are_isolated(self) -> None:
        self.assertEqual(
            SearchClient.DEFAULT_STORAGE_STATE_FILE,
            platform_default_storage_state_name("taobao"),
        )
        self.assertEqual(
            JDSearchClient.DEFAULT_STORAGE_STATE_FILE,
            platform_default_storage_state_name("jd"),
        )
        self.assertEqual(
            SearchClient.DEFAULT_USER_DATA_DIR_NAME,
            platform_default_profile_name("taobao"),
        )
        self.assertEqual(
            JDSearchClient.DEFAULT_USER_DATA_DIR_NAME,
            platform_default_profile_name("jd"),
        )
        taobao = SearchClient()
        jd = JDSearchClient()
        self.assertIn(
            taobao.storage_state_file.name,
            {
                platform_default_storage_state_name("taobao"),
                *platform_legacy_storage_state_names("taobao"),
            },
        )
        self.assertIn(
            jd.storage_state_file.name,
            {
                platform_default_storage_state_name("jd"),
                *platform_legacy_storage_state_names("jd"),
            },
        )
        self.assertIn(
            taobao.user_data_dir.name,
            {
                platform_default_profile_name("taobao"),
                *platform_legacy_profile_names("taobao"),
            },
        )
        self.assertIn(
            jd.user_data_dir.name,
            {
                platform_default_profile_name("jd"),
                *platform_legacy_profile_names("jd"),
            },
        )


if __name__ == "__main__":
    unittest.main()
