"""Standalone CLI for marketplace review collection."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from config import LOGGER_NAME, load_simple_dotenv, resolved_storage_state_file, resolved_user_data_dir
from data import Storage, normalize_url
from jd_review_scraper import JDReviewCrawler
from review_service import ReviewCollectionService
from scraper import load_url_lines
from taobao_review_scraper import TaobaoReviewCrawler
from tools import cleanup_global_browser

LOG = logging.getLogger(LOGGER_NAME)


def print_json(payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))


def _default_user_data_dir(platform: str = "taobao") -> str:
    return str(resolved_user_data_dir(platform))


def _default_storage_state_file(platform: str = "taobao") -> str:
    env_key = "JD_STORAGE_STATE_FILE" if platform == "jd" else "TAOBAO_STORAGE_STATE_FILE"
    return str(resolved_storage_state_file(platform, env_key=env_key))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone review collection for Taobao/Tmall/JD item URLs or item IDs.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "data"),
        help="Data directory",
    )
    parser.add_argument(
        "--playwright-headless",
        default=os.getenv("PLAYWRIGHT_HEADLESS", "0"),
        choices=["0", "1"],
    )
    parser.add_argument(
        "--playwright-cdp-url",
        default=os.getenv(
            "PLAYWRIGHT_CDP_URL",
            os.getenv("TAOBAO_CDP_ENDPOINT", ""),
        ),
        help="Optional CDP endpoint of an existing browser session.",
    )
    parser.add_argument(
        "--manual-wait-seconds",
        type=int,
        default=max(0, int(os.getenv("MANUAL_WAIT_SECONDS", "0"))),
        help="Pause after page load to allow manual login/captcha solve before extraction",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def _add_platform_args(target: argparse.ArgumentParser, *, platform: str) -> None:
        target.add_argument("target_name")
        target.add_argument("--item-url", action="append", default=[], help="Direct item URL; repeat for multiple")
        target.add_argument("--item-id", action="append", default=[], help="Direct item ID; repeat for multiple")
        target.add_argument(
            "--item-urls-file",
            default="",
            help="Text file containing one item URL or item ID per line",
        )
        target.add_argument("--months", type=int, default=2, help="Only keep reviews within the recent N months")
        target.add_argument("--days", type=int, default=0, help="Only keep reviews within the recent N days; overrides --months when > 0")
        target.add_argument("--limit", type=int, default=100, help="Maximum reviews per item after time filtering; use 0 for no cap within the time window")
        target.add_argument("--output-dir", default="", help="Optional output directory")
        if platform == "jd":
            target.add_argument(
                "--jd-browser-mode",
                default=os.getenv("JD_BROWSER_MODE", "cdp"),
                choices=["cdp", "persistent"],
            )
            target.add_argument(
                "--jd-storage-state-file",
                default=_default_storage_state_file("jd"),
            )
            target.add_argument(
                "--jd-user-data-dir",
                default=_default_user_data_dir("jd"),
            )
            target.add_argument(
                "--jd-manual-login-timeout-sec",
                type=int,
                default=max(30, int(os.getenv("JD_MANUAL_LOGIN_TIMEOUT_SEC", "300"))),
            )
        else:
            target.add_argument(
                "--taobao-browser-mode",
                default=os.getenv("TAOBAO_BROWSER_MODE", "cdp"),
                choices=["cdp", "persistent"],
            )
            target.add_argument(
                "--taobao-storage-state-file",
                default=_default_storage_state_file(),
            )
            target.add_argument(
                "--taobao-user-data-dir",
                default=_default_user_data_dir(),
            )
            target.add_argument(
                "--taobao-manual-login-timeout-sec",
                type=int,
                default=max(30, int(os.getenv("TAOBAO_MANUAL_LOGIN_TIMEOUT_SEC", "300"))),
            )

    _add_platform_args(sub.add_parser("taobao-reviews"), platform="taobao")
    _add_platform_args(sub.add_parser("jd-reviews"), platform="jd")
    return parser


def _resolve_targets(
    *,
    platform: str,
    item_urls: list[str],
    item_ids: list[str],
    item_urls_file: str,
) -> list[Any]:
    raw_inputs = list(item_urls or [])
    raw_inputs.extend(load_url_lines(item_urls_file or None))
    raw_inputs.extend([str(value or "").strip() for value in (item_ids or []) if str(value or "").strip()])
    if not raw_inputs:
        raise ValueError("provide at least one --item-url, --item-id, or --item-urls-file")

    default_platform = "jd" if platform == "jd" else "taobao"
    allowed_platforms = {default_platform}
    records = []
    seen: set[str] = set()
    for raw in raw_inputs:
        record = normalize_url(
            raw,
            default_platform=default_platform,
            allowed_platforms=allowed_platforms,
        )
        if record is None:
            continue
        if record.item_id in seen:
            continue
        seen.add(record.item_id)
        records.append(record)
    if not records:
        raise ValueError("no valid item URLs/IDs were found")
    return records


def _build_crawler(args: argparse.Namespace, *, storage: Storage, platform: str) -> Any:
    requested_headless = args.playwright_headless == "1"
    if requested_headless:
        LOG.warning("Headless mode was requested but is forced off to reduce anti-bot risk.")
    effective_headless = False
    if platform == "jd":
        return JDReviewCrawler(
            storage=storage,
            headless=effective_headless,
            browser_mode=str(args.jd_browser_mode or "cdp").strip().lower(),
            cdp_url=args.playwright_cdp_url or "",
            manual_wait_seconds=max(0, int(args.manual_wait_seconds)),
            storage_state_file=args.jd_storage_state_file,
            user_data_dir=args.jd_user_data_dir,
            manual_login_timeout_sec=max(30, int(args.jd_manual_login_timeout_sec)),
        )
    return TaobaoReviewCrawler(
        storage=storage,
        headless=effective_headless,
        browser_mode=str(args.taobao_browser_mode or "cdp").strip().lower(),
        cdp_url=args.playwright_cdp_url or "",
        manual_wait_seconds=max(0, int(args.manual_wait_seconds)),
        storage_state_file=args.taobao_storage_state_file,
        user_data_dir=args.taobao_user_data_dir,
        manual_login_timeout_sec=max(30, int(args.taobao_manual_login_timeout_sec)),
    )


async def _run_async(args: argparse.Namespace) -> dict[str, Any]:
    platform = "jd" if args.command == "jd-reviews" else "taobao"
    storage = Storage(data_dir=args.data_dir)
    targets = _resolve_targets(
        platform=platform,
        item_urls=list(args.item_url or []),
        item_ids=list(args.item_id or []),
        item_urls_file=args.item_urls_file or "",
    )
    crawler = _build_crawler(args, storage=storage, platform=platform)
    service = ReviewCollectionService(crawler, platform=platform, data_dir=storage.data_dir)
    return await service.run_async(
        target_name=args.target_name,
        targets=targets,
        months=max(1, int(args.months)),
        days=max(0, int(args.days)),
        limit=int(args.limit),
        output_dir=args.output_dir or None,
    )


def main(argv: list[str] | None = None) -> int:
    load_simple_dotenv(Path(".env"))
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        payload = asyncio.run(_run_async(args))
        print_json(payload)
        return 0 if bool(payload.get("ok", False)) else 2
    except Exception as exc:
        LOG.error("%s: %s", type(exc).__name__, exc)
        LOG.debug("Traceback:\n%s", traceback.format_exc())
        print_json({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        return 2
    finally:
        try:
            asyncio.run(cleanup_global_browser())
        except Exception as cleanup_exc:
            LOG.warning("Browser cleanup failed (non-critical): %s", cleanup_exc)


if __name__ == "__main__":
    raise SystemExit(main())
