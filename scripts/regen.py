import sys
from pathlib import Path


def main():
    # Use relative path to scripts directory
    root = Path(__file__).parent
    source_file = root / "run_pipeline.py"
    with open(source_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # --- SCRAPER.PY ---
    scraper_imports = """\"\"\"Web scraping and data extraction for Taobao/Tmall.\"\"\"

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from html import escape as html_escape, unescape as html_unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, quote_plus, unquote, urlencode, urljoin, urlparse

from analysis import SellingPointExtractor
from config import (
    ANTI_BOT_MARKERS,
    ANY_URL_RE,
    BRAND_RE,
    ITEM_ID_IN_TEXT_RE,
    ITEM_ID_RE,
    ITEM_URL_RE,
    JSON_ITEM_ID_RE,
    LINE_BREAK_RE,
    SEARCH_BLOCK_HINT,
    SHOP_NAME_RE,
    SKU_ID_RE,
    SKU_MAP_RE,
    UA,
)
from data import (
    ItemDetail,
    Storage,
    UrlRecord,
    clean_text,
    extract_candidate_item_urls,
    extract_json_object,
    has_valid_login_cookie,
    is_official_shop,
    load_valid_cookies,
    looks_like_tmall,
    normalize_brand_name,
    normalize_item_title,
    normalize_url,
    now_iso,
    parse_price_values,
    parse_sales_to_int,
    read_text_utf8_best,
    should_reuse_search_page,
)

LOG = logging.getLogger("taobao_insight")

"""
    # 697 to 742 (detect_*, load_url_lines)
    # Skip load_simple_dotenv (744-760) which is in config
    # 762 to 1107 (fetch_text_http to AsyncCdpPage)
    # 1528 to 3401 (SearchClient, Crawler)
    scraper_lines = lines[696:743] + lines[761:1107] + ["\n\n"] + lines[1527:3401]
    with open(root / "scraper.py", "w", encoding="utf-8") as f:
        f.write(scraper_imports)
        f.writelines(scraper_lines)

    # --- ANALYSIS.PY ---
    analysis_imports = """\"\"\"AI-driven analysis (Gemini), selling point extraction, and competitor comparisons.\"\"\"

from __future__ import annotations

import json
import logging
import os
import re
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
    now_iso,
    preserve_multiline_text,
    read_text_utf8_best,
)

LOG = logging.getLogger("taobao_insight")

"""
    # 3403 to 4913 (Gemini proxy, SellingPointExtractor, Analyzer)
    analysis_lines = lines[3402:4913]
    with open(root / "analysis.py", "w", encoding="utf-8") as f:
        f.write(analysis_imports)
        f.writelines(analysis_lines)

    # --- REPORT.PY ---
    report_imports = """\"\"\"CSV and HTML generation and export logic.\"\"\"

from __future__ import annotations

import csv
import logging
from html import escape as html_escape
from pathlib import Path
from typing import Any

from config import (
    EXTENDED_PRODUCT_COLUMNS,
    PRODUCT_COLUMNS,
)
from data import (
    Storage,
    csv_cell,
    now_iso,
    preserve_multiline_text,
    read_text_utf8_best,
)

LOG = logging.getLogger("taobao_insight")

"""
    # 5401 to 6082 (export_csv, export_html)
    report_lines = lines[5400:6083]
    with open(root / "report.py", "w", encoding="utf-8") as f:
        f.write(report_imports)
        f.writelines(report_lines)

    # --- PIPELINE.PY ---
    pipeline_imports = """\"\"\"Pipeline orchestration and CLI definition.\"\"\"

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
import json
from pathlib import Path
from typing import Any

from config import (
    load_simple_dotenv,
)
from data import (
    GroupService,
    ItemDetail,
    Storage,
    TaskTracker,
    UrlRecord,
    URLService,
    WorkbookService,
    clean_text,
    extract_json_object,
)
from scraper import (
    Crawler,
    SearchClient,
    load_url_lines,
)
from analysis import (
    Analyzer,
    SellingPointExtractor,
)
from report import (
    export_csv,
    export_html,
)

LOG = logging.getLogger("taobao_insight")

def _default_user_data_dir() -> str:
    from pathlib import Path
    appdata = os.getenv("APPDATA", "")
    if appdata:
        return str(Path(appdata) / "taobao_insight_profile")
    if sys.platform == "darwin":
        return str(Path.home() / "Library" / "Application Support" / "taobao_insight_profile")
    return str(Path.home() / ".config" / "taobao_insight_profile")

"""
    # 4968 to 5399 (Pipeline)
    # 6085 to 6105 (_default_storage_state_file, print_json)
    # skip _default_user_data_dir (6107-6118) because we redefine it in imports to avoid dependency issues on run_pipeline structure
    # 6120 to 6458 (CLI parser, setup_services, main)
    pipeline_lines = (
        lines[4967:5399] + ["\n\n"] + lines[6084:6106] + ["\n\n"] + lines[6120:6464]
    )

    # We must patch `sys.argv[0]` handling if necessary, but the logic should run fine.

    with open(root / "pipeline.py", "w", encoding="utf-8") as f:
        f.write(pipeline_imports)
        f.writelines(pipeline_lines)

    print("Success: Generated scraper.py, analysis.py, report.py, pipeline.py")


if __name__ == "__main__":
    main()
