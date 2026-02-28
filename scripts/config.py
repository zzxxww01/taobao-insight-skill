"""Global constants, regular expressions, and column definitions."""

import logging
import os
import re
from pathlib import Path

LOG = logging.getLogger("taobao_insight")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
)

WORKBOOK_COLUMNS = ["workbook_id", "workbook_name", "created_at", "updated_at"]
BACKEND_PRODUCT_COLUMNS = [
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
]
EXTENDED_PRODUCT_COLUMNS = [
    "search_rank",
    "sales_text",
    "official_store",
    "item_source_url",
    "source_type",
]
PRODUCT_COLUMNS = BACKEND_PRODUCT_COLUMNS + EXTENDED_PRODUCT_COLUMNS
MAX_TOP_N = 100

DEFAULT_GROUP_CODE = "1"
DEFAULT_GROUP_NAME = "default-group"
TAOBAO_DOMAINS = ("taobao.com", "tmall.com")
LOGIN_COOKIE_NAMES = {"cookie2", "unb", "_tb_token_", "_m_h5_tk"}

ITEM_URL_RE = re.compile(
    r"https?://(?:item\.taobao\.com|detail\.tmall\.com)/item\.htm\?[^\"'\s<>]+",
    re.IGNORECASE,
)
ANY_URL_RE = re.compile(r"https?://[^\"'\s<>]+", re.IGNORECASE)
ITEM_ID_IN_TEXT_RE = re.compile(
    r"(?:[?&]|%3[fF]|%26)id(?:=|%3[dD])(\d{6,})", re.IGNORECASE
)
JSON_ITEM_ID_RE = re.compile(
    r'"(?:itemId|item_id|nid|itemid)"\s*:\s*"?(?P<id>\d{6,})"?', re.IGNORECASE
)
ITEM_ID_RE = re.compile(r"(?:^|[?&])id=(\d{6,})", re.IGNORECASE)
SKU_ID_RE = re.compile(r"(?:^|[?&])skuId=(\d{6,})", re.IGNORECASE)
DIGITS_ONLY_RE = re.compile(r"^\d{6,}$")
TMALL_HINT_RE = re.compile(r"(tmall\.com|tmall\.hk|detail\.tmall)", re.IGNORECASE)
OFFICIAL_SHOP_MARKER_RE = re.compile(
    r"^(?!.*百亿补贴).*(官方|旗舰店|天猫.*自营|自营.*天猫)", re.IGNORECASE
)
SALES_TEXT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(万)?\+?\s*(?:人付款|已售)", re.IGNORECASE
)
PRICE_RE = re.compile(r"(?<!\d)(\d{1,6}(?:\.\d{1,2})?)(?!\d)")
SKU_MAP_RE = re.compile(
    r'"skuId"\s*:\s*"(?P<sku_id>\d{6,})".{0,260}?"price"\s*:\s*"?(?P<price>\d+(?:\.\d{1,2})?)"?',
    re.IGNORECASE | re.DOTALL,
)
SHOP_NAME_RE = re.compile(r'"(?:shopName|sellerNick)"\s*:\s*"(?P<name>[^"]+)"')
BRAND_RE = re.compile(r'"(?:brandName|brand)"\s*:\s*"(?P<brand>[^"]+)"', re.IGNORECASE)
LINE_BREAK_RE = re.compile(r"[\r\n]+")
WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
BANNED_SHOP_MARKER_RE = re.compile(r"(?!)")
TITLE_SITE_SUFFIX_RE = re.compile(
    r"(?:[-_｜|·•\s]*(?:tmall\.com天猫|taobao\.com淘宝网|tmall\.com|taobao\.com|天猫|淘宝网))+\s*$",
    re.IGNORECASE,
)
SHOP_SUFFIX_RE = re.compile(
    r"(官方旗舰店|旗舰店|官方店|专卖店|专营店|海外旗舰店|品牌店|天猫国际官方直营)$",
    re.IGNORECASE,
)
SHOP_TRAILING_CATEGORY_RE = re.compile(
    r"(美妆|彩妆|护肤|个护|家清|母婴|宠物|官方)$",
    re.IGNORECASE,
)
PROMO_TITLE_PREFIX_RE = re.compile(r"^(?:[\[【][^\]】]{1,24}[\]】]\s*)+")
BRAND_TOKEN_SPLIT_RE = re.compile(r"[\s/|·•,_-]+")
SEARCH_QUERY_PARAM_ALIASES = {"q", "keyword", "query"}
GENERIC_BRAND_TOKENS = {
    "天猫",
    "天猫超市",
    "天猫国际",
    "天猫国际官方直营",
    "淘宝",
    "淘宝网",
    "京东",
    "拼多多",
    "猫超",
    "官方店",
    "官方旗舰店",
    "旗舰店",
    "专卖店",
    "专营店",
    "直营店",
}
GENERIC_BRAND_TOKENS_LOWER = {token.lower() for token in GENERIC_BRAND_TOKENS}
BRAND_PRODUCT_KEYWORD_RE = re.compile(
    r"(粉饼|蜜粉|散粉|定妆|补妆|控油|遮瑕|持久|清透|雾面|哑光|柔焦|磨皮|防晒|防水|防汗|锁妆|礼物|套组|不卡粉|不脱妆|空气感)",
    re.IGNORECASE,
)
SEARCH_BLOCK_HINT = (
    "Taobao search was blocked by anti-bot or login validation. "
    "Recommended: use browser-use with a signed-in real browser, or run with "
    "--playwright-cdp-url http://127.0.0.1:9222 on a signed-in Chrome session, "
    "or pass direct item URLs by --item-url / --item-urls-file."
)
ANTI_BOT_MARKERS = [
    "RGV587_ERROR",
    "FAIL_SYS_USER_VALIDATE",
    "x5secdata",
    "_____tmd_____",
    "被挤爆",
    "captcha",
    "punish",
]
TITLE_POINT_RULES: list[tuple[str, str]] = [
    ("防水防汗", "强调防水防汗场景"),
    ("不卡粉", "强调服帖不卡粉"),
    ("持久", "主打持久妆效"),
    ("定妆", "强调定妆能力"),
    ("控油", "强调控油需求"),
    ("柔焦", "强调柔焦修饰"),
    ("磨皮", "强调磨皮修饰"),
    ("隐毛孔", "强调隐形毛孔修饰"),
    ("毛孔", "强调毛孔修饰"),
    ("遮瑕", "强调遮瑕修饰"),
    ("轻透", "强调轻透妆感"),
    ("清透", "强调清透妆感"),
    ("哑光", "强调哑光妆感"),
    ("雾面", "强调雾面妆感"),
    ("防水", "强调防水场景"),
    ("防汗", "强调防汗场景"),
    ("防晒", "强调防晒加成"),
]


def load_simple_dotenv(path: Path) -> None:
    if not path.exists():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return
    for line in raw.splitlines():
        item = line.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue
        key, value = item.split("=", 1)
        env_key = key.strip().lstrip("\ufeff")
        env_val = value.strip().strip('"').strip("'")
        if env_key and env_key not in os.environ:
            os.environ[env_key] = env_val
