#!/usr/bin/env python3
"""Environment check script for Taobao/Tmall/JD browser automation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_simple_dotenv

try:
    from tools.login_rules import LOGIN_COOKIE_NAMES as TAOBAO_LOGIN_COOKIE_NAMES
except Exception:
    TAOBAO_LOGIN_COOKIE_NAMES = {"cookie2", "unb", "_tb_token_", "_m_h5_tk"}

try:
    from tools.jd_login_rules import LOGIN_COOKIE_NAMES as JD_LOGIN_COOKIE_NAMES
except Exception:
    JD_LOGIN_COOKIE_NAMES = {"thor", "pin", "pinId", "TrackID", "unick"}


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    label: str
    domains: tuple[str, ...]
    login_cookie_names: set[str]
    browser_mode_env: str
    user_data_dir_env: str
    storage_state_env: str
    manual_login_timeout_env: str
    default_profile_name: str
    default_storage_state_name: str


PLATFORM_SPECS = {
    "taobao": PlatformSpec(
        key="taobao",
        label="Taobao/Tmall",
        domains=("taobao.com", "tmall.com"),
        login_cookie_names=set(TAOBAO_LOGIN_COOKIE_NAMES),
        browser_mode_env="TAOBAO_BROWSER_MODE",
        user_data_dir_env="TAOBAO_USER_DATA_DIR",
        storage_state_env="TAOBAO_STORAGE_STATE_FILE",
        manual_login_timeout_env="TAOBAO_MANUAL_LOGIN_TIMEOUT_SEC",
        default_profile_name="taobao_insight_profile",
        default_storage_state_name="taobao_storage_state.json",
    ),
    "jd": PlatformSpec(
        key="jd",
        label="JD",
        domains=("jd.com", "3.cn"),
        login_cookie_names=set(JD_LOGIN_COOKIE_NAMES),
        browser_mode_env="JD_BROWSER_MODE",
        user_data_dir_env="JD_USER_DATA_DIR",
        storage_state_env="JD_STORAGE_STATE_FILE",
        manual_login_timeout_env="JD_MANUAL_LOGIN_TIMEOUT_SEC",
        default_profile_name="jd_insight_profile",
        default_storage_state_name="jd_storage_state.json",
    ),
}


def print_step(message: str) -> None:
    print(f"\n[检查] {message}")


def load_env_file() -> Path:
    env_path = ROOT_DIR / ".env"
    load_simple_dotenv(env_path)
    return env_path


def check_python_dependencies() -> bool:
    print_step("Python 依赖")
    missing: list[str] = []
    try:
        import playwright  # noqa: F401

        print("[OK] 已安装 playwright")
    except ImportError:
        missing.append("playwright")

    try:
        from google import genai as google_genai  # noqa: F401

        print("[OK] 已安装 google-genai")
    except ImportError:
        missing.append("google-genai")

    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print(f"请先执行: pip install {' '.join(missing)}")
        return False
    return True


def _parse_int(value: str, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _default_user_data_dir(spec: PlatformSpec) -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        return Path(appdata) / spec.default_profile_name
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / spec.default_profile_name
    return Path.home() / ".config" / spec.default_profile_name


def _default_storage_state_file(spec: PlatformSpec) -> Path:
    env_value = (os.environ.get(spec.storage_state_env, "") or "").strip()
    if env_value:
        return Path(env_value)
    candidates = [
        ROOT_DIR / "backend" / "data" / spec.default_storage_state_name,
        ROOT_DIR / "data" / spec.default_storage_state_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _resolve_browser_mode(spec: PlatformSpec) -> str:
    mode = (os.environ.get(spec.browser_mode_env, "cdp") or "cdp").strip().lower()
    if mode not in {"cdp", "persistent"}:
        return "cdp"
    return mode


def _shared_cdp_url() -> str:
    return (
        os.environ.get("PLAYWRIGHT_CDP_URL", "")
        or os.environ.get("TAOBAO_CDP_ENDPOINT", "")
    ).strip()


def _count_valid_cookies(
    storage_state_file: Path,
    *,
    domains: tuple[str, ...],
    login_cookie_names: set[str],
) -> tuple[int, int, str]:
    if not storage_state_file.exists():
        return 0, 0, "missing"
    try:
        payload = json.loads(storage_state_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return 0, 0, f"invalid_json:{exc}"

    if not isinstance(payload, dict):
        return 0, 0, "invalid_payload"

    now_ts = int(time.time())
    valid_cookie_count = 0
    valid_login_cookie_count = 0
    for cookie in payload.get("cookies", []):
        if not isinstance(cookie, dict):
            continue
        domain = str(cookie.get("domain", "")).lower()
        if not any(item in domain for item in domains):
            continue
        expires = cookie.get("expires")
        if expires in (None, "", -1):
            valid = True
        else:
            try:
                valid = int(float(expires)) > now_ts + 120
            except Exception:
                valid = False
        if not valid:
            continue
        valid_cookie_count += 1
        name = str(cookie.get("name", "")).strip()
        if name in login_cookie_names:
            valid_login_cookie_count += 1
    return valid_cookie_count, valid_login_cookie_count, "ok"


def check_common_env(env_path: Path) -> bool:
    print_step("通用环境变量")
    if env_path.exists():
        print(f"[OK] 已加载同一个 .env 文件: {env_path}")
    else:
        print(f"[INFO] 未发现 .env 文件，当前使用进程环境变量: {env_path}")

    ok = True
    gemini_api_key = (os.environ.get("GEMINI_API_KEY", "") or "").strip()
    if gemini_api_key:
        print("[OK] 已配置 GEMINI_API_KEY")
    else:
        print("[ERROR] 未配置 GEMINI_API_KEY")
        ok = False

    gemini_proxy_url = (os.environ.get("GEMINI_PROXY_URL", "") or "").strip()
    if gemini_proxy_url:
        print(f"[INFO] 已配置 GEMINI_PROXY_URL: {gemini_proxy_url}")
    else:
        print("[INFO] 未配置 GEMINI_PROXY_URL")

    gemini_timeout = max(5, _parse_int(os.environ.get("GEMINI_TIMEOUT_SEC", "45"), 45))
    print(f"[INFO] GEMINI_TIMEOUT_SEC={gemini_timeout}")

    custom_browser_path = (os.environ.get("CUSTOM_BROWSER_PATH", "") or "").strip()
    if custom_browser_path:
        browser_path = Path(custom_browser_path)
        if browser_path.exists():
            print(f"[OK] CUSTOM_BROWSER_PATH 存在: {browser_path}")
        else:
            print(f"[WARN] CUSTOM_BROWSER_PATH 不存在: {browser_path}")

    return ok


def check_platform_env(spec: PlatformSpec) -> bool:
    print_step(f"{spec.label} 平台配置")
    browser_mode = _resolve_browser_mode(spec)
    storage_state_file = _default_storage_state_file(spec).resolve()
    user_data_dir = Path(
        (os.environ.get(spec.user_data_dir_env, "") or "").strip()
        or str(_default_user_data_dir(spec))
    ).resolve()
    recommended_storage_state = (
        ROOT_DIR / "data" / spec.default_storage_state_name
    ).resolve()
    recommended_user_data_dir = _default_user_data_dir(spec).resolve()
    manual_login_timeout = max(
        30,
        _parse_int(os.environ.get(spec.manual_login_timeout_env, "300"), 300),
    )

    print(f"浏览器模式: {browser_mode} ({spec.browser_mode_env})")
    print(f"登录等待超时: {manual_login_timeout}s ({spec.manual_login_timeout_env})")
    print(f"storage_state: {storage_state_file}")
    print(f"user_data_dir: {user_data_dir}")

    if not (os.environ.get(spec.storage_state_env, "") or "").strip():
        print(f"[INFO] 未显式设置 {spec.storage_state_env}，将使用默认路径")
    if not (os.environ.get(spec.user_data_dir_env, "") or "").strip():
        print(f"[INFO] 未显式设置 {spec.user_data_dir_env}，将使用默认 profile")
    if storage_state_file != recommended_storage_state:
        print(f"[INFO] 推荐固定 {spec.storage_state_env}={recommended_storage_state}")
    if user_data_dir != recommended_user_data_dir:
        print(f"[INFO] 推荐固定 {spec.user_data_dir_env}={recommended_user_data_dir}")

    valid_cookie_count, valid_login_cookie_count, status = _count_valid_cookies(
        storage_state_file,
        domains=spec.domains,
        login_cookie_names=spec.login_cookie_names,
    )
    if status == "missing":
        print("[INFO] 尚未发现 storage_state 文件，首次运行可能需要扫码或过验证")
    elif status.startswith("invalid_json") or status == "invalid_payload":
        print(f"[WARN] storage_state 文件不可读: {status}")
    elif valid_login_cookie_count > 0:
        print(
            f"[OK] 检测到可复用登录 cookie: "
            f"{valid_login_cookie_count} 个关键 cookie / {valid_cookie_count} 个站点 cookie"
        )
    else:
        print(
            "[INFO] 未检测到可复用登录 cookie，首次运行或登录过期时可能需要扫码或过验证"
        )
    return True


def check_browser_connection(selected_specs: list[PlatformSpec]) -> bool:
    print_step("浏览器连接")
    cdp_specs = [spec for spec in selected_specs if _resolve_browser_mode(spec) == "cdp"]
    if not cdp_specs:
        print("[OK] 所选平台均为 persistent 模式，运行时会自动启动浏览器")
        return True

    cdp_url = _shared_cdp_url()
    if not cdp_url:
        labels = ", ".join(spec.label for spec in cdp_specs)
        print(
            f"[OK] {labels} 使用 raw CDP 优先模式，未配置外部 CDP 端点时会由内置浏览器管理器自动拉起浏览器"
        )
        print("[INFO] 需要时也可以设置 PLAYWRIGHT_CDP_URL 复用现有浏览器实例")
        return True

    print(f"[INFO] 已配置共享外部 CDP 端点: {cdp_url}")
    check_endpoint = cdp_url.rstrip("/") + "/json/version"
    try:
        request = urllib.request.Request(check_endpoint, method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            if response.status == 200:
                print("[OK] 外部 Chrome CDP 端点可连接")
                return True
            print(f"[ERROR] 外部 Chrome CDP 端点响应异常: HTTP {response.status}")
            return False
    except urllib.error.URLError as exc:
        example_profile = _default_user_data_dir(cdp_specs[0]).resolve()
        print(f"[ERROR] 无法连接外部 Chrome CDP 端点: {exc.reason}")
        print("可选修复方式:")
        print("1. 清空 PLAYWRIGHT_CDP_URL，改用内置 raw CDP 自动拉起浏览器")
        print("2. 或手动启动 Chrome 并暴露 9222 端口，例如:")
        print(
            f'   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" '
            f'--remote-debugging-port=9222 --user-data-dir="{example_profile}"'
        )
        return False
    except Exception as exc:
        print(f"[ERROR] 检查外部 Chrome CDP 端点失败: {exc}")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check runtime environment for Taobao/Tmall/JD browser automation."
    )
    parser.add_argument(
        "--platform",
        choices=["taobao", "jd", "all"],
        default="all",
        help="Which platform config to inspect. Default: all",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    env_path = load_env_file()
    selected_specs = (
        list(PLATFORM_SPECS.values())
        if args.platform == "all"
        else [PLATFORM_SPECS[args.platform]]
    )

    labels = ", ".join(spec.label for spec in selected_specs)
    print(f"=== 环境检查: {labels} ===")

    deps_ok = check_python_dependencies()
    env_ok = check_common_env(env_path)
    platform_ok = True
    for spec in selected_specs:
        platform_ok = check_platform_env(spec) and platform_ok
    browser_ok = check_browser_connection(selected_specs)

    print("\n=== 检查结果 ===")
    if deps_ok and env_ok and platform_ok and browser_ok:
        print("[OK] 环境检查通过，可以开始运行抓取和分析流程")
        return 0

    print("[WARN] 环境存在问题，请先根据上面的提示修复后再运行")
    return 1


if __name__ == "__main__":
    sys.exit(main())
