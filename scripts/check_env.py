#!/usr/bin/env python3
"""
Environment check script for browser automation runtime.

Default mode is CDP-first. This script validates dependencies, environment
variables, cached login state, and CDP endpoint availability before pipeline
startup.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

try:
    from tools.login_rules import LOGIN_COOKIE_NAMES
except Exception:
    LOGIN_COOKIE_NAMES = {"cookie2", "unb", "_tb_token_", "_m_h5_tk"}


def print_step(msg: str) -> None:
    print(f"\n[检查项] {msg}...")


def check_python_dependencies() -> bool:
    print_step("检查 Python 依赖库")
    missing = []
    try:
        import playwright

        print("✅ playwright 已安装")
    except ImportError:
        missing.append("playwright")

    try:
        from google import genai as google_genai

        print("google-genai 已安装")
    except ImportError:
        missing.append("google-genai")

    if missing:
        print(f"❌ 缺少必须的依赖库: {', '.join(missing)}")
        print("请在终端中执行以安装:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True


def check_env_vars() -> bool:
    print_step("检查必要的环境变量")
    # 尝试加载当前目录的 .env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        print(f"✅ 发现 .env 文件 ({env_path})，请确保密钥已填充。")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(
            "⚠️ 警告: 未在运行环境中发现 GEMINI_API_KEY。如果它在 .env 文件中则可忽略此警告。"
        )
    else:
        print("✅ 已配置 GEMINI_API_KEY")

    return True


def _default_user_data_dir() -> Path:
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        return Path(appdata) / "taobao_insight_profile"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "taobao_insight_profile"
    return Path.home() / ".config" / "taobao_insight_profile"


def _default_storage_state_file() -> Path:
    env_value = (os.environ.get("TAOBAO_STORAGE_STATE_FILE", "") or "").strip()
    if env_value:
        return Path(env_value)
    candidates = [
        Path("backend/data/taobao_storage_state.json"),
        Path("data/taobao_storage_state.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def check_login_state_cache() -> bool:
    print_step("检查登录态缓存")
    storage_state_file = _default_storage_state_file().resolve()
    user_data_dir = Path(
        (os.environ.get("TAOBAO_USER_DATA_DIR", "") or "").strip()
        or str(_default_user_data_dir())
    ).resolve()
    recommended_storage_state = (Path(__file__).resolve().parent.parent / "data" / "taobao_storage_state.json").resolve()
    recommended_user_data_dir = _default_user_data_dir().resolve()

    print(f"storage_state 文件: {storage_state_file}")
    print(f"用户数据目录: {user_data_dir}")
    if storage_state_file != recommended_storage_state:
        print(f"⚠️ 建议固定 TAOBAO_STORAGE_STATE_FILE 为: {recommended_storage_state}")
    if user_data_dir != recommended_user_data_dir:
        print(f"⚠️ 建议固定 TAOBAO_USER_DATA_DIR 为: {recommended_user_data_dir}")

    login_cookie_names = set(LOGIN_COOKIE_NAMES)
    taobao_domains = ("taobao.com", "tmall.com")
    now_ts = int(time.time())
    valid_cookie_count = 0
    valid_login_cookie_count = 0

    if storage_state_file.exists():
        try:
            payload = json.loads(storage_state_file.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            for cookie in payload.get("cookies", []):
                if not isinstance(cookie, dict):
                    continue
                domain = str(cookie.get("domain", "")).lower()
                if not any(d in domain for d in taobao_domains):
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

    if valid_login_cookie_count > 0:
        print(
            f"✅ 检测到可用登录 cookie（{valid_login_cookie_count} 个关键登录 cookie，{valid_cookie_count} 个站点 cookie）"
        )
        print("ℹ️ 后续运行通常不需要重复扫码。")
        print("推荐固定路径命令（PowerShell）：")
        print(f"$env:TAOBAO_USER_DATA_DIR=\"{recommended_user_data_dir}\"")
        print(f"$env:TAOBAO_STORAGE_STATE_FILE=\"{recommended_storage_state}\"")
        return True

    print("⚠️ 未检测到可用登录 cookie，首次运行或登录态过期时可能需要扫码。")
    print("推荐固定路径命令（PowerShell）：")
    print(f"$env:TAOBAO_USER_DATA_DIR=\"{recommended_user_data_dir}\"")
    print(f"$env:TAOBAO_STORAGE_STATE_FILE=\"{recommended_storage_state}\"")
    return True


def resolve_browser_mode() -> str:
    mode = str(os.environ.get("TAOBAO_BROWSER_MODE", "cdp") or "cdp")
    mode = mode.strip().lower()
    if mode not in {"cdp", "persistent"}:
        return "cdp"
    return mode


def check_browser_connection(browser_mode: str) -> bool:
    if browser_mode == "persistent":
        print_step("检测浏览器模式")
        print("✅ 当前为 persistent 模式：运行时会自动启动浏览器并处理登录态。")
        print("ℹ️ 无需预先手动开启 9222 CDP 端口。")
        return True

    cdp_url = (
        os.environ.get("PLAYWRIGHT_CDP_URL", "")
        or os.environ.get("TAOBAO_CDP_ENDPOINT", "")
    ).strip()
    if not cdp_url:
        print_step("CDP 配置检查")
        print("✅ 当前为 cdp 模式：未设置外部 CDP 端点，将由原生 CDP 管理器自动拉起浏览器并完成连接。")
        return True

    print_step("检测 Chrome CDP 远程调试端口")
    print(f"正在尝试连接 CDP 端口: {cdp_url}")

    # 对于 /json/version，Chrome CDP 会返回当前支持的信息
    check_endpoint = cdp_url.rstrip("/") + "/json/version"
    try:
        request = urllib.request.Request(check_endpoint, method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            if response.status == 200:
                print("✅ 成功连接到 Chrome CDP 端口！浏览器环境就绪。")
                return True
            else:
                print(f"❌ CDP 端口响应异常，状态码: {response.status}")
                return False
    except urllib.error.URLError as e:
        print("❌ 无法连接到 Chrome CDP 端口！")
        print(f"   详细错误: {e.reason}")
        print("\n=== [CDP 模式需修复] ===")
        print(
            "您必须先关闭所有 Chrome 窗口，然后通过在命令行运行以下命令来彻底启动 Chrome 并暴露 9222 端口："
        )
        print("--------------------")
        print(
            '"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir=C:\\Users\\DELL\\AppData\\Roaming\\taobao_insight_profile'
        )
        print("--------------------")
        print("等待浏览器打开后，再重试此爬虫任务。")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False


def main() -> int:
    print("=== 开始淘宝市场调研环境检测 (Taobao Market Research Skill) ===")

    browser_mode = resolve_browser_mode()
    print(f"当前浏览器模式: {browser_mode}")
    deps_ok = check_python_dependencies()
    check_env_vars()
    check_login_state_cache()
    browser_ok = check_browser_connection(browser_mode)

    print("\n=== 检测结果 ===")
    if deps_ok and browser_ok:
        print("🎉 环境检测通过！您可以安全地开始执行爬虫流程了。")
        return 0
    else:
        print(
            "⚠️ 环境存在问题。请先根据上述红色的 ❌ 提示修复环境，否则爬虫将在此电脑上无法运行。"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

