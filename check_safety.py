#!/usr/bin/env python3
"""快速验证浏览器安全设置"""

import sys
import os

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from tools.cdp_browser import CDPBrowserManager
from tools.browser_launcher import BrowserLauncher

def check_safety_settings():
    """检查安全设置"""
    print("=" * 60)
    print("浏览器安全配置检查")
    print("=" * 60)

    # 检查 CDPBrowserManager 默认参数
    mgr = CDPBrowserManager()

    print("\n1. CDPBrowserManager 默认参数:")
    print("-" * 40)
    checks = [
        ("auto_close_browser", mgr.auto_close_browser, False, "不关闭浏览器"),
        ("safe_mode", mgr.safe_mode, True, "使用安全端口 (9330+)"),
        ("user_data_dir_template", mgr.user_data_dir_template, "taobao_cdp_profile", "独立数据目录"),
    ]

    all_ok = True
    for name, value, expected, description in checks:
        status = "[OK]" if value == expected else "[FAIL]"
        if value != expected:
            all_ok = False
        print(f"  {status} {name} = {repr(value)}")
        print(f"     -> {description}")

    # 检查端口选择
    print("\n2. Port Selection Test:")
    print("-" * 40)
    launcher = BrowserLauncher()

    port_standard = launcher.find_available_port(9222)
    port_safe = launcher.find_available_port(9330)

    print(f"  Standard mode (9222): {port_standard}")
    print(f"  Safe mode (9330): {port_safe}")

    if port_safe >= 9330:
        print(f"  [OK] Safe mode uses port {port_safe}, avoiding 9222")
    else:
        print(f"  [WARN] Port selection may have issues")

    # 总结
    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] All safety checks passed!")
        print("\nRecommended usage:")
        print("  python scripts/run_pipeline.py \\")
        print("    --taobao-browser-mode persistent \\")
        print("    --use-global-browser 1 \\")
        print("    final-csv \"粉饼\" --top-n 5")
    else:
        print("[FAIL] Some safety checks failed")

    print("=" * 60)
    return all_ok

if __name__ == "__main__":
    check_safety_settings()
