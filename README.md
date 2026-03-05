# 淘宝市场调研工具 (Taobao Insight)

> 一款基于 Claude Code 的 AI 市场调研工具。通过原生 CDP 自动管理浏览器与登录状态，抓取竞品数据，利用 Gemini AI 分析产品卖点，生成市场分析报告（Markdown + HTML 可视化）。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CDP](https://img.shields.io/badge/Chrome_DevTools_Protocol-Native-green.svg)](https://chromedevtools.github.io/devtools-protocol/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[功能特性](#功能特性) • [安装](#安装) • [快速开始](#快速开始) • [配置说明](#配置说明) • [常见问题](#常见问题)

---

## 功能特性

- **原生 CDP 浏览器控制** - 直接通过 Chrome DevTools Protocol 操控浏览器，CDP 失败时自动降级到 Playwright persistent 模式
- **自动扫码登录** - 首次运行自动弹出浏览器等待扫码，登录状态自动持久化
- **三种输入模式** - 支持关键词搜索、一个/多个商品链接直抓、以及 `.txt` 链接文件批量导入
- **AI 卖点提取** - 使用 Gemini AI 分析商品详情，提取核心卖点
- **市场分析报告** - 汇总竞品数据，生成市场空白点分析、卖点聚类等商业洞察
- **双格式输出** - 自动生成 Markdown 数据表和 HTML 可视化报告

---

## 安装

### 方法一：作为 Claude Code Skill 使用（推荐）

1. **克隆到 Claude Code skills 目录**

   ```bash
   # Windows: %USERPROFILE%\.claude\skills
   # macOS/Linux: ~/.claude/skills
   cd ~/.claude/skills
   git clone https://github.com/你的用户名/taobao-insight.git
   ```

2. **配置环境变量**

   ```bash
   cd taobao-insight
   cp .env.example .env
   # 编辑 .env 文件，填入你的 GEMINI_API_KEY
   ```

3. **重启 Claude Code** 即可使用

### 方法二：直接使用

1. 克隆仓库到本地
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   > Playwright 仅在 CDP 模式降级时需要，按需安装：`playwright install chromium`

3. 配置环境变量（同上方第 2 步）

---

## 快速开始

### 1. 配置 API Key

编辑 `.env` 文件：

```bash
GEMINI_API_KEY=你的_api_key_这里
```

### 2. 运行分析（支持 3 种输入）

首次运行时浏览器会自动启动并等待扫码登录，登录成功后自动保存状态。

#### 2.1 输入关键词（`keyword`）

- Claude Code 指令示例：
  ```
  帮我抓取淘宝上"粉饼"排名前 30 的产品，生成市场报告
  ```
- CLI 示例（PowerShell）：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"
  python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "粉饼" --top-n 30 --output "data/exports/粉饼-top30.md" --html-output "data/exports/粉饼-top30.html"
  ```

#### 2.2 输入一个或多个商品链接（`item_urls`）

- Claude Code 指令示例：
  ```
  帮我分析这几个链接商品：https://detail.tmall.com/item.htm?... https://item.taobao.com/item.htm?...
  ```
- CLI 示例（PowerShell）：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"
  python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "direct-items-20260306" --top-n 2 --item-url "<url_1>" --item-url "<url_2>" --output "data/exports/direct-items-20260306-top2.md" --html-output "data/exports/direct-items-20260306-top2.html"
  ```

#### 2.3 输入 txt 链接文件（`item_urls_file`）

- 文件要求：`txt` 每行 1 个商品链接，建议 UTF-8 编码。
- Claude Code 指令示例：
  ```
  帮我读取 C:\data\tmall_urls.txt 里的链接并做商品分析
  ```
- CLI 示例（PowerShell）：
  ```powershell
  $env:PYTHONIOENCODING="utf-8"
  python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "direct-file-20260306" --top-n 30 --item-urls-file "C:\data\tmall_urls.txt" --output "data/exports/direct-file-20260306-top30.md" --html-output "data/exports/direct-file-20260306-top30.html"
  ```

> 说明：`item_urls` / `item_urls_file` 模式会跳过搜索阶段，直接抓取商品详情并分析。

---

## 配置说明

### 基础配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `GEMINI_API_KEY` | Google Gemini API Key（必填） | - |
| `GEMINI_PROXY_URL` | API 代理地址（国内使用需配置） | - |
| `GEMINI_TIMEOUT_SEC` | API 超时时间（秒） | 180 |
| `TAOBAO_BROWSER_MODE` | 浏览器模式（`cdp` / `persistent`） | `cdp` |
| `CRAWL_WORKERS` | 爬虫并发数（建议保持 1） | 1 |
| `LLM_WORKERS` | AI 分析并发数（Flash 模型） | 64 |

### 完整配置示例

```bash
# Gemini AI 配置
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_PROXY_URL=http://127.0.0.1:7897
GEMINI_TIMEOUT_SEC=180

# 并发控制
CRAWL_WORKERS=1
LLM_WORKERS=64
LLM_WORKERS_MIN=32
```

> `TAOBAO_BROWSER_MODE` 默认为 `cdp`，一般无需手动设置。CDP 初始化失败时会自动降级到 `persistent`。

---

## 登录状态管理

### 保存位置

- **Windows**: `%APPDATA%\taobao_insight_profile`
- **macOS**: `~/Library/Application Support/taobao_insight_profile`
- **Linux**: `~/.config/taobao_insight_profile`

### 工作方式

- 首次运行自动启动浏览器，等待扫码登录（最多 300 秒）
- 登录成功后自动保存 cookies 到本地
- 后续运行自动复用登录状态
- 登录过期时会自动提示重新扫码

---

## 工作流程

1. **环境检查** - 检测 Python 依赖和 API 配置
2. **输入解析** - 自动识别 `keyword` / `item_urls` / `item_urls_file` 三种输入模式
3. **浏览器启动** - 通过原生 CDP 自动启动浏览器（失败时降级到 Playwright），首次运行提示扫码登录
4. **数据爬取** - 关键词模式先搜索再抓取；链接模式跳过搜索，直接抓取详情页数据
5. **AI 分析** - 使用 Gemini 清洗数据，提取产品卖点（爬虫与 LLM 流水线并行执行）
6. **报告生成** - 生成 Markdown 数据表和 HTML 可视化报告

---

## 常见问题

### Q: 首次运行需要做什么？

A: 无需额外准备。首次运行时会自动启动浏览器，弹出淘宝登录页面，用手机淘宝扫码登录即可。

### Q: 登录状态能保持多久？

A: 通常可以保持数天到数周。如果提示"需要重新登录"，扫码一次即可更新。

### Q: 为什么有时会跳转到登录页面？

A: 这是淘宝的风控机制。程序会自动检测并等待扫码登录，登录成功后继续执行。

### Q: Gemini API 超时怎么办？

A: 可以在 `.env` 中增加 `GEMINI_TIMEOUT_SEC` 到 180 秒以上，或配置 `GEMINI_PROXY_URL` 使用代理。

### Q: 浏览器启动失败怎么办？

A: 关闭所有 Chrome 进程后重试。如果系统找不到 Chrome/Edge，可设置 `CUSTOM_BROWSER_PATH` 环境变量指定浏览器路径。

### Q: 可以同时给关键词和链接吗？

A: 可以。系统会优先按链接模式执行（跳过搜索阶段），关键词仅用于任务命名。

### Q: txt 文件有什么格式要求？

A: 建议 UTF-8 编码，每行一个淘宝/天猫商品详情链接。空行和重复链接会被自动忽略。

---

## 项目结构

| 文件/目录 | 说明 |
|-----------|------|
| `SKILL.md` | Claude Code Skill 执行指南 |
| `scripts/pipeline.py` | 核心流水线 |
| `scripts/scraper.py` | 爬虫模块 |
| `scripts/analysis.py` | AI 分析模块 |
| `scripts/report.py` | 报告生成模块 |
| `tools/cdp_browser.py` | 原生 CDP 浏览器实现 |
| `tools/browser_manager.py` | 双模式浏览器管理器（CDP + Playwright 降级） |
| `tools/login_rules.py` | 集中化登录/风控检测规则 |
| `tools/taobao_login.py` | 登录恢复状态机 |
| `docs/login-implementation.md` | 浏览器登录模块技术文档 |

---

## License

[MIT License](LICENSE)
