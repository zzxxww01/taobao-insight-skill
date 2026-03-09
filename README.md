# 淘宝京东商品调研 (ecom-research-skill)

> 一款 **Claude Code Skill**，通过原生 CDP 自动管理浏览器与登录状态，抓取淘宝/天猫/京东竞品数据，利用 Gemini AI 提炼产品卖点，生成市场分析报告（Markdown + HTML）。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CDP](https://img.shields.io/badge/Chrome_DevTools_Protocol-Native-green.svg)](https://chromedevtools.github.io/devtools-protocol/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 目录

[作为 Skill 安装（推荐）](#作为-skill-安装推荐) • [直接使用](#直接使用) • [功能特性](#功能特性) • [配置说明](#配置说明) • [快速开始](#快速开始) • [评论抓取](#评论抓取) • [常见问题](#常见问题)

---

## 作为 Skill 安装（推荐）

这是本工具的主要使用方式。安装后，在 Claude Code 中用自然语言即可驱动完整的调研流程，无需手动输入命令。

### 第一步：克隆到 Skills 目录

```bash
# macOS / Linux
cd ~/.claude/skills
git clone https://github.com/zzxxww01/ecom-research-skill.git ecom-research-skill

# Windows（PowerShell）
cd "$env:USERPROFILE\.claude\skills"
git clone https://github.com/zzxxww01/ecom-research-skill.git ecom-research-skill
```

### 第二步：安装 Python 依赖

```bash
cd ecom-research-skill
pip install -r requirements.txt
```

> Playwright 仅在 CDP 模式降级时需要，可按需安装：`playwright install chromium`

### 第三步：配置 API Key

```bash
cp .env.example .env
```

用文本编辑器打开 `.env`，填入必填项：

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

> 国内用户若访问 Gemini 受限，可同时配置 `GEMINI_PROXY_URL=http://127.0.0.1:7897`

### 第四步：重启 Claude Code

重启后即可在对话中直接使用：

```
帮我调研淘宝上"粉饼"排名前 30 的竞品，生成市场报告
```

```
抓取这两个京东商品的近 7 天评论：
https://item.jd.com/100259348596.html
https://item.jd.com/8142476.html
```

> Claude Code 会自动识别意图、启动浏览器、等待扫码登录、执行抓取与分析，最后呈现报告摘要。

---

## 直接使用

不使用 Skill 机制时，也可以直接在命令行调用。

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境

```bash
cp .env.example .env
# 编辑 .env，至少填入 GEMINI_API_KEY
```

### 运行

```powershell
# Windows PowerShell（关键词调研）
$env:PYTHONIOENCODING="utf-8"
python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "粉饼" --top-n 30 --output "data/exports/粉饼-top30.md" --html-output "data/exports/粉饼-top30.html"
```

---

## 功能特性

- **双平台** — 淘宝/天猫 + 京东，共用同一套分析和导出能力
- **三种输入** — 关键词搜索 / 商品链接直抓 / `.txt` 链接文件批量导入
- **原生 CDP 浏览器** — 直接通过 Chrome DevTools Protocol 操控浏览器；CDP 失败时自动降级到 Playwright persistent 模式
- **自动登录管理** — 首次扫码后 Cookie 自动持久化，后续无感复用；风控拦截时自动暂停等待
- **Gemini AI 分析** — 支持多 Key 轮转，自动提炼产品卖点与市场洞察
- **独立评论抓取** — 单独导出评论原始数据（JSONL + CSV），不混入商品分析流程
- **双格式报告** — 自动生成 Markdown 数据表和 HTML 可视化报告

---

## 配置说明

所有配置通过项目根目录的 `.env` 文件管理。

### 必填

| 配置项 | 说明 |
|--------|------|
| `GEMINI_API_KEY` | Google Gemini API Key |

### 常用可选项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `GEMINI_API_KEY_FALLBACK` | 备用 Key，主 Key 限流时自动切换 | — |
| `GEMINI_PROXY_URL` | API 代理地址（国内建议配置） | — |
| `GEMINI_TIMEOUT_SEC` | API 超时时间（秒） | `45` |
| `CRAWL_WORKERS` | 爬虫并发数（建议保持 1） | `1` |
| `LLM_WORKERS` | AI 分析并发数 | `64` |
| `TAOBAO_BROWSER_MODE` | 浏览器模式 `cdp` / `persistent` | `cdp` |
| `JD_BROWSER_MODE` | 京东浏览器模式 | `cdp` |
| `CUSTOM_BROWSER_PATH` | 自定义浏览器路径（找不到 Chrome 时设置） | — |

完整配置项参见 [.env.example](.env.example)。

### 登录状态保存位置

| 平台 | Windows | macOS / Linux |
|------|---------|---------------|
| 淘宝/天猫 | `%APPDATA%\ecom_research_taobao_profile` | `~/.config/ecom_research_taobao_profile` |
| 京东 | `%APPDATA%\ecom_research_jd_profile` | `~/.config/ecom_research_jd_profile` |

> 若本机已存在旧目录 `taobao_insight_profile` / `jd_insight_profile` 或旧状态文件，程序会自动复用，避免重新扫码登录。

---

## 快速开始

### 作为 Skill 使用

安装完成后，直接在 Claude Code 对话框输入自然语言即可：

| 意图 | 示例指令 |
|------|----------|
| 淘宝关键词调研 | `调研淘宝"猫粮"前 20 名竞品` |
| 京东关键词调研 | `帮我调研京东"口红"TOP 20` |
| 商品链接直抓 | `分析这两个链接：https://...` |
| 批量链接文件 | `读取 C:\data\urls.txt 做竞品分析` |
| 抓取评论 | `抓取这个京东商品近 7 天的评论：https://item.jd.com/...` |

首次运行时浏览器会自动弹出，扫码登录后状态自动保存，后续无需重复操作。

### CLI 示例

<details>
<summary>淘宝关键词调研</summary>

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "粉饼" --top-n 30 --output "data/exports/粉饼-top30.md" --html-output "data/exports/粉饼-top30.html"
```
</details>

<details>
<summary>京东关键词调研</summary>

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/pipeline.py jd-final-csv "口红" --top-n 20 --output "data/exports/jd-口红-top20.md" --html-output "data/exports/jd-口红-top20.html"
```
</details>

<details>
<summary>商品链接直抓</summary>

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "my-task" --top-n 2 --item-url "https://detail.tmall.com/item.htm?..." --item-url "https://item.taobao.com/item.htm?..." --output "data/exports/my-task.md" --html-output "data/exports/my-task.html"
```
</details>

<details>
<summary>批量链接文件</summary>

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "batch-task" --top-n 30 --item-urls-file "C:\data\urls.txt" --output "data/exports/batch-task.md" --html-output "data/exports/batch-task.html"
```
</details>

---

## 评论抓取

只需抓评论原始数据时，使用独立入口 `scripts/review_pipeline.py`（不生成市场分析报告）。

### 常用命令

```powershell
# 京东：抓取近 7 天评论
$env:PYTHONIOENCODING="utf-8"
python scripts/review_pipeline.py jd-reviews "jd-task" --item-url "https://item.jd.com/100259348596.html" --days 7 --limit 0

# 淘宝：抓取近 2 个月、最多 200 条
python scripts/review_pipeline.py taobao-reviews "tb-task" --item-url "https://item.taobao.com/item.htm?id=..." --months 2 --limit 200
```

### 输入方式

```powershell
--item-url "<url>"          # 单个商品链接，可重复传入多个
--item-id <item_id>         # 直接传商品 ID
--item-urls-file "<path>"   # 从 txt 文件批量读取（每行一个链接）
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--days N` | 最近 N 天（优先于 `--months`） | `0` |
| `--months N` | 最近 N 个月 | `2` |
| `--limit N` | 每个商品最多保留 N 条；`0` 为不限 | `100` |
| `--output-dir` | 自定义输出目录 | 自动生成 |

### 输出结构

```
data/reviews_exports/<platform>/direct/<item_id>-d7-all-<timestamp>/
├── reviews.jsonl
├── reviews.csv
├── run-summary.json
└── run-summary.md
```

---

## 工作流程

```
输入 → 环境检查 → 浏览器启动（CDP / Playwright 降级）
     → 登录检测（首次扫码，后续自动复用）
     → 数据爬取（关键词搜索 → 详情页 / 链接直抓）
     → Gemini AI 分析（卖点提炼 / 市场洞察）
     → 导出（Markdown + HTML 报告）
```

---

## 项目结构

| 文件/目录 | 说明 |
|-----------|------|
| `SKILL.md` | Claude Code Skill 执行指南（Skill 核心） |
| `scripts/pipeline.py` | 商品调研主流水线 |
| `scripts/review_pipeline.py` | 评论抓取独立入口 |
| `scripts/scraper.py` | 淘宝/天猫爬虫 |
| `scripts/jd_scraper.py` | 京东爬虫 |
| `scripts/analysis.py` | Gemini AI 分析模块 |
| `scripts/report.py` | 报告生成模块 |
| `tools/cdp_browser.py` | 原生 CDP 浏览器实现 |
| `tools/browser_manager.py` | 双模式浏览器管理器 |
| `tools/taobao_login.py` | 登录状态机基类 + 淘宝实现 |
| `tools/jd_login.py` | 京东登录实现 |
| `tools/login_rules.py` | 淘宝登录/风控检测规则 |
| `tools/jd_login_rules.py` | 京东登录/风控检测规则 |
| `.env.example` | 环境变量模板 |

---

## 常见问题

**Q: 首次运行需要做什么？**
A: 无需额外准备。首次运行时浏览器会自动启动并弹出登录页面，用手机扫码登录后程序自动继续。

**Q: 登录状态能保持多久？**
A: 通常数天到数周。如果提示需要重新登录，扫码一次即可更新。

**Q: 浏览器启动失败怎么办？**
A: 关闭所有 Chrome/Edge 进程后重试。若系统找不到浏览器，在 `.env` 中设置 `CUSTOM_BROWSER_PATH` 指向浏览器可执行文件。

**Q: Gemini API 超时或无法访问？**
A: 在 `.env` 中设置 `GEMINI_PROXY_URL`（代理地址）或增大 `GEMINI_TIMEOUT_SEC`。也可以配置 `GEMINI_API_KEY_FALLBACK` 作为备用 Key。

**Q: 可以同时给关键词和链接吗？**
A: 可以。链接输入优先，关键词仅用于任务命名，搜索阶段会被跳过。

**Q: `.txt` 文件有什么格式要求？**
A: UTF-8 编码，每行一个商品链接。空行和重复链接自动忽略。

---

## License

[MIT License](LICENSE)
