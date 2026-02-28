# 淘宝市场调研工具 (Taobao Insight)

> 一款基于 Claude Code 的 AI 市场调研工具。自动管理登录状态，抓取竞品数据，利用 Gemini AI 分析产品卖点，生成市场分析报告（Markdown 数据 + HTML 可视化）。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Playwright](https://img.shields.io/badge/Playwright-Enabled-green.svg)](https://playwright.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[功能特性](#功能特性) • [安装](#安装) • [快速开始](#快速开始) • [配置说明](#配置说明) • [���见问题](#常见问题)

---

## 功能特性

- **持久化浏览器模式** - 自动启动并管理浏览器，保持登录状态，避免淘宝反爬限制
- **自动扫码登录** - 首次运行时自动弹出浏览器等待扫码登录，登录后自动保存状态
- **AI 卖点提取** - 使用 Gemini AI 分析商品详情，提取核心卖点
- **市场分析报告** - 汇总竞品数据，生成市场空白点分析、卖点聚类等商业分析
- **双格式输出** - 自动生成 CSV 数据表和 HTML 可视化报告

---

## 安装

### 方法一：作为 Claude Code Skill 使用（推荐）

1. **找到 Claude Code 的 skills 目录**

   - **Windows**: `%USERPROFILE%\.claude\skills`
   - **macOS**: `~/.claude/skills`
   - **Linux**: `~/.claude/skills`

2. **克隆本仓库到 skills 目录**

   ```bash
   cd ~/.claude/skills
   git clone https://github.com/你的用户名/taobao-insight.git
   ```

3. **配置环境变量**

   ```bash
   cd taobao-insight
   cp .env.example .env
   # 编辑 .env 文件，填入你的 GEMINI_API_KEY
   ```

4. **重启 Claude Code** 即可使用

### 方法二：直接使用

1. 克隆仓库到本地
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
3. 配置环境变量（参考上方方法一的第3步）

---

## 快速开始

### 1. 配置 API Key

编辑 `.env` 文件，填入你的 Gemini API Key：

```bash
GEMINI_API_KEY=你的_api_key_这里
```

### 2. 运行分析

在 Claude Code 中发送指令，例如：

```
帮我抓取淘宝上"粉饼"排名前 30 的产品，生成市场报告
```

**首次运行时浏览器会自动启动并等待扫码登录，登录成功后会自动保存登录��态。**

---

## 配置说明

### 基础配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `GEMINI_API_KEY` | Google Gemini API Key（必填） | - |
| `GEMINI_PROXY_URL` | API 代理地址（国内使用需配置） | - |
| `GEMINI_TIMEOUT_SEC` | API 超时时间（秒） | 180 |
| `TAOBAO_BROWSER_MODE` | 浏览器模式 | persistent |
| `CRAWL_WORKERS` | 爬虫并发数（建议保持 1） | 1 |
| `LLM_WORKERS` | AI 分析并发数 | 8 |

### 完整配置示例

```bash
# Gemini AI 配置
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_PROXY_URL=http://127.0.0.1:7897
GEMINI_TIMEOUT_SEC=180

# 浏览器配置
TAOBAO_BROWSER_MODE=persistent

# 并发控制
CRAWL_WORKERS=1
LLM_WORKERS=8
LLM_WORKERS_MIN=4
```

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
2. **浏览器启动** - 自动启动持久化浏览器，首次运行提示扫码登录
3. **数据爬取** - 控制浏览器获取商品列表和详情页数据
4. **AI 分析** - 使用 Gemini 清洗数据，提取产品卖点
5. **报告生成** - 生成 CSV 数据表和 HTML 可视化报告

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

---

## 项目结构

| 文件/目录 | 说明 |
|-----------|------|
| `SKILL.md` | Claude Code Skill 执行指南 |
| `scripts/pipeline.py` | 核心流水线 |
| `scripts/scraper.py` | 爬虫模块 |
| `scripts/analysis.py` | AI 分析模块 |
| `scripts/report.py` | 报告生成模块 |
| `tools/` | 浏览器管理工具 |

---

## License

[MIT License](LICENSE)
