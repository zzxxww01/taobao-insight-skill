---
name: taobao-insight
description: Taobao/Tmall/JD keyword product research, competitor analysis, and standalone review crawling. Use when user asks for 淘宝/天猫/京东商品调研、关键词选品、竞品分析、抓取N条商品、导出MD或HTML报告，或明确要求爬取/抓取淘宝或京东商品评论（例如：淘宝狗粮关键词调研20条、京东口红调研20条、抓取京东某商品近7天评论）。
---

# 淘宝/京东市场调研

原生 CDP 自动化抓取 + 大模型卖点提炼。按以下 5 阶段顺序执行，不要跳步。

> Windows 下所有 Python 命令前必须设置 `PYTHONIOENCODING=utf-8`。

---

## 京东支持补充

- 京东命令使用 `jd-analyze-keyword` 或 `jd-final-csv`
- 京东关键词示例：`口红`
- 京东商品链接示例：`https://item.jd.com/8142476.html`
- 京东登录/风控恢复必须沿用淘宝同一套浏览器设计：
  - 默认 `raw CDP`
  - CDP 初始化失败时自动降级到 `Playwright persistent`
  - 不要额外引入另一套登录浏览器主链路
- 京东默认使用独立会话参数：
  - `JD_BROWSER_MODE`
  - `JD_STORAGE_STATE_FILE`
  - `JD_USER_DATA_DIR`
  - `JD_MANUAL_LOGIN_TIMEOUT_SEC`
- 当用户明确是京东任务时：
  - 关键词模式运行 `python scripts/pipeline.py jd-final-csv "<keyword>" ...`
  - 链接模式运行 `python scripts/pipeline.py jd-final-csv "<placeholder_keyword>" --item-url "<url>" ...`
  - 不要混用淘宝的 `TAOBAO_*` session 路径

---

## 评论抓取独立入口

当用户明确说“爬取评论 / 抓评论 / 导出评论 / 京东评论 / 淘宝评论 / 评价抓取”时，不要走现有商品分析 pipeline，改走独立入口 `scripts/review_pipeline.py`。

评论抓取模式的原则：
- 只抓取并导出评论原始数据，不做商品分析、卖点总结、HTML/Markdown 调研报告
- 输入只接受商品 `URL/ID/URL 文件`
- 输出固定为 `JSONL + CSV + run-summary.json + run-summary.md`
- 默认输出目录为 `data/reviews_exports/<platform>/<direct|search>/<task>/`
- 单商品直抓时，目录名优先用 `item_id`，例如 `data/reviews_exports/jd/direct/100259348596-d5-all-20260309-001459`

常用命令：

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/review_pipeline.py jd-reviews <task_name> --item-url "<JD商品URL>" --days 7 --limit 0
python scripts/review_pipeline.py taobao-reviews <task_name> --item-url "<淘宝/天猫商品URL>" --days 7 --limit 0
```

更多输入方式：

```powershell
python scripts/review_pipeline.py jd-reviews <task_name> --item-id <item_id> --days 7 --limit 0
python scripts/review_pipeline.py jd-reviews <task_name> --item-urls-file "<urls.txt>" --days 7 --limit 0
```

参数语义：
- `--days N`：最近 N 天；大于 0 时优先于 `--months`
- `--months N`：最近 N 个月；默认 `2`
- `--limit N`：每个商品最多保留 N 条评论；`0` 表示时间窗口内尽量全抓
- `--output-dir <path>`：自定义输出目录
- `--jd-browser-mode cdp|persistent`
- `--taobao-browser-mode cdp|persistent`

评论抓取完成后，必须向用户回报：
- 输出目录
- `total_reviews`
- 每个商品的 `count`
- `stopped_reason`
- 如果发生 CDP 降级或登录恢复，也要说明

---

## 阶段 1: 环境检查

**被唤起后第一件事**就运行：
```powershell
$env:PYTHONIOENCODING="utf-8"; python scripts/check_env.py
```

失败时检查：`.env` 中 `GEMINI_API_KEY` 是否配置、`websockets`/`httpx` 等依赖是否安装。

浏览器**自动管理**——不需要用户手动启动 Chrome 或配置 CDP 端口。

---

## 阶段 2: 解析意图

先判断任务是不是“独立评论抓取”。

### 2.0 评论抓取触发规则

只要用户明确提到以下意图之一，就走评论抓取独立入口，不进入阶段 3 的分析 pipeline：
- “爬取评论”
- “抓评论”
- “导出评论”
- “淘宝评论”
- “京东评论”
- “评价抓取”

评论抓取时提取四类参数：
- `platform`: `taobao` / `jd`
- `input_mode`: `item_urls` / `item_ids` / `item_urls_file`
- `days_or_months`: 时间窗口，优先 `days`
- `limit`: 评论上限；未指定时默认 `100`，明确说“全部”时用 `0`

若命中评论抓取意图，则直接运行：

```powershell
python scripts/review_pipeline.py <taobao-reviews|jd-reviews> <task_name> ...
```

不要继续套用下面的关键词调研逻辑。

### 2.1 商品分析/调研任务

从用户指令中提取三类参数：
- `input_mode`: 输入模式（`keyword` / `item_urls` / `item_urls_file`）
- `keyword`: 关键词模式下的目标商品词；URL 模式下用于命名任务的占位词
- `top_n`: 抓取数量（关键词模式默认 30）

### 输入模式识别规则
- `keyword`：用户给的是搜索词，例如“调研排名前 20 的猫粮”
- `item_urls`：用户直接给了一个或多个商品链接（淘宝/天猫详情页）
- `item_urls_file`：用户给了一个 `.txt` 文件路径，文件内每行一个商品链接

### 每种输入的处理方法
- `keyword`
  - 参数：`keyword=<搜索词>`，`top_n=<用户指定或30>`
  - 示例：`keyword=猫粮, top_n=20`
  - 若未指定数量，必须告知用户："我将抓取前 30 名销售爆款并输出。"
- `item_urls`
  - 参数：提取所有 URL 到 `item_urls[]`；将 `top_n` 设为 `len(item_urls)`（用户另有指定则取二者较小值）
  - 关键词占位：`keyword` 使用可追踪字符串（如 `direct-items-<日期>` 或 `tmall-item-<item_id>`）
  - 说明：该模式会跳过搜索阶段，直接抓取商品详情并分析
- `item_urls_file`
  - 参数：读取 txt（UTF-8）后得到 `item_urls[]`，过滤空行与重复行；`top_n` 默认 `len(item_urls)`（最大不超过系统上限）
  - 关键词占位：同 URL 模式，使用可追踪占位词
  - 说明：该模式同样跳过搜索阶段

### 混合输入优先级
- 若同时出现关键词与 URL/URL 文件，优先使用 URL 输入（`item_urls` 或 `item_urls_file`），并在回复中明确“已按链接直抓，跳过关键词搜索”。

---

## 阶段 3: 执行抓取

本阶段只适用于商品分析/调研任务。
如果当前任务是评论抓取，跳过本阶段，改用上面的 `review_pipeline.py` 命令模板。

**运行前必须告知用户**：
> 爬虫即将开始，浏览器将自动启动。如果跳转到登录页面，程序会自动暂停等待您扫码，登录后自动继续。

**执行命令**（PowerShell）：
```powershell
cd skills/taobao-insight
$env:PYTHONIOENCODING="utf-8"
$env:TAOBAO_USER_DATA_DIR="$env:APPDATA\\taobao_insight_profile"
$env:TAOBAO_STORAGE_STATE_FILE="skills\\taobao-insight\\data\\taobao_storage_state.json"
python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "<keyword>" --top-n <top_n> --output "data/exports/<keyword>-top<top_n>.md" --html-output "data/exports/<keyword>-top<top_n>.html"
```

### 3.1 关键词输入（`keyword`）
```powershell
python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "<keyword>" --top-n <top_n> --output "data/exports/<keyword>-top<top_n>.md" --html-output "data/exports/<keyword>-top<top_n>.html"
```

### 3.2 一个或多个链接输入（`item_urls`）
```powershell
python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "<placeholder_keyword>" --top-n <top_n> --item-url "<url_1>" --item-url "<url_2>" --output "data/exports/<placeholder_keyword>-top<top_n>.md" --html-output "data/exports/<placeholder_keyword>-top<top_n>.html"
```
- `<placeholder_keyword>` 仅用于任务命名，不参与搜索。
- `top_n` 建议等于链接数量（或不超过链接数量）。

### 3.3 txt 文件输入（`item_urls_file`）
```powershell
python scripts/pipeline.py --crawl-workers 1 --llm-workers 64 final-csv "<placeholder_keyword>" --top-n <top_n> --item-urls-file "<urls.txt>" --output "data/exports/<placeholder_keyword>-top<top_n>.md" --html-output "data/exports/<placeholder_keyword>-top<top_n>.html"
```
- `<urls.txt>` 要求：每行 1 个商品链接，建议 UTF-8 编码。
- 处理时需过滤空行、重复链接；`top_n` 默认取有效链接条数。

**参数速查**：
| 参数 | 说明 |
|---|---|
| `--crawl-workers 1` | 必须为 1，避免并发连接问题 |
| `--llm-workers 64` | Flash 模型 64，Pro 模型改为 16 |
| `--top-n N` | 抓取前 N 个商品 |
| `--taobao-browser-mode` | 默认 `cdp`（自动降级 persistent），一般不用指定 |
| `--playwright-cdp-url` | 可选，连接已有浏览器实例 |
| `--item-url` | 可重复，多次传入多个商品链接（跳过搜索阶段） |
| `--item-urls-file` | 可选，传入 URL 文件（每行一个链接，跳过搜索阶段） |

**登录行为**：
- 首次运行弹出浏览器等待扫码（最多 300 秒），后续运行自动复用 cookies
- 扫码期间程序冻结等待，登录后自动继续
- 终端出现 `login page content detected` 是正常的扫码等待日志，不要重试

**环境变量持久化**（避免重复扫码）：
- 必须固定 `TAOBAO_USER_DATA_DIR`（建议 `%APPDATA%\taobao_insight_profile`）
- 必须固定 `TAOBAO_STORAGE_STATE_FILE`（建议 `data/taobao_storage_state.json`）

---

## 阶段 4: 监控进度

任务耗时较长时向用户汇报进展：
- 看到 `[INFO]` 日志 → "正在利用 Gemini 分析商品列表与提炼卖点，请稍候..."
- 看到 `login page content detected` → 正常扫码等待，静待即可
- 看到 `search phase is still on item detail page after recovery` → 页面未恢复，需重新运行

---

## 阶段 5: 产出报告

当 `pipeline.py` 退出码为 `0` 时：

1. 确认两个文件已生成：`data/exports/<keyword>-top<top_n>.html` 和 `.md`
2. **不要**额外创建任何临时导出文件
3. 读取 HTML 或 MD 中关键结论，挑 1-2 个亮点给用户预览
4. **必须**读取 `data/exports/<workbook_name>-run-summary.md`，从中提取以下字段向用户报告：
   - `total_runtime_sec` — 总耗时
   - `timings_sec` 下的 `search`、`crawl`、`llm_extract`、`llm_analyze`、`export` — 各阶段耗时
   - `success_rate`、`total_items`、`success_items`、`failed_items` — 成功率与样本统计
   - `top_errors` — 关键错误列表（每条含 `reason` 和 `count`）
   - `key_conclusion` — 关键结论
5. **必须**读取 `data/run_logs/<task_id>.jsonl`，若有失败需指出集中在哪个阶段
6. 若运行中出现兜底/降级行为（如 CDP 降级到 persistent、LLM 重试、页面恢复等），也需在报告中提及

**输出示例**：
```
淘宝大盘深度调研完成！

仪表盘: skills/taobao-insight/data/exports/<keyword>-top<N>.html
Markdown: skills/taobao-insight/data/exports/<keyword>-top<N>.md

市场缩影：
- 样本量：20 个商品（成功 18 / 失败 2）
- 成功率：90.0%
- 市场标签：粉饼、定妆、持久

各阶段耗时：
- 搜索(search)：12.3s
- 抓取(crawl)：89.4s
- LLM 提取(llm_extract)：45.2s
- LLM 分析(llm_analyze)：38.7s
- 导出(export)：2.1s
- 总耗时：207.6s

错误与异常：
- Page.goto net::ERR_ABORTED（2 次）
- Gemini API timeout（1 次）
（若无错误则输出"无"）

兜底/降级事件：
- CDP 初始化失败，已自动降级到 persistent 模式
（若无降级事件则省略此段）

关键结论：<从 run-summary.md 的 key_conclusion 字段提取>
```

---

## 错误处理

| 错误信息 | 处理方式 |
|---|---|
| 评论任务只抓到很少几条 | 优先检查是否走到了评论独立入口 `review_pipeline.py`，不要误用分析 pipeline |
| 环境检测失败 | 检查 Python 依赖和 `.env` 配置 |
| 浏览器启动失败 | 关闭所有 Chrome 进程后重试 |
| `Chrome/Edge executable not found` | 设置 `CUSTOM_BROWSER_PATH` 环境变量 |
| `CDP init failed, fallback to persistent` | 非阻塞提示，已自动降级 |
| `PermissionError: [WinError 5]` | 权限不足，需在可启动浏览器的环境执行 |
| `Search failed: login page content detected` | 300 秒扫码超时，提示用户重新运行 |
| `taobao item page blocked: redirected to login` | 抓取阶段登录失效，重跑流程 |
| `Exception: Gemini API ...` | 检查 `.env` 的 Proxy 配置或重启网络 |
| 编码错误 | 确保设置了 `PYTHONIOENCODING=utf-8` |
| 浏览器登录成功但程序超时 | 页面仍被判定为登录态，需复检搜索页并落盘 cookie |
| `Browser not initialized` | 阻塞错误，重新运行 |
| 反复 `Page.goto net::ERR_ABORTED` | 阻塞错误，重新运行 |
