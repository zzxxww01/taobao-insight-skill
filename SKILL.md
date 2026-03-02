---
name: taobao-insight
description: Taobao/Tmall keyword product research and competitor analysis. Use when user asks for 淘宝/天猫商品调研、关键词选品、竞品分析、抓取N条商品、导出MD或HTML报告（例如：淘宝狗粮关键词调研20条）。
---

# 淘宝市场调研

原生 CDP 自动化抓取 + 大模型卖点提炼。按以下 5 阶段顺序执行，不要跳步。

> Windows 下所有 Python 命令前必须设置 `PYTHONIOENCODING=utf-8`。

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

从用户指令中提取两个参数：
- `keyword`: 目标商品词
- `top_n`: 抓取数量（默认 30）

示例："调研排名前 20 的猫粮" → `keyword=猫粮, top_n=20`

如果用户没指定数量，告知："我将抓取前 30 名销售爆款并输出。"

---

## 阶段 3: 执行抓取

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

**参数速查**：
| 参数 | 说明 |
|---|---|
| `--crawl-workers 1` | 必须为 1，避免并发连接问题 |
| `--llm-workers 64` | Flash 模型 64，Pro 模型改为 16 |
| `--top-n N` | 抓取前 N 个商品 |
| `--taobao-browser-mode` | 默认 `cdp`（自动降级 persistent），一般不用指定 |
| `--playwright-cdp-url` | 可选，连接已有浏览器实例 |
| `--item-urls-file` | 可选，传入已有 URL 列表跳过搜索阶段 |

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
