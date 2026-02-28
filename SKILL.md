---
name: taobao-insight
description: Taobao/Tmall keyword product research and competitor analysis. Use when user asks for 淘宝/天猫商品调研、关键词选品、竞品分析、抓取N条商品、卖点提炼、导出CSV或HTML报告（例如：淘宝狗粮关键词调研20条）。
---

# 淘宝市场调研 (Taobao Market Research)

> **前置说明**: 本系统采用双引擎驱动：底层 Playwright 自动化抓取，上层大模型总结卖点。请严格遵守以下 5 个**循序渐进的执行阶段**，切勿跳过检查步骤擅自运行爬虫。
>
> **重要**: Windows 系统下执行命令时需要设置 UTF-8 编码环境，否则会出现编码错误。所有 Python 命令前需要设置 `PYTHONIOENCODING=utf-8`。

## 工作流程

你将按照以下 5 个阶段执行淘宝市场调研任务：

### 阶段 1: 运行环境检查

**目标**: 确保用户的环境健全且依赖已安装。

你必须在你被唤起的**第一句话**执行这个诊断（注意设置 UTF-8 编码）：
```bash
# Windows CMD
set PYTHONIOENCODING=utf-8 && python scripts/check_env.py

# PowerShell
$env:PYTHONIOENCODING="utf-8"; python scripts/check_env.py
```

**如果环境检测失败**：
- 检查是否缺少 Playwright 或其他依赖
- 检查 `.env` 文件中的 `GEMINI_API_KEY` 是否配置
- 根据错误提示安装缺失的依赖

**注意**: 本系统使用 GlobalBrowserManager 自动管理浏览器，**不需要**用户手动启动 Chrome CDP 端口。浏览器会在执行时自动启动。

---

### 阶段 2: 任务意图解析与参数补齐

**目标**: 确认用户想要调研的`目标商品词`和`抓取梳理数量`（默认 Top 30）。

1. 如果用户指令明确说了"调研排名前20的猫粮"，则记录: `keyword=猫粮`, `top_n=20`。
2. 如果用户只说"跑一下防晒霜市场"，则默认追问或提示："我将抓取前 30 名的销售爆款并输出。准备开跑了。"

---

### 阶段 3: 执行提取引擎并告知"扫码可能"

**目标**: 启动 `pipeline.py` 进行数据获取并告知用户人工介入点。

由于淘宝风控原因，在调用爬虫之前，你**必须**向用户输出这段预警信息：
> 💡 爬虫即将开始，浏览器将自动启动...
> *重要提示：如果浏览器跳转到淘宝登录页面，程序会自动暂停并等待您扫码登录。登录完成后程序会自动继续执行。*

**登录流程说明**：
- **首次运行**：会在搜索阶段自动检测登录页面，自动弹出浏览器窗口并等待扫码登录（最多 300 秒）
- **后续运行**：cookies 会自动保存，**不需要**重新登录
- 登录成功后程序会自动继续，**不需要**手动操作
- 扫码等待期间程序会进入**冻结模式**（不执行后台刷新/跳转），避免干扰扫码与登录回跳
- 若扫码成功但页面未自动跳转，程序会依据登录 cookie 判定成功并主动跳回目标搜索页
- 若搜索阶段误入商品页，程序会自动回跳到搜索页继续抓取
- 若抓取阶段会话过期跳到登录页，程序会明确标记为 session 失效并终止该商品抓取
- 程序结束时会自动保存登录状态到用户配置目录

**Cookies 保存位置**：
- Windows: `%APPDATA%\taobao_insight_profile`
- macOS: `~/Library/Application Support/taobao_insight_profile`
- Linux: `~/.config/taobao_insight_profile`

执行入口命令（**注意设置 UTF-8 编码**）：
```bash
# PowerShell 执行��式
cd skills/taobao-insight
$env:PYTHONIOENCODING="utf-8"
$env:TAOBAO_BROWSER_MODE="persistent"
$env:TAOBAO_USER_DATA_DIR="$env:APPDATA\\taobao_insight_profile"
$env:TAOBAO_STORAGE_STATE_FILE="skills\\taobao-insight\\data\\taobao_storage_state.json"
python scripts/pipeline.py --use-global-browser 1 --crawl-workers 1 --llm-workers 8 --taobao-browser-mode persistent final-csv "<用户提供的 keyword>" --top-n <用户指定的数字> --search-backend playwright --output "data/exports/<keyword>-top<top_n>.md" --html-output "data/exports/<keyword>-top<top_n>.html"
```

**关键参数说明**：
- `--taobao-browser-mode persistent`: 使用持久化浏览器模式（**自动启动浏览器**，不需要手动启动 CDP）
- `--use-global-browser 1`: 启用全局浏览器管理器（单一事件循环架构，登录只需一次）
- `--crawl-workers 1`: 使用单一事件循环（重要：避免 Playwright 连接问题）
- `--search-backend playwright`: 使用 Playwright 进行搜索（必须）
- `--top-n N`: 抓取前 N 个商品

#### 登录态持久化要求（避免重复扫码）

- 必须固定 `TAOBAO_USER_DATA_DIR` 到稳定目录（建议 `%APPDATA%\\taobao_insight_profile`）。
- 必须固定 `TAOBAO_STORAGE_STATE_FILE` 到稳定文件（建议 skill 的 `data/taobao_storage_state.json`）。
- 登录成功后立即保存 storage_state，不要等任务结束再保存。

#### 扫码登录问题复盘（2026-02-27）

1. **现象：扫码很多次没有反应**
   - 根因：任务进程未真正执行（命令被中断、权限被拦截，或进程已提前退出）。
   - 正确做法：确认终端日志持续输出且任务未退出后再扫码。
2. **现象：环境检查提示“必须开启 9222”**
   - 根因：把 CDP 模式的前提误用到了 persistent 模式。
   - 正确做法：`persistent` 模式无需手动开启 9222；仅 `cdp` 模式才需要。
3. **现象：扫码后进入商品页，而不是搜索页**
   - 根因：浏览器复用页或登录回跳导致页面上下文漂移。
   - 正确做法：搜索阶段必须识别“误入商品页”并强制回到目标搜索 URL（已内置）。
4. **现象：运行过程中突然变成登录页**
   - 根因：cookie 失效、风控二次校验或会话被刷新。
   - 正确做法：搜索阶段触发登录等待并恢复；抓取阶段识别为会话失效并给出明确错误。

#### 页面状态识别规则（必须遵守）

- **搜索阶段**：当前页应为 `s.taobao.com/search` 或 `list.tmall.com/search_product.htm`。
- **若识别到商品详情页 URL**（`detail.tmall.com/item.htm` 或 `item.taobao.com/item.htm`）：立即导航回目标搜索 URL。
- **若识别到登录/验证码页**（URL 或页面内容命中登录特征）：触发扫码等待流程。
- **恢复后复检**：必须再次确认处于搜索页；否则直接报错并提示用户重跑。
- **登录判定优先级**：Cookie 信号优先于页面跳转；当出现“扫码成功但不跳转”时按 cookie 驱动恢复。

---

### 阶段 4: 监控异步产出并跟踪状态

**目标**: 长耗时的爬虫应当向用户透明进展。

因为 Gemini 序列分析多个商品和提取详情需要可能数十秒甚至两分钟。你可以使用后台监控或者在提取途中穿插状态汇报。
例如，如果读取到 `[INFO]` 或看到浏览器正在操作，可以告诉用户：
`"正在利用大模型（Gemini）分析商品列表与提炼卖点文档，请稍候..."`

*(若终端打印 `detect_non_product_page: login page content detected` 时，不用惊慌也不用重试，这就是用户正在进行合法扫码，静待即可。)*
*(若终端提示 `search phase is still on item detail page after recovery`，说明页面上下文未恢复成功，需要重新运行并确保浏览器停留在搜索页。)*

---

### 阶段 5: 分析出炉与物料清点

**目标**: 向用户提供成果报告路径及汇总概览结论。

所有的数据都会保存在工作区的 `data/exports/` 目录中。当 `pipeline.py` 退出值为 `0` 时，整理输出：

1. 按执行命令中指定的路径确认两个目标文件已生成：`data/exports/<keyword>-top<top_n>.html` 与 `data/exports/<keyword>-top<top_n>.md`
2. 不要额外创建任何临时导出文件（例如 `*-products.*` 或其它额外副本）
3. （重要）读取刚刚生成的 HTML 或 MD 中的关键结论，挑选其中最震撼的 1-2 点给用户高能预览！

例如向用户展示：
```
✨ 淘宝大盘深度调研完成！

📊 仪表盘: skills/taobao-insight/data/exports/<商品名>-top<N>-<timestamp>.html
📝 Markdown: skills/taobao-insight/data/exports/<商品名>-top<N>-<timestamp>.md

📝 市场缩影：
- 样本量：5 个商品
- 已提取卖点：3 个
- 有价格信息：5 个
- 市场标签：粉饼、定妆、持久
```

---

## 错误处理边界法则

1. **环境检测挂了**: 检查 Python 依赖和 .env 配置
2. **浏览器启动失败**: 检查是否有其他浏览器进程占用，关闭所有 Chrome 进程后重试
3. **`PermissionError: [WinError 5] 拒绝访问`（Playwright 启动阶段）**: 表示运行环境权限不足（常见于受限沙箱），需在可启动浏览器的环境执行
4. **`RuntimeError: Search failed: login page content detected`**: 说明用户没能在 300 秒内完成手机扫码认证。提示用户：”300秒超时，登录未成功。请重新运行，程序会再次提示登录。”
5. **`RuntimeError: taobao item page blocked: redirected to Taobao login page`**: 说明抓取阶段登录态失效；应重新跑流程并在搜索阶段完成扫码
6. **`Exception: Gemini API ...`**: 大概率被封IP或触发网络限速，检查 `.env` 配置的 Proxy 或者重启网络
7. **编码错误**: 确保设置了 `PYTHONIOENCODING=utf-8`
8. **”浏览器显示登录成功但程序仍超时”**: 多数是页面仍被判定为登录态（未回到搜索页）。需要自动复检搜索结果页并在登录成功后立即落盘 cookie

记住：本技能的超凡之处在于 **跨越极高反爬壁垒拿到底层网页** 且 **内化了大模型卖点萃取逻辑**，用清晰的流水线汇报进展并保证终端产出物的可用与可读。
