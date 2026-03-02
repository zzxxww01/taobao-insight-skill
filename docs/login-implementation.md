# 浏览器登录与会话管理技术指南（通用）

## 1. 目标
这是一份可复用的工程指南，用于实现“浏览器自动化 + 登录态管理”模块。
核心要求只有两条：
- 主路径：原生 CDP（Chrome DevTools Protocol）
- 备路径：Persistent Context（同一套 profile 目录）

不允许并行维护多套主链路，不允许保留历史试验分支。

## 2. 不可妥协的原则
1. 默认模式必须是 `cdp`。
2. CDP 初始化失败时必须自动回退到 `persistent`，调用方不需要改命令。
3. 登录检测与恢复必须是独立模块，搜索/抓取/发布等业务层只调用，不复制逻辑。
4. 参数最小化：只保留真正生效且有明确语义的参数。
5. 任何“可传但无效果”的参数都要删除。

## 3. 推荐模块边界
1. `cdp_browser`：原生 CDP 连接与页面抽象（ws、target、session、evaluate、screenshot）。
2. `browser_manager`：统一入口，负责 `cdp -> persistent` 自动回退。
3. `login_rules`：登录页/风控页检测规则（URL + 标题 + 内容 + DOM）。
4. `login_handler`：登录恢复状态机（二维码等待、成功判定、超时退出）。
5. `business_xxx`：业务流程（搜索、抓取、发布等），只编排，不实现登录细节。

## 4. CDP 主链路（参考实现）
1. 启动浏览器并开启 `--remote-debugging-port`。
2. 轮询 `http://127.0.0.1:{port}/json/version` 获取 `webSocketDebuggerUrl`。
3. 建立 WebSocket 连接。
4. 通过 `Target.createTarget / Target.attachToTarget` 获取页面 session。
5. 以 session 维度执行 `Page/Runtime/DOM/Network` 指令。
6. 在 manager 层统一暴露 `new_page/get_page/cleanup`。

## 5. Persistent 备链路
1. 与 CDP 使用同一个 user data/profile 目录策略。
2. 仅在 CDP 初始化失败时启用，且自动切换。
3. 业务层无需感知当前是 CDP 还是 persistent。

## 6. 登录检测与恢复（通用状态机）
1. 先做快照：`url/title/body`。
2. 检测是否被登录墙/风控墙拦截。
3. 若被拦截，进入登录恢复：
   - 拉前浏览器窗口（可选）
   - 冻结等待人工扫码/验证
   - 周期性检查“页面状态 + 关键 cookie 指纹”
4. 成功条件：
   - 页面离开登录墙，或
   - 关键 cookie 发生有效变化并满足判定规则
5. 超时条件：达到 `login_timeout_sec` 仍未满足成功条件。

## 7. 会话持久化
1. Profile 目录使用用户级固定路径（Windows: `%APPDATA%`；Linux: `XDG_DATA_HOME`/`~/.local/share`）。
2. 每次登录恢复成功后立即持久化 storage state。
3. 同一机器/账号复用同一 profile，避免重复扫码。

## 8. 观测与诊断
必须输出结构化事件（例如 `login_recovery_events`），至少包含：
- `source`（search/crawl/publish）
- `stage`
- `ok`
- `reason`
- `final_state`
- `elapsed_sec`
- `url`
- `decision_trace`
- `updated_at`

## 9. 参数设计最小集合
建议仅保留：
- `browser_mode`（`cdp|persistent`，默认 `cdp`）
- `cdp_url`（可选，接入外部浏览器）
- `user_data_dir`
- `storage_state_file`
- `login_timeout_sec`
- `manual_wait_seconds`（仅页面加载后的额外等待）

## 10. 必须删除的反模式
1. 多个并行浏览器主链路（例如 raw CDP + playwright CDP socket + http mirror）。
2. 历史实验参数长期保留在 CLI/构造函数中。
3. 业务模块复制登录判断逻辑。
4. 登录失败后静默重试无限循环。
5. 依赖“人工切模式”而不是自动回退。

## 11. 交给 Claude Code 的执行模板
```text
请按以下约束实现浏览器登录模块（通用，不针对单一网站）：
1) 默认主路径为原生 CDP；失败时自动回退到 persistent。
2) 登录检测与恢复写成独立模块，业务层仅调用。
3) 删除所有无效参数和历史分支，不保留“可传但无效”的选项。
4) 输出结构化登录恢复事件，便于排障。
5) 保持 profile 与 storage state 可复用，避免重复登录。
6) 最终提供：代码修改、参数变更清单、运行验证结果。
```
