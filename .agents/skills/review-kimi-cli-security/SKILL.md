---
name: review-kimi-cli-security
description: Review Kimi Code CLI branches, PRs, or commits for security issues. Use when the user asks for 安全 review / security audit / 漏洞审查 of Kimi CLI changes, especially around prompts, approvals, AGENTS/skills, MCP, shell/file/web tools, providers, or data leakage.
---

# review-kimi-cli-security

用于审查 **Kimi Code CLI** 代码变更中的安全问题。

默认目标：
- 找出 **相对基线新引入** 的安全风险，而不是泛泛谈整个仓库
- 区分 **真实缺陷**、**设计取舍**、**仅可加固项**
- 明确是否存在 **严重安全隐患**

## 何时使用

当用户提出以下需求时使用：
- “帮我 review 这个分支/PR/commit，聚焦安全问题”
- “相比 main/master 有没有严重安全隐患？”
- “请做安全审计，重点看 prompt / tool / MCP / 权限 / 信息泄露”
- “这些改动是不是设计，不算漏洞？”

## 审查原则

1. **先确认基线**
   - 用户说 `master`，但仓库没有 `master` 时，先检查是否应使用 `main`
   - 明确比较对象后再下结论

2. **只看新增风险**
   - 先判断问题是否是当前分支新引入的
   - 如果基线已存在，同类问题不要算到本分支头上

3. **遵守用户 threat model**
   - 如果用户明确说“项目内 agent 配置可信”“自家 web service 可信”，就不要把这类设计默认上升为漏洞
   - 不要脱离用户前提做过度告警

4. **区分三类结论**
   - **安全缺陷**：存在清晰攻击路径、边界被突破、权限/数据暴露新增
   - **设计取舍**：行为更激进或更开放，但符合既定信任模型
   - **加固点**：理论上可更稳，但不足以算漏洞

5. **优先关注真实边界**
   - 审批边界
   - 文件系统边界
   - 网络边界
   - prompt / developer / system 指令边界
   - secrets / 错误体 / 日志 / UI 暴露边界

## Kimi CLI 的高风险热点

审查时优先看这些目录和模块：

- `src/kimi_cli/soul/`
  - `approval.py`：审批、auto approve、yolo
  - `agent.py`：runtime、AGENTS.md、skills、subagents
  - `kimisoul.py`：主循环、context 注入、tool 结果回灌、slash/skill/flow
  - `context.py` / `compaction.py`：上下文保存与压缩
- `src/kimi_cli/tools/`
  - `shell/`：命令执行、timeout、环境变量、审批
  - `web/`：搜索、抓取、service fallback、错误体
  - 文件工具：工作目录边界、绝对路径、写文件/替换
  - `multiagent/`：Task / subagent 边界
- `src/kimi_cli/ui/` 与 `src/kimi_cli/wire/`
  - 是否把敏感参数、token、错误体、完整命令暴露到终端/UI/IDE
- `src/kimi_cli/agents/`、`AGENTS.md`、`.agents/skills/`
  - 提示词、instructions、skills、slash command
- `packages/kosong/`
  - provider 适配层，尤其是 `system` / `developer` / `instructions` / tool conversion
- `src/kimi_cli/mcp.py`、MCP 加载相关逻辑
  - 外部工具加载、配置来源、信任边界

## 重点检查项

### 1. 审批与权限边界

重点问：
- 是否新增了绕过审批执行 Shell / 写文件 / 网络请求的路径？
- 是否扩大了 `yolo`、`auto_approve_actions` 或持久化状态的影响面？
- 子 agent / flow / slash command 是否能间接绕过原本的审批？

### 2. Prompt / 指令信任边界

重点问：
- 不可信内容是否进入了 `system` / `developer` / `instructions` 级别？
- tool 输出、error body、AGENTS、SKILL、用户输入是否被“升格”成更高信任级别？
- 这是否在用户给定 threat model 下仍然成立？

### 3. 文件系统与工作目录边界

重点问：
- 是否新增对工作目录外的读写执行？
- 是否把绝对路径、额外目录、share dir、home dir 引入更宽的访问面？
- 是否有路径规范化/校验缺失导致的越界？

### 4. 网络与外部服务

重点问：
- 是否新增 SSRF、任意 URL 访问、service fallback 的危险行为？
- 错误响应、header、trace id、URL 参数、token 是否被回显到上下文或 UI？
- 数据是否被发往新的 provider / service？

### 5. 信息泄露

重点问：
- 完整命令、API 错误、tool 参数、错误 body、日志是否暴露敏感信息？
- 这些信息只是本地可见，还是会继续喂给模型/第三方服务？

### 6. DoS / 资源风险（次优先级）

重点问：
- timeout、重试、后台任务、并发、无限循环是否明显放大资源占用？
- 这是安全问题，还是单纯可用性/体验取舍？

## 推荐工作流

### 第一步：确认比较范围

优先用 diff-first 方法，不要先通读全仓。

可参考：

```bash
git branch -a --list
git log --oneline --decorate --no-merges main..HEAD
git diff --stat main...HEAD
```

如果用户指定 `master` 但不存在：
- 明确告知仓库无 `master`
- 改为与 `main` 比较

### 第二步：按安全面聚类文件

把变更按下面几类分组，再优先读高风险文件：
- prompt / instructions / AGENTS / skills
- approval / shell / file / web tools
- provider / MCP / subagent
- UI / logging / diagnostics

### 第三步：对每个候选问题做五连问

对每个你怀疑的问题，都回答这 5 个问题：

1. **这是当前分支新引入的吗？**
2. **攻击者可控输入是什么？**
3. **穿越了哪条信任边界？**
4. **在当前 threat model 下，这条攻击路径真实吗？**
5. **这是缺陷、设计取舍，还是仅加固点？**

只要有一个问题回答不扎实，就不要上升为“严重漏洞”。

### 第四步：必要时补读调用链

当某个点可能是安全问题时，再补读调用链，例如：
- tool output 如何进入 message/context
- provider 参数如何传到第三方 API
- approval 如何在主 agent / subagent / flow 之间传播
- UI 是否会把结果展示给本地用户，还是回灌给模型

### 第五步：给出分级明确的结论

输出时优先使用简短表格，至少包含：
- 项目
- 定性（缺陷 / 设计 / 加固点）
- 风险级别
- 一句话理由

## 参考资料

当需要更细的检查项、分级标准或输出模板时，读取：
- [references/security-checklist.md](references/security-checklist.md)

## 结论模板

优先给用户一个短表格：

| 项目 | 定性 | 风险级别 | 结论 |
|---|---|---:|---|
| Prompt / instructions | 设计 / 缺陷 / 加固点 | 低/中/高 | 一句话 |
| Tool / approval | 设计 / 缺陷 / 加固点 | 低/中/高 | 一句话 |

然后补一句总评：

- **有严重安全隐患**：仅当存在明确、现实的高危攻击路径
- **未发现严重安全隐患**：可同时注明有哪些中低风险点或设计取舍

## 不要这样做

- 不要把“项目内可信配置”默认当成不可信输入
- 不要把所有 prompt 变化都算 prompt injection 漏洞
- 不要把本地 UI 多显示一点错误信息自动定成高危
- 不要把可用性问题直接说成严重安全问题
- 不要忽略基线，拿老问题指责新分支
- 不要只给抽象判断，不给文件和逻辑证据

## Kimi CLI 审查时常见的合理结论

以下情况常常更接近 **设计** 而非 **缺陷**：
- 项目内 `AGENTS.md`、agent spec、project skill 被视为可信配置
- 自家可信 web service 返回更多错误体用于诊断
- Shell/UI 仅改善本地展示或 timeout 策略，但未改变审批边界

以下情况更可能是 **真实缺陷**：
- 未经审批新增 Shell / 写文件 / 外部网络路径
- 不可信内容被升格为 `system` / `developer` / `instructions`
- 敏感数据从本地或服务端被无意回灌给模型或第三方
- 工作目录/额外目录边界被突破

## 最后提醒

这是一个 **Kimi CLI 定制安全审查 skill**，不是通用 AppSec 清单。

始终围绕这几个问题下结论：
1. 这是不是当前改动新引入的？
2. 它在 Kimi CLI 的真实架构里能否触发？
3. 它在当前 threat model 下算不算缺陷？
4. 是否足以称为“严重安全隐患”？
