# Security Review Checklist for Kimi CLI

适用于需要比 `SKILL.md` 更细颗粒度的安全审查。

当出现以下情况时再读本文件：
- 变更跨多个子系统，难以快速定性
- 需要给出更严格的证据链
- 需要把问题明确分成缺陷 / 设计 / 加固点
- 需要输出更完整的 review 结果

## 目录

1. [先定 threat model](#先定-threat-model)
2. [先定比较基线](#先定比较基线)
3. [按模块聚类变更](#按模块聚类变更)
4. [高风险检查面](#高风险检查面)
5. [证据链怎么补](#证据链怎么补)
6. [分级建议](#分级建议)
7. [常见误报](#常见误报)
8. [输出模板](#输出模板)

## 先定 threat model

先问清楚或从上下文确认这些前提：

- 项目内 `AGENTS.md` / agent spec / skill 是否视为可信配置？
- 自家 web service / MCP server / provider 是否在可信边界内？
- 本地终端/UI 暴露是否算安全问题，还是只算体验/运维问题？
- 是否重点关注远程可利用风险，而非本地可信用户可见信息？

如果用户已明确这些前提，就必须按该 threat model 审查，不要擅自上调风险。

## 先定比较基线

优先确认比较对象，而不是直接看 HEAD。

推荐命令：

```bash
git branch -a --list
git log --oneline --decorate --no-merges main..HEAD
git diff --stat main...HEAD
```

如果用户说的是 `master`，但仓库没有：
- 明确说明仓库无 `master`
- 改用 `main`

然后对每个候选问题先回答：

1. 这个行为是当前分支新增的吗？
2. 还是基线早就存在？

如果基线就有，不要算作“这个分支新引入的问题”。

## 按模块聚类变更

建议先按下面维度聚类文件：

### A. Prompt / 指令 / 配置面

重点目录：
- `src/kimi_cli/agents/`
- `src/kimi_cli/soul/agent.py`
- `src/kimi_cli/soul/kimisoul.py`
- `AGENTS.md`
- `.agents/skills/`
- `packages/kosong/src/kosong/contrib/chat_provider/`

优先看：
- `system` / `developer` / `instructions`
- AGENTS 加载方式
- skill / slash / flow 是否引入更高权限提示
- tool 输出是否被提升为系统级信息

### B. 审批 / 执行面

重点目录：
- `src/kimi_cli/soul/approval.py`
- `src/kimi_cli/tools/shell/`
- `src/kimi_cli/tools/`
- `src/kimi_cli/tools/multiagent/`
- `src/kimi_cli/soul/kimisoul.py`

优先看：
- 是否新增未经审批的 Shell / 写文件 / 网络调用
- `yolo` / auto approve 是否被扩大
- subagent / flow / slash 是否能旁路审批

### C. 文件系统边界

重点目录：
- `src/kimi_cli/tools/file*`
- `src/kimi_cli/tools/__init__.py`
- `src/kimi_cli/soul/agent.py`
- `src/kimi_cli/soul/slash.py`

优先看：
- 是否新增工作目录外访问
- share dir / additional dirs / home dir 是否放宽
- 路径规范化是否可靠

### D. 网络 / 数据外发

重点目录：
- `src/kimi_cli/tools/web/`
- `src/kimi_cli/mcp.py`
- `packages/kosong/`

优先看：
- 是否新增 URL 抓取、service fallback、SSRF 面
- 请求/响应是否把敏感 header、body、trace id 带回上下文
- 数据是否被发给新的 provider / 服务

### E. UI / 日志 / 诊断面

重点目录：
- `src/kimi_cli/ui/`
- `src/kimi_cli/wire/`
- `src/kimi_cli/tools/__init__.py`

优先看：
- 完整命令、参数、token、错误体是否暴露到本地 UI
- 这些信息是否只是本地显示，还是会继续进入模型上下文

## 高风险检查面

## 1. 审批边界

逐项确认：

- 是否仍然所有 Shell 都经过 `Approval.request(...)`
- 是否新增了可以间接触发 Shell 的路径
- 是否新增了默认批准、持久化批准、继承批准的意外扩大
- subagent 是否 share 了不该 share 的审批状态

危险信号：
- “工具调用前没有 approval”
- “某条新路径只在 flow/slash/subagent 场景下绕过审批”
- “新引入的工具默认被自动批准”

## 2. Prompt / instructions 边界

逐项确认：

- 用户可控、仓库可控、服务端返回的内容，是否被放进：
  - `system`
  - `developer`
  - `instructions`
- 这种提升是否符合用户声明的信任模型
- 是否有未转义嵌套协议标记（如 `<system>...</system>`）

危险信号：
- 不可信输入被升格到高优先级指令通道
- tool error/output 变成 system-style 提示
- 项目外部来源内容被无界注入高信任 prompt

## 3. 文件系统边界

逐项确认：

- 是否允许读取工作目录外文件
- 是否允许写入工作目录外路径
- `additional_dirs` 是否有新增越权面
- share dir、home dir、绝对路径是否被更广泛引入

危险信号：
- 新逻辑引入 `..`、absolute path、home 目录而无明确边界说明
- 文件工具不再限制工作区
- slash/skill 自动读入工作区外文件

## 4. Web / 网络边界

逐项确认：

- `FetchURL` / `SearchWeb` 的 fallback 行为是否改变
- 服务错误 body 是否回灌到 context
- 请求头里是否新增敏感数据传播
- MCP / provider 是否新增外发路径

危险信号：
- 原本不联网的路径现在会联网
- 原本只本地可见的数据现在会发给模型服务
- SSRF 范围明显扩大

## 5. 信息泄露

逐项确认：

- 完整命令是否出现在 UI
- tool arguments 是否带 token / cookie / path / secret
- provider 错误是否包含 body / request id / internal details
- 错误 output 只是给本地看，还是会继续喂给模型

危险信号：
- 敏感信息被二次转发给第三方模型服务
- 调试信息被系统化长期保存在 context/session/log 中

## 6. DoS / 资源风险

逐项确认：

- timeout 是否明显变大
- 是否新增后台任务 / retry / 并发
- 是否可能造成长时间阻塞、死循环、上下文爆炸

说明：
- 这类问题通常优先定为可用性问题
- 除非它能被外部稳定触发并产生明显安全后果，否则不要轻易升为高危安全问题

## 证据链怎么补

当某个点像真实问题时，顺着调用链补读，不要只盯 diff。

常见补读链路：

### tool output 回灌链

看：
- 工具如何生成 `ToolReturnValue`
- `tool_result_to_message()` 如何把 output/message 转回上下文
- 最终这些内容是只在 UI 显示，还是继续喂给模型

### provider 参数外发链

看：
- `Runtime.create()` 或 provider wrapper 如何构造参数
- `instructions` / `system_prompt` / headers / body 最终如何传到第三方 API

### approval 传播链

看：
- 主 agent、fixed subagent、dynamic subagent 是否 share approval
- flow / slash / skill 是否经过相同审批路径

### UI 暴露链

看：
- 数据是仅本地 terminal 显示
- 还是 wire / ACP / context 中也会携带

## 分级建议

### 严重 / 高

只有在满足下面任一类情况时才考虑：
- 明确新增未经审批的命令执行或写文件
- 明确新增工作区边界突破
- 明确新增可现实利用的数据外传通道
- 明确新增高信任 prompt 边界突破，且输入不可信

### 中

适用于：
- 有较清晰攻击/泄露路径，但需要额外前提
- 对可信边界有一定侵蚀，但尚未形成直接高危后果

### 低

适用于：
- 本地显示面扩大
- 调试信息略多
- timeout / 资源策略变化
- 更偏可用性、体验、运维问题

### 不算缺陷

可归入：
- 明确的设计取舍
- 符合用户给定 threat model 的信任输入
- 基线早已存在的问题

## 常见误报

下面这些在 Kimi CLI 里经常容易被误报：

1. **把项目内 agent config / skill / AGENTS 当成不可信输入**
   - 如果用户明确说可信，就不要继续按 prompt injection 漏洞报

2. **把本地 UI 展示增强直接说成严重泄露**
   - 先判断是否仅本地可见
   - 再判断是否继续回灌到模型/第三方

3. **把 timeout 变大直接说成安全漏洞**
   - 多数情况下这是可用性/体验取舍

4. **忽略基线**
   - 如果 main 已经有同样行为，就不是当前分支新引入

5. **把 provider 行为变化自动上升为漏洞**
   - 先看是不是同一信任边界内的实现调整

## 输出模板

优先给用户一个短表格：

| 项目 | 定性 | 风险级别 | 说明 |
|---|---|---:|---|
| 审批边界 | 缺陷 / 设计 / 加固点 | 低/中/高 | 一句话 |
| Prompt 边界 | 缺陷 / 设计 / 加固点 | 低/中/高 | 一句话 |
| UI / 诊断 | 缺陷 / 设计 / 加固点 | 低/中/高 | 一句话 |

然后给一句总评：

- **未发现严重安全隐患**
- **存在中低风险问题，但不足以称为严重隐患**
- **发现严重安全隐患**

如果用户想一项一项过：
- 主动把每一项拆成“设计 / 缺陷 / 加固点”
- 根据用户给的 threat model 动态下调或上调定性
- 最后再收敛成极简表格
