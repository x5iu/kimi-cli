---
name: review-origin-main
description: Review new commits on `origin/main` for the current branch using a movable review-base tag. Use when the user wants to know what from `origin/main` is worth cherry-picking or manually porting, without merge/rebase, and wants the review tag advanced afterward.
---

# review-origin-main

用于持续审查 `origin/main` 相对某个工作分支的新改动，并维护一个“已 review 到哪里”的 tag。

这个 skill 的目标不是制造假的 ancestry，而是：
- 找出 `origin/main` 上新增、且对当前分支有价值的改动
- 明确哪些适合 `cherry-pick`，哪些只适合手工移植，哪些应忽略
- 在 review 完成后，把 review-base tag 前移到新的基线 commit

## 适用场景

当用户提出类似需求时使用：
- “看看 `origin/main` 最近有什么值得合到我的分支”
- “我不想 merge/rebase，只想持续跟踪 main 的新增改动”
- “帮我 review 从上次基线之后的 main 改动，并更新 tag”
- “只关心 shell/core，看看哪些值得抄”

## 核心约定

### 1. review-base tag 的含义

`review-base/<branch>-main` 表示：
- **当前分支已经 review 到 `origin/main` 的这个 commit 为止**
- **不表示这个 commit 已经被 merge/rebase 到当前分支**
- **也不表示所有代码都被完整吸收**；可能只是部分 `cherry-pick` 或手工移植

例子：
- `review-base/allstar-main`

### 2. 不要伪造历史

为了“标记已经看过”，不要用这些手法：
- 不要 `merge`
- 不要 `rebase`
- 不要 `merge -s ours`
- 不要把 review-base tag 解释成“已经合入”

### 3. git mutation 必须先确认

创建或移动 tag 前，必须先用 `AskUserQuestion` 确认：
- tag 名称
- 指向哪个 commit

## 推荐流程

### 1. 确认目标分支与审查范围

先确定：
- 当前要服务的分支名，默认用当前分支
- 审查范围是否有限制

常见范围：
- 全量 review
- `shell`
- `core`
- `shell + core`
- 用户指定路径

如果用户只关心 shell/core，优先过滤这些路径：
- `src/kimi_cli/ui/shell`
- `src/kimi_cli/loop`
- `src/kimi_cli/eventbus`
- `src/kimi_cli/app.py`
- `src/kimi_cli/cli.py`
- `packages/llmkit/src/llmkit/contrib/chat_provider/openai_responses.py`
- 对应测试目录

### 2. 拉取最新 `origin/main`

```bash
git fetch origin
```

不要在旧的远端状态上做 review。

### 3. 找到上次 review 基线

默认 tag 名：

```bash
BRANCH="$(git branch --show-current)"
TAG="review-base/${BRANCH}-main"
```

查看 tag 是否存在：

```bash
git rev-parse -q --verify "refs/tags/${TAG}^{commit}"
```

处理规则：
- 如果 tag 存在：用 `TAG..origin/main` 作为本次增量 review 范围
- 如果 tag 不存在：这是第一次 bootstrap review。改用 `git merge-base "$BRANCH" origin/main` 作为临时起点，并在结论里明确说明“这是首次建基线”

示例：

```bash
BASE=$(git rev-parse -q --verify "refs/tags/${TAG}^{commit}" 2>/dev/null || true)
if [ -z "$BASE" ]; then
  BASE=$(git merge-base "$BRANCH" origin/main)
fi
```

### 4. 列出新增 commit

全量：

```bash
git log --reverse --no-merges --format='%H%x09%s' "$BASE..origin/main"
```

只看 shell/core 相关路径：

```bash
git log --reverse --no-merges --format='%H%x09%s' "$BASE..origin/main" -- \
  src/kimi_cli/ui/shell \
  src/kimi_cli/loop \
  src/kimi_cli/eventbus \
  src/kimi_cli/app.py \
  src/kimi_cli/cli.py \
  packages/llmkit/src/llmkit/contrib/chat_provider/openai_responses.py
```

### 5. 先按 commit 和文件做粗分组

先用 `diff-first` 方法，不要一上来通读所有文件。

推荐顺序：
1. `git log --oneline --decorate "$BASE..origin/main"`
2. `git show --stat <commit>`
3. 只对候选 commit 再看 `git show --patch --unified=3 <commit> -- <relevant-paths>`

优先关注：
- `shell` 交互行为变化
- `soul` 主循环、context、compaction、steer、plan mode
- `wire` 事件协议变化
- provider 行为变化

默认降级关注：
- docs
- web/vis/setup/oauth
- 纯版本 bump
- 和用户 scope 无关的扩展功能

## 推荐判定标签

每个候选 commit 必须给出以下几类之一：
- `直接 cherry-pick`
- `手工移植`
- `已被当前分支覆盖`
- `不相关`
- `暂缓`

### `直接 cherry-pick`

适用条件：
- 改动小而集中
- 语义独立
- 与当前分支冲突面小
- 当前分支没有明显的本地重写

### `手工移植`

适用条件：
- 思路值得要，但当前分支在同一区域已经深改
- upstream commit 带了大量 UI/TUI 重构，不宜原样拿
- 只想吸收其中一部分逻辑

### `已被当前分支覆盖`

适用条件：
- 当前分支已有等价修复或更完整实现
- upstream 方案更旧，或者与本分支路线冲突

### `不相关`

适用条件：
- 不在用户关心范围内
- 只是 docs / bump / web / vis / setup 等旁支内容

### `暂缓`

适用条件：
- 值得要，但依赖更多上游 commit
- 当前证据不足，需要补读调用链或实测

## 冲突风险检查

遇到候选 commit 时，检查当前分支是否已在同一区域深改。

```bash
MB=$(git merge-base "$BRANCH" origin/main)
comm -12 \
  <(git diff --name-only "$MB..$BRANCH" | sort) \
  <(git diff --name-only "$MB..origin/main" | sort)
```

如果 commit 主要落在重叠文件里：
- 默认提高冲突风险
- 更倾向 `手工移植`
- 不要轻易建议直接 `cherry-pick`

`git cherry -v "$BRANCH" origin/main` 可以作为提示，但不能替代代码判断。

## 输出模板

结果优先用表格，按“值得处理”优先排序：

| Commit | 范围 | 建议 | 风险 | 一句话理由 |
|---|---|---|---|---|
| `abc12345` | `shell` | `直接 cherry-pick` | 低 | 小修复、依赖少、当前分支未覆盖 |
| `def67890` | `core` | `手工移植` | 中 | 思路值得要，但当前分支同文件已有深改 |
| `ghi11111` | `docs` | `不相关` | 低 | 不在本次关心范围 |

然后补一个总评：
- 哪 1 到 3 个最值得拿
- 哪些不要直接拿
- 如果用户只关心某一块，结论必须围绕该块，不要发散

## 何时前移 tag

只有在以下条件满足时，才建议前移 review-base tag：
- 本次 review 范围已经明确
- 已经把“值得拿/不值得拿”的结论说清楚
- 用户认可要把基线推进到某个 commit

注意：
- tag 可以指向 `origin/main` 当前 tip
- 也可以只指向一个中间 commit
- 如果本次只 review 到一半，只推进到“已经 review 完”的那个 commit，不要盲目打到 tip

## 打 tag 的标准做法

创建或更新 annotated tag：

```bash
TAG="review-base/${BRANCH}-main"
TARGET="<reviewed-commit>"

git tag -fa "$TAG" "$TARGET" \
  -m "$BRANCH reviewed through origin/main@$TARGET; not merged/rebased; selected changes may be cherry-picked or manually ported"
```

创建后，展示给用户确认结果：

```bash
git show --no-patch --decorate "$TAG"
```

## 下次 review 的起点

下次直接从 tag 之后开始：

```bash
git fetch origin
git log --oneline "$TAG..origin/main"
```

如果要限定路径：

```bash
git log --oneline "$TAG..origin/main" -- src/kimi_cli/ui/shell src/kimi_cli/loop
```

## 最后提醒

这个 skill 解决的是“review 进度基线”问题，不是“代码合并状态”问题。

始终明确区分：
- **review 到哪里了**：看 `review-base/<branch>-main`
- **代码实际拿了哪些**：看 `cherry-pick`、手工 patch、当前分支真实 diff

不要把二者混为一谈。
