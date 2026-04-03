# Kimi Code CLI Tools

## Guidelines

- Except for `Task` tool, tools should not refer to any types in `kimi_cli/eventbus/`. When importing things like `ToolReturnValue`, `DisplayBlock`, import from `llmkit.tooling`.
