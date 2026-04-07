A structured search tool wrapping ripgrep. Use as a fallback when the Shell tool with `rg` is unavailable or when you prefer a structured interface with built-in pagination.

**Tips:**
- When a `<system-hint>` gives you an `rg` path, prefer Shell with `rg` for maximum flexibility.
- Use this tool when Shell is not available, or for simple searches where structured output (files_with_matches, count_matches) is convenient.
- Use the ripgrep pattern syntax, not grep syntax. E.g. you need to escape braces like `\\{` to search for `{`.
- When context budget is tight, set `head_limit` ≤ 50 and use `glob` or `type` to narrow the search scope.
- Hidden files (dotfiles like `.gitlab-ci.yml`, `.eslintrc.json`) are always searched. To also search files excluded by `.gitignore` (e.g. `node_modules`, build outputs), set `include_ignored` to `true`. Sensitive files (such as `.env`) are still skipped for safety, even when `include_ignored` is `true`.
