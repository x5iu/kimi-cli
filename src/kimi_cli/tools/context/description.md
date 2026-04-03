Recall details from previously compacted conversation context.

After context compaction, the summary may not contain every detail. Use this tool to retrieve specific information from earlier in the conversation — file paths, error messages, code snippets, function names, design decisions, or user instructions that were discussed before compaction.

When to use:
- You reference something discussed earlier but the details are missing from the current context
- The user refers to prior work, decisions, or instructions ("as we discussed", "like before")
- You need exact error messages, stack traces, or code from previous steps
- You want to verify what was already tried or decided

What it does:
- Lists available compacted-context archives with summaries and key topics
- Searches archived pre-compaction messages by targeted keywords
- Returns small, relevant excerpts instead of the whole archive

Tips:
- Leave `query` empty first to see available archives, their summaries, and suggested search terms
- Use specific queries: file paths, function names, error strings, IDs, or distinctive keywords
- Use `archive_id` to narrow search when you know which archive is relevant
- Prefer this tool over guessing or asking the user to repeat themselves
