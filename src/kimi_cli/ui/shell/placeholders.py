from __future__ import annotations

from .placeholder_cache import (
    AttachmentCache,
    CachedAttachment,
    CachedAttachmentKind,
    parse_attachment_kind,
)
from .placeholder_manager import (
    PromptPlaceholderManager,
    ResolvedPromptCommand,
    build_pasted_text_placeholder,
    count_text_lines,
    normalize_pasted_text,
    sanitize_surrogates,
    should_placeholderize_pasted_text,
)

_parse_attachment_kind = parse_attachment_kind

__all__ = [
    "AttachmentCache",
    "CachedAttachment",
    "CachedAttachmentKind",
    "PromptPlaceholderManager",
    "ResolvedPromptCommand",
    "_parse_attachment_kind",
    "build_pasted_text_placeholder",
    "count_text_lines",
    "normalize_pasted_text",
    "parse_attachment_kind",
    "sanitize_surrogates",
    "should_placeholderize_pasted_text",
]
