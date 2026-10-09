"""Conversation -> dense archive text for snapcompact.

Faithful, bounded serialization of the discarded middle: tool results truncated
head+tail, tool-call arguments capped per value and per call, tool output kept
distinct from conversation. No paraphrasing, no LLM — the archive text is what
gets rasterized onto the frames, and (bounded) what later compactions re-render
from, so prior archives survive re-compaction without OCR.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    _MERGED_SUMMARY_DELIMITER,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _SUMMARY_END_MARKER,
    is_compaction_summary_message,
)

SNAPCOMPACT_ARCHIVE_KEY = "_snapcompact_archive"


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if part.get("type") == "text":
            return str(part.get("text", "") or "")
        if part.get("type") in ("image_url", "input_image", "image"):
            return "[image attached]"
    return ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for t in (_part_text(p) for p in content) if t)
    return "" if content is None else str(content)


def _trunc_head_tail(text: str, max_chars: int, head_ratio: float) -> str:
    """Keep head+tail of overlong text with an explicit elision marker."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * head_ratio)
    tail = max(0, max_chars - head)
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n...[elided {omitted} chars]...\n"
        + (text[-tail:] if tail else "")
    )


def _trunc_args(args: Any, per_value: int, per_call: int) -> str:
    """JSON-dump tool-call args with each string value capped at ``per_value``."""
    def cap(v: Any) -> Any:
        if isinstance(v, str) and len(v) > per_value:
            return v[:per_value] + f"...[{len(v) - per_value} chars elided]"
        if isinstance(v, dict):
            return {k: cap(x) for k, x in v.items()}
        if isinstance(v, list):
            return [cap(x) for x in v]
        return v

    try:
        if isinstance(args, str):
            import json

            args = json.loads(args) if args.strip() else {}
        capped = cap(args)
        import json

        out = json.dumps(capped, ensure_ascii=False, default=str)
    except Exception:
        out = str(args)
    if len(out) > per_call:
        out = out[:per_call] + f"...[{len(out) - per_call} chars elided]"
    return out


def _tool_name(msg: Dict[str, Any]) -> str:
    return str(msg.get("name") or msg.get("tool_call_id") or "tool")


def serialize_messages(
    middle: List[Dict[str, Any]], *, tool_result_max_chars: int = 2_000,
    tool_result_head_ratio: float = 0.6, tool_arg_max_chars: int = 500,
    tool_call_max_chars: int = 2_000,
) -> str:
    """Serialize the compressible middle into conversation-dense archive text."""
    blocks: List[str] = []
    for msg in middle:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        # Prior compaction summaries: snapcompact archives re-embed their bounded
        # source text; foreign/merged summaries embed their text (minus end
        # marker). The merged-carrier delimiter is the durable signal — metadata
        # keys don't survive the merge path, the delimiter does.
        merged_at = -1
        content = msg.get("content")
        if isinstance(content, str):
            merged_at = content.find(_MERGED_SUMMARY_DELIMITER)
        elif isinstance(content, list):
            merged_at = next(
                (i for i, p in enumerate(content)
                 if isinstance(p, dict) and _MERGED_SUMMARY_DELIMITER in str(p.get("text", ""))),
                -1,
            )
        if msg.get(SNAPCOMPACT_ARCHIVE_KEY) or is_compaction_summary_message(msg) or merged_at >= 0:
            archive = msg.get(SNAPCOMPACT_ARCHIVE_KEY) or {}
            source = archive.get("text") if isinstance(archive, dict) else None
            if not source:
                # Merged carrier: summary text lives after the delimiter.
                def _after_delim(text: str) -> str:
                    idx = text.find(_MERGED_SUMMARY_DELIMITER)
                    if idx < 0:
                        return ""
                    after = text[idx + len(_MERGED_SUMMARY_DELIMITER):]
                    return after.split(_SUMMARY_END_MARKER)[0].strip()

                if merged_at >= 0:
                    if isinstance(content, str):
                        source = _after_delim(content)
                    else:
                        source = "\n".join(filter(None, (
                            _after_delim(str(p.get("text", "") if isinstance(p, dict) else p))
                            for p in content[merged_at:]
                        )))
                if not source:
                    source = _content_text(content).split(_SUMMARY_END_MARKER)[0]
                    source = source.split(_MERGED_PRIOR_CONTEXT_HEADER)[-1]
            blocks.append(f"[PRIOR CONTEXT SUMMARY]\n{source}")
            continue

        text = _content_text(msg.get("content")).strip()
        if role == "tool":
            body = _trunc_head_tail(text, tool_result_max_chars, tool_result_head_ratio)
            blocks.append(f"[TOOL {_tool_name(msg)}]\n{body}" if body else "")
        elif role == "assistant":
            parts = []
            if text:
                parts.append(text)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name") or "?"
                args = _trunc_args(fn.get("arguments"), tool_arg_max_chars, tool_call_max_chars)
                parts.append(f"[TOOL_CALL {name}({args})]")
            if parts:
                blocks.append("[ASSISTANT]\n" + "\n".join(parts))
        elif role in ("user", "system", "developer"):
            if text:
                blocks.append(f"[{role.upper()}]\n{text}")
        elif text:
            blocks.append(f"[{str(role).upper()}]\n{text}")
    return "\n\n".join(b for b in blocks if b)


def bound_archive_text(text: str, max_chars: int, head_ratio: float = 0.34) -> str:
    """Bound the persistent archive source (kept in the summary row metadata)."""
    return _trunc_head_tail(text, max_chars, head_ratio)
