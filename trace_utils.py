"""
Utilities for extracting and rendering chat traces from AutoGen agents.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=True)
    return str(content)


def _normalize_function_call(call: Any) -> Optional[Dict[str, Any]]:
    if not call:
        return None
    return {
        "name": _get(call, "name"),
        "arguments": str(_get(call, "arguments", "")).strip(),
    }


def _normalize_tool_call(call: Any) -> Dict[str, Any]:
    func = _get(call, "function")
    return {
        "id": _get(call, "id"),
        "type": _get(call, "type"),
        "function": _normalize_function_call(func),
    }


def _normalize_message(message: Any) -> Dict[str, Any]:
    return {
        "role": _get(message, "role"),
        "name": _get(message, "name"),
        "tool_call_id": _get(message, "tool_call_id"),
        "content": _stringify_content(_get(message, "content")),
        "function_call": _normalize_function_call(_get(message, "function_call")),
        "tool_calls": [
            _normalize_tool_call(call) for call in _get(message, "tool_calls", []) or []
        ],
    }


def _resolve_message_log(
    assistant: Any, user_proxy: Any
) -> Iterable[Any]:
    pairs = (
        (getattr(user_proxy, "chat_messages", None), _get(assistant, "name")),
        (getattr(assistant, "chat_messages", None), _get(user_proxy, "name")),
    )
    for messages, peer_name in pairs:
        if isinstance(messages, dict):
            if peer_name and messages.get(peer_name):
                return messages[peer_name]
            for value in messages.values():
                if value:
                    return value
        if isinstance(messages, list) and messages:
            return messages
    return []


def collect_trace(assistant: Any, user_proxy: Any) -> List[Dict[str, Any]]:
    """
    Extract a normalized trace from the conversation between the two agents.
    """
    raw_messages = _resolve_message_log(assistant, user_proxy)
    return [_normalize_message(msg) for msg in raw_messages]


def render_trace(trace: List[Dict[str, Any]]) -> str:
    """
    Produce a human-readable trace showing thinking and tool calls.
    """
    lines: List[str] = []
    for idx, msg in enumerate(trace, start=1):
        role = msg.get("role") or "unknown"
        name = msg.get("name")
        header = f"{idx:02d}. {role}"
        if name:
            header += f" [{name}]"
        if msg.get("tool_call_id"):
            header += f" (tool_call_id={msg['tool_call_id']})"
        lines.append(header)

        if msg.get("function_call"):
            fn = msg["function_call"]
            lines.append(f"    fn -> {fn.get('name')}({fn.get('arguments')})")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            lines.append(
                f"    tool -> {fn.get('name')}({fn.get('arguments')}) [id={call.get('id')}]"
            )
        content = msg.get("content")
        if content:
            for line in content.splitlines():
                lines.append(f"    {line}")
        if not msg.get("content") and not msg.get("function_call") and not msg.get("tool_calls"):
            lines.append("    (no content)")
    return "\n".join(lines)


def write_trace_json(trace: List[Dict[str, Any]], path: str) -> None:
    """
    Write the trace to disk as JSON lines for downstream analysis.
    """
    with open(path, "w", encoding="utf-8") as f:
        for entry in trace:
            json.dump(entry, f, ensure_ascii=True)
            f.write("\n")
