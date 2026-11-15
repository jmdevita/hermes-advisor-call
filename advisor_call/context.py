"""Per-session conversation capture and transcript rendering.

Capture comes from two places: the ``pre_llm_call`` hook (once per turn:
history plus the user message, sender and platform) and the ``llm_request``
middleware (every main-loop request, including tool results from the
current turn). The newer snapshot wins. Both store references; nothing is
deep-copied.

``render_transcript`` turns either snapshot into plain text for the
advisor. It accepts OpenAI-style messages (Hermes' internal format) and
Anthropic-style content blocks.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

MAX_SESSIONS = 256
TOOL_ARGS_CHARS = 400
OMITTED_PREFIX = "...[earlier conversation omitted]...\n"
ADVISOR_TOOL_PREFIX = "advisor_call_"

Redactor = Callable[[str], str]


# --------------------------------------------------------------------------- capture


@dataclass
class Snapshot:
    messages: list[Any]
    seq: int


@dataclass
class SessionContext:
    hook: Snapshot | None = None
    request: Snapshot | None = None
    sender_id: str = ""
    platform: str = ""
    seen: bool = False  # pre_llm_call has run for this session

    def latest(self) -> list[Any]:
        candidates = [s for s in (self.hook, self.request) if s is not None]
        if not candidates:
            return []
        return max(candidates, key=lambda s: s.seq).messages


class ContextStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._max = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, SessionContext] = OrderedDict()
        self._seq = itertools.count(1)

    def _get(self, session_id: str) -> SessionContext:
        ctx = self._sessions.get(session_id)
        if ctx is None:
            ctx = SessionContext()
            self._sessions[session_id] = ctx
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
        return ctx

    def record_turn(
        self,
        session_id: str | None,
        *,
        conversation_history: Any = None,
        user_message: Any = None,
        sender_id: Any = None,
        platform: Any = None,
    ) -> None:
        """From pre_llm_call."""
        if not session_id:
            return
        history = list(conversation_history) if isinstance(conversation_history, list) else []
        if user_message:
            last = history[-1] if history else None
            already = (
                isinstance(last, dict)
                and last.get("role") == "user"
                and last.get("content") == user_message
            )
            if not already:
                history.append({"role": "user", "content": user_message})
        with self._lock:
            ctx = self._get(session_id)
            ctx.hook = Snapshot(history, next(self._seq))
            ctx.seen = True
            if sender_id:
                ctx.sender_id = str(sender_id)
            if platform:
                ctx.platform = str(platform)

    def record_request(self, session_id: str | None, messages: Any) -> None:
        """From llm_request middleware. Messages are stored by reference."""
        if not session_id or not isinstance(messages, list):
            return
        with self._lock:
            ctx = self._get(session_id)
            ctx.request = Snapshot(messages, next(self._seq))

    def messages(self, session_id: str | None) -> list[Any]:
        if not session_id:
            return []
        with self._lock:
            ctx = self._sessions.get(session_id)
            return ctx.latest() if ctx else []

    def identity(self, session_id: str | None) -> tuple[str, str] | None:
        """(sender_id, platform) for a session seen by pre_llm_call, or None
        when this session has not been seen (or was evicted)."""
        if not session_id:
            return None
        with self._lock:
            ctx = self._sessions.get(session_id)
            return (ctx.sender_id, ctx.platform) if ctx is not None and ctx.seen else None

    def forget(self, session_id: str | None) -> None:
        if session_id:
            with self._lock:
                self._sessions.pop(session_id, None)


# --------------------------------------------------------------------------- rendering


def _cut_middle(text: str, limit: int) -> str:
    """Keep the head and the tail of text, at most `limit` characters of it."""
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    head = limit // 2
    tail = limit - head
    return f"{text[:head]}\n...[{omitted} characters omitted]...\n{text[-tail:]}"


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _args_text(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(args)


def _tool_call_name_args(tc: Any) -> tuple[str, str]:
    if not isinstance(tc, dict):
        return "?", ""
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name") or "?"
    args = fn.get("arguments", tc.get("arguments", tc.get("input", "")))
    return str(name), _args_text(args)


def _same(text: str) -> str:
    return text


def _parts_text(
    content: Any, tool_chars: int, skip_ids: set[str], clean: Redactor = _same
) -> list[str]:
    """Render a content value: a string or a list of parts/blocks. Every
    piece is redacted (clean) before anything cuts it, so a cut can never
    split a secret and leave part of it unrecognisable."""
    if content is None:
        return []
    if isinstance(content, str):
        return [clean(content)] if content else []
    if not isinstance(content, list):
        return [clean(str(content))]
    out: list[str] = []
    for part in content:
        if isinstance(part, str):
            out.append(clean(part))
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            if part.get("text"):
                out.append(clean(str(part["text"])))
        elif kind in ("image", "image_url", "input_image"):
            out.append("[image]")
        elif kind == "tool_use":  # Anthropic-style assistant block
            if part.get("id") in skip_ids:
                continue
            args = _short(clean(_args_text(part.get("input", ""))), TOOL_ARGS_CHARS)
            out.append(f"[tool call] {part.get('name', '?')}({args})")
        elif kind == "tool_result":  # Anthropic-style user block
            if part.get("tool_use_id") in skip_ids:
                continue
            inner = "\n".join(_parts_text(part.get("content"), tool_chars, skip_ids, clean))
            out.append(f"[tool result]\n{_cut_middle(inner, tool_chars)}")
        elif kind in ("thinking", "redacted_thinking"):
            continue
        elif part.get("text"):
            out.append(clean(str(part["text"])))
    return out


def _in_flight_advisor_ids(messages: list[Any]) -> set[str]:
    """Ids of advisor_call_* tool calls that have no result yet."""
    calls: set[str] = set()
    answered: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            name, _ = _tool_call_name_args(tc)
            if name.startswith(ADVISOR_TOOL_PREFIX) and isinstance(tc, dict) and tc.get("id"):
                calls.add(str(tc["id"]))
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            answered.add(str(msg["tool_call_id"]))
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if (
                    part.get("type") == "tool_use"
                    and str(part.get("name", "")).startswith(ADVISOR_TOOL_PREFIX)
                    and part.get("id")
                ):
                    calls.add(str(part["id"]))
                if part.get("type") == "tool_result" and part.get("tool_use_id"):
                    answered.add(str(part["tool_use_id"]))
    return calls - answered


def render_transcript(
    messages: list[Any], *, limit_chars: int, tool_chars: int, redact: Redactor | None = None
) -> str:
    """Plain-text transcript of a session for the advisor.

    System messages are skipped, tool output is cut in the middle to
    tool_chars, the in-flight advisor tool call is left out, and when the
    total is over limit_chars the tail is kept. Redaction runs on every
    piece before any cut, and once more over the final text."""
    skip_ids = _in_flight_advisor_ids(messages)
    clean: Redactor = redact or _same
    blocks: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown")
        if role in ("system", "developer"):
            continue
        if role == "tool":
            if str(msg.get("tool_call_id", "")) in skip_ids:
                continue
            body = "\n".join(_parts_text(msg.get("content"), tool_chars, skip_ids, clean))
            name = msg.get("name")
            label = f"### TOOL RESULT{f' ({name})' if name else ''}"
            blocks.append(f"{label}\n{_cut_middle(body, tool_chars)}")
            continue
        lines = _parts_text(msg.get("content"), tool_chars, skip_ids, clean)
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict) and str(tc.get("id", "")) in skip_ids:
                continue
            name, args = _tool_call_name_args(tc)
            lines.append(f"[tool call] {name}({_short(clean(args), TOOL_ARGS_CHARS)})")
        if not lines:
            continue
        blocks.append(f"### {role.upper()}\n" + "\n".join(lines))

    text = "\n\n".join(blocks)
    if limit_chars > 0 and len(text) > limit_chars:
        keep = max(0, limit_chars - len(OMITTED_PREFIX))
        tail = text[-keep:] if keep else ""
        # Start at a message boundary when one is close by.
        boundary = tail.find("\n### ")
        if 0 <= boundary < len(tail) // 2:
            tail = tail[boundary + 1 :]
        text = OMITTED_PREFIX + tail
    if redact is not None:
        text = redact(text)
    return text


# --------------------------------------------------------------------------- redaction

_FALLBACK_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{16,}=*"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
]


def fallback_redact(text: str) -> str:
    """Minimal redaction used only if Hermes' redactor cannot be imported."""
    for pattern in _FALLBACK_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def hermes_redactor() -> Redactor:
    """Hermes' public redactor, or the built-in fallback."""
    try:
        from agent.redact import redact_sensitive_text

        def redact(text: str) -> str:
            # This text leaves the machine: redact even if the user turned
            # Hermes' own redaction off, and strip credentials from URLs.
            return redact_sensitive_text(text, force=True, redact_url_credentials=True)

        return redact
    except Exception:  # noqa: BLE001
        log.warning("advisor-call: Hermes redactor unavailable; using built-in fallback")
        return fallback_redact
