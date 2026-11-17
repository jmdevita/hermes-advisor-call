"""Auto-suggest: tell the user when Hermes flags a tool loop and the executor
does not call the advisor.

Hermes has its own loop guardrails (agent/tool_guardrails.py): it appends a
warning to the tool result when the executor repeats a failing or identical
call. This module does no detection of its own and never touches a tool
result. It reads Hermes' warning from the next model request (llm_request
middleware) and, if the turn ends without an advisor call, the plugin adds
one line for the user to the reply. Hermes' own settings decide when a loop
is flagged; turn its warnings off and this goes quiet too. DESIGN.md 2.8.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_SESSIONS = 256
MAX_PER_SESSION = 3  # user notices per session
UNATTENDED_PLATFORMS = frozenset({"cron"})
TAIL_CHARS = 2000  # Hermes appends its warning at the end of the result

# The start of each note Hermes appends (hermes-agent 0.23: tool_guardrails.py
# append_toolguard_guidance, _IDENTICAL_CALL_NOTICE, _IDENTICAL_CYCLE_NOTICE).
_WARNING = re.compile(
    r"\[Tool loop (?:warning|hard stop): (?P<code>[a-z_]+);"
    r"|\[hermes note: (?P<call>this is the \S+ consecutive identical call)"
    r"|\[hermes note: (?P<cycle>the last \d+ rounds repeated the same cycle)"
)


def _text(content: Any) -> str:
    """The text of a message's content: a string, or a list of parts (which
    may nest, as in Anthropic tool_result blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(p) for p in content)
    if isinstance(content, dict):
        return _text(content.get("text") or content.get("content") or "")
    return ""


def hermes_warning(messages: Any) -> str | None:
    """The kind of loop warning Hermes attached to the newest tool results
    (the messages after the last assistant message), or None."""
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") == "assistant":
            break
        text = _text(msg.get("content")).rstrip()
        if not text.endswith("]"):
            continue
        matches = list(_WARNING.finditer(text[-TAIL_CHARS:]))
        if matches:
            found = matches[-1]
            if found.group("code"):
                return found.group("code")
            return "identical_call" if found.group("call") else "identical_cycle"
    return None


@dataclass
class _State:
    pending: bool = False  # a warning this turn, not followed by a call
    called: bool = False  # the executor called the advisor this turn
    notices: int = 0


class Tracker:
    """Per-session flags. Thread-safe; bounded to MAX_SESSIONS (LRU)."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._max = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _State] = OrderedDict()

    def _get(self, session_id: str) -> _State:
        state = self._sessions.get(session_id)
        if state is None:
            state = self._sessions[session_id] = _State()
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
        return state

    def new_turn(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            state = self._get(session_id)
            state.pending = state.called = False

    def flagged(self, session_id: str, allowed: Callable[[], bool] = lambda: True) -> bool:
        """Hermes flagged a loop. True the first time in a turn that a notice
        will be owed (at most MAX_PER_SESSION per session, and only when
        allowed() says a call could happen)."""
        with self._lock:
            state = self._get(session_id)
            if state.pending or state.called or state.notices >= MAX_PER_SESSION:
                return False
            if not allowed():
                return False
            state.pending = True
            state.notices += 1
            return True

    def advisor_called(self, session_id: str | None) -> bool:
        """The executor called the advisor. True if that followed a warning
        this turn."""
        if not session_id:
            return False
        with self._lock:
            state = self._get(session_id)
            followed, state.pending, state.called = state.pending, False, True
            return followed

    def take_notice(self, session_id: str | None) -> bool:
        """Whether to tell the user at the end of the turn, once."""
        if not session_id:
            return False
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or not state.pending:
                return False
            state.pending = False
            return True
