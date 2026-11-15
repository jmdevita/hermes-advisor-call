"""Auto-suggest: notice when the executor looks stuck, and nudge.

Deterministic: counters and fixed rules over tool results, no model calls.
It never calls the advisor itself and never blocks a tool; it appends a
one-line hint to the tool result that tripped a rule (the executor decides
what to do) and, optionally, tells the user at the end of the turn if the
executor did not call. Design and the research behind it: DESIGN.md 2.8.

Signals, per session, reset when the user sends a new message and after an
advisor call:

- error streak: N tool errors in a row;
- same error: the same (normalised) error message N times in the turn,
  even with other calls in between;
- loop: the same tool, arguments *and* result N times in a row (a repeated
  call whose result changes, such as polling, is not a loop);
- cycle: a repeating sequence of 2 to 5 such calls, seen twice in a row.

Identical calls from one model response (one batch) count once.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

MAX_SESSIONS = 256
HISTORY = 12  # recent fingerprints kept per session
ADVISOR_TOOL_PREFIX = "advisor_call_"
# Hermes' own tool-search bridge: catalogue reads, not work.
BRIDGE_TOOLS = frozenset({"tool_search", "tool_describe"})
UNATTENDED_PLATFORMS = frozenset({"cron"})
MODES = ("off", "executor", "user", "both")

NUDGE_PREFIX = "[advisor-call]"


@dataclass(frozen=True)
class SuggestSettings:
    mode: str = "both"
    error_streak: int = 3
    same_error: int = 3
    loop: int = 3
    cycles: bool = True
    cooldown_calls: int = 10
    max_per_session: int = 3
    ignore_tools: frozenset[str] = frozenset()

    @property
    def nudges_executor(self) -> bool:
        return self.mode in ("executor", "both")

    @property
    def tells_user(self) -> bool:
        return self.mode in ("user", "both")


# --------------------------------------------------------------------------- fingerprints

_VOLATILE = [
    (re.compile(r"0x[0-9a-f]+", re.I), "0x#"),
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "#hex"),
    (re.compile(r"\d+"), "#"),
    (re.compile(r"\s+"), " "),
]


def normalize_error(message: str) -> str:
    """Error text with line numbers, ids, addresses and spacing smoothed out,
    so the same failure compares equal across attempts."""
    text = message.strip().lower()
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return text[:300]


def normalize_args(args: Any) -> str:
    """Arguments as canonical JSON (sorted keys, no spacing). Never raises."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return args
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(args)


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def fingerprint(tool_name: str, args: Any, result: Any) -> tuple[str, str, str]:
    return (tool_name, _digest(normalize_args(args)), _digest(str(result)))


# --------------------------------------------------------------------------- tracker


@dataclass
class _State:
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    last_batch: str = ""
    errors_in_row: int = 0
    error_counts: dict[str, int] = field(default_factory=dict)
    calls_since_nudge: int = 10**9
    nudges: int = 0
    pending_notice: str = ""  # reason, for the user, if the executor does not call
    open_suggestion: bool = False  # a suggestion this turn, not yet followed by a call


@dataclass(frozen=True)
class Detection:
    reason: str  # factual, one line, for the hint and the user notice
    kind: str  # error_streak | same_error | loop | cycle


class Tracker:
    """Per-session counters. Thread-safe; bounded to MAX_SESSIONS (LRU)."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._max = max_sessions
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _State] = OrderedDict()

    def _get(self, session_id: str) -> _State:
        state = self._sessions.get(session_id)
        if state is None:
            state = _State()
            self._sessions[session_id] = state
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
        return state

    def new_turn(self, session_id: str | None) -> None:
        """The user sent a message: start counting afresh (keep the session's
        nudge count, so the per-session limit holds)."""
        if not session_id:
            return
        with self._lock:
            state = self._get(session_id)
            state.history.clear()
            state.last_batch = ""
            state.errors_in_row = 0
            state.error_counts.clear()
            state.pending_notice = ""
            state.open_suggestion = False

    def advisor_called(self, session_id: str | None) -> bool:
        """An advisor call happened: reset the counters. Returns True if it
        followed a suggestion made this turn."""
        if not session_id:
            return False
        with self._lock:
            state = self._get(session_id)
            followed = state.open_suggestion
            state.history.clear()
            state.errors_in_row = 0
            state.error_counts.clear()
            state.pending_notice = ""
            state.open_suggestion = False
            return followed

    def take_notice(self, session_id: str | None) -> str:
        """The reason to tell the user at the end of the turn, once."""
        if not session_id:
            return ""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or not state.pending_notice:
                return ""
            reason, state.pending_notice = state.pending_notice, ""
            return reason

    def observe(
        self,
        session_id: str,
        settings: SuggestSettings,
        *,
        tool_name: str,
        args: Any,
        result: Any,
        failed: bool,
        error_message: str,
        batch: str,
        allowed: Callable[[], bool] = lambda: True,
    ) -> Detection | None:
        """Record one tool result; return a Detection when a rule trips and a
        suggestion may be made now: cooldown and session limit respected, and
        allowed() true (checked only then; a refused suggestion is not
        counted against the cooldown)."""
        fp = fingerprint(tool_name, args, result)
        with self._lock:
            state = self._get(session_id)
            # Identical calls from the same model response count once.
            if batch and batch == state.last_batch and state.history and state.history[-1] == fp:
                return None
            state.last_batch = batch
            state.history.append(fp)
            state.calls_since_nudge += 1
            if failed:
                state.errors_in_row += 1
                key = normalize_error(error_message or str(result)[:300])
                state.error_counts[key] = state.error_counts.get(key, 0) + 1
                same = state.error_counts[key]
            else:
                state.errors_in_row = 0
                same = 0

            detection = self._detect(state, settings, tool_name, failed, same, error_message)
            if detection is None:
                return None
            if state.nudges >= settings.max_per_session:
                return None
            if state.calls_since_nudge < settings.cooldown_calls:
                return None
            if not allowed():
                return None
            state.nudges += 1
            state.calls_since_nudge = 0
            state.open_suggestion = True
            if settings.tells_user:
                state.pending_notice = detection.reason
            return detection

    @staticmethod
    def _detect(
        state: _State,
        settings: SuggestSettings,
        tool_name: str,
        failed: bool,
        same: int,
        error_message: str,
    ) -> Detection | None:
        history = list(state.history)
        n = settings.loop
        if n > 0 and len(history) >= n and len(set(history[-n:])) == 1:
            return Detection(
                f"'{tool_name}' was called {n} times in a row with the same "
                "arguments and got the same result",
                "loop",
            )
        if settings.cycles:
            for length in range(2, 6):
                if (
                    len(history) >= 2 * length
                    and history[-length:] == history[-2 * length : -length]
                    and len(set(history[-length:])) > 1
                ):
                    names = " -> ".join(fp[0] for fp in history[-length:])
                    return Detection(
                        f"a repeating cycle of tool calls ({names}) came back "
                        "with the same results",
                        "cycle",
                    )
        if failed and settings.same_error > 0 and same >= settings.same_error:
            short = " ".join((error_message or "").split())[:120]
            detail = f": {short}" if short else ""
            return Detection(
                f"the same error came back {same} times this turn{detail}", "same_error"
            )
        if failed and settings.error_streak > 0 and state.errors_in_row >= settings.error_streak:
            return Detection(f"{state.errors_in_row} tool calls failed in a row", "error_streak")
        return None


NOTE_FIELD = "advisor_call_note"


def hint_text(reason: str) -> str:
    """Factual: says what happened and offers options, does not prescribe a fix."""
    return (
        f"{NUDGE_PREFIX} {reason}. If you are stuck, you can get a second opinion with "
        "advisor_call_open (the user must approve it), or ask the user for guidance."
    )


def annotate(result: str, reason: str) -> str:
    """The tool result with the hint added. A JSON object result stays valid
    JSON (the hint goes in an extra field); anything else gets a final line."""
    text = hint_text(reason)
    try:
        parsed = json.loads(result)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and NOTE_FIELD not in parsed:
        parsed[NOTE_FIELD] = text
        return json.dumps(parsed, ensure_ascii=False)
    return f"{result}\n\n{text}"
