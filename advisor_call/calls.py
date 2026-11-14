"""Call state: one open call per session, caps, idle expiry, locking.

All state is in memory and keyed by the session_id Hermes passes. There is
no fallback to "the most recent session" anywhere: a lookup either names
the caller's own session or fails.

The lock is never held across the advisor request. An exchange is
``reserve`` (under the lock: checks, mark in flight) -> network call
(outside the lock) -> ``settle`` (under the lock: record cost, append
messages, maybe end the call).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .cost import Usage

log = logging.getLogger(__name__)

MAX_SESSIONS = 64

EndReason = Literal[
    "",
    "hangup",
    "user_hangup",
    "bill_cap",
    "day_cap",
    "month_cap",
    "turn_limit",
    "idle",
    "session_end",
    "error",
]

END_REASON_TEXT: dict[str, str] = {
    "hangup": "hung up",
    "user_hangup": "the user ended the call",
    "bill_cap": "bill cap reached",
    "day_cap": "today's advisor allowance is used up",
    "month_cap": "this cycle's advisor budget is used up",
    "turn_limit": "message limit reached",
    "idle": "ended after being idle",
    "session_end": "session ended",
    "error": "ended after an error",
}


def end_reason_text(reason: str) -> str:
    return END_REASON_TEXT.get(reason, reason or "ended")


def new_call_id() -> str:
    return f"c-{secrets.token_hex(4)}"


@dataclass
class Call:
    call_id: str
    session_id: str
    line: str
    started_at: float
    last_activity: float
    cap_usd: float
    initiated_by: Literal["executor", "user"]
    status: Literal["open", "ended"] = "open"
    end_reason: str = ""
    ended_at: float | None = None
    messages: list[dict[str, str]] = field(default_factory=list)  # advisor-side conversation
    exchanges: int = 0
    cost_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)
    in_flight: bool = False
    estimated: bool = False  # part of the bill is an estimate (cache usage hidden)
    # Prompt prefix (estimated tokens) the last successful exchange left in
    # Anthropic's cache, and when. Used to estimate cache reads.
    cached_est: float = 0.0
    cached_at: float | None = None
    # How the token estimator compared with what Hermes reported on this
    # call (>1 means the text is denser than estimated). Scales the next
    # pre-flight estimate up, never down.
    token_scale: float = 1.0

    @property
    def is_open(self) -> bool:
        return self.status == "open"


@dataclass
class Grant:
    """A /second-opinion allow pre-authorization for one session."""

    line: str
    cap_usd: float | None
    granted_at: float
    notified: bool = False


@dataclass
class Reservation:
    res_id: int
    call: Call
    text: str  # the executor message being sent
    messages: list[dict[str, str]]  # call.messages + the pending user message
    est_cost: float


@dataclass
class Settled:
    call: Call
    ended_now: str  # end reason if this exchange ended the call, else ""


class CallError(Exception):
    """A lookup or state problem the tool reports to the executor.

    code: no_session | no_call | wrong_call | busy | ended
    """

    def __init__(self, code: str, message: str, call: Call | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.call = call


# Preflight: (call, messages incl. the pending one, reserved_elsewhere) ->
# (end_reason or "", estimated cost of this exchange)
Preflight = Callable[[Call, list[dict[str, str]], float], tuple[str, float]]


class CallManager:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        max_sessions: int = MAX_SESSIONS,
        on_call_end: Callable[[Call], None] | None = None,
    ) -> None:
        self._clock = clock
        self._max = max_sessions
        self._on_call_end = on_call_end
        self._lock = threading.Lock()
        self._open: OrderedDict[str, Call] = OrderedDict()
        self._ended: OrderedDict[str, Call] = OrderedDict()  # last ended call per session
        self._grants: OrderedDict[str, Grant] = OrderedDict()
        self._activity: OrderedDict[str, None] = OrderedDict()
        self._reservations: dict[int, Reservation] = {}
        self._next_res = 0
        self.idle_timeout_s: float = 30 * 60.0

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _check_session(session_id: str | None) -> str:
        if not session_id or not isinstance(session_id, str):
            raise CallError("no_session", "no session id; cannot identify the call")
        return session_id

    @staticmethod
    def _touch(od: OrderedDict, key: str, value, limit: int) -> None:
        od[key] = value
        od.move_to_end(key)
        while len(od) > limit:
            od.popitem(last=False)

    def _end_locked(self, call: Call, reason: str, ended: list[Call]) -> None:
        if not call.is_open:
            return
        call.status = "ended"
        call.end_reason = reason
        call.ended_at = self._clock()
        if self._open.get(call.session_id) is call:
            del self._open[call.session_id]
        self._touch(self._ended, call.session_id, call, self._max)
        ended.append(call)

    def _expire_idle_locked(self, ended: list[Call]) -> None:
        now = self._clock()
        for call in list(self._open.values()):
            if not call.in_flight and now - call.last_activity > self.idle_timeout_s:
                self._end_locked(call, "idle", ended)

    def _notify(self, ended: list[Call]) -> None:
        if not self._on_call_end:
            return
        for call in ended:
            try:
                self._on_call_end(call)
            except Exception:  # noqa: BLE001 - persistence must not break a tool call
                log.exception("advisor-call: on_call_end failed for %s", call.call_id)

    def _resolve_locked(self, session_id: str, call_id: str | None) -> Call:
        call = self._open.get(session_id)
        if call is not None and (not call_id or call_id == call.call_id):
            return call
        last = self._ended.get(session_id)
        if call_id and last is not None and last.call_id == call_id:
            raise CallError(
                "ended", f"call {call_id} has ended: {end_reason_text(last.end_reason)}", last
            )
        if call_id:
            # Never resolve an id outside the caller's own session.
            raise CallError("wrong_call", f"no call {call_id!r} in this session")
        if last is not None and last.end_reason in (
            "user_hangup",
            "idle",
            "bill_cap",
            "day_cap",
            "month_cap",
            "turn_limit",
        ):
            raise CallError(
                "ended", f"the last call has ended: {end_reason_text(last.end_reason)}", last
            )
        raise CallError("no_call", "no open advisor call in this session")

    # ----------------------------------------------------------------- calls

    def open(
        self,
        session_id: str | None,
        *,
        line: str,
        cap_usd: float,
        initiated_by: Literal["executor", "user"] = "executor",
    ) -> tuple[Call, bool]:
        """(call, created). An existing open call is returned unchanged."""
        sid = self._check_session(session_id)
        ended: list[Call] = []
        with self._lock:
            self._expire_idle_locked(ended)
            existing = self._open.get(sid)
            if existing is not None:
                result = (existing, False)
            else:
                now = self._clock()
                call = Call(
                    call_id=new_call_id(),
                    session_id=sid,
                    line=line,
                    started_at=now,
                    last_activity=now,
                    cap_usd=cap_usd,
                    initiated_by=initiated_by,
                )
                while len(self._open) >= self._max:
                    _, oldest = next(iter(self._open.items()))
                    self._end_locked(oldest, "session_end", ended)
                self._open[sid] = call
                result = (call, True)
        self._notify(ended)
        return result

    def get(self, session_id: str | None, call_id: str | None = None) -> Call:
        """The caller's open call, or CallError."""
        sid = self._check_session(session_id)
        ended: list[Call] = []
        try:
            with self._lock:
                self._expire_idle_locked(ended)
                return self._resolve_locked(sid, call_id)
        finally:
            self._notify(ended)

    def latest(self, session_id: str | None) -> Call | None:
        """The open call, else the last ended call, for this session."""
        sid = self._check_session(session_id)
        ended: list[Call] = []
        with self._lock:
            self._expire_idle_locked(ended)
            call = self._open.get(sid) or self._ended.get(sid)
        self._notify(ended)
        return call

    def end(self, session_id: str | None, reason: str, call_id: str | None = None) -> Call:
        """End the caller's open call. Raises CallError if there is none."""
        sid = self._check_session(session_id)
        ended: list[Call] = []
        try:
            with self._lock:
                self._expire_idle_locked(ended)
                call = self._resolve_locked(sid, call_id)
                self._end_locked(call, reason, ended)
                return call
        finally:
            self._notify(ended)

    # ----------------------------------------------------------------- exchanges

    def reserve(
        self, session_id: str | None, call_id: str | None, text: str, preflight: Preflight
    ) -> Reservation:
        """Check the caps and mark the call in flight. If preflight refuses,
        the call is ended with that reason and CallError("ended") is raised."""
        sid = self._check_session(session_id)
        ended: list[Call] = []
        try:
            with self._lock:
                self._expire_idle_locked(ended)
                call = self._resolve_locked(sid, call_id)
                if call.in_flight:
                    raise CallError(
                        "busy", "a message on this call is still waiting for a reply", call
                    )
                pending = [*call.messages, {"role": "user", "content": text}]
                elsewhere = sum(r.est_cost for r in self._reservations.values())
                reason, est = preflight(call, pending, elsewhere)
                if reason:
                    self._end_locked(call, reason, ended)
                    raise CallError("ended", f"call ended: {end_reason_text(reason)}", call)
                self._next_res += 1
                res = Reservation(self._next_res, call, text, pending, est)
                self._reservations[res.res_id] = res
                call.in_flight = True
                call.last_activity = self._clock()
                return res
        finally:
            self._notify(ended)

    def settle(
        self,
        res: Reservation,
        *,
        reply: str | None,
        cost_usd: float,
        usage: Usage,
        ok: bool,
        estimated: bool = False,
        cached_prefix_est: float = 0.0,
        token_scale: float | None = None,
    ) -> Settled:
        """Record one exchange. On success the executor message and the reply
        are appended; on failure the conversation is left unchanged. Cost is
        added either way. Ends the call when it reaches its cap.

        cached_prefix_est: on success, the prompt prefix (estimated tokens)
        this request left in the cache, through its last breakpoint. After a
        failure we assume nothing is cached, which can only raise the next
        estimate. token_scale: the measured estimator scale, if any."""
        ended: list[Call] = []
        call = res.call
        with self._lock:
            self._reservations.pop(res.res_id, None)
            call.in_flight = False
            call.last_activity = self._clock()
            call.cost_usd += max(0.0, cost_usd)
            call.usage.add(usage)
            call.estimated = call.estimated or estimated
            if token_scale is not None and token_scale > call.token_scale:
                call.token_scale = token_scale
            if ok and reply:
                call.cached_est = max(0.0, cached_prefix_est)
                call.cached_at = call.last_activity
            else:
                call.cached_est, call.cached_at = 0.0, None
            if ok and reply:
                call.messages.extend(
                    [{"role": "user", "content": res.text}, {"role": "assistant", "content": reply}]
                )
                call.exchanges += 1
            elif cost_usd > 0:
                call.exchanges += 1  # a billed attempt still counts
            ended_now = ""
            if call.is_open and call.cost_usd >= call.cap_usd:
                self._end_locked(call, "bill_cap", ended)
                ended_now = "bill_cap"
            elif not call.is_open:
                ended.append(call)  # ended mid-flight (user hangup): persist the final cost
        self._notify(ended)
        return Settled(call, ended_now)

    def release(self, res: Reservation) -> None:
        """Drop a reservation without an exchange (the request never went out)."""
        with self._lock:
            self._reservations.pop(res.res_id, None)
            res.call.in_flight = False

    # ----------------------------------------------------------------- sessions

    def end_session(self, session_id: str | None) -> Call | None:
        if not session_id:
            return None
        ended: list[Call] = []
        with self._lock:
            call = self._open.get(session_id)
            if call is not None:
                self._end_locked(call, "session_end", ended)
            self._grants.pop(session_id, None)
            self._activity.pop(session_id, None)
        self._notify(ended)
        return call

    def grant(self, session_id: str | None, *, line: str, cap_usd: float | None) -> Grant:
        sid = self._check_session(session_id)
        with self._lock:
            g = Grant(line=line, cap_usd=cap_usd, granted_at=self._clock())
            self._touch(self._grants, sid, g, self._max)
            return g

    def revoke(self, session_id: str | None) -> bool:
        sid = self._check_session(session_id)
        with self._lock:
            return self._grants.pop(sid, None) is not None

    def grant_for(self, session_id: str | None) -> Grant | None:
        if not session_id:
            return None
        with self._lock:
            return self._grants.get(session_id)

    def take_grant_notice(self, session_id: str | None) -> Grant | None:
        """The grant, the first time it is asked for; None afterwards."""
        if not session_id:
            return None
        with self._lock:
            g = self._grants.get(session_id)
            if g is None or g.notified:
                return None
            g.notified = True
            return g

    def mark_activity(self, session_id: str | None) -> None:
        if session_id:
            with self._lock:
                self._touch(self._activity, session_id, None, self._max)

    def take_activity(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        with self._lock:
            if session_id in self._activity:
                del self._activity[session_id]
                return True
            return False
