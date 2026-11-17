"""The /second-opinion slash command.

The handler is sync on purpose: /second-opinion ask waits for the advisor, and
Hermes bounds async command results at 30 s on the CLI (DESIGN.md, O-8).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

from .calls import Call, CallError, end_reason_text
from .cost import call_cap
from .hooks import money, sender_allowed
from .ledger import LedgerError
from .prompts import COMMAND, for_platform
from .runtime import Runtime
from .settings import Settings
from .tools import open_call

log = logging.getLogger(__name__)

FALLBACK_COMMAND = "advisorcall"
ARGS_HINT = "[status|allow|revoke|ask|hangup|bill|transcript]"
DESCRIPTION = "Advisor calls: status, pre-authorize, ask directly, hang up, see the bill"
CHAT_LIMIT = 3500  # stay well under Telegram's 4096-character message limit

USAGE = f"""\
/{COMMAND} or /{COMMAND} status   advisor, open call, today's and this cycle's spend
/{COMMAND} allow [$cap]           pre-authorize advisor calls for this session
/{COMMAND} revoke                 remove the pre-authorization
/{COMMAND} ask <question>         ask the advisor yourself (one message, counts toward the budget)
/{COMMAND} hangup                 end the open call now
/{COMMAND} bill                   spend today, this cycle and the last 7 days
/{COMMAND} transcript             the last call in this session"""

_CAP = re.compile(r"^\$?(\d+(?:\.\d+)?)$")


def _sender(source: Any) -> str:
    uid = getattr(source, "user_id", None) if source is not None else None
    return str(uid) if uid else ""


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _clip(text: str, limit: int = CHAT_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 20].rstrip() + "\n…[truncated]"


class SecondOpinionCommand:
    def __init__(self, rt: Runtime) -> None:
        self.rt = rt

    def __call__(
        self,
        raw_args: Any = "",
        session_id: Any = None,
        session_key: Any = None,
        platform: Any = None,
        source: Any = None,
        **_: Any,
    ) -> str:
        try:
            text = self.dispatch(str(raw_args or ""), session_id or None, source)
        except Exception as exc:  # noqa: BLE001 - a command must never crash the chat
            log.exception("advisor-call: /second-opinion failed")
            text = f"/{COMMAND} failed: {type(exc).__name__}: {exc}"
        return for_platform(text, platform)

    # ----------------------------------------------------------------- dispatch

    def dispatch(self, raw: str, session_id: str | None, source: Any) -> str:
        words = raw.strip().split(maxsplit=1)
        sub = words[0].lower() if words else "status"
        rest = words[1] if len(words) > 1 else ""
        settings = self.rt.settings()
        # Subcommands that spend money or show the advisor conversation are
        # limited to allowed_senders on the gateway. CLI and TUI have no
        # source and are the local owner.
        guarded = sub in ("allow", "ask", "transcript")
        if guarded and source is not None:
            identity = (_sender(source), "gateway")
            if not sender_allowed(settings, identity):
                return "You are not allowed to use advisor calls."
        handler = {
            "status": self.status,
            "allow": self.allow,
            "revoke": self.revoke,
            "ask": self.ask,
            "hangup": self.hangup,
            "bill": self.bill,
            "transcript": self.transcript,
        }.get(sub)
        if handler is None:
            return USAGE
        if sub not in ("status", "bill") and not session_id:
            return "No session yet. Send a message first, then try again."
        return handler(settings, session_id, rest)

    # ----------------------------------------------------------------- helpers

    def _budget_lines(self, settings: Settings) -> list[str]:
        b = settings.budget
        if b.cycle_error:
            return [b.cycle_error]
        try:
            state = self.rt.budget(settings)
        except LedgerError as exc:
            return [f"Spend cannot be read: {exc}"]
        today = self.rt.today()
        est = ""
        try:
            if self.rt.ledger.has_estimates(state.cycle.start, today):
                est = " (includes estimates)"
        except LedgerError:
            pass
        return [
            f"Today: {money(state.spent_today)} of {money(state.allowance)}",
            f"This cycle: {money(state.spent_cycle)} of {money(state.per_month)}{est}",
            state.cycle.describe(today),
        ]

    def _call_line(self, call: Call) -> str:
        cost = f"~{money(call.cost_usd)} (estimate)" if call.estimated else money(call.cost_usd)
        status = "open" if call.is_open else f"ended ({end_reason_text(call.end_reason)})"
        return (
            f"Call {call.call_id} ({call.line}): {status}, {plural(call.exchanges, 'message')}, "
            f"{cost} of {money(call.cap_usd)}"
        )

    # ----------------------------------------------------------------- subcommands

    def status(self, settings: Settings, session_id: str | None, _: str) -> str:
        line = settings.advisor
        ok = "" if line.usable else f" (cannot be used: {line.problem})"
        out = [f"Advisor: {line.name} ({line.model}){ok}"]
        out += self._budget_lines(settings)
        if not settings.budget.cycle_error:
            try:
                cap = call_cap(settings.budget, self.rt.budget(settings))
                out.append(f"Per-call cap: {money(cap)}")
            except (ValueError, LedgerError):
                pass
        if settings.suggest:
            try:
                today = self.rt.today()
                made, followed = self.rt.ledger.suggestion_stats(
                    today - dt.timedelta(days=6), today
                )
                if made:
                    out.append(
                        f"Auto-suggest, last 7 days: {plural(made, 'loop warning')}, "
                        f"{followed} led to a call"
                    )
            except LedgerError:
                pass
        if session_id:
            call = self.rt.manager.latest(session_id)
            if call is not None:
                out.append(self._call_line(call))
            grant = self.rt.manager.grant_for(session_id)
            if grant is not None:
                cap_text = f" (cap {money(grant.cap_usd)})" if grant.cap_usd is not None else ""
                out.append(f"This session is pre-authorized{cap_text}.")
            else:
                out.append("This session is not pre-authorized; each call asks for approval.")
        return "\n".join(out)

    def allow(self, settings: Settings, session_id: str, rest: str) -> str:
        cap = None
        words = rest.split()
        if words:
            m = _CAP.match(words[0])
            if m is None or len(words) > 1:
                return f"Usage: /{COMMAND} allow [$cap]"
            cap = float(m.group(1))
        line = settings.advisor
        if not line.usable:
            return f"Advisor model {line.model!r} cannot be used: {line.problem}."
        if settings.budget.cycle_error:
            return settings.budget.cycle_error
        if cap is not None and cap <= 0:
            return "The cap must be more than $0."
        try:
            state = self.rt.budget(settings)
        except (ValueError, LedgerError) as exc:
            return f"Advisor budget cannot be checked: {exc}"
        if cap is not None:
            cap = min(cap, state.available)  # clamped to what is left today
        self.rt.manager.grant(session_id, cap_usd=cap)
        shown = money(cap) if cap is not None else money(call_cap(settings.budget, state))
        return (
            f"Advisor calls are pre-authorized for this session ({line.name}, cap "
            f"{shown} per call). The executor is told on its next turn. /{COMMAND} revoke to undo."
        )

    def revoke(self, settings: Settings, session_id: str, _: str) -> str:
        if self.rt.manager.revoke(session_id):
            return "Pre-authorization removed. Each advisor call will ask for approval again."
        return "This session was not pre-authorized."

    def ask(self, settings: Settings, session_id: str, rest: str) -> str:
        question = rest.strip()
        if not question:
            return f"Usage: /{COMMAND} ask <question>"
        try:
            existing = self.rt.manager.get(session_id)
        except CallError:
            existing = None
        if existing is not None:
            return (
                f"The executor has a call open ({existing.call_id}). /{COMMAND} hangup "
                "first, then ask."
            )
        framed = f"(This message is from the user, asking you directly.)\n{question}"
        result = open_call(self.rt, {"question": framed}, session_id, initiated_by="user")
        call = self.rt.manager.latest(session_id)
        if call is not None and call.initiated_by != "user":
            # The executor opened a call between the check above and ours.
            return (
                f"The executor opened a call meanwhile ({call.call_id}). /{COMMAND} "
                "hangup first, then ask."
            )
        if call is not None and call.is_open:
            self.rt.manager.end(session_id, "hangup", call.call_id)
        self.rt.manager.take_activity(session_id)  # no executor footer for the user's own ask
        if "error" in result and "reply" not in result:
            return f"Advisor unavailable: {result['error']}"
        # Just the answer: the call's cost shows in /second-opinion status and bill.
        return _clip(str(result.get("reply", "")).strip())

    def hangup(self, settings: Settings, session_id: str, _: str) -> str:
        try:
            call = self.rt.manager.end(session_id, "user_hangup")
        except CallError:
            return "No advisor call is open in this session."
        return f"Call ended. {self._call_line(call)}"

    def bill(self, settings: Settings, session_id: str | None, _: str) -> str:
        out = self._budget_lines(settings)
        if settings.budget.cycle_error:
            return "\n".join(out)
        try:
            rows = self.rt.ledger.daily_totals(self.rt.today(), 7)
        except LedgerError as exc:
            return "\n".join([*out, f"Daily totals unavailable: {exc}"])
        out.append("Last 7 days:")
        out += [f"  {d:%a %b} {d.day}: {money(v)}" for d, v in rows]
        return "\n".join(out)

    def transcript(self, settings: Settings, session_id: str, _: str) -> str:
        call = self.rt.manager.latest(session_id)
        if call is None or not call.messages:
            return "No advisor call in this session yet."
        lines = [self._call_line(call), ""]
        for i, msg in enumerate(call.messages):
            text = msg["content"]
            if i == 0:  # the opening: show the question, not the whole transcript
                m = re.search(r"<executor_message>\n?(.*?)\n?</executor_message>", text, re.S)
                text = m.group(1) if m else text
            who = "Executor" if msg["role"] == "user" else "Advisor"
            lines.append(f"{who}: {text}")
            lines.append("")
        return _clip("\n".join(lines).strip())
