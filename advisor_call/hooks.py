"""Hermes hooks and middleware.

- pre_tool_call: the approval gate for advisor_call_open (DESIGN.md 2.5). Fast:
  a config read and an indexed ledger read, no network.
- pre_llm_call: remember the conversation and sender per session; tell the
  executor once that the user pre-authorized calls.
- llm_request middleware: capture the latest assembled request (mid-turn
  tool results). Never modifies it.
- transform_llm_output: append the bill footer on turns with advisor activity.
- on_session_finalize: close the session's open call.

Every callback takes **kwargs (Hermes' plugin doctor requires it) and
treats every kwarg as optional.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .calls import Call, CallError, end_reason_text
from .cost import call_cap
from .ledger import LedgerError
from .runtime import Runtime
from .settings import Settings
from .tools import OPEN, SAY

log = logging.getLogger(__name__)

REASON_CHARS = 300
# Control characters, line/paragraph separators, zero-width and bidi
# controls. Built from code points so the source holds no invisible text.
_STRIP_CODEPOINTS = [
    *range(0x00, 0x20),
    *range(0x7F, 0xA0),
    0x2028,
    0x2029,
    *range(0x200B, 0x2010),
    *range(0x202A, 0x202F),
    *range(0x2066, 0x206A),
]
_CONTROL = re.compile("[" + "".join(re.escape(chr(c)) for c in _STRIP_CODEPOINTS) + "]")


def sanitize_reason(raw: Any) -> str:
    """Model-written text for the approval prompt: one line, no control or
    bidi characters, at most REASON_CHARS."""
    text = raw if isinstance(raw, str) else ""
    text = _CONTROL.sub(" ", text)
    text = " ".join(text.split())
    if len(text) > REASON_CHARS:
        text = text[: REASON_CHARS - 1].rstrip() + "…"
    return text or "(no reason given)"


def _block(message: str) -> dict[str, str]:
    return {"action": "block", "message": message}


# Platforms that run on the owner's own machine. Hermes passes an empty
# platform for CLI agents (DESIGN.md, Phase 3 notes).
LOCAL_PLATFORMS = frozenset({"", "cli", "tui", "acp", "local"})


def sender_allowed(settings: Settings, identity: tuple[str, str] | None) -> bool:
    """identity is (sender_id, platform) from the session's captured context,
    or None when the session has not been seen. allowed_senders applies when
    it is non-empty: a known sender must be on it; a local session with no
    sender is the owner; anything else (an unseen session, a gateway session
    without a sender id) is refused."""
    if not settings.allowed_senders:
        return True
    if identity is None:
        return False
    sender, platform = identity
    if sender:
        return sender in settings.allowed_senders
    return platform.lower() in LOCAL_PLATFORMS


def money(value: float) -> str:
    return f"${value:.2f}"


class Hooks:
    def __init__(self, rt: Runtime) -> None:
        self.rt = rt

    # ----------------------------------------------------------------- approval

    def pre_tool_call(
        self, tool_name: Any = None, args: Any = None, session_id: Any = None, **_: Any
    ) -> dict[str, str] | None:
        try:
            if tool_name == SAY:
                return self._check_say(session_id)
            if tool_name != OPEN:
                return None
            return self._check_open(args if isinstance(args, dict) else {}, session_id)
        except Exception as exc:  # noqa: BLE001 - fail closed on any bug
            log.exception("advisor-call: approval hook failed")
            return _block(
                f"advisor-call could not check this call ({type(exc).__name__}); "
                "no call was made. Continue on your own judgment and tell the user."
            )

    def _check_say(self, session_id: Any) -> dict[str, str] | None:
        if not session_id:
            return None  # the handler reports the missing session
        settings = self.rt.settings()
        if not sender_allowed(settings, self.rt.context.identity(session_id)):
            return _block("This sender is not allowed to use advisor calls.")
        return None

    def _check_open(self, args: dict[str, Any], session_id: Any) -> dict[str, str] | None:
        rt = self.rt
        # 1. Never guess the session.
        if not session_id or not isinstance(session_id, str):
            return _block("advisor-call: no session id, so no call was made.")
        settings = rt.settings()
        # 2. Sender.
        if not sender_allowed(settings, rt.context.identity(session_id)):
            return _block("This sender is not allowed to use advisor calls.")
        # 3. An open call is returned by the handler without a new request.
        # The handler will not open a *new* call on this authorization.
        try:
            rt.manager.get(session_id)
            rt.authorize(session_id, "existing")
            return None
        except CallError:
            pass
        # 4. Line and cycle configuration.
        line_name = args.get("line") if isinstance(args.get("line"), str) else ""
        line = settings.line(line_name.strip() or None)
        if line is None or not line.usable:
            valid = ", ".join(settings.usable_lines()) or "none"
            return _block(
                f"Advisor line {(line_name or settings.default_line)!r} is "
                f"misconfigured. Valid lines: {valid}."
            )
        if settings.budget.cycle_error:
            return _block(settings.budget.cycle_error)
        # 5. Budget (fail closed if spend cannot be read).
        try:
            state = rt.budget(settings)
        except (ValueError, LedgerError) as exc:
            return _block(
                f"Advisor budget cannot be checked ({exc}). Continue on your own "
                "judgment and tell the user."
            )
        grant = rt.manager.grant_for(session_id)
        if grant is not None and grant.line != line.name:
            grant = None  # a pre-authorization covers only the line it was given for
        cap = call_cap(settings.budget, state, grant.cap_usd if grant is not None else None)
        if state.available <= 0 or cap <= 0:
            return _block(
                f"Advisor budget for today is used up ({money(state.spent_today)} of "
                f"{money(state.allowance)}; {money(state.remaining_cycle)} left this cycle). "
                "Continue on your own judgment and tell the user."
            )
        # 6. Approvals off: only an explicit user grant allows a call.
        if (
            settings.refuse_when_approvals_off
            and settings.approvals_mode == "off"
            and grant is None
        ):
            return _block(
                "Approvals are off, so advisor calls need `/second-opinion allow` from the user."
            )
        # 7. Pre-authorized by the user for this session and line.
        if grant is not None:
            rt.authorize(session_id, "grant")
            return None
        # 8. Ask the human. The session id in rule_key makes "Always" match
        # this session only, so a new session always asks again. The figures
        # come first and the model-written reason is quoted last, so the
        # reason cannot pass itself off as the cap.
        reason = sanitize_reason(args.get("reason")).replace('"', "'")
        rt.authorize(session_id, "approved")
        return {
            "action": "approve",
            "message": (
                f"Advisor call ({line.name}, {line.model}): bill cap {money(cap)}, today "
                f"{money(state.spent_today)} of {money(state.allowance)}. "
                f'Reason given by the model: "{reason}"'
            ),
            "rule_key": f"advisor_call:{line.name}:{session_id}",
        }

    # ----------------------------------------------------------------- capture

    def pre_llm_call(
        self,
        session_id: Any = None,
        conversation_history: Any = None,
        user_message: Any = None,
        sender_id: Any = None,
        platform: Any = None,
        **_: Any,
    ) -> dict[str, str] | None:
        try:
            self.rt.context.record_turn(
                session_id,
                conversation_history=conversation_history,
                user_message=user_message,
                sender_id=sender_id,
                platform=platform,
            )
            grant = self.rt.manager.take_grant_notice(session_id)
            if grant is None:
                return None
            settings = self.rt.settings()
            cap = grant.cap_usd
            if cap is None:
                try:
                    cap = call_cap(settings.budget, self.rt.budget(settings))
                except (ValueError, LedgerError):
                    cap = None
            cap_text = f", cap {money(cap)}" if cap is not None else ""
            return {
                "context": "[advisor-call] The user has pre-authorized advisor calls for "
                f"this session (line: {grant.line}{cap_text})."
            }
        except Exception:  # noqa: BLE001 - capture must never break a turn
            log.exception("advisor-call: pre_llm_call failed")
            return None

    def llm_request(self, request: Any = None, session_id: Any = None, **_: Any) -> None:
        try:
            if isinstance(request, dict):
                self.rt.context.record_request(session_id, request.get("messages"))
        except Exception:  # noqa: BLE001 - fail open, never touch the request
            log.debug("advisor-call: llm_request capture failed", exc_info=True)
        return None

    # ----------------------------------------------------------------- footer

    def transform_llm_output(
        self, response_text: Any = None, session_id: Any = None, **_: Any
    ) -> str | None:
        try:
            if not self.rt.manager.take_activity(session_id):
                return None
            settings = self.rt.settings()
            if not settings.bill_footer or not isinstance(response_text, str):
                return None
            call = self.rt.manager.latest(session_id)
            if call is None:
                return None
            return f"{response_text.rstrip()}\n\n{self.footer(call, settings)}"
        except Exception:  # noqa: BLE001
            log.exception("advisor-call: bill footer failed")
            return None

    def footer(self, call: Call, settings: Settings) -> str:
        cost = f"~{money(call.cost_usd)}" if call.estimated else money(call.cost_usd)
        parts = [
            f"📞 Advisor call ({call.line}): {call.exchanges} "
            f"message{'s' if call.exchanges != 1 else ''}",
            f"{cost} of {money(call.cap_usd)}" + (" (estimate)" if call.estimated else ""),
            "still open" if call.is_open else f"ended ({end_reason_text(call.end_reason)})",
        ]
        try:
            state = self.rt.budget(settings)
            parts.append(f"today {money(state.spent_today)} of {money(state.allowance)}")
        except (ValueError, LedgerError):
            pass
        return " · ".join(parts)

    # ----------------------------------------------------------------- teardown

    def on_session_finalize(
        self, session_id: Any = None, old_session_id: Any = None, **_: Any
    ) -> None:
        try:
            for sid in {session_id, old_session_id}:
                if isinstance(sid, str) and sid:
                    self.rt.manager.end_session(sid)
                    self.rt.context.forget(sid)
        except Exception:  # noqa: BLE001
            log.exception("advisor-call: on_session_finalize failed")
        return None
