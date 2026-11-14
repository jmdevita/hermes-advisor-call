"""The three call tools: advisor_call_open, advisor_call_say, advisor_call_hangup.

Every handler has the shape ``handler(args, session_id=None, **_)`` and
returns a JSON string. A handler never raises. A missing session_id is an
error: the session is never guessed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from .advisor import AdvisorError, AdvisorRequest, ask, build_request
from .calls import Call, CallError, end_reason_text
from .context import render_transcript
from .cost import (
    Charge,
    Usage,
    call_cap,
    charge,
    estimate_cost,
    preflight_refusal,
    reusable_prefix_chars,
)
from .ledger import LedgerError
from .prompts import GUIDANCE_ON_ERROR, NO_TRANSCRIPT_NOTE, OPENING_TEMPLATE
from .runtime import Runtime
from .settings import Line, Settings

log = logging.getLogger(__name__)

TOOLSET = "advisor_call"
OPEN, SAY, HANGUP = "advisor_call_open", "advisor_call_say", "advisor_call_hangup"

OPEN_SCHEMA: dict[str, Any] = {
    "name": OPEN,
    "description": (
        "Phone a stronger senior model for advice on a hard problem. The user must approve "
        "the call. Use it when you are stuck after real attempts, facing an architecture or "
        "debugging decision with costly consequences, or when the user asks you to get a "
        "second opinion. Do not use it for routine work. The advisor has no tools and cannot "
        "see your files: put the exact error text, relevant code and what you already tried "
        "in `question`. After opening, continue with advisor_call_say and finish with "
        "advisor_call_hangup."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "One sentence shown to the user on the approval prompt: why "
                "you need the call.",
            },
            "question": {
                "type": "string",
                "description": "Your first message to the advisor: the problem, exact errors, "
                "relevant code, what you tried, and your specific question.",
            },
            "line": {
                "type": "string",
                "description": "Advisor line to call. Omit for the default.",
            },
        },
        "required": ["reason", "question"],
    },
}

SAY_SCHEMA: dict[str, Any] = {
    "name": SAY,
    "description": (
        "Send your next message on the open advisor call and get the advisor's reply. Answer "
        "the advisor's questions with exact output from your own tools. Returns the reply and "
        "the running bill."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Your message to the advisor."},
            "call_id": {
                "type": "string",
                "description": "The call to speak on. Omit for this session's open call.",
            },
        },
        "required": ["message"],
    },
}

HANGUP_SCHEMA: dict[str, Any] = {
    "name": HANGUP,
    "description": "End the advisor call. Makes no request to the advisor; returns the bill.",
    "parameters": {
        "type": "object",
        "properties": {
            "call_id": {
                "type": "string",
                "description": "The call to end. Omit for this session's open call.",
            },
        },
    },
}


# --------------------------------------------------------------------------- results


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"error": message, "guidance": GUIDANCE_ON_ERROR, **extra}


def _ended_notice(reason: str) -> str:
    return (
        f"The call has ended ({end_reason_text(reason)}). Continue on your own judgment "
        "and tell the user what the advisor said and what it cost."
    )


def _call_error(
    rt: Runtime, settings: Settings, err: CallError, session_id: str | None, call_id: str | None
) -> dict[str, Any]:
    if err.code == "ended" and err.call is not None:
        return _error(
            err.message,
            status="ended",
            call_id=err.call.call_id,
            end_reason=err.call.end_reason,
            bill=rt.bill(err.call, settings),
            notice=_ended_notice(err.call.end_reason),
        )
    if err.code == "wrong_call" and call_id and session_id:
        try:
            if rt.ledger.open_call_record(call_id, session_id):
                return _error("call dropped (gateway restarted); open a new call")
        except LedgerError:
            pass
    if err.code == "no_call":
        return _error(
            "No open advisor call in this session. Open one with advisor_call_open "
            "(the user must approve)."
        )
    return _error(err.message)


def _str_arg(args: Any, key: str) -> str:
    value = args.get(key) if isinstance(args, dict) else None
    return value.strip() if isinstance(value, str) else ""


def _usable_line(settings: Settings, name: str) -> tuple[Line | None, str]:
    line = settings.line(name or None)
    if line is None or not line.usable:
        valid = ", ".join(settings.usable_lines()) or "none"
        which = name or settings.default_line
        detail = f": {line.problem}" if line is not None and line.problem else ""
        return None, f"advisor line {which!r} is misconfigured{detail}. Valid lines: {valid}."
    return line, ""


# --------------------------------------------------------------------------- exchange


def _bill_for(
    usage: Usage, line: Line, settings: Settings, req: AdvisorRequest, reused: int
) -> Charge:
    if not (
        usage.input_tokens
        or usage.output_tokens
        or usage.cache_read_tokens
        or usage.cache_write_tokens
    ):
        # The provider answered but no usage came back: bill the upper bound.
        return Charge(
            estimate_cost(
                req.prompt_chars,
                settings.max_tokens,
                line.prices,
                caching=settings.caching,
                reused_chars=reused,
            ),
            estimated=True,
        )
    return charge(
        usage,
        line.prices,
        caching=settings.caching,
        prompt_chars=req.prompt_chars,
        uncached_chars=req.uncached_chars,
        reused_chars=reused,
    )


def _record(
    rt: Runtime, call: Call, line: Line, model: str, usage: Usage, ch: Charge, ok: bool
) -> None:
    try:
        rt.ledger.record_exchange(
            ts=rt.clock.time(),
            day=rt.today(),
            session_id=call.session_id,
            call_id=call.call_id,
            line=line.name,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd=ch.cost_usd,
            ok=ok,
            estimated=ch.estimated,
        )
    except LedgerError as exc:
        # The call's own total still holds this cost; the day total will lag.
        log.warning("advisor-call: could not record exchange on %s: %s", call.call_id, exc)


def _log_transcript(
    rt: Runtime, settings: Settings, call: Call, text: str, reply: str | None
) -> None:
    if not settings.log_transcripts:
        return
    try:
        rt.ledger.append_transcript(
            call.call_id, {"ts": rt.clock.time(), "executor": text, "advisor": reply}
        )
    except LedgerError as exc:
        log.warning("advisor-call: transcript log failed: %s", exc)


def exchange(
    rt: Runtime, settings: Settings, line: Line, session_id: str, call_id: str | None, text: str
) -> dict[str, Any]:
    """Send one executor message on the session's open call: reserve under
    the caps, ask the advisor outside the lock, settle, record."""
    captured: dict[str, Any] = {}

    def preflight(call: Call, pending: list[dict[str, str]], elsewhere: float):
        req = build_request(pending, settings.cache_ttl)
        reused = reusable_prefix_chars(
            call.cached_chars, call.cached_at, rt.clock.time(), settings.cache_ttl
        )
        est = estimate_cost(
            req.prompt_chars,
            settings.max_tokens,
            line.prices,
            caching=settings.caching,
            reused_chars=reused,
        )
        state = rt.budget(settings)  # raises -> no exchange
        captured.update(req=req, reused=reused)
        reason = preflight_refusal(
            call_cost=call.cost_usd,
            call_cap_usd=call.cap_usd,
            exchanges=call.exchanges,
            max_messages=settings.budget.max_messages_per_call,
            est_cost=est,
            state=state,
            reserved_elsewhere=elsewhere,
        )
        return reason, est

    try:
        res = rt.manager.reserve(session_id, call_id, text, preflight)
    except CallError as err:
        return _call_error(rt, settings, err, session_id, call_id)
    except (ValueError, LedgerError) as exc:
        return _error(f"cannot check the advisor budget: {exc}")

    call = res.call
    req: AdvisorRequest = captured["req"]
    reused: int = captured["reused"]
    rt.manager.mark_activity(session_id)
    try:
        reply = ask(rt.llm(), line, settings, req)
    except AdvisorError as err:
        usage = err.usage or Usage()
        billed = err.usage is not None and (usage.input_tokens or usage.output_tokens)
        ch = _bill_for(usage, line, settings, req, reused) if billed else Charge(0.0, False)
        settled = rt.manager.settle(
            res, reply=None, cost_usd=ch.cost_usd, usage=usage, ok=False, estimated=ch.estimated
        )
        if billed:
            _record(rt, call, line, line.model, usage, ch, ok=False)
        _log_transcript(rt, settings, call, text, None)
        result = _error(
            f"advisor unavailable: {err.message}",
            call_id=call.call_id,
            status="open" if call.is_open else "ended",
            bill=rt.bill(call, settings),
        )
        if settled.ended_now:
            result.update(end_reason=settled.ended_now, notice=_ended_notice(settled.ended_now))
        return result
    except Exception as exc:  # noqa: BLE001 - e.g. ctx.llm unavailable; never leave it in flight
        rt.manager.release(res)
        log.exception("advisor-call: advisor request could not be made")
        return _error(
            f"advisor unavailable: {type(exc).__name__}: {exc}",
            call_id=call.call_id,
            status="open" if call.is_open else "ended",
            bill=rt.bill(call, settings),
        )

    ch = _bill_for(reply.usage, line, settings, req, reused)
    settled = rt.manager.settle(
        res,
        reply=reply.text,
        cost_usd=ch.cost_usd,
        usage=reply.usage,
        ok=True,
        estimated=ch.estimated,
        cached_prefix_chars=req.cached_prefix_chars if settings.caching else 0,
    )
    _record(rt, call, line, reply.model, reply.usage, ch, ok=True)
    _log_transcript(rt, settings, call, text, reply.text)
    result: dict[str, Any] = {
        "call_id": call.call_id,
        "line": line.name,
        "advisor_model": reply.model or line.model,
        "reply": reply.text,
        "bill": rt.bill(call, settings),
        "status": "open" if call.is_open else "ended",
    }
    if not call.is_open:
        result["end_reason"] = call.end_reason
        result["notice"] = _ended_notice(call.end_reason)
    return result


# --------------------------------------------------------------------------- handlers


def opening_message(rt: Runtime, settings: Settings, session_id: str, question: str) -> str:
    raw = rt.context.messages(session_id)
    redact = rt.redactor() if settings.redact_secrets else None
    transcript = ""
    if raw:
        try:
            transcript = render_transcript(
                raw,
                limit_chars=settings.opening_transcript_chars,
                tool_chars=settings.tool_output_chars,
                redact=redact,
            )
        except Exception:  # noqa: BLE001 - an odd history must not block the call
            log.exception("advisor-call: transcript rendering failed")
            transcript = ""
    if not transcript:
        transcript = NO_TRANSCRIPT_NOTE
    if redact is not None:
        question = redact(question)
    return OPENING_TEMPLATE.format(transcript=transcript, question=question)


def open_call(
    rt: Runtime, args: Any, session_id: str | None, *, initiated_by: str = "executor"
) -> dict[str, Any]:
    if not session_id:
        return _error("no session id; cannot open an advisor call")
    settings = rt.settings()
    question = _str_arg(args, "question")
    if not question:
        return _error(
            "`question` is required: the problem, exact errors, code and what you already tried"
        )
    line, problem = _usable_line(settings, _str_arg(args, "line"))
    if line is None:
        return _error(problem)
    if settings.budget.cycle_error:
        return _error(settings.budget.cycle_error)

    try:
        existing = rt.manager.get(session_id)
    except CallError:
        existing = None
    if existing is not None:
        rt.manager.mark_activity(session_id)
        return {
            "call_id": existing.call_id,
            "line": existing.line,
            "status": "open",
            "note": "A call is already open in this session; continue with advisor_call_say.",
            "bill": rt.bill(existing, settings),
        }

    try:
        state = rt.budget(settings)
    except (ValueError, LedgerError) as exc:
        return _error(f"cannot check the advisor budget, so no call was opened: {exc}")

    grant = rt.manager.grant_for(session_id)
    custom = grant.cap_usd if grant is not None and grant.line == line.name else None
    cap = call_cap(settings.budget, state, custom)
    if cap <= 0:
        return _error(
            f"Advisor budget for today is used up (${state.spent_today:.2f} of "
            f"${state.allowance:.2f}; ${state.remaining_cycle:.2f} left this cycle)."
        )

    call, created = rt.manager.open(
        session_id, line=line.name, cap_usd=cap, initiated_by=initiated_by
    )  # type: ignore[arg-type]
    if not created:
        return {
            "call_id": call.call_id,
            "line": call.line,
            "status": "open",
            "note": "A call is already open in this session; continue with advisor_call_say.",
            "bill": rt.bill(call, settings),
        }
    rt._persist_call(call)  # an open row lets a restart be told apart from a wrong id

    result = exchange(
        rt,
        settings,
        line,
        session_id,
        call.call_id,
        opening_message(rt, settings, session_id, question),
    )
    if "error" in result and call.is_open and not call.messages:
        # The opening never reached the advisor: end the call rather than
        # leave one open without its transcript.
        rt.manager.end(session_id, "error", call.call_id)
        result["status"] = "ended"
        result["note"] = "No call is open. You may try advisor_call_open again."
    return result


def say(rt: Runtime, args: Any, session_id: str | None) -> dict[str, Any]:
    if not session_id:
        return _error("no session id; cannot find the advisor call")
    settings = rt.settings()
    message = _str_arg(args, "message")
    if not message:
        return _error("`message` is required")
    call_id = _str_arg(args, "call_id") or None
    try:
        call = rt.manager.get(session_id, call_id)
    except CallError as err:
        return _call_error(rt, settings, err, session_id, call_id)
    line, problem = _usable_line(settings, call.line)
    if line is None:
        return _error(problem, call_id=call.call_id)
    if settings.budget.cycle_error:
        return _error(settings.budget.cycle_error, call_id=call.call_id)
    if settings.redact_secrets:
        redact = rt.redactor()
        if redact is not None:
            message = redact(message)
    return exchange(rt, settings, line, session_id, call.call_id, message)


def hangup(rt: Runtime, args: Any, session_id: str | None) -> dict[str, Any]:
    if not session_id:
        return _error("no session id; cannot find the advisor call")
    settings = rt.settings()
    call_id = _str_arg(args, "call_id") or None
    try:
        call = rt.manager.end(session_id, "hangup", call_id)
    except CallError as err:
        return _call_error(rt, settings, err, session_id, call_id)
    rt.manager.mark_activity(session_id)
    return {
        "call_id": call.call_id,
        "status": "ended",
        "end_reason": "hangup",
        "bill": rt.bill(call, settings),
    }


# --------------------------------------------------------------------------- wiring


def make_handler(rt: Runtime, fn: Callable[..., dict[str, Any]]) -> Callable[..., str]:
    """Wrap a tool function into a Hermes handler that never raises."""

    def handler(args: Any, session_id: str | None = None, **_: Any) -> str:
        try:
            return _json(fn(rt, args if isinstance(args, dict) else {}, session_id))
        except Exception as exc:  # noqa: BLE001
            log.exception("advisor-call: tool handler failed")
            return _json(_error(f"advisor-call internal error: {type(exc).__name__}: {exc}"))

    handler.__name__ = f"{fn.__name__}_handler"
    return handler


TOOLS: list[tuple[str, dict[str, Any], Callable[..., dict[str, Any]], str]] = [
    (OPEN, OPEN_SCHEMA, open_call, "📞"),
    (SAY, SAY_SCHEMA, say, "📞"),
    (HANGUP, HANGUP_SCHEMA, hangup, "📞"),
]
