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
    MIN_CALIBRATION_TOKENS,
    Charge,
    Usage,
    call_cap,
    charge,
    estimate_cost,
    preflight_refusal,
    reusable_prefix,
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
        "Get a second opinion from a stronger senior model on a hard problem. The user must "
        "approve the call. Use it when you are stuck after real attempts, facing an "
        "architecture or debugging decision with costly consequences, or when the user asks "
        "you to get a second opinion. Do not use it for routine work. The advisor has no "
        "tools and cannot see your files: put the exact error text, relevant code and what "
        "you already tried in `question`. After opening, continue with advisor_call_say and "
        "finish with advisor_call_hangup."
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
    usage: Usage,
    line: Line,
    settings: Settings,
    req: AdvisorRequest,
    reused: float,
    est_cost: float,
) -> Charge:
    if not usage.reported:
        # The provider answered but no usage came back: bill the upper bound.
        return Charge(est_cost, estimated=True)
    return charge(
        usage,
        line.prices,
        caching=settings.caching,
        prompt_est=req.prompt_est,
        uncached_est=req.uncached_est,
        reused_est=reused,
    )


def _measured_scale(usage: Usage, settings: Settings, req: AdvisorRequest) -> float | None:
    """How much denser the text was than estimate_tokens() assumed, measured
    on whatever part of the prompt Hermes reported exactly."""
    if not usage.reported:
        return None
    exact_total = not settings.caching or usage.cache_read_tokens or usage.cache_write_tokens
    if exact_total:
        seen = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        basis = req.prompt_est
    else:
        seen, basis = usage.input_tokens, req.uncached_est
    if basis < MIN_CALIBRATION_TOKENS or seen <= 0:
        return None
    return seen / basis


_estimate_warned = False


def _warn_estimate_once() -> None:
    global _estimate_warned
    if not _estimate_warned:
        _estimate_warned = True
        log.warning(
            "advisor-call: advisor bills are estimates while prompt caching is on: Hermes does "
            "not pass Anthropic's cache token counts to plugins. Set "
            "plugins.entries.advisor-call.advisor.cache_ttl: off for exact bills (DESIGN.md 2.1)."
        )


def _record(
    rt: Runtime, call: Call, line: Line, model: str, usage: Usage, ch: Charge, ok: bool
) -> None:
    """Write the exchange to the ledger. Called while the exchange's
    reservation is still held, so other sessions always see its cost in one
    place or the other. If the write fails, the cost is kept in memory and
    still counted against today's allowance and the cycle budget."""
    if ch.estimated:
        _warn_estimate_once()
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
        rt.add_unrecorded(ch.cost_usd)
        log.warning(
            "advisor-call: could not record exchange on %s (%s); $%.4f kept in memory "
            "and still counted against the caps",
            call.call_id,
            exc,
            ch.cost_usd,
        )


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


def _failure(
    rt: Runtime, settings: Settings, call: Call, message: str, ended_now: str = ""
) -> dict[str, Any]:
    result = _error(
        message,
        call_id=call.call_id,
        status="open" if call.is_open else "ended",
        bill=rt.bill(call, settings),
    )
    if ended_now:
        result.update(end_reason=ended_now, notice=_ended_notice(ended_now))
    return result


def exchange(
    rt: Runtime, settings: Settings, line: Line, session_id: str, call_id: str | None, text: str
) -> dict[str, Any]:
    """Send one executor message on the session's open call: reserve under
    the caps, ask the advisor outside the lock, record, then settle."""
    captured: dict[str, Any] = {}

    def preflight(call: Call, pending: list[dict[str, str]], elsewhere: float):
        req = build_request(pending, settings.cache_ttl)
        reused = reusable_prefix(
            call.cached_est, call.cached_at, rt.clock.time(), settings.cache_ttl
        )
        scale = max(1.0, call.token_scale)  # text on this call measured denser: estimate up
        est = estimate_cost(
            req.prompt_est * scale,
            line.max_tokens,
            line.prices,
            caching=settings.caching,
            reused_est=reused * scale,
        )
        state = rt.budget(settings)  # raises -> no exchange
        captured.update(req=req, reused=reused, est=est)
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
    reused: float = captured["reused"]
    est: float = captured["est"]
    rt.manager.mark_activity(session_id)

    try:
        llm = rt.llm()
    except Exception as exc:  # noqa: BLE001 - nothing was sent: release, bill nothing
        rt.manager.release(res)
        log.exception("advisor-call: ctx.llm unavailable")
        return _failure(rt, settings, call, f"advisor unavailable: {type(exc).__name__}: {exc}")

    try:
        reply = ask(llm, line, settings, req)
    except AdvisorError as err:
        usage = err.usage or Usage()
        if not err.billable:
            # Answered by another provider (Hermes' fallback): record its
            # usage for the record, but do not price it as this line.
            ch = Charge(0.0, False)
            _record(rt, call, line, err.provider or "fallback", usage, ch, ok=False)
        else:
            if err.bill_estimate:
                ch = Charge(est, estimated=True)  # may have been processed and billed
            elif usage.reported:
                ch = _bill_for(usage, line, settings, req, reused, est)
            else:
                ch = Charge(0.0, False)
            if ch.cost_usd > 0:
                _record(rt, call, line, line.model, usage, ch, ok=False)
        settled = rt.manager.settle(
            res,
            reply=None,
            cost_usd=ch.cost_usd,
            usage=usage,
            ok=False,
            estimated=ch.estimated,
            token_scale=_measured_scale(usage, settings, req),
        )
        _log_transcript(rt, settings, call, text, None)
        return _failure(
            rt, settings, call, f"advisor unavailable: {err.message}", settled.ended_now
        )
    except Exception as exc:  # noqa: BLE001 - a bug after the request may have gone out
        log.exception("advisor-call: unexpected error during the advisor request")
        ch = Charge(est, estimated=True)
        _record(rt, call, line, line.model, Usage(), ch, ok=False)
        settled = rt.manager.settle(
            res, reply=None, cost_usd=ch.cost_usd, usage=Usage(), ok=False, estimated=True
        )
        return _failure(
            rt,
            settings,
            call,
            f"advisor unavailable: {type(exc).__name__}: {exc}",
            settled.ended_now,
        )

    ch = _bill_for(reply.usage, line, settings, req, reused, est)
    _record(rt, call, line, reply.model, reply.usage, ch, ok=True)
    settled = rt.manager.settle(
        res,
        reply=reply.text,
        cost_usd=ch.cost_usd,
        usage=reply.usage,
        ok=True,
        estimated=ch.estimated,
        cached_prefix_est=req.cached_prefix_est if settings.caching else 0.0,
        token_scale=_measured_scale(reply.usage, settings, req),
    )
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

    # The approval hook leaves a one-shot authorization for this tool call.
    # A new (paid) call needs one that came from a human approval or a
    # matching /second-opinion allow; "a call is already open" is not enough,
    # because that call may have ended while this tool call was waiting.
    authorization = rt.take_authorization(session_id) if initiated_by == "executor" else "user"

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

    if authorization not in ("approved", "grant", "user"):
        return _error(
            "This new call was not approved (the previous call ended while the request was "
            "waiting). Call advisor_call_open again; the user will be asked to approve."
        )

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
    rt.persist_call(call)  # an open row lets a restart be told apart from a wrong id

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
