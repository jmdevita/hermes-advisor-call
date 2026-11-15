"""U-16 to U-20 through the tool handlers, plus caps, billing and restarts."""

import json

import pytest
from conftest import gated_handlers, reply

from advisor_call.ledger import Ledger
from advisor_call.runtime import Runtime
from advisor_call.settings import parse_settings
from advisor_call.tools import HANGUP, OPEN, SAY

SID = "session-1"


def call(tools, name, args, session_id=SID):
    return json.loads(tools[name](args, session_id=session_id))


def open_(tools, question="Why does the build fail?", **kw):
    return call(tools, OPEN, {"reason": "stuck", "question": question, **kw})


def test_open_returns_reply_and_bill(tools, fake_llm, runtime):
    # U-18
    runtime.context.record_turn(
        SID,
        conversation_history=[
            {"role": "user", "content": "please fix the build"},
            {"role": "assistant", "content": "trying make"},
        ],
        user_message="any luck?",
    )
    fake_llm.script(reply("Run make clean first."))
    out = open_(tools)
    assert out["status"] == "open" and out["reply"] == "Run make clean first."
    assert out["call_id"].startswith("c-") and out["line"] == "sonnet"
    assert out["advisor_model"] == "claude-sonnet-5-5"
    bill = out["bill"]
    assert bill["messages"] == 1 and bill["max_messages"] == 20
    assert bill["cost_usd"] > 0 and bill["cap_usd"] > 0
    assert bill["day_usd"] == pytest.approx(bill["cost_usd"])
    assert bill["cycle_cap_usd"] == 100.0 and bill["estimate"] is True
    opening = fake_llm.calls[0].messages[1]["content"][0]["text"]
    assert "please fix the build" in opening and "Why does the build fail?" in opening


def test_transcript_sent_once(tools, fake_llm, runtime):
    runtime.context.record_turn(SID, conversation_history=[], user_message="UNIQUE-HISTORY")
    fake_llm.script(reply("a1"), reply("a2"))
    open_(tools)
    out = call(tools, SAY, {"message": "here is the log"})
    assert out["reply"] == "a2" and out["bill"]["messages"] == 2
    second = fake_llm.calls[1].messages
    assert second[-1] == {"role": "user", "content": "here is the log"}
    assert sum("UNIQUE-HISTORY" in json.dumps(m) for m in second) == 1  # only the opening


def test_no_transcript_is_noted(tools, fake_llm):
    fake_llm.script(reply())
    open_(tools)
    assert "No transcript" in fake_llm.calls[0].messages[1]["content"][0]["text"]


def test_empty_reply_is_error_and_not_appended(tools, fake_llm, runtime):
    # U-16 (tool side)
    fake_llm.script(reply("first"), reply("", input_tokens=50, output_tokens=4000))
    open_(tools)
    c = runtime.manager.get(SID)
    before = list(c.messages)
    out = call(tools, SAY, {"message": "more?"})
    assert "empty" in out["error"] and out["status"] == "open"
    assert c.messages == before
    assert out["bill"]["messages"] == 2  # billed attempt counts


def test_advisor_exception_keeps_call_open(tools, fake_llm, runtime):
    # U-17
    fake_llm.script(reply("first"), ConnectionError("connection refused"))
    open_(tools)
    out = call(tools, SAY, {"message": "next"})
    assert out["error"].startswith("advisor unavailable") and out["status"] == "open"
    assert runtime.manager.get(SID).is_open
    assert out["bill"]["messages"] == 1  # an unbilled failure does not count


def test_timeout_is_billed_at_the_estimate(tools, fake_llm, runtime):
    # A timed-out request may still have been processed and billed.
    fake_llm.script(reply("first"), TimeoutError("read timed out"))
    first = open_(tools)
    out = call(tools, SAY, {"message": "next"})
    assert out["status"] == "open" and out["bill"]["messages"] == 2
    assert out["bill"]["cost_usd"] > first["bill"]["cost_usd"] and out["bill"]["estimate"]


def test_handlers_never_raise(tools, runtime):
    runtime._llm = lambda: (_ for _ in ()).throw(RuntimeError("ctx.llm unavailable"))
    for name in (OPEN, SAY, HANGUP):
        for args in (None, [], {"question": 5}, {"message": None}, "junk"):
            out = call(tools, name, args)
            assert "error" in out
    out = open_(tools)
    assert "advisor unavailable" in out["error"] and out["status"] == "ended"


def test_hangup_makes_no_llm_call(tools, fake_llm):
    # U-19
    fake_llm.script(reply())
    open_(tools)
    out = call(tools, HANGUP, {})
    assert out["status"] == "ended" and out["end_reason"] == "hangup"
    assert out["bill"]["messages"] == 1 and len(fake_llm.calls) == 1
    again = call(tools, SAY, {"message": "hello?"})
    assert "No open advisor call" in again["error"]


def test_missing_session_id(tools, fake_llm):
    # U-20
    for name, args in (
        (OPEN, {"reason": "r", "question": "q"}),
        (SAY, {"message": "m"}),
        (HANGUP, {}),
    ):
        for sid in (None, ""):
            out = call(tools, name, args, session_id=sid)
            assert "no session id" in out["error"]
            # The handler itself (reached when the hook lets it through) gives guidance.
            assert "guidance" in out or out.get("blocked")
    assert fake_llm.calls == []


def test_cycle_day_unset_refuses(tools, fake_llm, config):
    config["budget"].pop("cycle_start_day")
    out = open_(tools)
    assert "cycle_start_day" in out["error"] and fake_llm.calls == []


def test_model_not_allowed(tools, fake_llm):
    out = open_(tools, model="opus")
    assert "not available" in out["error"] and "Allowed: sonnet" in out["error"]
    assert fake_llm.calls == []


def test_tiny_cap_ends_before_sending(tools, fake_llm, config):
    config["budget"].update(per_call_usd=0.001, pacing="fixed")
    out = open_(tools)
    assert out["status"] == "ended" and out["end_reason"] == "bill_cap"
    assert fake_llm.calls == []


def test_cap_reached_after_reply(tools, fake_llm, config):
    config["budget"].update(per_call_usd=0.10, pacing="fixed")
    config["advisor"] = {"max_tokens": 100, "cache_ttl": "off"}
    fake_llm.script(reply("big", input_tokens=20_000, output_tokens=7_000))
    out = open_(tools)
    assert out["reply"] == "big" and out["status"] == "ended"
    assert out["end_reason"] == "bill_cap" and "Continue on your own" in out["notice"]


def test_exact_bill_when_caching_off(tools, fake_llm, config, db_path):
    config["advisor"] = {"cache_ttl": "off"}
    fake_llm.script(reply(input_tokens=10_000, output_tokens=1_000))
    out = open_(tools)
    assert out["bill"]["cost_usd"] == pytest.approx((10_000 * 2 + 1_000 * 10) / 1e6)
    assert "estimate" not in out["bill"]
    assert all(isinstance(m["content"], str) for m in fake_llm.calls[0].messages)
    led = Ledger.at_path(db_path)
    assert led.day_total(runtime_today()) == pytest.approx(0.03)


def runtime_today():
    import datetime as dt

    return dt.date(2026, 11, 14)


def test_ledger_rows_and_estimate_flag(tools, fake_llm, db_path):
    fake_llm.script(reply(), reply())
    open_(tools)
    call(tools, SAY, {"message": "again"})
    led = Ledger.at_path(db_path)
    assert led.has_estimates(runtime_today(), runtime_today())
    assert led.day_total(runtime_today()) > 0


def test_day_allowance_used_up(tools, fake_llm, config, runtime):
    config["budget"].update(pacing="fixed", per_day_usd=0.05)
    runtime.ledger.record_exchange(
        ts=0,
        day=runtime_today(),
        session_id="other",
        call_id="c-x",
        line="sonnet",
        model="m",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.05,
        ok=True,
    )
    out = open_(tools)
    assert "used up" in out["error"] and fake_llm.calls == []


def test_ledger_failure_fails_closed(tools, fake_llm, runtime):
    # U-5 through the tool: cannot read spend -> no call.
    def broken():
        raise OSError("disk gone")

    runtime.ledger = Ledger(broken)
    out = open_(tools)
    assert "budget cannot be checked" in out["error"] and fake_llm.calls == []  # hook blocks


def test_custom_cap_from_grant(tools, fake_llm, runtime):
    runtime.manager.grant(SID, line="sonnet", cap_usd=0.2)
    fake_llm.script(reply())
    out = open_(tools)
    assert out["bill"]["cap_usd"] == pytest.approx(0.2)


def test_existing_call_returned_with_note(tools, fake_llm):
    fake_llm.script(reply())
    first = open_(tools)
    second = open_(tools, question="another")
    assert second["call_id"] == first["call_id"] and "already open" in second["note"]
    assert len(fake_llm.calls) == 1


def test_session_isolation_through_tools(tools, fake_llm):
    fake_llm.script(reply())
    a = open_(tools)
    out = call(tools, SAY, {"message": "hijack", "call_id": a["call_id"]}, session_id="B")
    assert "no call" in out["error"] and "reply" not in out
    out = call(tools, HANGUP, {"call_id": a["call_id"]}, session_id="B")
    assert "error" in out
    assert len(fake_llm.calls) == 1


def test_restart_is_reported(fake_llm, db_path, clock, config):
    def make():
        return Runtime(
            llm=lambda: fake_llm,
            ledger=Ledger.at_path(db_path),
            settings=lambda: parse_settings(config),
            clock=clock,
            redactor=lambda: None,
        )

    rt = make()
    first = gated_handlers(rt)
    fake_llm.script(reply())
    opened = json.loads(first[OPEN]({"reason": "r", "question": "q"}, session_id=SID))
    assert rt.manager.get(SID).is_open
    rt2 = make()
    restarted = gated_handlers(rt2)
    out = json.loads(restarted[SAY]({"message": "m", "call_id": opened["call_id"]}, session_id=SID))
    assert "gateway restarted" in out["error"]


def test_opening_failure_ends_call(tools, fake_llm, runtime):
    fake_llm.script(ConnectionError("refused"))
    out = open_(tools)
    assert out["status"] == "ended" and "try advisor_call_open again" in out["note"]
    assert runtime.manager.latest(SID).end_reason == "error"
    fake_llm.script(reply("second try"))
    assert open_(tools)["reply"] == "second try"


def test_redaction_applied_to_messages(fake_llm, db_path, clock, config):
    rt = Runtime(
        llm=lambda: fake_llm,
        ledger=Ledger.at_path(db_path),
        settings=lambda: parse_settings(config),
        clock=clock,
        redactor=lambda: lambda t: t.replace("hunter2", "[REDACTED]"),
    )
    handlers = gated_handlers(rt)
    fake_llm.script(reply(), reply())
    handlers[OPEN]({"reason": "r", "question": "pw hunter2"}, session_id=SID)
    handlers[SAY]({"message": "still hunter2"}, session_id=SID)
    sent = json.dumps([c.messages for c in fake_llm.calls])
    assert "hunter2" not in sent and "[REDACTED]" in sent


def test_reuse_within_ttl_is_billed_lower(tools, fake_llm, clock):
    """The same follow-up costs less within the cache TTL (reused prefix at
    the input price) than after it (the prefix is rewritten)."""
    big = "x" * 30_000
    deltas = []
    for sid, wait in (("A", 0), ("B", 10)):
        fake_llm.script(
            reply(input_tokens=1, output_tokens=100), reply(input_tokens=500, output_tokens=100)
        )
        opened = call(tools, OPEN, {"reason": "r", "question": big}, session_id=sid)
        clock.advance(minutes=wait)
        said = call(tools, SAY, {"message": "y" * 2_000}, session_id=sid)
        deltas.append(said["bill"]["cost_usd"] - opened["bill"]["cost_usd"])
    within, after = deltas
    assert 0 < within < after
    # The tail measured 4 chars per token, so ~31,600 reused chars are ~7,900
    # tokens, billed at 2.00 instead of 2.50: about 0.4 cents cheaper.
    assert after - within == pytest.approx(31_600 / 4 * 0.5 / 1e6, rel=0.1)


def test_estimate_warning_logged_once(tools, fake_llm, caplog, monkeypatch):
    import advisor_call.tools as tools_module

    monkeypatch.setattr(tools_module, "_estimate_warned", False)
    fake_llm.script(reply(), reply())
    with caplog.at_level("WARNING", logger="advisor_call.tools"):
        open_(tools)
        call(tools, SAY, {"message": "again"})
    assert sum("bills are estimates" in r.message for r in caplog.records) == 1
