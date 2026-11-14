"""Regression tests for the pre-release review findings. Each test is the
reviewer's demonstration, turned around to assert the fixed behaviour.
(Findings 2, 4 and 7 are covered in test_hooks.py and test_context.py.)"""

import json
import threading
from types import SimpleNamespace

import pytest
from conftest import FakeLlm, FakeResult, FakeUsage, gated_handlers, reply

from advisor_call.commands import SecondOpinionCommand
from advisor_call.cost import Usage, charge
from advisor_call.hooks import Hooks
from advisor_call.ledger import Ledger, LedgerError
from advisor_call.runtime import Runtime
from advisor_call.settings import Prices, parse_settings
from advisor_call.tools import OPEN, SAY, TOOLS, make_handler

SID = "s1"
QUESTION = "x" * 75_000  # estimated ~25,000 tokens (~$0.05 at Sonnet input prices)


def make(tmp_path, clock, cfg, llm=None, ledger=None):
    llm = llm or FakeLlm()
    rt = Runtime(
        llm=lambda: llm,
        ledger=ledger or Ledger.at_path(tmp_path / "l.db"),
        settings=lambda: parse_settings(cfg, None, {"mode": "manual"}),
        clock=clock,
        redactor=lambda: None,
    )
    return rt, llm, gated_handlers(rt)


def j(text):
    return json.loads(text)


def test_dense_text_does_not_overshoot_the_cap(tmp_path, clock):
    # Finding 1: 30,000 CJK characters cost more than chars/3 suggests.
    cfg = {
        "budget": {"cycle_start_day": 1, "pacing": "fixed", "per_call_usd": 0.05},
        "advisor": {"cache_ttl": "off", "max_tokens": 1000},
    }
    rt, llm, tools = make(tmp_path, clock, cfg)
    out = j(tools[OPEN]({"reason": "r", "question": "漢" * 30_000}, session_id=SID))
    assert out["status"] == "ended" and out["end_reason"] == "bill_cap"
    assert llm.calls == []  # refused before sending


def test_cached_opening_of_dense_text_is_not_underbilled():
    # Finding 1: with caching on, the opening's cached part is sized with the
    # character-aware estimate, so ~1 token per character CJK is not billed at 1/3.
    prices = Prices(2, 10, 0.1, 2.5)
    from advisor_call.cost import estimate_tokens

    est = estimate_tokens("漢" * 30_000)
    billed = charge(Usage(3, 500), prices, caching=True, prompt_est=est).cost_usd
    real = (30_000 * 1.0 * 2.5 + 3 * 2 + 500 * 10) / 1e6
    assert billed >= real


def test_measured_density_raises_the_next_estimate(tmp_path, clock):
    # Finding 1: after Hermes reports more tokens than estimated, the
    # pre-flight check for the next message on that call scales up.
    # Cap $0.20: the opening is estimated at ~$0.045 but billed $0.12. Unscaled,
    # the next message (~$0.045) would still fit; scaled 3x (~$0.125) it does not.
    cfg = {
        "budget": {"cycle_start_day": 1, "pacing": "fixed", "per_call_usd": 0.20},
        "advisor": {"cache_ttl": "off", "max_tokens": 500},
    }
    rt, llm, tools = make(tmp_path, clock, cfg)
    llm.script(reply("a", input_tokens=60_000, output_tokens=10))  # 3x the estimate
    j(tools[OPEN]({"reason": "r", "question": "x" * 60_000}, session_id=SID))
    assert rt.manager.get(SID).token_scale == pytest.approx(3.0, rel=0.05)
    out = j(tools[SAY]({"message": "next"}, session_id=SID))
    assert out.get("end_reason") == "bill_cap" and len(llm.calls) == 1


def test_no_new_call_without_fresh_approval(tmp_path, clock):
    # Finding 3: the hook let the call through because one was open; it ended
    # before the handler ran; the handler must not open a new, unapproved call.
    rt, llm, tools = make(tmp_path, clock, {"budget": {"cycle_start_day": 1}})
    llm.script(reply("a1"), reply("a2"))
    first = j(tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID))
    hooks = Hooks(rt)
    assert hooks.pre_tool_call(tool_name=OPEN, args={"question": "q2"}, session_id=SID) is None
    SecondOpinionCommand(rt)("hangup", session_id=SID)
    raw = {n: make_handler(rt, fn) for n, _, fn, _ in TOOLS}  # the handler, after the hook
    second = j(raw[OPEN]({"reason": "r", "question": "q2"}, session_id=SID))
    assert "not approved" in second["error"] and len(llm.calls) == 1
    assert first["call_id"]
    # Without any hook run at all, the handler refuses too.
    assert "not approved" in j(raw[OPEN]({"question": "q3"}, session_id="other"))["error"]


def test_concurrent_sessions_cannot_exceed_the_allowance(tmp_path, clock):
    # Finding 5: the ledger row is written before the reservation is released.
    cfg = {
        "budget": {
            "cycle_start_day": 1,
            "pacing": "fixed",
            "per_day_usd": 0.10,
            "per_call_usd": 0.10,
        },
        "advisor": {"cache_ttl": "off", "max_tokens": 1000},
    }
    ledger = Ledger.at_path(tmp_path / "l.db")
    rt, llm, tools = make(tmp_path, clock, cfg, ledger=ledger)
    original = ledger.record_exchange
    state = {"done": False}

    def racing(**kw):
        if not state["done"]:
            state["done"] = True
            tools[OPEN]({"reason": "r", "question": QUESTION}, session_id="s2")  # in the window
        return original(**kw)

    ledger.record_exchange = racing
    # Each exchange: estimated ~$0.06, really $0.054. Two would be $0.108.
    llm.script(
        reply("a", input_tokens=22_000, output_tokens=1_000),
        reply("b", input_tokens=22_000, output_tokens=1_000),
    )
    tools[OPEN]({"reason": "r", "question": QUESTION}, session_id="s1")
    assert ledger.day_total(rt.today()) <= 0.10 + 1e-9
    assert len(llm.calls) == 1  # the second session was refused (day_cap)


def test_failed_ledger_writes_still_count(tmp_path, clock):
    # Finding 6: spend the ledger could not store is kept and counted.
    cfg = {
        "budget": {
            "cycle_start_day": 1,
            "pacing": "fixed",
            "per_day_usd": 0.10,
            "per_call_usd": 0.10,
        },
        "advisor": {"cache_ttl": "off", "max_tokens": 1000},
    }
    ledger = Ledger.at_path(tmp_path / "l.db")
    rt, llm, tools = make(tmp_path, clock, cfg, ledger=ledger)

    def fail(**kw):
        raise LedgerError("disk full")

    ledger.record_exchange = fail
    spent = 0.0
    for i in range(5):
        llm.script(reply("a", input_tokens=22_000, output_tokens=1_000))
        out = j(tools[OPEN]({"reason": "r", "question": QUESTION}, session_id=f"s{i}"))
        spent += out.get("bill", {}).get("cost_usd", 0.0)
    assert spent <= 0.10 + 1e-9
    assert len(llm.calls) == 1  # unrecorded spend still counted: the rest refused


def test_fallback_reply_is_recorded_unbilled(tmp_path, clock):
    # Finding 8: recorded with ok=0 and its tokens, cost 0 for this line.
    ledger = Ledger.at_path(tmp_path / "l.db")
    rt, llm, tools = make(tmp_path, clock, {"budget": {"cycle_start_day": 1}}, ledger=ledger)
    llm.script(reply("x", provider="openrouter", input_tokens=5_000, output_tokens=4_000))
    out = j(tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID))
    assert "fell back" in out["error"]
    rows = (
        ledger._db().execute("select ok, cost_usd, model, input_tokens from exchanges").fetchall()
    )
    assert rows == [(0, 0.0, "openrouter", 5_000)]


def test_unusable_usage_values_do_not_make_a_reply_free(tmp_path, clock):
    # Finding 9: infinite usage no longer raises; the reply is billed by estimate.
    rt, llm, tools = make(tmp_path, clock, {"budget": {"cycle_start_day": 1}})
    llm.script(FakeResult(text="hi", usage=FakeUsage(float("inf"), 10)))
    out = j(tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID))
    assert out["reply"] == "hi" and out["bill"]["cost_usd"] > 0 and out["bill"]["estimate"]


def test_ask_race_with_executor_call(tmp_path, clock, monkeypatch):
    # Plausible finding: the executor opens a call between /second-opinion
    # ask's check and its own open.
    rt, llm, tools = make(tmp_path, clock, {"budget": {"cycle_start_day": 1}})
    llm.script(reply("executor advice"))
    cmd = SecondOpinionCommand(rt)
    import advisor_call.commands as commands

    real_open = commands.open_call

    def executor_first(rt_, args, session_id, **kw):
        tools[OPEN]({"reason": "r", "question": "q"}, session_id=session_id)
        return real_open(rt_, args, session_id, **kw)

    monkeypatch.setattr(commands, "open_call", executor_first)
    out = cmd("ask hello", session_id=SID)
    assert "executor opened a call meanwhile" in out
    assert rt.manager.get(SID).initiated_by == "executor"  # left alone


def test_transcript_is_sender_checked(tmp_path, clock):
    cfg = {"budget": {"cycle_start_day": 1}, "allowed_senders": ["alice"]}
    rt, llm, tools = make(tmp_path, clock, cfg)
    cmd = SecondOpinionCommand(rt)
    assert "not allowed" in cmd(
        "transcript", session_id=SID, source=SimpleNamespace(user_id="mallory")
    )
    assert "No advisor call" in cmd(
        "transcript", session_id=SID, source=SimpleNamespace(user_id="alice")
    )


def test_authorization_expires(tmp_path, clock):
    rt, _, _ = make(tmp_path, clock, {"budget": {"cycle_start_day": 1}})
    rt.authorize(SID, "approved")
    clock.advance(minutes=16)
    assert rt.take_authorization(SID) is None
    rt.authorize(SID, "approved")
    assert rt.take_authorization(SID) == "approved"
    assert rt.take_authorization(SID) is None  # one-shot


def test_concurrent_opens_one_call_one_charge(tmp_path, clock):
    rt, llm, tools = make(tmp_path, clock, {"budget": {"cycle_start_day": 1}})
    llm.script(*[reply(f"r{i}") for i in range(8)])
    barrier = threading.Barrier(4)
    results = []

    def worker():
        barrier.wait()
        results.append(j(tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({r.get("call_id") for r in results if r.get("call_id")}) == 1
