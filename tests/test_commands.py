"""U-28 /phone subcommands; U-37 (command part)."""

from types import SimpleNamespace

import pytest
from conftest import reply

from advisor_call.commands import USAGE, PhoneCommand
from advisor_call.tools import OPEN, SAY, TOOLS, make_handler

SID = "session-1"


@pytest.fixture
def phone(runtime):
    return PhoneCommand(runtime)


@pytest.fixture
def tools(runtime):
    return {name: make_handler(runtime, fn) for name, _, fn, _ in TOOLS}


def run(phone, args="", session_id=SID, source=None):
    return phone(args, session_id=session_id, session_key="k", platform="cli", source=source)


def test_status(phone):
    out = run(phone)
    assert "Advisor line: sonnet (claude-sonnet-5-5)" in out
    assert "Today: $0.00 of $" in out and "budget left $100.00 / 19 days" in out
    assert "Cycle: Nov 3 to Dec 2 (19 days left)" in out
    assert "Per-call cap today: $0.79" in out
    assert "not pre-authorized" in out
    assert run(phone, "status") == out


def test_status_without_cycle_day(phone, config, fake_llm):
    # U-37: /phone status shows the setup message; allow and ask refuse.
    config["budget"].pop("cycle_start_day")
    assert "cycle_start_day" in run(phone)
    assert "cycle_start_day" in run(phone, "allow")
    assert "cycle_start_day" in run(phone, "ask what now?")
    assert fake_llm.calls == []


def test_allow_with_custom_cap_clamped(phone, runtime, config):
    config["budget"].update(pacing="fixed", per_day_usd=0.30)
    out = run(phone, "allow $5")
    assert "pre-authorized" in out and "cap $0.30" in out  # clamped to today's remainder
    assert runtime.manager.grant_for(SID).cap_usd == pytest.approx(0.30)
    assert "This session is pre-authorized (line sonnet, cap $0.30)" in run(phone)


def test_allow_forms(phone, runtime):
    assert "computed per call" in run(phone, "allow")
    assert "cap $0.25" in run(phone, "allow sonnet 0.25")
    assert "not usable" in run(phone, "allow opus")
    assert "more than $0" in run(phone, "allow $0")
    assert "Usage" in run(phone, "allow a b")


def test_revoke(phone, runtime):
    assert "was not pre-authorized" in run(phone, "revoke")
    run(phone, "allow")
    assert "removed" in run(phone, "revoke") and runtime.manager.grant_for(SID) is None


def test_ask(phone, runtime, fake_llm):
    runtime.context.record_turn(SID, conversation_history=[], user_message="context here")
    fake_llm.script(reply("Direct answer."))
    out = run(phone, "ask is this approach sound?")
    assert out.startswith("Direct answer.") and "📞 Call c-" in out and "ended (hung up)" in out
    sent = fake_llm.calls[0].messages[1]["content"][0]["text"]
    assert "from the user" in sent and "context here" in sent
    assert runtime.manager.latest(SID).initiated_by == "user"
    assert not runtime.manager.take_activity(SID)  # no executor footer for it
    assert runtime.ledger.day_total(runtime.today()) > 0  # counts toward the budget


def test_ask_refused_while_executor_call_open(phone, tools, fake_llm):
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    assert "hangup" in run(phone, "ask anything?")
    assert "Usage" in run(phone, "ask")


def test_ask_failure(phone, fake_llm):
    fake_llm.script(RuntimeError("overloaded"))
    assert run(phone, "ask hi").startswith("Advisor unavailable")


def test_hangup_tells_executor(phone, tools, fake_llm):
    assert "No advisor call" in run(phone, "hangup")
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    out = run(phone, "hangup")
    assert out.startswith("Call ended.") and "the user ended the call" in out
    import json

    said = json.loads(tools[SAY]({"message": "still there?"}, session_id=SID))
    assert "the user ended the call" in said["error"]


def test_bill(phone, runtime):
    runtime.ledger.record_exchange(
        ts=0,
        day=runtime.today(),
        session_id="s",
        call_id="c",
        line="sonnet",
        model="m",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.42,
        ok=True,
        estimated=True,
    )
    out = run(phone, "bill")
    assert "Today: $0.42 of $" in out and "(includes estimates)" in out
    assert "Days left in cycle: 19" in out and "Last 7 days:" in out
    assert "Sat Nov 14: $0.42" in out and "Sun Nov 8: $0.00" in out


def test_transcript(phone, tools, fake_llm):
    assert "No advisor call" in run(phone, "transcript")
    fake_llm.script(reply("first advice"), reply("second advice"))
    tools[OPEN]({"reason": "r", "question": "the real question"}, session_id=SID)
    tools[SAY]({"message": "follow-up"}, session_id=SID)
    out = run(phone, "transcript")
    assert "Executor: the real question" in out and "<transcript>" not in out
    assert "Advisor: second advice" in out and "Executor: follow-up" in out


def test_transcript_truncated_for_chat(phone, tools, fake_llm):
    fake_llm.script(reply("z" * 10_000))
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    out = run(phone, "transcript")
    assert len(out) <= 3500 and out.endswith("[truncated]")


def test_sender_check_on_spending_subcommands(phone, config, fake_llm):
    config["allowed_senders"] = ["111"]
    stranger = SimpleNamespace(user_id="999")
    assert "not allowed" in run(phone, "allow", source=stranger)
    assert "not allowed" in run(phone, "ask hi", source=stranger)
    assert "not allowed" in run(phone, "allow", source=SimpleNamespace(user_id=None))
    assert "Advisor line" in run(phone, "status", source=stranger)  # read-only is fine
    assert "pre-authorized" in run(phone, "allow", source=SimpleNamespace(user_id=111))
    assert fake_llm.calls == []


def test_no_session_and_unknown(phone):
    assert "Send a message first" in run(phone, "allow", session_id="")
    assert "Advisor line" in run(phone, "status", session_id=None)
    assert run(phone, "dance") == USAGE


def test_command_never_raises(phone, runtime):
    runtime._settings = lambda: 1 / 0
    assert "failed: ZeroDivisionError" in run(phone, "status")
