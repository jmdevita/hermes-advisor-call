"""U-28 /second-opinion subcommands; U-37 (command part)."""

from types import SimpleNamespace

import pytest
from conftest import reply

from advisor_call.commands import USAGE, SecondOpinionCommand
from advisor_call.tools import OPEN, SAY

SID = "session-1"


@pytest.fixture
def cmd(runtime):
    return SecondOpinionCommand(runtime)


def run(cmd, args="", session_id=SID, source=None):
    return cmd(args, session_id=session_id, session_key="k", platform="cli", source=source)


def test_status(cmd):
    out = run(cmd)
    assert "Advisor: sonnet (claude-sonnet-5-5)" in out
    assert "Today: $0.00 of $3.00" in out and "This cycle: $0.00 of $100.00" in out
    assert "Cycle: Nov 3 to Dec 2 (19 days left)" in out
    assert "Per-call cap: $0.50" in out
    assert "not pre-authorized" in out
    assert run(cmd, "status") == out


def test_status_without_cycle_day(cmd, config, fake_llm):
    # U-37: /second-opinion status shows the setup message; allow and ask refuse.
    config["budget"].pop("cycle_start_day")
    assert "cycle_start_day" in run(cmd)
    assert "cycle_start_day" in run(cmd, "allow")
    assert "cycle_start_day" in run(cmd, "ask what now?")
    assert fake_llm.calls == []


def test_allow_with_custom_cap_clamped(cmd, runtime, config):
    config["budget"].update(per_day_usd=0.30)
    out = run(cmd, "allow $5")
    assert "pre-authorized" in out and "cap $0.30" in out  # clamped to today's remainder
    assert runtime.manager.grant_for(SID).cap_usd == pytest.approx(0.30)
    assert "This session is pre-authorized (cap $0.30)" in run(cmd)


def test_allow_forms(cmd, runtime):
    assert "cap $0.50 per call" in run(cmd, "allow")
    assert "cap $0.25" in run(cmd, "allow 0.25")
    assert "more than $0" in run(cmd, "allow $0")
    assert "Usage" in run(cmd, "allow opus")  # one advisor model: nothing to choose
    assert "Usage" in run(cmd, "allow 1 2")


def test_revoke(cmd, runtime):
    assert "was not pre-authorized" in run(cmd, "revoke")
    run(cmd, "allow")
    assert "removed" in run(cmd, "revoke") and runtime.manager.grant_for(SID) is None


def test_ask(cmd, runtime, fake_llm):
    runtime.context.record_turn(SID, conversation_history=[], user_message="context here")
    fake_llm.script(reply("Direct answer."))
    out = run(cmd, "ask is this approach sound?")
    assert out == "Direct answer."  # just the answer; the cost shows in status and bill
    assert "ended (hung up), 1 message," in run(cmd, "status")  # not "1 messages"
    sent = fake_llm.calls[0].messages[1]["content"][0]["text"]
    assert "from the user" in sent and "context here" in sent
    assert runtime.manager.latest(SID).initiated_by == "user"
    assert not runtime.manager.take_activity(SID)  # no executor footer for it
    assert runtime.ledger.day_total(runtime.today()) > 0  # counts toward the budget


def test_ask_refused_while_executor_call_open(cmd, tools, fake_llm):
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    assert "hangup" in run(cmd, "ask anything?")
    assert "Usage" in run(cmd, "ask")


def test_ask_failure(cmd, fake_llm):
    fake_llm.script(RuntimeError("overloaded"))
    assert run(cmd, "ask hi").startswith("Advisor unavailable")


def test_hangup_tells_executor(cmd, tools, fake_llm):
    assert "No advisor call" in run(cmd, "hangup")
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    out = run(cmd, "hangup")
    assert out.startswith("Call ended.") and "the user ended the call" in out
    import json

    said = json.loads(tools[SAY]({"message": "still there?"}, session_id=SID))
    assert "the user ended the call" in said["error"]


def test_bill(cmd, runtime):
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
    out = run(cmd, "bill")
    assert "Today: $0.42 of $" in out and "(includes estimates)" in out
    assert "Cycle: Nov 3 to Dec 2 (19 days left)" in out and "Last 7 days:" in out
    assert "Days left in cycle" not in out  # the Cycle line already says it
    assert "Sat Nov 14: $0.42" in out and "Sun Nov 8: $0.00" in out


def test_transcript(cmd, tools, fake_llm):
    assert "No advisor call" in run(cmd, "transcript")
    fake_llm.script(reply("first advice"), reply("second advice"))
    tools[OPEN]({"reason": "r", "question": "the real question"}, session_id=SID)
    tools[SAY]({"message": "follow-up"}, session_id=SID)
    out = run(cmd, "transcript")
    assert "Executor: the real question" in out and "<transcript>" not in out
    assert "Advisor: second advice" in out and "Executor: follow-up" in out


def test_transcript_truncated_for_chat(cmd, tools, fake_llm):
    fake_llm.script(reply("z" * 10_000))
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    out = run(cmd, "transcript")
    assert len(out) <= 3500 and out.endswith("[truncated]")


def test_sender_check_on_spending_subcommands(cmd, config, fake_llm):
    config["allowed_senders"] = ["111"]
    stranger = SimpleNamespace(user_id="999")
    assert "not allowed" in run(cmd, "allow", source=stranger)
    assert "not allowed" in run(cmd, "ask hi", source=stranger)
    assert "not allowed" in run(cmd, "allow", source=SimpleNamespace(user_id=None))
    assert "Advisor:" in run(cmd, "status", source=stranger)  # read-only is fine
    assert "pre-authorized" in run(cmd, "allow", source=SimpleNamespace(user_id=111))
    assert fake_llm.calls == []


def test_no_session_and_unknown(cmd):
    assert "Send a message first" in run(cmd, "allow", session_id="")
    assert "Advisor:" in run(cmd, "status", session_id=None)
    assert run(cmd, "dance") == USAGE


def test_command_never_raises(cmd, runtime):
    runtime._settings = lambda: 1 / 0
    assert "failed: ZeroDivisionError" in run(cmd, "status")


def test_command_name_matches_the_platform(cmd, runtime):
    def on(platform, args):
        return cmd(args, session_id=SID, session_key="k", platform=platform, source=None)

    telegram = on("telegram", "allow")
    assert "/second_opinion revoke" in telegram and "/second-opinion" not in telegram
    assert "/second_opinion ask" in on("telegram", "dance")  # the usage text too
    assert "/second-opinion" in on("cli", "dance")
