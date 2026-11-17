"""Auto-suggest: reading Hermes' own loop warnings, and the line for the user."""

import json

import pytest
from conftest import reply

from advisor_call.commands import SecondOpinionCommand
from advisor_call.hooks import Hooks
from advisor_call.settings import parse_settings
from advisor_call.suggest import MAX_PER_SESSION, Tracker, hermes_warning
from advisor_call.tools import OPEN

SID = "s1"
# Verbatim shapes from hermes-agent 0.23 (agent/tool_guardrails.py).
LOOP_WARNING = (
    "\n\n[Tool loop warning: same_tool_failure_warning; count=3; terminal has failed "
    "3 times this turn. This looks like a loop.]"
)
HARD_STOP = "\n\n[Tool loop hard stop: exact_failure_block; count=5; stopped.]"
IDENTICAL_CALL = (
    "\n\n[hermes note: this is the 3rd consecutive identical call to read_file with "
    "identical arguments returning the same result. Do not repeat it — change arguments, "
    "use a different tool, or proceed with what you have.]"
)
IDENTICAL_CYCLE = (
    "\n\n[hermes note: the last 3 rounds repeated the same cycle of 2 tool calls (ending "
    "with terminal) with identical arguments and identical results. Do not repeat the "
    "batch — change arguments, use a different tool, or proceed with what you have.]"
)


def convo(*tool_results, fmt="openai"):
    """A request: user message, assistant tool calls, then the newest tool results."""
    msgs = [{"role": "user", "content": "fix it"}, {"role": "assistant", "content": "trying"}]
    for i, text in enumerate(tool_results):
        if fmt == "openai":
            msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": text})
        else:  # Anthropic: tool results are blocks in a user message
            block = {"type": "tool_result", "tool_use_id": f"t{i}", "content": text}
            msgs.append({"role": "user", "content": [block]})
    return msgs


# ------------------------------------------------------------------- reading the warning


@pytest.mark.parametrize(
    ("suffix", "kind"),
    [
        (LOOP_WARNING, "same_tool_failure_warning"),
        (HARD_STOP, "exact_failure_block"),
        (IDENTICAL_CALL, "identical_call"),
        (IDENTICAL_CYCLE, "identical_cycle"),
    ],
)
def test_hermes_warnings_are_recognised(suffix, kind):
    assert hermes_warning(convo('{"exit_code": 1}' + suffix)) == kind
    assert hermes_warning(convo("fine", '{"exit_code": 1}' + suffix, fmt="anthropic")) == kind
    parts = [{"type": "text", "text": "out" + suffix}]
    assert hermes_warning(convo(parts)) == kind


def test_no_warning():
    assert hermes_warning(convo('{"exit_code": 1}', "plain output")) is None
    assert hermes_warning([{"role": "user", "content": "hello"}]) is None
    for junk in (None, "text", [], [None, 3], [{"role": "tool", "content": None}]):
        assert hermes_warning(junk) is None


def test_only_the_newest_results_count():
    # A warning from an earlier step (before the last assistant message) is old news.
    msgs = convo("x" + LOOP_WARNING) + [{"role": "assistant", "content": "next"}]
    assert hermes_warning(msgs + [{"role": "tool", "content": "ok"}]) is None


def test_warning_text_inside_a_result_is_not_a_warning():
    # e.g. the executor reads Hermes' source: the marker is not at the end of the result.
    assert hermes_warning(convo(f"docs say{LOOP_WARNING}\nand more text after")) is None


# ------------------------------------------------------------------- tracker


def test_tracker_once_per_turn_and_session_limit():
    t = Tracker()
    assert t.flagged(SID) and not t.flagged(SID)  # once per turn
    assert t.take_notice(SID) and not t.take_notice(SID)  # told once
    for _ in range(MAX_PER_SESSION - 1):
        t.new_turn(SID)
        assert t.flagged(SID)
    t.new_turn(SID)
    assert not t.flagged(SID)  # session limit


def test_tracker_refused_is_not_counted():
    t = Tracker()
    assert not t.flagged(SID, lambda: False)
    assert not t.take_notice(SID)
    assert t.flagged(SID)


def test_tracker_call_clears_and_silences_the_turn():
    t = Tracker()
    t.flagged(SID)
    assert t.advisor_called(SID) is True
    assert not t.take_notice(SID) and not t.flagged(SID)  # quiet for the rest of the turn
    t.new_turn(SID)
    assert t.advisor_called(SID) is False  # no warning came first


# ------------------------------------------------------------------- settings


def test_suggest_setting():
    assert parse_settings({}).suggest is True
    assert parse_settings({"suggest": False}).suggest is False
    bad = parse_settings({"suggest": {"mode": "both"}})
    assert bad.suggest is True and any("suggest" in w for w in bad.warnings)


# ------------------------------------------------------------------- hook wiring


@pytest.fixture
def hooks(runtime):
    return Hooks(runtime)


def start_turn(hooks, platform="telegram", sid=SID):
    hooks.pre_llm_call(
        session_id=sid,
        conversation_history=[],
        user_message="fix it",
        sender_id="1",
        platform=platform,
    )


def loop_flagged(hooks, sid=SID, suffix=LOOP_WARNING):
    request = {"messages": convo('{"exit_code": 1}' + suffix)}
    before = json.dumps(request)
    assert hooks.llm_request(request=request, session_id=sid) is None
    assert json.dumps(request) == before  # the request is never changed


def final(hooks, sid=SID, platform="telegram"):
    return hooks.transform_llm_output(
        response_text="I could not fix it.", session_id=sid, platform=platform
    )


def test_user_told_when_the_executor_does_not_call(hooks, runtime):
    start_turn(hooks)
    loop_flagged(hooks)
    loop_flagged(hooks, suffix=IDENTICAL_CALL)  # a second warning in the turn: still one line
    out = final(hooks)
    assert out.count("💡") == 1
    assert "💡 Hermes flagged a tool loop this turn and the agent didn't call the advisor" in out
    assert "/second_opinion ask" in out  # platform-aware command name
    assert final(hooks) is None  # once
    assert runtime.ledger.suggestion_stats(runtime.today(), runtime.today()) == (1, 0)


def test_no_line_without_a_warning(hooks):
    start_turn(hooks)
    hooks.llm_request(request={"messages": convo("all good")}, session_id=SID)
    assert final(hooks) is None


def test_a_call_after_the_warning_clears_the_line_and_is_recorded(hooks, runtime, tools, fake_llm):
    start_turn(hooks)
    loop_flagged(hooks)
    fake_llm.script(reply("try this"))
    out = json.loads(tools[OPEN]({"reason": "stuck", "question": "help"}, session_id=SID))
    assert out["reply"] == "try this"
    text = final(hooks)
    assert "💡" not in text and "📞 Advisor call" in text
    assert runtime.ledger.suggestion_stats(runtime.today(), runtime.today()) == (1, 1)
    status = SecondOpinionCommand(runtime)("status", session_id=SID, platform="cli")
    assert "Auto-suggest, last 7 days: 1 loop warning, 1 led to a call" in status


def test_off(hooks, config):
    config["suggest"] = False
    start_turn(hooks)
    loop_flagged(hooks)
    assert final(hooks) is None


def test_no_line_when_a_call_could_not_happen(hooks, runtime, config, tools, fake_llm):
    # A call is already open.
    start_turn(hooks)
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    final(hooks)  # takes the bill line
    start_turn(hooks)
    loop_flagged(hooks)
    assert final(hooks) is None
    # Today's allowance is used up.
    config["budget"].update(per_day_usd=0.000001)
    start_turn(hooks, sid="s2")
    loop_flagged(hooks, sid="s2")
    assert final(hooks, sid="s2") is None
    # cycle_start_day missing.
    config["budget"] = {}
    start_turn(hooks, sid="s3")
    loop_flagged(hooks, sid="s3")
    assert final(hooks, sid="s3") is None


def test_unattended_needs_a_grant(hooks, runtime):
    start_turn(hooks, platform="cron")
    loop_flagged(hooks)
    assert final(hooks, platform="cron") is None
    start_turn(hooks, platform="cron", sid="s2")
    runtime.manager.grant("s2", cap_usd=None)
    loop_flagged(hooks, sid="s2")
    assert "💡" in final(hooks, sid="s2", platform="cron")


def test_fails_open(hooks, runtime):
    runtime._settings = lambda: 1 / 0
    assert (
        hooks.llm_request(request={"messages": convo("x" + LOOP_WARNING)}, session_id=SID) is None
    )
    assert hooks.llm_request(request={"messages": "junk"}, session_id=None) is None
