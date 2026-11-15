"""Auto-suggest: detection rules (suggest.Tracker) and the hook wiring."""

import json

import pytest
from conftest import reply

from advisor_call.commands import SecondOpinionCommand
from advisor_call.hooks import Hooks
from advisor_call.settings import parse_settings
from advisor_call.suggest import (
    NUDGE_PREFIX,
    SuggestSettings,
    Tracker,
    normalize_args,
    normalize_error,
)
from advisor_call.tools import OPEN

SID = "s1"
DEFAULT = SuggestSettings(cooldown_calls=0)
# Genuinely different errors (numbers alone do not make errors different: they are normalised).
ERRORS = ["error: file not found", "error: permission denied", "error: connection refused"]


def feed(tracker, calls, settings=DEFAULT, sid=SID):
    """calls: (tool, args, result, failed[, batch]) tuples. Returns detections."""
    out = []
    for i, call in enumerate(calls):
        tool, args, result, failed = call[:4]
        batch = call[4] if len(call) > 4 else f"r{i}"
        out.append(
            tracker.observe(
                sid,
                settings,
                tool_name=tool,
                args=args,
                result=result,
                failed=failed,
                error_message=result if failed else "",
                batch=batch,
            )
        )
    return out


# ------------------------------------------------------------------- rules


def test_error_streak():
    t = Tracker()
    got = feed(t, [("a", {"n": i}, ERRORS[i], True) for i in range(3)])
    assert got[:2] == [None, None] and got[2].kind == "error_streak"
    assert "3 tool calls failed in a row" in got[2].reason


def test_success_breaks_the_streak():
    t = Tracker()
    got = feed(
        t,
        [
            ("a", {"n": 1}, ERRORS[0], True),
            ("a", {"n": 2}, ERRORS[1], True),
            ("b", {}, "fine", False),
            ("a", {"n": 3}, ERRORS[2], True),
        ],
    )
    assert got == [None, None, None, None]


def test_same_error_with_other_calls_between():
    t = Tracker()
    err = "ImportError at line {n}: no module named foo"
    calls = []
    for n in (10, 20, 30):
        calls += [("run", {"n": n}, err.format(n=n), True), ("read", {"n": n}, f"ok {n}", False)]
    got = feed(t, calls)
    hits = [d for d in got if d]
    assert hits and hits[0].kind == "same_error"  # line numbers are normalised away
    assert "no module named foo" in hits[0].reason


def test_loop_needs_the_same_result():
    # Polling: same call, changing result, is not a loop (research: OpenHands false positives).
    t = Tracker()
    assert not any(feed(t, [("status", {"job": 1}, f"running {i}%", False) for i in range(5)]))
    t = Tracker()
    got = feed(t, [("status", {"job": 1}, "running", False)] * 3)
    assert got[2].kind == "loop" and "'status' was called 3 times" in got[2].reason


def test_cycle_detection():
    t = Tracker()
    got = feed(t, [("edit", {"f": "x"}, "done", False), ("test", {}, "1 failed", False)] * 2)
    assert got[3].kind == "cycle" and "edit -> test" in got[3].reason


def test_arguments_are_normalised():
    assert normalize_args({"a": 1, "b": 2}) == normalize_args('{"b": 2, "a": 1}')
    t = Tracker()
    got = feed(
        t,
        [
            ("x", {"a": 1, "b": 2}, "r", False),
            ("x", '{"b":2,"a":1}', "r", False),
            ("x", '{ "a": 1, "b": 2 }', "r", False),
        ],
    )
    assert got[2].kind == "loop"


def test_error_normalisation():
    a = normalize_error("Timeout after 30s at 0x7ffe12 (req 9f8e7d6c5b4a)")
    b = normalize_error("timeout after 45s at 0x1a2b (req 0123456789ab)")
    assert a == b


def test_one_batch_counts_once():
    # Identical calls from one model response are one attempt.
    t = Tracker()
    got = feed(t, [("x", {}, "same", False, "batch-1")] * 3)
    assert not any(got)
    got = feed(t, [("x", {}, "same", False, "batch-2"), ("x", {}, "same", False, "batch-3")])
    assert got[1].kind == "loop"


def test_cooldown_and_session_limit():
    t = Tracker()
    s = SuggestSettings(cooldown_calls=5, max_per_session=2)
    got = feed(t, [("x", {}, "same", False)] * 3, s)
    assert got[2] is not None
    assert not any(feed(t, [("x", {}, "same", False)] * 4, s))  # cooling down
    assert feed(t, [("x", {}, "same", False)], s)[0] is not None  # 5th call after the nudge
    t.new_turn(SID)
    assert not any(feed(t, [("x", {}, "same", False)] * 10, s))  # session limit (2) reached


def test_new_turn_resets_counters():
    t = Tracker()
    feed(t, [("a", {"n": i}, "e", True) for i in range(2)])
    t.new_turn(SID)
    assert feed(t, [("a", {"n": 9}, "e2", True)]) == [None]


def test_refused_suggestion_is_not_counted():
    t = Tracker()
    s = SuggestSettings(cooldown_calls=10)
    calls = [("x", {}, "same", False)] * 3
    got = [
        t.observe(
            SID,
            s,
            tool_name=c[0],
            args=c[1],
            result=c[2],
            failed=False,
            error_message="",
            batch=f"b{i}",
            allowed=lambda: False,
        )
        for i, c in enumerate(calls)
    ]
    assert not any(got)
    # Allowed now: no cooldown was started by the refused one.
    assert (
        t.observe(
            SID,
            s,
            tool_name="x",
            args={},
            result="same",
            failed=False,
            error_message="",
            batch="b9",
        )
        is not None
    )


def test_rules_can_be_turned_off():
    t = Tracker()
    s = SuggestSettings(cooldown_calls=0, loop=0, cycles=False)
    assert not any(feed(t, [("x", {}, "same", False)] * 6, s))


# ------------------------------------------------------------------- settings


def test_suggest_settings():
    s = parse_settings({"budget": {"cycle_start_day": 3}}).suggest
    assert s == SuggestSettings()  # mode both, 3/3/3, cooldown 10, 3 per session
    assert parse_settings({"suggest": False}).suggest.mode == "off"
    custom = parse_settings(
        {"suggest": {"mode": "user", "loop": 4, "ignore_tools": ["poll_build"]}}
    ).suggest
    assert (custom.mode, custom.loop, custom.ignore_tools) == ("user", 4, frozenset({"poll_build"}))
    bad = parse_settings({"suggest": {"mode": "loud", "loop": -1}})
    assert bad.suggest.mode == "both" and bad.suggest.loop == 3 and len(bad.warnings) >= 2


# ------------------------------------------------------------------- hook wiring


@pytest.fixture
def hooks(runtime):
    return Hooks(runtime)


def tool_result(hooks, tool="terminal", args=None, result="error: boom", failed=True, n=0, sid=SID):
    return hooks.transform_tool_result(
        tool_name=tool,
        args=args if args is not None else {"cmd": f"make {n}"},
        result=result,
        session_id=sid,
        status="error" if failed else "ok",
        error_message=result if failed else None,
        api_request_id=f"req-{n}",
    )


def three_errors(hooks, sid=SID):
    return [tool_result(hooks, n=i, result=ERRORS[i], sid=sid) for i in range(3)]


def start_turn(hooks, platform="telegram", sid=SID):
    hooks.pre_llm_call(
        session_id=sid,
        conversation_history=[],
        user_message="fix it",
        sender_id="1",
        platform=platform,
    )


def test_hint_appended_to_the_tripping_result(hooks, runtime):
    start_turn(hooks)
    out = three_errors(hooks)
    assert out[:2] == [None, None]
    assert out[2].startswith(ERRORS[2]) and NUDGE_PREFIX in out[2]
    assert "advisor_call_open" in out[2] and "ask the user" in out[2]


def test_user_told_when_the_executor_does_not_call(hooks):
    start_turn(hooks)
    three_errors(hooks)
    final = hooks.transform_llm_output(
        response_text="I could not fix it.", session_id=SID, platform="telegram"
    )
    assert "💡 The agent looked stuck (3 tool calls failed in a row)" in final
    assert "/second_opinion ask" in final  # platform-aware command name
    assert hooks.transform_llm_output(response_text="next", session_id=SID) is None  # once


def test_following_the_hint_clears_the_notice_and_is_recorded(hooks, runtime, tools, fake_llm):
    start_turn(hooks)
    three_errors(hooks)
    fake_llm.script(reply("try this"))
    out = json.loads(tools[OPEN]({"reason": "stuck", "question": "help"}, session_id=SID))
    assert out["reply"] == "try this"
    final = hooks.transform_llm_output(response_text="done", session_id=SID)
    assert "💡" not in final and "📞 Advisor call" in final
    assert runtime.ledger.suggestion_stats(runtime.today(), runtime.today()) == (1, 1)
    status = SecondOpinionCommand(runtime)("status", session_id=SID, platform="cli")
    assert "Auto-suggest, last 7 days: 1 suggestion, 1 led to a call" in status


def test_user_mode_does_not_touch_tool_results(hooks, config):
    config["suggest"] = {"mode": "user"}
    start_turn(hooks)
    assert three_errors(hooks) == [None, None, None]
    assert "💡" in hooks.transform_llm_output(response_text="x", session_id=SID)


def test_executor_mode_does_not_tell_the_user(hooks, config):
    config["suggest"] = {"mode": "executor"}
    start_turn(hooks)
    assert NUDGE_PREFIX in three_errors(hooks)[2]
    assert hooks.transform_llm_output(response_text="x", session_id=SID) is None


def test_off_and_skipped_tools(hooks, config):
    start_turn(hooks)
    for tool in ("advisor_call_say", "tool_search", "tool_describe"):
        assert not any(tool_result(hooks, tool=tool, n=i) for i in range(4))
    config["suggest"] = {"ignore_tools": ["poll_build"]}
    assert not any(tool_result(hooks, tool="poll_build", n=i) for i in range(4))
    config["suggest"] = {"mode": "off"}
    assert three_errors(hooks) == [None, None, None]


def test_no_suggestion_when_a_call_could_not_happen(hooks, runtime, config, tools, fake_llm):
    # A call is already open.
    start_turn(hooks)
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    assert three_errors(hooks)[2] is None
    # Today's allowance is used up.
    start_turn(hooks, sid="s2")
    config["budget"].update(pacing="fixed", per_day_usd=0.000001)
    assert three_errors(hooks, sid="s2")[2] is None
    # cycle_start_day missing.
    config["budget"] = {}
    start_turn(hooks, sid="s3")
    assert three_errors(hooks, sid="s3")[2] is None


def test_unattended_needs_a_grant(hooks, runtime):
    start_turn(hooks, platform="cron")
    assert three_errors(hooks)[2] is None
    start_turn(hooks, platform="cron", sid="s2")
    runtime.manager.grant("s2", line="sonnet", cap_usd=None)
    assert NUDGE_PREFIX in three_errors(hooks, sid="s2")[2]


def test_hook_fails_open(hooks, runtime):
    runtime._settings = lambda: 1 / 0
    assert tool_result(hooks) is None
    assert hooks.transform_tool_result(tool_name="x", result=b"bytes", session_id=SID) is None
    assert hooks.transform_tool_result() is None


def test_numbers_alone_do_not_make_errors_different():
    t = Tracker()
    got = feed(t, [("a", {"n": i}, f"boom at line {i}", True) for i in range(3)])
    assert got[2].kind == "same_error"  # the more specific signal wins over the streak


def test_json_results_stay_valid_json(hooks):
    start_turn(hooks)
    results = [json.dumps({"error": e, "content": ""}) for e in ERRORS]
    out = [tool_result(hooks, n=i, result=r) for i, r in enumerate(results)]
    parsed = json.loads(out[2])  # still valid JSON
    assert parsed["error"] == ERRORS[2] and "advisor_call_open" in parsed["advisor_call_note"]
