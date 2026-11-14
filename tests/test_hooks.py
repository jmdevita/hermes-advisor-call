"""U-21 to U-27, U-29, U-36, U-37 (hook part): approval gate, capture,
grant notice, bill footer, session teardown."""

import json

import pytest
from conftest import reply

from advisor_call.hooks import Hooks, sanitize_reason
from advisor_call.tools import OPEN, SAY, TOOLS, make_handler

SID = "session-1"


@pytest.fixture
def hooks(runtime):
    return Hooks(runtime)


@pytest.fixture
def tools(runtime):
    return {name: make_handler(runtime, fn) for name, _, fn, _ in TOOLS}


def gate(hooks, args=None, session_id=SID, tool=OPEN):
    return hooks.pre_tool_call(
        tool_name=tool,
        args=args or {"reason": "stuck", "question": "q"},
        session_id=session_id,
        task_id="t",
        tool_call_id="x",
    )


def test_other_tools_pass(hooks):
    # U-21
    for tool in ("terminal", "read_file", "advisor_call_hangup", "tool_call"):
        assert gate(hooks, tool=tool) is None


def test_approve_directive_and_rule_key(hooks):
    # U-22
    d = gate(hooks)
    assert d["action"] == "approve"
    assert d["rule_key"] == f"advisor_call:sonnet:{SID}"
    assert "Call the sonnet advisor (claude-sonnet-5-5). Reason: stuck" in d["message"]
    assert "Bill cap $" in d["message"] and "today $0.00 of $" in d["message"]


def test_rule_key_differs_per_session(hooks):
    assert gate(hooks)["rule_key"] != gate(hooks, session_id="session-2")["rule_key"]


def test_blocks_when_budget_used_up(hooks, runtime, config):
    # U-23 (day)
    config["budget"].update(pacing="fixed", per_day_usd=0.10)
    runtime.ledger.record_exchange(
        ts=0,
        day=runtime.today(),
        session_id="x",
        call_id="c-x",
        line="sonnet",
        model="m",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.10,
        ok=True,
    )
    d = gate(hooks)
    assert d["action"] == "block"
    assert "used up ($0.10 of $0.10;" in d["message"] and "left this cycle" in d["message"]


def test_blocks_when_cycle_budget_used_up(hooks, runtime, config):
    # U-23 (cycle)
    config["budget"].update(per_month_usd=1.0)
    runtime.ledger.record_exchange(
        ts=0,
        day=runtime.today().replace(day=5),
        session_id="x",
        call_id="c-x",
        line="sonnet",
        model="m",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=1.0,
        ok=True,
    )
    d = gate(hooks)
    assert d["action"] == "block" and "$0.00 left this cycle" in d["message"]


def test_grant_skips_prompt_and_revoke_restores_it(hooks, runtime):
    # U-24
    runtime.manager.grant(SID, line="sonnet", cap_usd=None)
    assert gate(hooks) is None
    runtime.manager.revoke(SID)
    assert gate(hooks)["action"] == "approve"


def test_grant_for_other_line_does_not_apply(hooks, runtime, config):
    config["lines"] = {
        "sonnet": {},
        "opus": {
            "provider": "anthropic",
            "model": "claude-opus-5-5",
            "prices_per_mtok": {"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5},
        },
    }
    runtime.manager.grant(SID, line="sonnet", cap_usd=None)
    # The session grant skips the prompt for any line; only its custom cap is
    # tied to the line it was given for.
    assert gate(hooks) is None
    assert gate(hooks, {"reason": "r", "question": "q", "line": "opus"}) is None


def test_blocks_sender_outside_allowed_list(hooks, runtime, config):
    # U-25
    config["allowed_senders"] = ["111"]
    runtime.context.record_turn(
        SID, conversation_history=[], user_message="hi", sender_id="999", platform="telegram"
    )
    assert gate(hooks)["action"] == "block"
    assert gate(hooks, tool=SAY)["action"] == "block"
    runtime.context.record_turn(
        SID, conversation_history=[], user_message="hi", sender_id="111", platform="telegram"
    )
    assert gate(hooks)["action"] == "approve"


def test_cli_session_without_sender_is_owner(hooks, config):
    config["allowed_senders"] = ["111"]
    assert gate(hooks)["action"] == "approve"


def test_approvals_off_needs_grant(hooks, runtime, approvals):
    # U-26
    approvals["mode"] = False  # YAML `off`
    d = gate(hooks)
    assert d["action"] == "block" and "/phone allow" in d["message"]
    runtime.manager.grant(SID, line="sonnet", cap_usd=None)
    assert gate(hooks) is None


def test_approvals_off_allowed_when_opted_out(hooks, approvals, config):
    approvals["mode"] = "off"
    config["approval"] = {"refuse_when_approvals_off": False}
    assert gate(hooks)["action"] == "approve"  # Hermes will auto-approve; caps still apply


def test_reason_sanitized(hooks):
    # U-27
    rlo, zwsp = chr(0x202E), chr(0x200B)  # right-to-left override, zero-width space
    evil = "line one\nline two\x1b[31m" + rlo + "evil" + zwsp + "x" * 500
    d = gate(hooks, {"reason": evil, "question": "q"})
    reason = d["message"].split("Reason: ", 1)[1].split(" | Bill cap", 1)[0]
    assert "\n" not in reason and "\x1b" not in reason and rlo not in reason
    assert zwsp not in reason and len(reason) <= 300
    assert reason.startswith("line one line two")
    assert sanitize_reason(None) == "(no reason given)"


def test_missing_session_and_misconfig_block(hooks, config):
    assert gate(hooks, session_id=None)["action"] == "block"
    assert gate(hooks, {"reason": "r", "question": "q", "line": "nope"})["action"] == "block"
    config["budget"].pop("cycle_start_day")
    d = gate(hooks)
    # U-37 (hook part)
    assert d["action"] == "block" and "cycle_start_day" in d["message"]


def test_open_call_passes_without_prompt(hooks, tools, fake_llm):
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    assert gate(hooks) is None


def test_ledger_failure_blocks(hooks, runtime):
    from advisor_call.ledger import Ledger

    runtime.ledger = Ledger(lambda: (_ for _ in ()).throw(OSError("gone")))
    d = gate(hooks)
    assert d["action"] == "block" and "cannot be checked" in d["message"]


def test_hook_bug_fails_closed(hooks, runtime):
    runtime._settings = lambda: 1 / 0
    d = gate(hooks)
    assert d["action"] == "block" and "no call was made" in d["message"]


def test_cap_shown_matches_call_cap(hooks, tools, fake_llm):
    # U-36: the cap on the approval prompt is the cap the call gets, and it
    # does not change mid-call.
    shown = gate(hooks)["message"].split("Bill cap $", 1)[1].split(",", 1)[0]
    fake_llm.script(reply(), reply())
    opened = json.loads(tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID))
    assert f"{opened['bill']['cap_usd']:.2f}" == shown
    said = json.loads(tools[SAY]({"message": "m"}, session_id=SID))
    assert said["bill"]["cap_usd"] == opened["bill"]["cap_usd"]


# ------------------------------------------------------------------- capture


def test_pre_llm_call_records_and_notifies_once(hooks, runtime):
    out = hooks.pre_llm_call(
        session_id=SID,
        conversation_history=[],
        user_message="hello",
        sender_id="42",
        platform="telegram",
        is_first_turn=True,
    )
    assert out is None
    assert runtime.context.sender(SID) == ("42", "telegram")
    runtime.manager.grant(SID, line="sonnet", cap_usd=0.4)
    out = hooks.pre_llm_call(session_id=SID, conversation_history=[], user_message="next")
    assert out == {
        "context": "[advisor-call] The user has pre-authorized advisor calls for "
        "this session (line: sonnet, cap $0.40)."
    }
    assert hooks.pre_llm_call(session_id=SID, conversation_history=[], user_message="x") is None


def test_llm_request_capture_never_modifies(hooks, runtime):
    request = {"model": "qwen", "messages": [{"role": "tool", "content": "fresh output"}]}
    before = json.dumps(request)
    assert hooks.llm_request(request=request, session_id=SID, turn_id="t") is None
    assert json.dumps(request) == before
    assert runtime.context.messages(SID) == request["messages"]
    assert hooks.llm_request(request="junk", session_id=SID) is None
    assert hooks.llm_request() is None


# ------------------------------------------------------------------- footer (U-29)


def test_footer_only_after_activity(hooks, tools, fake_llm):
    assert hooks.transform_llm_output(response_text="plain", session_id=SID) is None
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    out = hooks.transform_llm_output(
        response_text="Here is the fix.", session_id=SID, model="qwen", platform="cli"
    )
    first, footer = out.split("\n\n", 1)
    assert first == "Here is the fix."
    assert footer.startswith("📞 Advisor call (sonnet): 1 message · ~$")
    assert "(estimate)" in footer and "still open" in footer and "today $" in footer
    assert hooks.transform_llm_output(response_text="next turn", session_id=SID) is None


def test_footer_after_hangup(hooks, tools, fake_llm):
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    hooks.transform_llm_output(response_text="x", session_id=SID)
    tools["advisor_call_hangup"]({}, session_id=SID)
    out = hooks.transform_llm_output(response_text="done", session_id=SID)
    assert "ended (hung up)" in out


def test_footer_disabled(hooks, tools, fake_llm, config):
    config["bill_footer"] = False
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    assert hooks.transform_llm_output(response_text="quiet", session_id=SID) is None


def test_session_finalize_ends_call(hooks, tools, fake_llm, runtime):
    fake_llm.script(reply())
    tools[OPEN]({"reason": "r", "question": "q"}, session_id=SID)
    runtime.manager.grant(SID, line="sonnet", cap_usd=None)
    hooks.on_session_finalize(session_id=SID, platform="cli", reason="exit")
    call = runtime.manager.latest(SID)
    assert call.end_reason == "session_end" and runtime.manager.grant_for(SID) is None
    hooks.on_session_finalize()  # all kwargs optional
    hooks.on_session_finalize(old_session_id="unknown", new_session_id="n")
