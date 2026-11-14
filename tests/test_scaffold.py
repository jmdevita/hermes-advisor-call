"""Phase 0 smoke tests: the fixtures work and Hermes is not required."""

import sys

import pytest


def test_hermes_not_imported():
    # The suite must run without Hermes installed (CI has none), so nothing
    # under test may import Hermes at module level.
    assert not any(m == "hermes_cli" or m.startswith("hermes_cli.") for m in sys.modules)
    assert "agent" not in sys.modules and "plugins" not in sys.modules


def test_fake_llm_scripted_replies(fake_llm):
    fake_llm.script("first", RuntimeError("boom"))
    assert fake_llm.complete([{"role": "user", "content": "hi"}], task="t").text == "first"
    with pytest.raises(RuntimeError):
        fake_llm.complete([], task="t")
    assert [c.kwargs["task"] for c in fake_llm.calls] == ["t", "t"]


def test_fake_ctx_records(fake_ctx):
    fake_ctx.register_hook("pre_tool_call", print)
    fake_ctx.register_auxiliary_task("advisor_call_sonnet", display_name="x")
    assert fake_ctx.hooks["pre_tool_call"] == [print]
    assert "advisor_call_sonnet" in fake_ctx.auxiliary_tasks


def test_fake_clock(clock):
    start = clock.time()
    clock.advance(minutes=31)
    assert clock.time() - start == pytest.approx(31 * 60)
