"""U-13 to U-16 (layout, markers, empty replies) and the ask() contract."""

import pytest
from conftest import FakeLlm, reply

from advisor_call.advisor import AdvisorError, ask, build_request
from advisor_call.cost import estimate_tokens
from advisor_call.prompts import ADVISOR_SYSTEM
from advisor_call.settings import parse_settings

SETTINGS = parse_settings({"budget": {"cycle_start_day": 3}})
LINE = SETTINGS.advisor


def conv(n_exchanges: int) -> list[dict[str, str]]:
    """Opening, then alternating replies and executor messages, ending with
    the pending executor message of exchange n."""
    out = [{"role": "user", "content": "<transcript>T</transcript> first question"}]
    for i in range(1, n_exchanges):
        out.append({"role": "assistant", "content": f"reply {i}"})
        out.append({"role": "user", "content": f"executor message {i + 1}"})
    return out


def markers(messages):
    found = []
    for i, m in enumerate(messages):
        if isinstance(m["content"], list):
            for part in m["content"]:
                if "cache_control" in part:
                    found.append((i, part["cache_control"]))
    return found


def test_opening_layout_breakpoint_a():
    # U-13
    req = build_request(conv(1), "5m")
    assert req.messages[0] == {"role": "system", "content": ADVISOR_SYSTEM}
    assert req.messages[1]["role"] == "user"
    assert req.messages[1]["content"][0]["text"].startswith("<transcript>")
    assert markers(req.messages) == [(1, {"type": "ephemeral"})]
    assert req.uncached_est == 0 and req.cached_prefix_est == req.prompt_est


def test_third_exchange_breakpoints_a_and_b_only():
    # U-14
    req = build_request(conv(3), "5m")
    roles = [m["role"] for m in req.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert [i for i, _ in markers(req.messages)] == [1, 4]  # opening and the latest reply
    assert req.messages[-1] == {"role": "user", "content": "executor message 3"}
    assert req.uncached_est == pytest.approx(estimate_tokens("executor message 3"))
    assert req.cached_prefix_est == pytest.approx(req.prompt_est - req.uncached_est)


@pytest.mark.parametrize("n", range(1, 12))
def test_never_more_than_two_breakpoints(n):
    assert len(markers(build_request(conv(n), "1h").messages)) <= 2


def test_cache_off_and_1h():
    # U-15
    off = build_request(conv(3), "off")
    assert markers(off.messages) == []
    assert all(isinstance(m["content"], str) for m in off.messages)
    assert off.uncached_est == off.prompt_est and off.cached_prefix_est == 0
    hour = build_request(conv(2), "1h")
    assert {c["ttl"] for _, c in markers(hour.messages)} == {"1h"}


def test_layout_requires_pending_executor_message():
    with pytest.raises(ValueError):
        build_request([], "5m")
    with pytest.raises(ValueError):
        build_request(conv(2)[:-1], "5m")


def test_ask_passes_task_and_limits():
    llm = FakeLlm().script(reply("ok"))
    out = ask(llm, LINE, SETTINGS, build_request(conv(1), "5m"))
    assert out.text == "ok" and out.usage.input_tokens == 500
    kwargs = llm.calls[0].kwargs
    assert kwargs == {
        "task": "advisor_call_sonnet",
        "max_tokens": 4000,
        "timeout": 180.0,
        "purpose": "advisor-call",
    }
    assert "provider" not in kwargs and "model" not in kwargs  # no trust overrides


def test_empty_reply_is_an_error_with_usage():
    # U-16 (client side): empty text is a failure, but usage is kept for billing.
    llm = FakeLlm().script(reply("   ", input_tokens=10, output_tokens=4000))
    with pytest.raises(AdvisorError) as err:
        ask(llm, LINE, SETTINGS, build_request(conv(1), "5m"))
    assert "empty" in err.value.message and err.value.usage.output_tokens == 4000


def test_fallback_provider_is_rejected_and_not_billable():
    llm = FakeLlm().script(reply("from elsewhere", provider="ollama", input_tokens=7))
    with pytest.raises(AdvisorError) as err:
        ask(llm, LINE, SETTINGS, build_request(conv(1), "5m"))
    assert "fell back" in err.value.message
    assert not err.value.billable and err.value.provider == "ollama"
    assert err.value.usage.input_tokens == 7  # kept for the record


def test_timeout_is_flagged_for_estimate_billing():
    for exc in (TimeoutError("slow"), RuntimeError("Request timed out after 180s")):
        llm = FakeLlm().script(exc)
        with pytest.raises(AdvisorError) as err:
            ask(llm, LINE, SETTINGS, build_request(conv(1), "5m"))
        assert err.value.bill_estimate
    llm = FakeLlm().script(ConnectionError("refused"))
    with pytest.raises(AdvisorError) as err:
        ask(llm, LINE, SETTINGS, build_request(conv(1), "5m"))
    assert not err.value.bill_estimate


def test_exception_becomes_advisor_error_with_hint():
    llm = FakeLlm().script(RuntimeError("Your credit balance is too low"))
    with pytest.raises(AdvisorError) as err:
        ask(llm, LINE, SETTINGS, build_request(conv(1), "5m"))
    assert "RuntimeError" in err.value.message and "Console" in err.value.message


def test_org_scoped_key_error_gets_a_hint():
    msg = (
        "400 invalid_request_error: This API key is not scoped to a workspace, so this "
        "request must include the anthropic-workspace-id header"
    )
    llm = FakeLlm().script(RuntimeError(msg))
    with pytest.raises(AdvisorError) as err:
        ask(llm, LINE, SETTINGS, build_request(conv(1), "5m"))
    assert "Scope set to a workspace" in err.value.message
