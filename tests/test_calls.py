"""U-6 to U-11: call state, caps through preflight, idle, isolation."""

import threading

import pytest

from advisor_call.calls import CallError, CallManager
from advisor_call.cost import Usage


def ok_preflight(est=0.01):
    return lambda call, messages, elsewhere: ("", est)


def make(clock, **kw):
    ended = []
    mgr = CallManager(clock=clock.time, on_call_end=ended.append, **kw)
    return mgr, ended


def exchange(mgr, sid, text="q", *, reply="a", cost=0.01, preflight=None, call_id=None):
    res = mgr.reserve(sid, call_id, text, preflight or ok_preflight())
    return mgr.settle(res, reply=reply, cost_usd=cost, usage=Usage(10, 5), ok=True)


def test_open_and_exchange_appends_on_success(clock):
    mgr, _ = make(clock)
    call, created = mgr.open("s1", line="sonnet", cap_usd=0.5)
    assert created and call.is_open and call.call_id.startswith("c-")
    exchange(mgr, "s1", "first question", reply="advice")
    assert call.messages == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "advice"},
    ]
    assert call.exchanges == 1 and call.cost_usd == pytest.approx(0.01)


def test_failed_exchange_leaves_conversation_but_bills(clock):
    mgr, _ = make(clock)
    call, _ = mgr.open("s1", line="sonnet", cap_usd=0.5)
    res = mgr.reserve("s1", None, "q", ok_preflight())
    mgr.settle(res, reply=None, cost_usd=0.02, usage=Usage(10, 0), ok=False)
    assert call.messages == [] and call.exchanges == 1 and call.cost_usd == pytest.approx(0.02)
    res = mgr.reserve("s1", None, "q", ok_preflight())
    mgr.settle(res, reply=None, cost_usd=0.0, usage=Usage(), ok=False)
    assert call.exchanges == 1  # an unbilled failure does not count


def test_bill_cap_preflight_ends_call_and_sends_nothing(clock):
    # U-6
    mgr, ended = make(clock)
    call, _ = mgr.open("s1", line="sonnet", cap_usd=0.05)
    exchange(mgr, "s1", cost=0.04)

    def preflight(c, messages, elsewhere):
        return ("bill_cap" if c.cost_usd + 0.02 > c.cap_usd else "", 0.02)

    with pytest.raises(CallError) as err:
        mgr.reserve("s1", None, "again", preflight)
    assert err.value.code == "ended" and call.end_reason == "bill_cap"
    assert not call.is_open and ended == [call]
    assert len(call.messages) == 2  # the refused message was never added


def test_cap_reached_after_reply_ends_call(clock):
    mgr, ended = make(clock)
    call, _ = mgr.open("s1", line="sonnet", cap_usd=0.05)
    settled = exchange(mgr, "s1", cost=0.06)
    assert settled.ended_now == "bill_cap" and not call.is_open
    assert call.messages[-1]["content"] == "a"  # the reply is still delivered
    assert ended == [call]


def test_day_cap_across_two_calls(clock):
    # U-7: preflight sees spend from an earlier call the same day.
    mgr, _ = make(clock)
    spent = {"today": 0.0}
    allowance = 0.05

    def preflight(c, messages, elsewhere):
        est = 0.02
        return ("day_cap" if spent["today"] + elsewhere + est > allowance else "", est)

    for sid in ("s1", "s2"):
        mgr.open(sid, line="sonnet", cap_usd=0.5)
    res = mgr.reserve("s1", None, "q", preflight)
    mgr.settle(res, reply="a", cost_usd=0.02, usage=Usage(), ok=True)
    spent["today"] += 0.02
    mgr.end("s1", "hangup")
    res = mgr.reserve("s2", None, "q", preflight)
    mgr.settle(res, reply="a", cost_usd=0.02, usage=Usage(), ok=True)
    spent["today"] += 0.02
    with pytest.raises(CallError):
        mgr.reserve("s2", None, "q", preflight)
    assert mgr.latest("s2").end_reason == "day_cap"


def test_in_flight_reservations_count_against_others(clock):
    mgr, _ = make(clock)
    for sid in ("s1", "s2"):
        mgr.open(sid, line="sonnet", cap_usd=0.5)
    seen = []

    def preflight(c, messages, elsewhere):
        seen.append(elsewhere)
        return "", 0.03

    r1 = mgr.reserve("s1", None, "q", preflight)
    mgr.reserve("s2", None, "q", preflight)
    assert seen == [0.0, pytest.approx(0.03)]
    mgr.release(r1)
    assert not r1.call.in_flight


def test_turn_limit_via_preflight(clock):
    # U-8: the manager counts exchanges; preflight compares with the limit.
    mgr, _ = make(clock)
    call, _ = mgr.open("s1", line="sonnet", cap_usd=5)

    def preflight(c, messages, elsewhere):
        return ("turn_limit" if c.exchanges >= 3 else "", 0.01)

    for _ in range(3):
        exchange(mgr, "s1", preflight=preflight)
    with pytest.raises(CallError):
        mgr.reserve("s1", None, "q", preflight)
    assert call.end_reason == "turn_limit" and call.exchanges == 3


def test_idle_expiry(clock):
    # U-9
    mgr, ended = make(clock)
    mgr.idle_timeout_s = 30 * 60
    call, _ = mgr.open("s1", line="sonnet", cap_usd=0.5)
    clock.advance(minutes=29)
    assert mgr.get("s1") is call
    clock.advance(minutes=31)
    with pytest.raises(CallError) as err:
        mgr.get("s1")
    assert err.value.code == "ended" and "idle" in err.value.message
    assert call.end_reason == "idle" and ended == [call]
    new, created = mgr.open("s1", line="sonnet", cap_usd=0.5)
    assert created and new is not call


def test_in_flight_call_does_not_idle_out(clock):
    mgr, _ = make(clock)
    mgr.idle_timeout_s = 60
    call, _ = mgr.open("s1", line="sonnet", cap_usd=0.5)
    res = mgr.reserve("s1", None, "q", ok_preflight())
    clock.advance(minutes=10)
    assert mgr.get("s1") is call
    mgr.settle(res, reply="a", cost_usd=0.01, usage=Usage(), ok=True)
    assert call.is_open


def test_one_open_call_per_session(clock):
    # U-10
    mgr, _ = make(clock)
    first, created1 = mgr.open("s1", line="sonnet", cap_usd=0.5)
    second, created2 = mgr.open("s1", line="sonnet", cap_usd=0.9)
    assert created1 and not created2 and second is first and first.cap_usd == 0.5


def test_busy_while_in_flight(clock):
    mgr, _ = make(clock)
    mgr.open("s1", line="sonnet", cap_usd=0.5)
    mgr.reserve("s1", None, "q", ok_preflight())
    with pytest.raises(CallError) as err:
        mgr.reserve("s1", None, "q2", ok_preflight())
    assert err.value.code == "busy"


def test_session_isolation(clock):
    # U-11
    mgr, _ = make(clock)
    a, _ = mgr.open("session-A", line="sonnet", cap_usd=0.5)
    exchange(mgr, "session-A")
    # B has no call, and cannot reach A's call even by id.
    for fn in (
        lambda: mgr.get("session-B"),
        lambda: mgr.get("session-B", a.call_id),
        lambda: mgr.reserve("session-B", a.call_id, "x", ok_preflight()),
        lambda: mgr.end("session-B", "hangup", a.call_id),
        lambda: mgr.end("session-B", "hangup"),
    ):
        with pytest.raises(CallError) as err:
            fn()
        assert err.value.code in ("no_call", "wrong_call")
        assert err.value.call is None  # nothing about A leaks out
    assert mgr.latest("session-B") is None
    assert a.is_open and len(a.messages) == 2
    # A wrong id in A's own session does not fall back to A's open call either.
    with pytest.raises(CallError) as err:
        mgr.get("session-A", "c-deadbeef")
    assert err.value.code == "wrong_call"


def test_missing_session_id_is_an_error(clock):
    mgr, _ = make(clock)
    for sid in (None, ""):
        with pytest.raises(CallError) as err:
            mgr.open(sid, line="sonnet", cap_usd=0.5)
        assert err.value.code == "no_session"


def test_user_hangup_mid_flight_still_records_cost(clock):
    mgr, ended = make(clock)
    call, _ = mgr.open("s1", line="sonnet", cap_usd=0.5)
    res = mgr.reserve("s1", None, "q", ok_preflight())
    mgr.end("s1", "user_hangup")
    mgr.settle(res, reply="late", cost_usd=0.03, usage=Usage(), ok=True)
    assert call.cost_usd == pytest.approx(0.03) and call.end_reason == "user_hangup"
    assert ended == [call, call]  # persisted at hangup, then again with the final cost
    with pytest.raises(CallError) as err:
        mgr.reserve("s1", None, "q", ok_preflight())
    assert "user ended" in err.value.message


def test_lru_bound(clock):
    mgr, ended = make(clock, max_sessions=3)
    calls = [mgr.open(f"s{i}", line="sonnet", cap_usd=0.5)[0] for i in range(4)]
    assert calls[0].end_reason == "session_end" and ended == [calls[0]]
    assert all(c.is_open for c in calls[1:])


def test_grants_and_notice(clock):
    mgr, _ = make(clock)
    assert mgr.grant_for("s1") is None
    mgr.grant("s1", line="sonnet", cap_usd=0.4)
    assert mgr.grant_for("s1").cap_usd == 0.4
    assert mgr.take_grant_notice("s1") is not None
    assert mgr.take_grant_notice("s1") is None  # only once
    assert mgr.grant_for("s2") is None
    assert mgr.revoke("s1") and mgr.grant_for("s1") is None


def test_activity_flag(clock):
    mgr, _ = make(clock)
    assert not mgr.take_activity("s1")
    mgr.mark_activity("s1")
    assert mgr.take_activity("s1") and not mgr.take_activity("s1")


def test_end_session_clears_everything(clock):
    mgr, ended = make(clock)
    call, _ = mgr.open("s1", line="sonnet", cap_usd=0.5)
    mgr.grant("s1", line="sonnet", cap_usd=None)
    mgr.mark_activity("s1")
    assert mgr.end_session("s1") is call
    assert call.end_reason == "session_end" and mgr.grant_for("s1") is None
    assert not mgr.take_activity("s1")


def test_concurrent_opens_create_one_call(clock):
    mgr, _ = make(clock)
    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(mgr.open("s1", line="sonnet", cap_usd=0.5))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(created for _, created in results) == 1
    assert len({id(call) for call, _ in results}) == 1


def test_on_call_end_failure_is_contained(clock):
    def boom(call):
        raise RuntimeError("db gone")

    mgr = CallManager(clock=clock.time, on_call_end=boom)
    mgr.open("s1", line="sonnet", cap_usd=0.5)
    assert mgr.end("s1", "hangup").end_reason == "hangup"
